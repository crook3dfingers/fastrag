#!/usr/bin/env python3
"""
Benchmark runner for fastrag vs Docling vs Unstructured.

Measures wall-clock time and peak RSS via /usr/bin/time -v,
then prints a markdown summary table.

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --iterations 3 --tools fastrag docling
    python benchmarks/bench.py --formats pdf html --table
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIXTURES_DIR = SCRIPT_DIR / "fixtures"
RESULTS_DIR = SCRIPT_DIR / "results"
VENV_BIN = SCRIPT_DIR / ".venv" / "bin"

# ── Tool definitions ─────────────────────────────────────────────────────────


def _venv_or_path(name: str) -> str:
    """Return the venv binary path if it exists, otherwise the bare name for PATH lookup."""
    venv_path = VENV_BIN / name
    if venv_path.is_file():
        return str(venv_path)
    return name


TOOLS = {
    "fastrag": {
        "check": str(PROJECT_ROOT / "target" / "release" / "fastrag"),
        "cmd": [str(PROJECT_ROOT / "target" / "release" / "fastrag"), "parse", "{file}", "-f", "json", "-o", "{outdir}"],
        "formats": ["pdf", "html", "markdown", "csv", "text", "xml", "xlsx", "docx", "pptx"],
    },
    "docling": {
        "check": _venv_or_path("docling"),
        "cmd": [_venv_or_path("docling"), "--to", "json", "--output", "{outdir}", "{file}"],
        "formats": ["pdf", "html"],
    },
    "unstructured": {
        "check": _venv_or_path("unstructured-ingest"),
        "cmd": [_venv_or_path("unstructured-ingest"), "local", "--input-path", "{file}", "--output-dir", "{outdir}"],
        "formats": ["pdf", "html"],
    },
}

# ── Fixture manifest ─────────────────────────────────────────────────────────

FIXTURES = {
    "pdf": ["small.pdf", "medium.pdf", "large.pdf"],
    "html": ["small.html", "medium.html", "large.html"],
    "markdown": ["small.md", "medium.md"],
    "csv": ["small.csv", "medium.csv", "large.csv"],
    "text": ["small.txt", "large.txt"],
    "xml": ["small.xml", "medium.xml", "large.xml"],
    "xlsx": ["small.xlsx", "medium.xlsx", "large.xlsx"],
    "docx": ["small.docx", "medium.docx", "large.docx"],
    "pptx": ["small.pptx", "medium.pptx", "large.pptx"],
}


def is_tool_available(tool_name: str) -> bool:
    check = TOOLS[tool_name]["check"]
    if os.path.isabs(check):
        return os.path.isfile(check) and os.access(check, os.X_OK)
    return shutil.which(check) is not None


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def parse_gnu_time(stderr: str) -> dict:
    """Extract wall-clock time and peak RSS from /usr/bin/time -v output."""
    wall = None
    rss = None

    for line in stderr.splitlines():
        line = line.strip()
        if "Elapsed (wall clock) time" in line:
            # Try h:mm:ss or h:mm:ss.ss first (3 colon-separated groups)
            match = re.search(r"(\d+):(\d+):(\d+(?:\.\d+)?)", line)
            if match:
                h, m, s = match.groups()
                wall = int(h) * 3600 + int(m) * 60 + float(s)
            else:
                # Try m:ss.cc format (e.g. "0:00.00", "1:23.45")
                match = re.search(r"(\d+):(\d+\.\d+)", line)
                if match:
                    m, s = match.groups()
                    wall = int(m) * 60 + float(s)
        if "Maximum resident set size" in line:
            match = re.search(r"(\d+)", line)
            if match:
                rss = int(match.group(1))  # in KB

    return {"wall_seconds": wall, "peak_rss_kb": rss}


def run_benchmark(tool_name: str, filepath: Path, outdir: Path) -> dict | None:
    """Run a single benchmark iteration using /usr/bin/time -v."""
    cmd_template = TOOLS[tool_name]["cmd"]
    cmd = [
        c.replace("{file}", str(filepath)).replace("{outdir}", str(outdir))
        for c in cmd_template
    ]

    time_cmd = ["/usr/bin/time", "-v"] + cmd

    try:
        result = subprocess.run(
            time_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per run
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "wall_seconds": 300, "peak_rss_kb": None}
    except Exception as e:
        return {"error": str(e), "wall_seconds": None, "peak_rss_kb": None}

    parsed = parse_gnu_time(result.stderr)

    if result.returncode != 0 and parsed["wall_seconds"] is None:
        return {
            "error": f"exit code {result.returncode}",
            "wall_seconds": None,
            "peak_rss_kb": None,
            "stderr": result.stderr[-500:],
        }

    return parsed


def aggregate(measurements: list[dict]) -> dict:
    """Compute median, mean, stddev from a list of measurement dicts."""
    times = [m["wall_seconds"] for m in measurements if m.get("wall_seconds") is not None]
    rss_vals = [m["peak_rss_kb"] for m in measurements if m.get("peak_rss_kb") is not None]

    result = {}
    if times:
        result["median_time"] = statistics.median(times)
        result["mean_time"] = statistics.mean(times)
        result["stddev_time"] = statistics.stdev(times) if len(times) > 1 else 0.0
        result["min_time"] = min(times)
        result["max_time"] = max(times)
    if rss_vals:
        result["median_rss_kb"] = statistics.median(rss_vals)
    return result


def build_fastrag():
    """Build fastrag release binary."""
    print("Building fastrag (release)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  cargo build failed:\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    print("  done.")


def download_fixtures():
    """Run the download script if fixtures are missing."""
    script = SCRIPT_DIR / "download_fixtures.sh"
    if not any(FIXTURES_DIR.rglob("*.*")):
        print("Downloading fixtures...")
        subprocess.run(["bash", str(script)], check=True)


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds == 0.0:
        return "<10ms"
    if seconds < 0.01:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 1:
        return f"{seconds:.3f}s"
    return f"{seconds:.2f}s"


def fmt_speedup(fastrag_time: float | None, other_time: float | None) -> str:
    if fastrag_time is None or other_time is None:
        return "—"
    # Use 5ms floor for sub-resolution times (< 10ms from /usr/bin/time)
    effective = max(fastrag_time, 0.005)
    ratio = other_time / effective
    if ratio < 1.5:
        return "~1x"
    return f"{ratio:.0f}x"


def main():
    parser = argparse.ArgumentParser(description="Benchmark fastrag vs competitors")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Iterations per benchmark (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs to discard (default: 1)")
    parser.add_argument("--tools", nargs="+", choices=list(TOOLS.keys()), default=None, help="Tools to benchmark")
    parser.add_argument("--formats", nargs="+", choices=list(FIXTURES.keys()), default=None, help="Formats to benchmark")
    parser.add_argument("--table", action="store_true", help="Print only the markdown table")
    parser.add_argument("--skip-build", action="store_true", help="Skip cargo build")
    parser.add_argument("--skip-download", action="store_true", help="Skip fixture download")
    args = parser.parse_args()

    # Resolve tools
    tool_names = args.tools or list(TOOLS.keys())
    available_tools = []
    for t in tool_names:
        if t == "fastrag" and not args.skip_build:
            build_fastrag()
        if is_tool_available(t):
            available_tools.append(t)
        else:
            print(f"  {t}: not found, skipping")

    if not available_tools:
        print("No tools available. Install at least one tool to benchmark.", file=sys.stderr)
        sys.exit(1)

    # Resolve formats
    formats = args.formats or list(FIXTURES.keys())

    # Download fixtures
    if not args.skip_download:
        download_fixtures()

    # Verify fixtures exist
    test_files = []
    for fmt in formats:
        for filename in FIXTURES.get(fmt, []):
            fpath = FIXTURES_DIR / fmt / filename
            if fpath.exists():
                test_files.append((fmt, filename, fpath))
            else:
                print(f"  warning: {fpath} not found, skipping")

    if not test_files:
        print("No test files found. Run download_fixtures.sh first.", file=sys.stderr)
        sys.exit(1)

    total_runs = len(available_tools) * len(test_files) * (args.warmup + args.iterations)
    print(f"\nBenchmarking {len(available_tools)} tools × {len(test_files)} files × {args.warmup + args.iterations} runs = {total_runs} total runs\n")

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results = {}  # key: (tool, fmt, filename) -> aggregated stats

    for fmt, filename, fpath in test_files:
        fsize = file_size_bytes(fpath)
        label = f"{fmt}/{filename}"

        for tool_name in available_tools:
            if fmt not in TOOLS[tool_name]["formats"]:
                continue

            print(f"  {tool_name:15s} | {label:30s} | ", end="", flush=True)

            measurements = []
            for i in range(args.warmup + args.iterations):
                with tempfile.TemporaryDirectory() as tmpdir:
                    result = run_benchmark(tool_name, fpath, Path(tmpdir))
                    if i >= args.warmup:
                        measurements.append(result)

                    if result.get("error"):
                        print(f"E", end="", flush=True)
                    else:
                        print(f".", end="", flush=True)

            stats = aggregate(measurements)
            stats["file_size"] = fsize
            stats["file_label"] = label
            stats["tool"] = tool_name
            stats["measurements"] = measurements
            all_results[(tool_name, fmt, filename)] = stats

            med = fmt_time(stats.get("median_time"))
            rss = f"{stats.get('median_rss_kb', 0) / 1024:.1f} MB" if stats.get("median_rss_kb") else "—"
            print(f" median={med}, RSS={rss}")

    # ── Build markdown table ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")

    # Shared formats table (competitive comparison)
    shared_formats = [f for f in formats if f in ("pdf", "html")]
    if shared_formats:
        print("### Shared Formats (fastrag vs Docling vs Unstructured)\n")
        header = "| Document | Size | fastrag | Docling | Unstructured | vs Docling | vs Unstructured |"
        sep = "|----------|------|---------|---------|--------------|------------|-----------------|"
        print(header)
        print(sep)

        for fmt in shared_formats:
            for filename in FIXTURES.get(fmt, []):
                fpath = FIXTURES_DIR / fmt / filename
                if not fpath.exists():
                    continue
                label = f"{fmt}/{filename}"
                fsize = human_size(file_size_bytes(fpath))

                ft = all_results.get(("fastrag", fmt, filename), {}).get("median_time")
                dt = all_results.get(("docling", fmt, filename), {}).get("median_time")
                ut = all_results.get(("unstructured", fmt, filename), {}).get("median_time")

                row = (
                    f"| {label} | {fsize} "
                    f"| {fmt_time(ft)} | {fmt_time(dt)} | {fmt_time(ut)} "
                    f"| {fmt_speedup(ft, dt)} | {fmt_speedup(ft, ut)} |"
                )
                print(row)
        print()

    # fastrag-only formats table
    fastrag_formats = [f for f in formats if f in ("markdown", "csv", "text", "xml", "xlsx", "docx", "pptx")]
    if fastrag_formats and "fastrag" in available_tools:
        print("### fastrag-Only Formats (Throughput)\n")
        header = "| Document | Size | Time | Throughput (MB/s) | Peak RSS |"
        sep = "|----------|------|------|-------------------|----------|"
        print(header)
        print(sep)

        for fmt in fastrag_formats:
            for filename in FIXTURES.get(fmt, []):
                fpath = FIXTURES_DIR / fmt / filename
                if not fpath.exists():
                    continue
                label = f"{fmt}/{filename}"
                stats = all_results.get(("fastrag", fmt, filename), {})
                fsize_bytes = file_size_bytes(fpath)
                fsize = human_size(fsize_bytes)
                med = stats.get("median_time")
                throughput = f"{(fsize_bytes / 1_000_000) / med:.1f}" if med and med > 0 else "—"
                rss = f"{stats.get('median_rss_kb', 0) / 1024:.1f} MB" if stats.get("median_rss_kb") else "—"

                print(f"| {label} | {fsize} | {fmt_time(med)} | {throughput} | {rss} |")
        print()

    # ── Save raw results ─────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = RESULTS_DIR / f"bench_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for key, val in all_results.items():
        tool, fmt, filename = key
        skey = f"{tool}/{fmt}/{filename}"
        serializable[skey] = {k: v for k, v in val.items() if k != "measurements"}
        serializable[skey]["measurements"] = [
            {k: v for k, v in m.items() if k != "stderr"} for m in val.get("measurements", [])
        ]

    raw = {
        "timestamp": timestamp,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "tools": available_tools,
        "results": serializable,
    }

    with open(outfile, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw results saved to {outfile}")


if __name__ == "__main__":
    main()
