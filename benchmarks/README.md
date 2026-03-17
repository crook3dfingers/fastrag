# Benchmarks

Reproducible benchmark suite comparing fastrag against Docling (IBM) and Unstructured across shared formats (PDF, HTML), plus fastrag-only formats (Markdown, CSV, Text).

## Setup

### Install fastrag

```bash
cargo build --release
```

### Install competitors (optional)

```bash
pip install docling
pip install "unstructured[all-docs]"
```

### Download fixtures

```bash
bash benchmarks/download_fixtures.sh
```

## Running benchmarks

```bash
# Full run (all installed tools, 5 iterations + 1 warmup)
python benchmarks/bench.py

# Quick smoke test (fastrag only, 2 iterations)
python benchmarks/bench.py --iterations 2 --warmup 0 --tools fastrag

# Specific formats
python benchmarks/bench.py --formats pdf html

# Print markdown table only
python benchmarks/bench.py --table
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations N` | 5 | Measured iterations per (tool, file) pair |
| `--warmup N` | 1 | Warmup runs discarded from stats |
| `--tools` | all installed | Filter tools: `fastrag`, `docling`, `unstructured` |
| `--formats` | all | Filter formats: `pdf`, `html`, `markdown`, `csv`, `text` |
| `--table` | off | Print only the markdown summary table |
| `--skip-build` | off | Skip `cargo build --release` |
| `--skip-download` | off | Skip fixture download |

## Methodology

Each (tool, file) pair is run N times (default 5) after a warmup run. Every invocation is wrapped with `/usr/bin/time -v` to capture:

- **Wall-clock time** — primary speed metric
- **Peak RSS (KB)** — memory efficiency
- **Throughput (MB/s)** — file size divided by wall time

Results report the **median** across iterations to reduce noise from OS scheduling and I/O caching.

### Test documents

**Shared formats** (competitive comparison):

| Label | Source | ~Size |
|-------|--------|-------|
| `pdf/small.pdf` | W3C dummy PDF | 13 KB |
| `pdf/medium.pdf` | "Attention Is All You Need" (arXiv) | 2 MB |
| `pdf/large.pdf` | US Public Law 117-167 | 3 MB |
| `html/small.html` | example.com | 1 KB |
| `html/medium.html` | Wikipedia: Rust (programming language) | 500 KB |
| `html/large.html` | HTML spec parsing section | 2 MB |

**fastrag-only formats** (throughput showcase):

| Label | Source | ~Size |
|-------|--------|-------|
| `markdown/small.md` | Rust lang README | ~10 KB |
| `markdown/medium.md` | sindresorhus/awesome | ~30 KB |
| `csv/small.csv` | airtravel.csv | <1 KB |
| `csv/medium.csv` | Generated: 10K rows × 10 cols | ~600 KB |
| `csv/large.csv` | Generated: 100K rows × 10 cols | ~6 MB |
| `text/small.txt` | Project Gutenberg: Sherlock Holmes | ~600 KB |
| `text/large.txt` | Project Gutenberg: Complete Shakespeare | ~5 MB |

## Fairness notes

- Docling and Unstructured are Python tools. Python process startup adds ~50-100ms of overhead that Rust binaries do not have. This is part of real-world performance but worth noting.
- Docling and Unstructured perform deeper semantic analysis (OCR, layout detection) on PDFs than fastrag currently does. The comparison measures raw parsing speed, not feature parity.
- All tools are run as subprocesses from the same machine. No network calls are involved.

## Results

Results from running `python benchmarks/bench.py` are printed to stdout as markdown tables and saved as JSON in `benchmarks/results/`.

Run the benchmarks on your own hardware for accurate numbers — results vary by machine.
