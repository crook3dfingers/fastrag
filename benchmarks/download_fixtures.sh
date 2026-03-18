#!/usr/bin/env bash
# Download and generate benchmark fixtures.
# Idempotent — skips files that already exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES="$SCRIPT_DIR/fixtures"

UA="Mozilla/5.0 (compatible; fastrag-bench/1.0)"

fetch() {
    local dest="$1" url="$2"
    if [ -f "$dest" ]; then
        echo "  skip $dest (exists)"
        return
    fi
    echo "  fetch $dest"
    curl -fsSL --retry 3 -A "$UA" -o "$dest" "$url"
}

# ── PDF ──────────────────────────────────────────────────────────────────────
mkdir -p "$FIXTURES/pdf"
echo "==> PDF fixtures"
fetch "$FIXTURES/pdf/small.pdf"  "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
fetch "$FIXTURES/pdf/medium.pdf" "https://arxiv.org/pdf/1706.03762"
fetch "$FIXTURES/pdf/dense_text.pdf"  "https://www.congress.gov/117/plaws/publ167/PLAW-117publ167.pdf"

# ── HTML ─────────────────────────────────────────────────────────────────────
mkdir -p "$FIXTURES/html"
echo "==> HTML fixtures"
fetch "$FIXTURES/html/small.html"  "https://example.com"
fetch "$FIXTURES/html/medium.html" "https://en.wikipedia.org/wiki/Rust_(programming_language)"
fetch "$FIXTURES/html/large.html"  "https://html.spec.whatwg.org/multipage/parsing.html"

# ── Markdown ─────────────────────────────────────────────────────────────────
mkdir -p "$FIXTURES/markdown"
echo "==> Markdown fixtures"
fetch "$FIXTURES/markdown/small.md"  "https://raw.githubusercontent.com/rust-lang/rust/master/README.md"
fetch "$FIXTURES/markdown/medium.md" "https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md"

# ── CSV ──────────────────────────────────────────────────────────────────────
mkdir -p "$FIXTURES/csv"
echo "==> CSV fixtures"
fetch "$FIXTURES/csv/small.csv" "https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv"

if [ ! -f "$FIXTURES/csv/medium.csv" ]; then
    echo "  generate medium.csv (10K rows)"
    python3 -c "
import csv, random, string, sys
random.seed(42)
cols = [f'col_{i}' for i in range(10)]
w = csv.writer(sys.stdout)
w.writerow(cols)
for _ in range(10_000):
    w.writerow([random.randint(0,100_000) for _ in cols])
" > "$FIXTURES/csv/medium.csv"
fi

if [ ! -f "$FIXTURES/csv/large.csv" ]; then
    echo "  generate large.csv (100K rows)"
    python3 -c "
import csv, random, sys
random.seed(42)
cols = [f'col_{i}' for i in range(10)]
w = csv.writer(sys.stdout)
w.writerow(cols)
for _ in range(100_000):
    w.writerow([random.randint(0,100_000) for _ in cols])
" > "$FIXTURES/csv/large.csv"
fi

# ── Text ─────────────────────────────────────────────────────────────────────
mkdir -p "$FIXTURES/text"
echo "==> Text fixtures"
fetch "$FIXTURES/text/small.txt" "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
fetch "$FIXTURES/text/large.txt" "https://www.gutenberg.org/cache/epub/100/pg100.txt"

echo ""
echo "All fixtures ready in $FIXTURES/"
ls -lhR "$FIXTURES/" | head -60
