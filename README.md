# FastRAG

**100x faster document parsing for AI/RAG pipelines.**

FastRAG is a Rust-based, single-binary document parser that replaces Python's `unstructured` with better performance and zero runtime dependencies.

## Features

- **Blazing fast** — native Rust performance, no Python/Java/Tika overhead
- **Single binary** — no runtime dependencies to install
- **Multiple formats** — PDF, HTML, Markdown, CSV, XML, DOCX, XLSX, PPTX, EPUB, RTF, Email (EML), plain text
- **Structured output** — Markdown, JSON, JSONL, HTML, or plain text with rich metadata
- **Language detection** — document-level and per-element language identification
- **Parallel processing** — batch parse entire directories with configurable workers
- **PDF intelligence** — optional table detection, image extraction, form field extraction, footnote extraction, multi-column reading order, and OCR for scanned pages
- **Corpus retrieval** — semantic indexing, corpus queries, metadata summaries, and an HTTP query server
- **Footnote extraction** — detects footnotes/endnotes in HTML documents and in PDF pages
- **Multi-column PDF layout** — reorders text from multi-column PDF layouts into correct left-to-right reading order (feature flag: `pdf-column-detect`)
- **Streaming output** — `--stream` flag emits elements incrementally as JSONL for low-latency pipelines

## Installation

```bash
cargo install fastrag-cli
```

Or build from source:

```bash
git clone https://github.com/crook3dfingers/fastrag.git
cd fastrag
cargo build --release
```

## Usage

### CLI

```bash
# Parse a single file to markdown (default)
fastrag parse document.pdf

# Parse to JSON
fastrag parse document.pdf --format json

# Parse to JSONL (one JSON object per element, one per line)
fastrag parse document.pdf --format jsonl

# Detect language (document-level and per-element)
fastrag parse document.pdf --detect-language

# Parse an entire directory with 8 workers
fastrag parse ./documents/ --output ./parsed/ -j 8

# List supported formats
fastrag formats
```

### Cargo.toml

```rust
use fastrag::parse;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let doc = parse("document.pdf")?;

    // Structured markdown
    println!("{}", doc.to_markdown());

    // JSON with metadata
    println!("{}", doc.to_json()?);

    // Access elements directly
    for element in &doc.elements {
        println!("{:?}: {}", element.kind, element.text);
    }

    Ok(())
}
```

## Supported Formats

| Format   | Status | Crate |
|----------|--------|-------|
| PDF      | v0.1   | `fastrag-pdf` |
| HTML     | v0.1   | `fastrag-html` |
| Markdown | v0.1   | `fastrag-markdown` |
| CSV      | v0.1   | `fastrag-csv` |
| Text     | v0.1   | `fastrag-text` |
| XML      | v0.1   | `fastrag-xml` |
| DOCX     | v0.1   | `fastrag-docx` |
| XLSX     | v0.1   | `fastrag-xlsx` |
| PPTX     | v0.1   | `fastrag-pptx` |
| EPUB     | v0.1   | `fastrag-epub` |
| RTF      | v0.1   | `fastrag-rtf` |
| Email    | v0.1   | `fastrag-email` |

## Output Formats

FastRAG renders parsed documents in five output formats:

| Format | Flag | Extension | Description |
|--------|------|-----------|-------------|
| Markdown | `--format markdown` | `.md` | Structured markdown with headings, code fences, tables (default) |
| JSON | `--format json` | `.json` | Full document as pretty-printed JSON with metadata and elements |
| JSONL | `--format jsonl` | `.jsonl` | One JSON object per element, one per line — suited for streaming pipelines |
| Plain text | `--format text` | `.txt` | Concatenated element text with no formatting |
| HTML | `--format html` | `.html` | Semantic HTML with proper tags for each element kind |

## Language Detection

FastRAG detects document language using the `whatlang` crate (feature-gated behind `language-detection`, enabled by default).

With `--detect-language`:
- **Document-level**: stores ISO 639-1 code and confidence in document metadata
- **Per-element**: detects language for individual Paragraph, Title, Heading, BlockQuote, and ListItem elements (text ≥ 20 chars), storing `language` and `language_confidence` in element attributes

Useful for multilingual documents where sections may be in different languages.

## PDF Feature Flags

The PDF parser supports optional capabilities via feature flags:

| Feature | Flag | Dependencies | Description |
|---------|------|-------------|-------------|
| Table detection | `pdf-table-detect` | None | Detects tables from text positions, outputs markdown tables |
| Image extraction | `pdf-images` | None | Extracts embedded images with chart/figure classification |
| Form fields | `pdf-forms` | None | Extracts interactive form fields (name, type, value) from AcroForm dictionaries |
| Footnote extraction | `pdf-footnotes` | `pdf-table-detect` | Detects footnotes in bottom 15% of page using numeric/bracket/superscript markers |
| Column detection | `pdf-column-detect` | `pdf-table-detect` | Reorders multi-column text into left-to-right reading order |
| OCR | `pdf-ocr` | `pdfium-render`, `tesseract` | OCR for scanned (image-only) pages |

Enable in your `Cargo.toml`:

```toml
[dependencies]
fastrag = { version = "0.1", features = ["pdf-images", "pdf-table-detect"] }
```

OCR requires system packages (`tesseract-ocr`, `tesseract-ocr-eng`) and links against PDFium statically.

## HTML Footnotes

The HTML parser extracts footnotes and endnotes from common patterns:
- Footnote sections: `section.footnotes`, `div.footnotes`, `div.endnotes`, `[role=doc-endnotes]`
- Footnote items: `li[id]` within those sections
- Inline references: `<sup><a href="#fn...">` patterns in body paragraphs

Footnotes become `Footnote` elements with a `reference_id` attribute. Paragraphs containing footnote references get a `footnote_refs` attribute listing the referenced IDs.

## Streaming Output

The `--stream` flag writes each element as a separate JSONL line to stdout, flushing after each element:

```bash
fastrag parse document.pdf --stream
```

Streaming mode skips hierarchy building and caption association, yielding elements as they are extracted from each page. This is useful for large documents where you want to process elements as they arrive.

The MCP `parse_file` tool also accepts a `stream` parameter.

## Chunking for RAG

FastRAG provides three chunking strategies for splitting parsed documents into RAG-ready pieces:

| Strategy | Description |
|----------|-------------|
| `basic` | Accumulates elements up to the character limit, then starts a new chunk |
| `by-title` | Splits on Title/Heading boundaries, sub-chunks large sections |
| `recursive` | Recursive character splitting — tries separators in order, falling back to finer granularity |
| `semantic` | Splits on embedding similarity boundaries — groups similar sentences together, splits where topics change |

All strategies support **chunk overlap**, which repeats characters from the end of one chunk at the start of the next, preserving context across boundaries.

### CLI

```bash
# Basic chunking with 500-char limit
fastrag parse document.pdf --chunk-strategy basic --chunk-size 500

# By-title with 200-char overlap
fastrag parse document.pdf --chunk-strategy by-title --chunk-size 1000 --chunk-overlap 200

# Recursive splitting (default separators: \n\n, \n, ". ", " ", "")
fastrag parse document.pdf --chunk-strategy recursive --chunk-size 500

# Recursive with custom separators (comma-separated, use \n for newline)
fastrag parse document.pdf --chunk-strategy recursive --chunk-size 500 --chunk-separators "\n\n,\n,. , ,"

# Semantic chunking with similarity threshold
fastrag parse document.pdf --chunk-strategy semantic --chunk-size 1000 --similarity-threshold 0.3
```

## Corpus Retrieval

FastRAG can build and query a persisted semantic corpus when the `retrieval` feature is enabled.

### CLI

```bash
# Index a file or directory into a corpus directory
fastrag index ./documents --corpus ./corpus

# Index with run-wide metadata (applied to every file)
fastrag index ./documents --corpus ./corpus \
    --metadata customer=acme --metadata year=2024

# Query the indexed corpus
fastrag query "invoice payment terms" --corpus ./corpus --top-k 5

# Filter by metadata at query time (AND-combined equality)
fastrag query "privilege escalation" --corpus ./corpus \
    --filter customer=acme,severity=high

# Show corpus metadata
fastrag corpus-info --corpus ./corpus

# Start the HTTP query server
fastrag serve-http --corpus ./corpus --port 8081
```

Alongside each input file, an optional `<name>.meta.json` sidecar carrying a flat
`{"key":"string"}` object attaches per-file metadata. Sidecar values override any
`--metadata` flags on the same key. Query-time filters pass as `--filter k=v,k=v`
on the CLI, `&filter=k=v,k=v` on the HTTP `/query` endpoint, or as a `filter`
object on the MCP `search_corpus` tool.

The CLI accepts an optional local model path with `--model-path`. If omitted, FastRAG loads the default BGE-small-en-v1.5 embedder and caches it under `dirs::cache_dir()/fastrag/models/bge-small-en-v1.5`.

### Library

Enable retrieval from `Cargo.toml`:

```toml
[dependencies]
fastrag = { version = "0.1", features = ["retrieval"] }
```

Retrieval uses the shared `ChunkingStrategy` API, persists the corpus under `manifest.json`, `index.bin`, and `entries.bin`, and returns deterministic search hits sorted by score.
The MCP server exposes the same search capability as `search_corpus` when built with the `mcp-search` feature.

Quality, latency, and footprint baselines for the retrieval pipeline are tracked in [`docs/eval-baselines.md`](docs/eval-baselines.md). Regenerate them with `scripts/run-eval.sh`.

### Library

```rust
use fastrag::{parse, ChunkingStrategy, default_separators};

let doc = parse("document.pdf")?;

// Basic chunking
let chunks = doc.chunk(&ChunkingStrategy::Basic {
    max_characters: 500,
    overlap: 0,
});

// Recursive character splitting with overlap
let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
    max_characters: 500,
    overlap: 100,
    separators: default_separators(),
});

// Semantic chunking
let chunks = doc.chunk(&ChunkingStrategy::Semantic {
    max_characters: 1000,
    similarity_threshold: Some(0.3),
    percentile_threshold: None,
});

for chunk in &chunks {
    println!("Chunk {}: {} chars", chunk.index, chunk.char_count);
    println!("{}", chunk.text);
}
```

## Deployment

The `serve-http` subcommand exposes a small operational surface for production use.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe — returns `{"status":"ok"}` |
| `GET /query?q=...&top_k=N` | Semantic corpus search |
| `GET /metrics` | Prometheus text-format metrics |

### Metrics

| Name | Type | Description |
|------|------|-------------|
| `fastrag_query_total` | counter | Total `/query` requests served |
| `fastrag_query_duration_seconds` | histogram | `/query` latency distribution |
| `fastrag_index_entries` | gauge | Number of entries in the loaded corpus |

### Logging

Logs go to stdout via `tracing`. Set `FASTRAG_LOG_FORMAT=json` for one-line JSON logs in production, or leave unset for the pretty terminal format. Filter levels with `FASTRAG_LOG=info,fastrag_cli=debug`.

### systemd

A sample unit file lives at `deploy/fastrag.service`. Copy it to `/etc/systemd/system/`, create the `fastrag` user and `/var/lib/fastrag/corpus`, then run `systemctl enable --now fastrag`.

### Docker

The repo ships a multi-stage `Dockerfile` that produces a distroless image (~50 MB):

```bash
docker build -t fastrag .
docker run -p 8081:8081 -v /srv/fastrag:/var/lib/fastrag fastrag
```

## Architecture

FastRAG uses a workspace of small, focused crates:

- **`fastrag-core`** — Core types (`Document`, `Element`, `Parser` trait)
- **`fastrag-pdf`**, **`fastrag-html`**, etc. — Format-specific parsers
- **`fastrag`** — Facade library with `ParserRegistry` and auto-detection
- **`fastrag-cli`** — Command-line interface
- **`fastrag-embed`** — BGE-small embedding implementation and test mock
- **`fastrag-index`** — HNSW corpus index and persistence
- **`fastrag-mcp`** — MCP server for AI assistant integration

Each parser is feature-gated, so you only compile what you need.

## Benchmarks

The PDF parser includes criterion benchmarks:

```bash
cargo bench -p fastrag-pdf --bench pdf_parsing --features images,table-detect
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).
