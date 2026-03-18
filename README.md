# FastRAG

**100x faster document parsing for AI/RAG pipelines.**

FastRAG is a Rust-based, single-binary document parser that replaces Python's `unstructured` with better performance and zero runtime dependencies.

## Features

- **Blazing fast** — native Rust performance, no Python/Java/Tika overhead
- **Single binary** — no runtime dependencies to install
- **Multiple formats** — PDF, HTML, Markdown, CSV, XML, DOCX, XLSX, PPTX, plain text
- **Structured output** — Markdown, JSON, or plain text with rich metadata
- **Parallel processing** — batch parse entire directories with configurable workers
- **PDF intelligence** — optional table detection, image extraction, and OCR for scanned pages

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

# Parse an entire directory with 8 workers
fastrag parse ./documents/ --output ./parsed/ -j 8

# List supported formats
fastrag formats
```

### Library

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

## PDF Feature Flags

The PDF parser supports optional capabilities via feature flags:

| Feature | Flag | Dependencies | Description |
|---------|------|-------------|-------------|
| Table detection | `pdf-table-detect` | None | Detects tables from text positions, outputs markdown tables |
| Image extraction | `pdf-images` | None | Extracts embedded images with chart/figure classification |
| OCR | `pdf-ocr` | `pdfium-render`, `tesseract` | OCR for scanned (image-only) pages |

Enable in your `Cargo.toml`:

```toml
[dependencies]
fastrag = { version = "0.1", features = ["pdf-images", "pdf-table-detect"] }
```

OCR requires system packages (`tesseract-ocr`, `tesseract-ocr-eng`) and links against PDFium statically.

## Chunking for RAG

FastRAG provides three chunking strategies for splitting parsed documents into RAG-ready pieces:

| Strategy | Description |
|----------|-------------|
| `basic` | Accumulates elements up to the character limit, then starts a new chunk |
| `by-title` | Splits on Title/Heading boundaries, sub-chunks large sections |
| `recursive` | Recursive character splitting — tries separators in order, falling back to finer granularity |

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
```

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

for chunk in &chunks {
    println!("Chunk {}: {} chars", chunk.index, chunk.char_count);
    println!("{}", chunk.text);
}
```

## Architecture

FastRAG uses a workspace of small, focused crates:

- **`fastrag-core`** — Core types (`Document`, `Element`, `Parser` trait)
- **`fastrag-pdf`**, **`fastrag-html`**, etc. — Format-specific parsers
- **`fastrag`** — Facade library with `ParserRegistry` and auto-detection
- **`fastrag-cli`** — Command-line interface

Each parser is feature-gated, so you only compile what you need.

## Benchmarks

The PDF parser includes criterion benchmarks:

```bash
cargo bench -p fastrag-pdf --bench pdf_parsing --features images,table-detect
```

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE).
