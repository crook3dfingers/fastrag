#[cfg(feature = "retrieval")]
pub mod corpus;
#[cfg(feature = "store")]
pub mod filter;
#[cfg(feature = "hygiene")]
pub mod hygiene;
#[cfg(feature = "store")]
pub mod ingest;
pub mod ops;
pub mod registry;

// Re-export core types
pub use fastrag_core::*;

// Re-export parsers when feature-enabled
#[cfg(feature = "csv")]
pub use fastrag_csv::CsvParser;
#[cfg(feature = "docx")]
pub use fastrag_docx::DocxParser;
#[cfg(feature = "email")]
pub use fastrag_email::EmailParser;
#[cfg(all(feature = "embedding", feature = "legacy-candle"))]
pub use fastrag_embed::BgeSmallEmbedder;
#[cfg(feature = "embedding")]
pub use fastrag_embed::{DynEmbedder, DynEmbedderTrait, EmbedError as EmbedderError, Embedder};
#[cfg(feature = "html")]
pub use fastrag_html::HtmlParser;
#[cfg(feature = "index")]
pub use fastrag_index::{
    CorpusManifest, HnswIndex, IndexError, ManifestChunkingStrategy, VectorEntry, VectorHit,
    VectorIndex,
};
#[cfg(feature = "markdown")]
pub use fastrag_markdown::MarkdownParser;
#[cfg(feature = "pdf")]
pub use fastrag_pdf::PdfParser;
#[cfg(feature = "pptx")]
pub use fastrag_pptx::PptxParser;
#[cfg(feature = "rerank")]
pub use fastrag_rerank::{RerankError, Reranker};
#[cfg(feature = "rtf")]
pub use fastrag_rtf::RtfParser;
#[cfg(feature = "text")]
pub use fastrag_text::TextParser;
#[cfg(feature = "xlsx")]
pub use fastrag_xlsx::XlsxParser;
#[cfg(feature = "xml")]
pub use fastrag_xml::XmlParser;

use registry::ParserRegistry;
use std::path::Path;

/// Parse a file at the given path using automatic format detection.
pub fn parse(path: impl AsRef<Path>) -> Result<Document, FastRagError> {
    let registry = ParserRegistry::default();
    registry.parse_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv_fixture() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let doc = parse(format!("{fixtures}/sample.csv")).unwrap();
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::Table));
    }
}
