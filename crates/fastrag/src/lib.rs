pub mod registry;

// Re-export core types
pub use fastrag_core::*;

// Re-export parsers when feature-enabled
#[cfg(feature = "csv")]
pub use fastrag_csv::CsvParser;
#[cfg(feature = "html")]
pub use fastrag_html::HtmlParser;
#[cfg(feature = "markdown")]
pub use fastrag_markdown::MarkdownParser;
#[cfg(feature = "pdf")]
pub use fastrag_pdf::PdfParser;
#[cfg(feature = "text")]
pub use fastrag_text::TextParser;

use registry::ParserRegistry;
use std::path::Path;

/// Parse a file at the given path using automatic format detection.
pub fn parse(path: impl AsRef<Path>) -> Result<Document, FastRagError> {
    let registry = ParserRegistry::default();
    registry.parse_file(path)
}
