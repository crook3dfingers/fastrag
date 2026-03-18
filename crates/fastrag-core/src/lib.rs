pub mod chunking;
pub mod document;
pub mod error;
pub mod format;
#[cfg(feature = "language-detection")]
pub mod language;
pub mod output;

pub use chunking::{Chunk, ChunkingStrategy, default_separators};
pub use document::{Document, Element, ElementKind, Metadata, is_caption_text};
pub use error::FastRagError;
pub use format::{FileFormat, SourceInfo};
pub use output::OutputFormat;

/// Every format parser implements this trait.
pub trait Parser: Send + Sync {
    /// Returns the file formats this parser can handle.
    fn supported_formats(&self) -> &[FileFormat];

    /// Parse raw bytes into a structured Document.
    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError>;
}
