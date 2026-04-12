pub mod chunking;
pub mod document;
pub mod error;
pub mod format;
#[cfg(feature = "language-detection")]
pub mod language;
pub mod output;

pub use chunking::{
    Chunk, ChunkingStrategy, ContextInjection, cosine_similarity, default_embedder,
    default_separators,
};
pub use document::{BoundingBox, Document, Element, ElementKind, Metadata, is_caption_text};
pub use error::FastRagError;
pub use format::{FileFormat, SourceInfo};
pub use output::OutputFormat;

/// A parser that emits multiple `Document` objects from a single source file.
///
/// Used for formats like NVD JSON feeds where one file encodes many
/// independent records. Implement this trait alongside (not instead of)
/// `Parser` when a format requires multi-doc emission.
pub trait MultiDocParser: Send + Sync {
    /// Parse the file at `path` and return one `Document` per logical record.
    fn parse_all(&self, path: &std::path::Path) -> Result<Vec<Document>, FastRagError>;
}

/// Every format parser implements this trait.
pub trait Parser: Send + Sync {
    /// Returns the file formats this parser can handle.
    fn supported_formats(&self) -> &[FileFormat];

    /// Parse raw bytes into a structured Document.
    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError>;

    /// Stream elements incrementally instead of building a complete Document.
    ///
    /// The default implementation calls `parse()` then yields elements one by one.
    /// Parsers can override this for true incremental processing (e.g., page-by-page).
    ///
    /// Note: streaming mode skips `build_hierarchy()` and `associate_captions()`.
    fn parse_stream<'a>(
        &'a self,
        input: &'a [u8],
        source: &'a SourceInfo,
    ) -> Result<Box<dyn Iterator<Item = Result<Element, FastRagError>> + 'a>, FastRagError> {
        let doc = self.parse(input, source)?;
        Ok(Box::new(doc.elements.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod multi_doc_parser_tests {
    use super::*;
    use std::path::Path;

    struct StubMultiParser;

    impl MultiDocParser for StubMultiParser {
        fn parse_all(&self, _path: &Path) -> Result<Vec<Document>, FastRagError> {
            Ok(vec![
                Document {
                    metadata: Metadata::new(FileFormat::Text),
                    elements: vec![],
                },
                Document {
                    metadata: Metadata::new(FileFormat::Text),
                    elements: vec![],
                },
            ])
        }
    }

    #[test]
    fn multi_doc_parser_trait_is_implementable() {
        let parser = StubMultiParser;
        let docs = parser.parse_all(Path::new("dummy")).unwrap();
        assert_eq!(docs.len(), 2);
    }
}
