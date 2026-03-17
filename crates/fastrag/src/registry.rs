use std::collections::HashMap;
use std::path::Path;

use fastrag_core::{Document, FastRagError, FileFormat, Parser, SourceInfo};

/// Registry that maps file formats to their parsers and handles dispatch.
pub struct ParserRegistry {
    parsers: HashMap<FileFormat, Box<dyn Parser>>,
}

impl Default for ParserRegistry {
    fn default() -> Self {
        let mut registry = Self {
            parsers: HashMap::new(),
        };

        #[cfg(feature = "text")]
        registry.register(Box::new(fastrag_text::TextParser));

        #[cfg(feature = "csv")]
        registry.register(Box::new(fastrag_csv::CsvParser::default()));

        #[cfg(feature = "markdown")]
        registry.register(Box::new(fastrag_markdown::MarkdownParser));

        #[cfg(feature = "html")]
        registry.register(Box::new(fastrag_html::HtmlParser));

        #[cfg(feature = "pdf")]
        registry.register(Box::new(fastrag_pdf::PdfParser));

        registry
    }
}

impl ParserRegistry {
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
        }
    }

    /// Register a parser for a specific format.
    pub fn register_for_format(&mut self, format: FileFormat, parser: Box<dyn Parser>) {
        self.parsers.insert(format, parser);
    }

    /// Register a parser for all formats it supports.
    pub fn register(&mut self, parser: Box<dyn Parser>) {
        let formats: Vec<FileFormat> = parser.supported_formats().to_vec();
        if let Some(format) = formats.into_iter().next() {
            self.parsers.insert(format, parser);
        }
    }

    /// List all supported formats.
    pub fn supported_formats(&self) -> Vec<FileFormat> {
        self.parsers.keys().copied().collect()
    }

    /// Parse a file at the given path.
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<Document, FastRagError> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let first_bytes = &data[..data.len().min(1024)];
        let format = FileFormat::detect(path, first_bytes);

        self.parse_bytes(&data, format, path.to_string_lossy().to_string())
    }

    /// Parse raw bytes with a known format.
    pub fn parse_bytes(
        &self,
        data: &[u8],
        format: FileFormat,
        filename: String,
    ) -> Result<Document, FastRagError> {
        let parser = self
            .parsers
            .get(&format)
            .ok_or_else(|| FastRagError::UnsupportedFormat(format.to_string()))?;

        let source = SourceInfo::new(format).with_filename(filename);
        parser.parse(data, &source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::ElementKind;

    #[test]
    fn new_registry_is_empty() {
        let reg = ParserRegistry::new();
        assert!(reg.supported_formats().is_empty());
    }

    #[test]
    fn register_text_parser() {
        let mut reg = ParserRegistry::new();
        reg.register(Box::new(fastrag_text::TextParser));
        assert!(reg.supported_formats().contains(&FileFormat::Text));
    }

    #[test]
    fn register_for_format() {
        let mut reg = ParserRegistry::new();
        reg.register_for_format(FileFormat::Text, Box::new(fastrag_text::TextParser));
        assert!(reg.supported_formats().contains(&FileFormat::Text));
    }

    #[test]
    fn parse_bytes_with_registered_parser() {
        let mut reg = ParserRegistry::new();
        reg.register(Box::new(fastrag_text::TextParser));
        let doc = reg
            .parse_bytes(b"Hello world", FileFormat::Text, "test.txt".to_string())
            .unwrap();
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[0].text, "Hello world");
    }

    #[test]
    fn parse_bytes_unsupported_format() {
        let reg = ParserRegistry::new();
        let result = reg.parse_bytes(b"data", FileFormat::Pdf, "test.pdf".to_string());
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::UnsupportedFormat(_) => {}
            other => panic!("expected UnsupportedFormat, got: {other}"),
        }
    }

    #[test]
    fn parse_file_sample_txt() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let reg = ParserRegistry::default();
        let doc = reg.parse_file(format!("{fixtures}/sample.txt")).unwrap();
        assert!(!doc.elements.is_empty());
    }

    #[test]
    fn parse_file_nonexistent() {
        let reg = ParserRegistry::default();
        let result = reg.parse_file("nonexistent.txt");
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Io(_) => {}
            other => panic!("expected Io error, got: {other}"),
        }
    }

    #[test]
    fn default_registry_has_5_formats() {
        let reg = ParserRegistry::default();
        assert_eq!(reg.supported_formats().len(), 5);
    }
}
