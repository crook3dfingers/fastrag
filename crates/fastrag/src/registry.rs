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
    /// Note: Since Box<dyn Parser> can only be stored once, this registers
    /// for the first supported format only. Use `register_for_format` for
    /// parsers supporting multiple formats.
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
