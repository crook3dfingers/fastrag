use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::format::FileFormat;

/// A parsed document containing metadata and structured elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub metadata: Metadata,
    pub elements: Vec<Element>,
}

/// Metadata about the source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub source_file: Option<String>,
    pub format: FileFormat,
    pub title: Option<String>,
    pub author: Option<String>,
    pub page_count: Option<usize>,
    pub created_at: Option<String>,
    #[serde(flatten)]
    pub custom: HashMap<String, String>,
}

impl Metadata {
    pub fn new(format: FileFormat) -> Self {
        Self {
            source_file: None,
            format,
            title: None,
            author: None,
            page_count: None,
            created_at: None,
            custom: HashMap::new(),
        }
    }
}

/// A single structural element extracted from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub kind: ElementKind,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    pub depth: u8,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, String>,
}

impl Element {
    pub fn new(kind: ElementKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            page: None,
            section: None,
            depth: 0,
            attributes: HashMap::new(),
        }
    }

    pub fn with_depth(mut self, depth: u8) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_page(mut self, page: usize) -> Self {
        self.page = Some(page);
        self
    }

    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }
}

/// The kind of structural element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    Title,
    Heading,
    Paragraph,
    Table,
    Code,
    List,
    ListItem,
    Image,
    BlockQuote,
    HorizontalRule,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_new_defaults() {
        let el = Element::new(ElementKind::Paragraph, "hello");
        assert_eq!(el.kind, ElementKind::Paragraph);
        assert_eq!(el.text, "hello");
        assert_eq!(el.depth, 0);
        assert_eq!(el.page, None);
        assert_eq!(el.section, None);
        assert!(el.attributes.is_empty());
    }

    #[test]
    fn element_with_depth() {
        let el = Element::new(ElementKind::Heading, "h").with_depth(2);
        assert_eq!(el.depth, 2);
    }

    #[test]
    fn element_with_page() {
        let el = Element::new(ElementKind::Paragraph, "p").with_page(5);
        assert_eq!(el.page, Some(5));
    }

    #[test]
    fn element_with_section() {
        let el = Element::new(ElementKind::Paragraph, "p").with_section("intro");
        assert_eq!(el.section, Some("intro".to_string()));
    }

    #[test]
    fn element_builder_chaining() {
        let el = Element::new(ElementKind::Code, "x = 1")
            .with_depth(1)
            .with_page(3)
            .with_section("code");
        assert_eq!(el.depth, 1);
        assert_eq!(el.page, Some(3));
        assert_eq!(el.section, Some("code".to_string()));
        assert_eq!(el.text, "x = 1");
    }

    #[test]
    fn metadata_new_defaults() {
        let m = Metadata::new(FileFormat::Html);
        assert_eq!(m.format, FileFormat::Html);
        assert_eq!(m.source_file, None);
        assert_eq!(m.title, None);
        assert_eq!(m.author, None);
        assert_eq!(m.page_count, None);
        assert_eq!(m.created_at, None);
        assert!(m.custom.is_empty());
    }
}
