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
