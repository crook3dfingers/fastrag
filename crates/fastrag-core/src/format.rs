use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Supported file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    Pdf,
    Html,
    Markdown,
    Csv,
    Text,
    Unknown,
}

impl fmt::Display for FileFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pdf => write!(f, "PDF"),
            Self::Html => write!(f, "HTML"),
            Self::Markdown => write!(f, "Markdown"),
            Self::Csv => write!(f, "CSV"),
            Self::Text => write!(f, "Text"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl FileFormat {
    /// Detect the file format from a path (extension) with magic-byte fallback.
    pub fn detect(path: &Path, first_bytes: &[u8]) -> Self {
        // Try extension first
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "pdf" => return Self::Pdf,
                "html" | "htm" | "xhtml" => return Self::Html,
                "md" | "markdown" | "mkd" => return Self::Markdown,
                "csv" | "tsv" => return Self::Csv,
                "txt" | "text" | "log" => return Self::Text,
                _ => {}
            }
        }

        // Magic byte fallback
        if first_bytes.starts_with(b"%PDF") {
            return Self::Pdf;
        }

        // Check for HTML-like content
        let start = String::from_utf8_lossy(&first_bytes[..first_bytes.len().min(512)]);
        let trimmed = start.trim_start();
        if trimmed.starts_with("<!DOCTYPE")
            || trimmed.starts_with("<html")
            || trimmed.starts_with("<HTML")
        {
            return Self::Html;
        }

        Self::Unknown
    }

    /// List all known formats.
    pub fn all_known() -> &'static [FileFormat] {
        &[Self::Pdf, Self::Html, Self::Markdown, Self::Csv, Self::Text]
    }
}

/// Information about the source of the data being parsed.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub filename: Option<String>,
    pub format: FileFormat,
}

impl SourceInfo {
    pub fn new(format: FileFormat) -> Self {
        Self {
            filename: None,
            format,
        }
    }

    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }
}
