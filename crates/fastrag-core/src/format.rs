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
    Docx,
    Pptx,
    Xlsx,
    Xml,
    Epub,
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
            Self::Docx => write!(f, "DOCX"),
            Self::Pptx => write!(f, "PPTX"),
            Self::Xlsx => write!(f, "XLSX"),
            Self::Xml => write!(f, "XML"),
            Self::Epub => write!(f, "EPUB"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl FileFormat {
    /// Detect the file format from a path (extension) with magic-byte fallback.
    pub fn detect(path: &Path, first_bytes: &[u8]) -> Self {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext.to_lowercase().as_str() {
                "pdf" => return Self::Pdf,
                "html" | "htm" | "xhtml" => return Self::Html,
                "md" | "markdown" | "mkd" => return Self::Markdown,
                "csv" | "tsv" => return Self::Csv,
                "txt" | "text" | "log" => return Self::Text,
                "docx" => return Self::Docx,
                "pptx" => return Self::Pptx,
                "xlsx" => return Self::Xlsx,
                "xml" => return Self::Xml,
                "epub" => return Self::Epub,
                _ => {}
            }
        }

        if first_bytes.starts_with(b"%PDF") {
            return Self::Pdf;
        }

        if !first_bytes.is_empty() && first_bytes.len() >= 5 {
            let start = String::from_utf8_lossy(&first_bytes[..first_bytes.len().min(512)]);
            let trimmed = start.trim_start();
            if trimmed.starts_with("<?xml") {
                return Self::Xml;
            }
        }

        if !first_bytes.is_empty() {
            let start = String::from_utf8_lossy(&first_bytes[..first_bytes.len().min(512)]);
            let trimmed = start.trim_start();
            if trimmed.starts_with("<!DOCTYPE")
                || trimmed.starts_with("<html")
                || trimmed.starts_with("<HTML")
            {
                return Self::Html;
            }
        }

        Self::Unknown
    }

    /// List all known formats.
    pub fn all_known() -> &'static [FileFormat] {
        &[
            Self::Pdf,
            Self::Html,
            Self::Markdown,
            Self::Csv,
            Self::Text,
            Self::Docx,
            Self::Pptx,
            Self::Xlsx,
            Self::Xml,
            Self::Epub,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- FileFormat::detect extension tests ---

    #[test]
    fn detect_pdf_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("doc.pdf"), &[]),
            FileFormat::Pdf
        );
    }

    #[test]
    fn detect_html_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("page.html"), &[]),
            FileFormat::Html
        );
    }

    #[test]
    fn detect_htm_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("page.htm"), &[]),
            FileFormat::Html
        );
    }

    #[test]
    fn detect_xhtml_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("page.xhtml"), &[]),
            FileFormat::Html
        );
    }

    #[test]
    fn detect_md_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("readme.md"), &[]),
            FileFormat::Markdown
        );
    }

    #[test]
    fn detect_mkd_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("doc.mkd"), &[]),
            FileFormat::Markdown
        );
    }

    #[test]
    fn detect_csv_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("data.csv"), &[]),
            FileFormat::Csv
        );
    }

    #[test]
    fn detect_tsv_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("data.tsv"), &[]),
            FileFormat::Csv
        );
    }

    #[test]
    fn detect_txt_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("notes.txt"), &[]),
            FileFormat::Text
        );
    }

    #[test]
    fn detect_log_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("app.log"), &[]),
            FileFormat::Text
        );
    }

    #[test]
    fn detect_unknown_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("file.xyz"), &[]),
            FileFormat::Unknown
        );
    }

    // --- Magic byte tests ---

    #[test]
    fn detect_pdf_magic_bytes() {
        assert_eq!(
            FileFormat::detect(Path::new("noext"), b"%PDF-1.5 rest"),
            FileFormat::Pdf
        );
    }

    #[test]
    fn detect_html_doctype_magic() {
        assert_eq!(
            FileFormat::detect(Path::new("noext"), b"<!DOCTYPE html>"),
            FileFormat::Html,
        );
    }

    #[test]
    fn detect_html_tag_magic() {
        assert_eq!(
            FileFormat::detect(Path::new("noext"), b"<html><head>"),
            FileFormat::Html,
        );
    }

    // --- Extension takes priority over magic bytes ---

    #[test]
    fn extension_overrides_magic_bytes() {
        // .txt extension should win even if content looks like PDF
        assert_eq!(
            FileFormat::detect(Path::new("file.txt"), b"%PDF-1.5"),
            FileFormat::Text
        );
    }

    // --- Display tests ---

    #[test]
    fn display_pdf() {
        assert_eq!(format!("{}", FileFormat::Pdf), "PDF");
    }

    #[test]
    fn display_html() {
        assert_eq!(format!("{}", FileFormat::Html), "HTML");
    }

    #[test]
    fn display_markdown() {
        assert_eq!(format!("{}", FileFormat::Markdown), "Markdown");
    }

    #[test]
    fn display_csv() {
        assert_eq!(format!("{}", FileFormat::Csv), "CSV");
    }

    #[test]
    fn display_text() {
        assert_eq!(format!("{}", FileFormat::Text), "Text");
    }

    #[test]
    fn display_unknown() {
        assert_eq!(format!("{}", FileFormat::Unknown), "Unknown");
    }

    // --- all_known tests ---

    #[test]
    fn detect_docx_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("doc.docx"), &[]),
            FileFormat::Docx
        );
    }

    #[test]
    fn detect_pptx_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("slides.pptx"), &[]),
            FileFormat::Pptx
        );
    }

    #[test]
    fn detect_xlsx_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("data.xlsx"), &[]),
            FileFormat::Xlsx
        );
    }

    #[test]
    fn detect_xml_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("config.xml"), &[]),
            FileFormat::Xml
        );
    }

    #[test]
    fn detect_xml_magic_bytes() {
        assert_eq!(
            FileFormat::detect(Path::new("noext"), b"<?xml version=\"1.0\"?>"),
            FileFormat::Xml
        );
    }

    #[test]
    fn display_docx() {
        assert_eq!(format!("{}", FileFormat::Docx), "DOCX");
    }

    #[test]
    fn display_pptx() {
        assert_eq!(format!("{}", FileFormat::Pptx), "PPTX");
    }

    #[test]
    fn display_xlsx() {
        assert_eq!(format!("{}", FileFormat::Xlsx), "XLSX");
    }

    #[test]
    fn display_xml() {
        assert_eq!(format!("{}", FileFormat::Xml), "XML");
    }

    #[test]
    fn all_known_length() {
        assert_eq!(FileFormat::all_known().len(), 10);
    }

    #[test]
    fn all_known_contains_all_variants() {
        let known = FileFormat::all_known();
        assert!(known.contains(&FileFormat::Pdf));
        assert!(known.contains(&FileFormat::Html));
        assert!(known.contains(&FileFormat::Markdown));
        assert!(known.contains(&FileFormat::Csv));
        assert!(known.contains(&FileFormat::Text));
        assert!(known.contains(&FileFormat::Docx));
        assert!(known.contains(&FileFormat::Pptx));
        assert!(known.contains(&FileFormat::Xlsx));
        assert!(known.contains(&FileFormat::Xml));
        assert!(known.contains(&FileFormat::Epub));
    }

    #[test]
    fn detect_epub_extension() {
        assert_eq!(
            FileFormat::detect(Path::new("book.epub"), &[]),
            FileFormat::Epub
        );
    }

    #[test]
    fn display_epub() {
        assert_eq!(format!("{}", FileFormat::Epub), "EPUB");
    }

    // --- SourceInfo tests ---

    #[test]
    fn source_info_new_defaults() {
        let si = SourceInfo::new(FileFormat::Csv);
        assert_eq!(si.format, FileFormat::Csv);
        assert_eq!(si.filename, None);
    }

    #[test]
    fn source_info_with_filename() {
        let si = SourceInfo::new(FileFormat::Text).with_filename("test.txt");
        assert_eq!(si.filename, Some("test.txt".to_string()));
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
