use crate::document::{Document, ElementKind};

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Markdown,
    Json,
    PlainText,
}

impl Document {
    /// Render the document as structured markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        // Only emit metadata title if first element isn't already a title
        let first_is_title = self
            .elements
            .first()
            .is_some_and(|e| e.kind == ElementKind::Title);
        if !first_is_title && let Some(title) = &self.metadata.title {
            out.push_str(&format!("# {title}\n\n"));
        }

        for element in &self.elements {
            match &element.kind {
                ElementKind::Title => {
                    out.push_str(&format!("# {}\n\n", element.text));
                }
                ElementKind::Heading => {
                    let prefix = "#".repeat(element.depth.max(1) as usize + 1);
                    out.push_str(&format!("{prefix} {}\n\n", element.text));
                }
                ElementKind::Paragraph => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::Table => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::Code => {
                    let lang = element
                        .attributes
                        .get("language")
                        .map_or("", |s| s.as_str());
                    out.push_str(&format!("```{lang}\n{}\n```\n\n", element.text));
                }
                ElementKind::List => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::ListItem => {
                    out.push_str(&format!("- {}\n", element.text));
                }
                ElementKind::Image => {
                    let alt = element.attributes.get("alt").map_or("", |s| s.as_str());
                    out.push_str(&format!("![{alt}]({})\n\n", element.text));
                }
                ElementKind::BlockQuote => {
                    for line in element.text.lines() {
                        out.push_str(&format!("> {line}\n"));
                    }
                    out.push('\n');
                }
                ElementKind::HorizontalRule => {
                    out.push_str("---\n\n");
                }
                ElementKind::Unknown => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
            }
        }

        out.trim_end().to_string()
    }

    /// Render the document as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render the document as plain text (all element text concatenated).
    pub fn to_plain_text(&self) -> String {
        let mut out = String::new();

        for element in &self.elements {
            if !element.text.is_empty() {
                out.push_str(&element.text);
                out.push('\n');
            }
        }

        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use crate::document::*;
    use crate::format::FileFormat;

    fn sample_doc() -> Document {
        Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![
                Element::new(ElementKind::Title, "My Document"),
                Element::new(ElementKind::Paragraph, "First paragraph."),
                Element::new(ElementKind::Heading, "Section One").with_depth(1),
                Element::new(ElementKind::Paragraph, "Second paragraph."),
                Element::new(ElementKind::Code, "let x = 42;"),
                Element::new(ElementKind::HorizontalRule, ""),
                Element::new(ElementKind::BlockQuote, "A wise quote."),
            ],
        }
    }

    #[test]
    fn test_to_markdown() {
        let doc = sample_doc();
        let md = doc.to_markdown();
        assert!(md.contains("# My Document"));
        assert!(md.contains("## Section One"));
        assert!(md.contains("First paragraph."));
        assert!(md.contains("```\nlet x = 42;\n```"));
        assert!(md.contains("---"));
        assert!(md.contains("> A wise quote."));
    }

    #[test]
    fn test_to_json() {
        let doc = sample_doc();
        let json = doc.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["elements"].is_array());
        assert_eq!(parsed["elements"][0]["kind"], "title");
        assert_eq!(parsed["metadata"]["format"], "text");
    }

    #[test]
    fn test_to_plain_text() {
        let doc = sample_doc();
        let text = doc.to_plain_text();
        assert!(text.contains("My Document"));
        assert!(text.contains("First paragraph."));
        assert!(text.contains("let x = 42;"));
        // Should not contain markdown formatting
        assert!(!text.contains("# "));
        assert!(!text.contains("```"));
    }
}
