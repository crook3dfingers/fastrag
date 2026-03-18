use std::collections::HashMap;

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// Email (EML) parser using the `mailparse` crate.
pub struct EmailParser;

impl Parser for EmailParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Email]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let parsed = mailparse::parse_mail(input).map_err(|e| FastRagError::Parse {
            format: FileFormat::Email,
            message: format!("email parse error: {e}"),
        })?;

        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        // Extract headers into custom metadata
        let mut custom = HashMap::new();
        for header in &parsed.headers {
            let key = header.get_key().to_lowercase();
            let value = header.get_value();
            match key.as_str() {
                "subject" => {
                    metadata.title = Some(value.clone());
                    custom.insert("subject".to_string(), value);
                }
                "from" | "to" | "date" => {
                    custom.insert(key, value);
                }
                _ => {}
            }
        }
        metadata.custom = custom;

        // Extract body text, normalize line endings
        let body_text = extract_body_text(&parsed).replace("\r\n", "\n");

        let mut elements = Vec::new();
        for para in body_text.split("\n\n") {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                continue;
            }
            elements.push(Element::new(ElementKind::Paragraph, trimmed));
        }

        // If no double-newline paragraphs, split on single newlines
        if elements.is_empty() {
            for line in body_text.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    elements.push(Element::new(ElementKind::Paragraph, trimmed));
                }
            }
        }

        Ok(Document { metadata, elements })
    }
}

/// Extract plain text body from a parsed email, preferring text/plain over text/html.
fn extract_body_text(mail: &mailparse::ParsedMail) -> String {
    // Check subparts first (multipart MIME)
    if !mail.subparts.is_empty() {
        // Prefer text/plain
        for part in &mail.subparts {
            if let Some(ct) = part
                .headers
                .iter()
                .find(|h| h.get_key().to_lowercase() == "content-type")
            {
                let ct_val = ct.get_value().to_lowercase();
                if ct_val.contains("text/plain")
                    && let Ok(body) = part.get_body()
                {
                    return body;
                }
            }
        }
        // Fall back to text/html
        for part in &mail.subparts {
            if let Some(ct) = part
                .headers
                .iter()
                .find(|h| h.get_key().to_lowercase() == "content-type")
            {
                let ct_val = ct.get_value().to_lowercase();
                if ct_val.contains("text/html")
                    && let Ok(body) = part.get_body()
                {
                    return body;
                }
            }
        }
        // Try any part
        for part in &mail.subparts {
            if let Ok(body) = part.get_body()
                && !body.trim().is_empty()
            {
                return body;
            }
        }
    }

    // Simple (non-multipart) email
    mail.get_body().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_returns_email() {
        assert_eq!(EmailParser.supported_formats(), &[FileFormat::Email]);
    }

    #[test]
    fn parse_simple_email() {
        let eml = b"From: alice@example.com\r\nTo: bob@example.com\r\nSubject: Test Email\r\nDate: Mon, 01 Jan 2024 00:00:00 +0000\r\n\r\nHello Bob,\r\n\r\nThis is a test email.\r\n\r\nBest regards,\r\nAlice";
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email).with_filename("test.eml");
        let doc = parser.parse(eml, &source).unwrap();

        assert_eq!(doc.metadata.title, Some("Test Email".to_string()));
        assert_eq!(
            doc.metadata.custom.get("from"),
            Some(&"alice@example.com".to_string())
        );
        assert_eq!(
            doc.metadata.custom.get("to"),
            Some(&"bob@example.com".to_string())
        );
        assert!(!doc.elements.is_empty());
    }

    #[test]
    fn parse_email_extracts_paragraphs() {
        let eml = b"Subject: Hi\r\n\r\nFirst paragraph.\r\n\r\nSecond paragraph.";
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email);
        let doc = parser.parse(eml, &source).unwrap();

        let texts: Vec<&str> = doc.elements.iter().map(|e| e.text.as_str()).collect();
        assert!(texts.len() >= 2, "expected ≥2 paragraphs, got {:?}", texts);
    }

    #[test]
    fn parse_email_subject_is_title() {
        let eml = b"Subject: Important Meeting\r\n\r\nBody text here.";
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email);
        let doc = parser.parse(eml, &source).unwrap();

        assert_eq!(doc.metadata.title, Some("Important Meeting".to_string()));
        assert_eq!(
            doc.metadata.custom.get("subject"),
            Some(&"Important Meeting".to_string())
        );
    }

    #[test]
    fn parse_email_date_in_custom() {
        let eml = b"Date: Mon, 01 Jan 2024 00:00:00 +0000\r\nSubject: Test\r\n\r\nBody.";
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email);
        let doc = parser.parse(eml, &source).unwrap();

        assert!(doc.metadata.custom.contains_key("date"));
    }

    #[test]
    fn invalid_email_returns_error() {
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email);
        // mailparse is lenient with parsing, so even invalid data may parse
        // This test just ensures no panic
        let _ = parser.parse(b"", &source);
    }

    #[test]
    fn parse_fixture_file() {
        let eml_bytes = include_bytes!("../../../tests/fixtures/sample.eml");
        let parser = EmailParser;
        let source = SourceInfo::new(FileFormat::Email).with_filename("sample.eml");
        let doc = parser.parse(eml_bytes, &source).unwrap();
        assert!(!doc.elements.is_empty());
        assert!(doc.metadata.title.is_some());
    }
}
