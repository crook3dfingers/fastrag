use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// Plain text parser. Splits on blank lines into paragraphs,
/// detects heading-like lines (ALL CAPS or short lines ending with colon).
pub struct TextParser;

impl Parser for TextParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Text]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let text =
            String::from_utf8(input.to_vec()).map_err(|e| FastRagError::Encoding(e.to_string()))?;

        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let mut elements = Vec::new();
        let mut current_block = String::new();

        for line in text.lines() {
            if line.trim().is_empty() {
                if !current_block.is_empty() {
                    elements.push(classify_block(&current_block));
                    current_block.clear();
                }
            } else {
                if !current_block.is_empty() {
                    current_block.push('\n');
                }
                current_block.push_str(line);
            }
        }

        if !current_block.is_empty() {
            elements.push(classify_block(&current_block));
        }

        // Use first Title or Heading as document title
        for el in &elements {
            if el.kind == ElementKind::Title || el.kind == ElementKind::Heading {
                let mut meta = metadata.clone();
                meta.title = Some(el.text.clone());
                return Ok(Document {
                    metadata: meta,
                    elements,
                });
            }
        }

        Ok(Document { metadata, elements })
    }
}

fn classify_block(block: &str) -> Element {
    let trimmed = block.trim();

    // Single line, all uppercase, reasonably short → Title/Heading
    if !trimmed.contains('\n') && trimmed.len() <= 100 {
        let alpha_chars: Vec<char> = trimmed.chars().filter(|c| c.is_alphabetic()).collect();
        if !alpha_chars.is_empty() && alpha_chars.iter().all(|c| c.is_uppercase()) {
            return Element::new(ElementKind::Title, trimmed);
        }

        // Short line ending with colon → Heading
        if trimmed.len() <= 80 && trimmed.ends_with(':') {
            return Element::new(ElementKind::Heading, trimmed.trim_end_matches(':')).with_depth(1);
        }
    }

    Element::new(ElementKind::Paragraph, trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_text(input: &str) -> Document {
        let parser = TextParser;
        let source = SourceInfo::new(FileFormat::Text).with_filename("test.txt");
        parser.parse(input.as_bytes(), &source).unwrap()
    }

    #[test]
    fn test_basic_paragraphs() {
        let doc = parse_text("First paragraph.\n\nSecond paragraph.");
        assert_eq!(doc.elements.len(), 2);
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[0].text, "First paragraph.");
        assert_eq!(doc.elements[1].text, "Second paragraph.");
    }

    #[test]
    fn test_uppercase_title() {
        let doc = parse_text("INTRODUCTION\n\nSome text here.");
        assert_eq!(doc.elements[0].kind, ElementKind::Title);
        assert_eq!(doc.elements[0].text, "INTRODUCTION");
        assert_eq!(doc.metadata.title, Some("INTRODUCTION".to_string()));
    }

    #[test]
    fn test_colon_heading() {
        let doc = parse_text("Summary:\n\nThe results are in.");
        assert_eq!(doc.elements[0].kind, ElementKind::Heading);
        assert_eq!(doc.elements[0].text, "Summary");
    }

    #[test]
    fn test_multiline_paragraph() {
        let doc = parse_text("Line one\nLine two\nLine three\n\nNew paragraph.");
        assert_eq!(doc.elements.len(), 2);
        assert!(
            doc.elements[0]
                .text
                .contains("Line one\nLine two\nLine three")
        );
    }

    #[test]
    fn test_empty_input() {
        let doc = parse_text("");
        assert!(doc.elements.is_empty());
    }
}
