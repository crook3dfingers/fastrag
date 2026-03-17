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

    if !trimmed.contains('\n') && trimmed.len() <= 100 {
        let alpha_chars: Vec<char> = trimmed.chars().filter(|c| c.is_alphabetic()).collect();
        if !alpha_chars.is_empty() && alpha_chars.iter().all(|c| c.is_uppercase()) {
            return Element::new(ElementKind::Title, trimmed);
        }

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
    fn supported_formats_returns_text() {
        assert_eq!(TextParser.supported_formats(), &[FileFormat::Text]);
    }

    #[test]
    fn two_paragraphs_separated_by_blank_line() {
        let doc = parse_text("First paragraph.\n\nSecond paragraph.");
        assert_eq!(doc.elements.len(), 2);
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[0].text, "First paragraph.");
        assert_eq!(doc.elements[1].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[1].text, "Second paragraph.");
    }

    #[test]
    fn all_caps_line_becomes_title() {
        let doc = parse_text("INTRODUCTION\n\nSome text here.");
        assert_eq!(doc.elements[0].kind, ElementKind::Title);
        assert_eq!(doc.elements[0].text, "INTRODUCTION");
    }

    #[test]
    fn colon_ending_line_becomes_heading() {
        let doc = parse_text("Summary:\n\nThe results are in.");
        assert_eq!(doc.elements[0].kind, ElementKind::Heading);
        assert_eq!(doc.elements[0].text, "Summary");
        assert_eq!(doc.elements[0].depth, 1);
    }

    #[test]
    fn multiline_paragraph_joined_with_newline() {
        let doc = parse_text("Line one\nLine two\nLine three\n\nNew paragraph.");
        assert_eq!(doc.elements.len(), 2);
        assert_eq!(doc.elements[0].text, "Line one\nLine two\nLine three");
    }

    #[test]
    fn empty_input_no_elements() {
        let doc = parse_text("");
        assert!(doc.elements.is_empty());
    }

    #[test]
    fn title_sets_metadata_title() {
        let doc = parse_text("INTRODUCTION\n\nSome text.");
        assert_eq!(doc.metadata.title, Some("INTRODUCTION".to_string()));
    }

    #[test]
    fn no_title_metadata_is_none() {
        let doc = parse_text("Just a paragraph.\n\nAnother one.");
        assert_eq!(doc.metadata.title, None);
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_text("Hello");
        assert_eq!(doc.metadata.source_file, Some("test.txt".to_string()));
    }

    #[test]
    fn multiple_blank_lines_no_empty_elements() {
        let doc = parse_text("A\n\n\n\nB");
        assert_eq!(doc.elements.len(), 2);
        assert!(doc.elements.iter().all(|e| !e.text.is_empty()));
    }

    #[test]
    fn all_caps_over_100_chars_is_paragraph() {
        let long = "A".repeat(101);
        let doc = parse_text(&long);
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
    }

    #[test]
    fn invalid_utf8_returns_encoding_error() {
        let parser = TextParser;
        let source = SourceInfo::new(FileFormat::Text);
        let bad_bytes: &[u8] = &[0xFF, 0xFE, 0x80, 0x81];
        let result = parser.parse(bad_bytes, &source);
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Encoding(_) => {}
            other => panic!("expected Encoding error, got: {other}"),
        }
    }
}
