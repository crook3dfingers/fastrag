use comrak::nodes::NodeValue;
use comrak::{Arena, Options, parse_document};

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// Markdown parser using comrak (GFM-compatible).
pub struct MarkdownParser;

impl Parser for MarkdownParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Markdown]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let text =
            String::from_utf8(input.to_vec()).map_err(|e| FastRagError::Encoding(e.to_string()))?;

        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let arena = Arena::new();
        let mut options = Options::default();
        options.extension.table = true;
        options.extension.strikethrough = true;
        options.extension.tasklist = true;

        let root = parse_document(&arena, &text, &options);

        let mut elements = Vec::new();
        collect_elements(root, &mut elements);

        // Extract title from first heading
        for el in &elements {
            if (el.kind == ElementKind::Heading || el.kind == ElementKind::Title) && el.depth == 0 {
                metadata.title = Some(el.text.clone());
                break;
            }
        }

        Ok(Document { metadata, elements })
    }
}

fn collect_elements<'a>(node: &'a comrak::nodes::AstNode<'a>, elements: &mut Vec<Element>) {
    let data = node.data.borrow();
    match &data.value {
        NodeValue::Heading(heading) => {
            let text = collect_text(node);
            let depth = heading.level.saturating_sub(1);
            if depth == 0 {
                elements.push(Element::new(ElementKind::Title, text).with_depth(depth));
            } else {
                elements.push(Element::new(ElementKind::Heading, text).with_depth(depth));
            }
            return; // Don't recurse into heading children
        }
        NodeValue::Paragraph => {
            let text = collect_text(node);
            if !text.is_empty() {
                elements.push(Element::new(ElementKind::Paragraph, text));
            }
            return;
        }
        NodeValue::CodeBlock(cb) => {
            let mut el = Element::new(ElementKind::Code, cb.literal.trim_end());
            if !cb.info.is_empty() {
                el.attributes
                    .insert("language".to_string(), cb.info.clone());
            }
            elements.push(el);
            return;
        }
        NodeValue::BlockQuote => {
            let text = collect_text(node);
            if !text.is_empty() {
                elements.push(Element::new(ElementKind::BlockQuote, text));
            }
            return;
        }
        NodeValue::List(_) => {
            // Collect each item
            for child in node.children() {
                let child_data = child.data.borrow();
                if matches!(child_data.value, NodeValue::Item(_)) {
                    let text = collect_text(child);
                    if !text.is_empty() {
                        elements.push(Element::new(ElementKind::ListItem, text));
                    }
                }
            }
            return;
        }
        NodeValue::ThematicBreak => {
            elements.push(Element::new(ElementKind::HorizontalRule, ""));
            return;
        }
        NodeValue::Table(_) => {
            let text = render_table_text(node);
            elements.push(Element::new(ElementKind::Table, text));
            return;
        }
        _ => {}
    }
    drop(data);

    // Recurse for container nodes
    for child in node.children() {
        collect_elements(child, elements);
    }
}

fn collect_text<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut text = String::new();
    collect_text_inner(node, &mut text);
    text.trim().to_string()
}

fn collect_text_inner<'a>(node: &'a comrak::nodes::AstNode<'a>, out: &mut String) {
    let data = node.data.borrow();
    match &data.value {
        NodeValue::Text(t) => out.push_str(t),
        NodeValue::Code(c) => {
            out.push('`');
            out.push_str(&c.literal);
            out.push('`');
        }
        NodeValue::SoftBreak | NodeValue::LineBreak => out.push(' '),
        _ => {}
    }
    drop(data);

    for child in node.children() {
        collect_text_inner(child, out);
    }
}

fn render_table_text<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();

    for child in node.children() {
        let child_data = child.data.borrow();
        if matches!(child_data.value, NodeValue::TableRow(_)) {
            let mut cells = Vec::new();
            drop(child_data);
            for cell in child.children() {
                cells.push(collect_text(cell));
            }
            rows.push(cells);
        }
    }

    if rows.is_empty() {
        return String::new();
    }

    let mut out = String::new();

    // Header
    if let Some(header) = rows.first() {
        out.push('|');
        for cell in header {
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
        out.push('|');
        for _ in header {
            out.push_str(" --- |");
        }
        out.push('\n');
    }

    // Data rows
    for row in rows.iter().skip(1) {
        out.push('|');
        for cell in row {
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
    }

    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_md(input: &str) -> Document {
        let parser = MarkdownParser;
        let source = SourceInfo::new(FileFormat::Markdown).with_filename("test.md");
        parser.parse(input.as_bytes(), &source).unwrap()
    }

    #[test]
    fn test_headings_and_paragraphs() {
        let doc = parse_md("# Title\n\nSome text.\n\n## Section\n\nMore text.");
        assert_eq!(doc.elements[0].kind, ElementKind::Title);
        assert_eq!(doc.elements[0].text, "Title");
        assert_eq!(doc.elements[1].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[2].kind, ElementKind::Heading);
        assert_eq!(doc.elements[2].depth, 1);
        assert_eq!(doc.metadata.title, Some("Title".to_string()));
    }

    #[test]
    fn test_code_block() {
        let doc = parse_md("```rust\nfn main() {}\n```");
        assert_eq!(doc.elements[0].kind, ElementKind::Code);
        assert_eq!(doc.elements[0].text, "fn main() {}");
        assert_eq!(
            doc.elements[0].attributes.get("language"),
            Some(&"rust".to_string())
        );
    }

    #[test]
    fn test_list_items() {
        let doc = parse_md("- Item one\n- Item two\n- Item three");
        assert_eq!(doc.elements.len(), 3);
        assert_eq!(doc.elements[0].kind, ElementKind::ListItem);
        assert_eq!(doc.elements[0].text, "Item one");
    }

    #[test]
    fn test_blockquote() {
        let doc = parse_md("> This is a quote.");
        assert_eq!(doc.elements[0].kind, ElementKind::BlockQuote);
        assert!(doc.elements[0].text.contains("This is a quote."));
    }

    #[test]
    fn test_horizontal_rule() {
        let doc = parse_md("---");
        assert_eq!(doc.elements[0].kind, ElementKind::HorizontalRule);
    }
}
