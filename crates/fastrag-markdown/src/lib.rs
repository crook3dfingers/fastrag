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
            return;
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
    fn supported_formats_returns_markdown() {
        assert_eq!(MarkdownParser.supported_formats(), &[FileFormat::Markdown]);
    }

    #[test]
    fn h1_becomes_title_depth0() {
        let doc = parse_md("# Title");
        assert_eq!(doc.elements[0].kind, ElementKind::Title);
        assert_eq!(doc.elements[0].text, "Title");
        assert_eq!(doc.elements[0].depth, 0);
    }

    #[test]
    fn h2_becomes_heading_depth1() {
        let doc = parse_md("## Section");
        assert_eq!(doc.elements[0].kind, ElementKind::Heading);
        assert_eq!(doc.elements[0].depth, 1);
        assert_eq!(doc.elements[0].text, "Section");
    }

    #[test]
    fn h3_becomes_heading_depth2() {
        let doc = parse_md("### Sub");
        assert_eq!(doc.elements[0].kind, ElementKind::Heading);
        assert_eq!(doc.elements[0].depth, 2);
    }

    #[test]
    fn plain_paragraph_text() {
        let doc = parse_md("Hello world.");
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[0].text, "Hello world.");
    }

    #[test]
    fn code_block_with_language() {
        let doc = parse_md("```rust\nfn main() {}\n```");
        assert_eq!(doc.elements[0].kind, ElementKind::Code);
        assert_eq!(doc.elements[0].text, "fn main() {}");
        assert_eq!(
            doc.elements[0].attributes.get("language"),
            Some(&"rust".to_string())
        );
    }

    #[test]
    fn code_block_without_language() {
        let doc = parse_md("```\ncode here\n```");
        assert_eq!(doc.elements[0].kind, ElementKind::Code);
        assert_eq!(doc.elements[0].text, "code here");
        assert!(doc.elements[0].attributes.get("language").is_none());
    }

    #[test]
    fn list_items() {
        let doc = parse_md("- A\n- B\n- C");
        assert_eq!(doc.elements.len(), 3);
        assert_eq!(doc.elements[0].kind, ElementKind::ListItem);
        assert_eq!(doc.elements[0].text, "A");
        assert_eq!(doc.elements[1].text, "B");
        assert_eq!(doc.elements[2].text, "C");
    }

    #[test]
    fn blockquote() {
        let doc = parse_md("> A wise quote.");
        assert_eq!(doc.elements[0].kind, ElementKind::BlockQuote);
        assert!(doc.elements[0].text.contains("A wise quote."));
    }

    #[test]
    fn horizontal_rule() {
        let doc = parse_md("---");
        assert_eq!(doc.elements[0].kind, ElementKind::HorizontalRule);
    }

    #[test]
    fn table_produces_table_element() {
        let doc = parse_md("| A | B |\n|---|---|\n| 1 | 2 |");
        assert_eq!(doc.elements[0].kind, ElementKind::Table);
        assert!(doc.elements[0].text.contains("| A |"));
        assert!(doc.elements[0].text.contains("| 1 |"));
    }

    #[test]
    fn title_sets_metadata_title() {
        let doc = parse_md("# My Doc\n\nSome text.");
        assert_eq!(doc.metadata.title, Some("My Doc".to_string()));
    }

    #[test]
    fn inline_code_preserved() {
        let doc = parse_md("Use `println!` for output.");
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert!(doc.elements[0].text.contains("`println!`"));
    }

    #[test]
    fn complex_doc_fixture_content() {
        let input = "# FastRAG Sample Document\n\nThis is a sample markdown document for testing.\n\n## Features\n\n- Fast parsing\n- Multiple format support\n- Structured output\n\n## Code Example\n\n```rust\nfn main() {\n    println!(\"Hello, FastRAG!\");\n}\n```\n\n## Table\n\n| Feature | Status |\n|---------|--------|\n| PDF     | Done   |\n| HTML    | Done   |\n| Markdown| Done   |\n\n---\n\n> FastRAG: 100x faster document parsing.";
        let doc = parse_md(input);
        assert_eq!(
            doc.metadata.title,
            Some("FastRAG Sample Document".to_string())
        );
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::Title));
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::ListItem));
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::Code));
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::Table));
        assert!(
            doc.elements
                .iter()
                .any(|e| e.kind == ElementKind::HorizontalRule)
        );
        assert!(
            doc.elements
                .iter()
                .any(|e| e.kind == ElementKind::BlockQuote)
        );
    }
}
