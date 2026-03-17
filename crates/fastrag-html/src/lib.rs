use scraper::{Html, Selector};

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// HTML parser using scraper. Strips nav, script, style elements and extracts content.
pub struct HtmlParser;

impl Parser for HtmlParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Html]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let text =
            String::from_utf8(input.to_vec()).map_err(|e| FastRagError::Encoding(e.to_string()))?;

        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let html = Html::parse_document(&text);

        // Extract title from <title> tag
        if let Some(title_el) = Selector::parse("title")
            .ok()
            .and_then(|s| html.select(&s).next())
        {
            let title_text: String = title_el.text().collect();
            let title_text = title_text.trim().to_string();
            if !title_text.is_empty() {
                metadata.title = Some(title_text);
            }
        }

        let mut elements = Vec::new();

        // Content selectors in order of priority
        let content_selectors = [
            ("h1", ElementKind::Title, 0u8),
            ("h2", ElementKind::Heading, 1),
            ("h3", ElementKind::Heading, 2),
            ("h4", ElementKind::Heading, 3),
            ("h5", ElementKind::Heading, 4),
            ("h6", ElementKind::Heading, 5),
        ];

        // Try to find a main content area
        let body_selector = Selector::parse("article, main, [role='main'], .content, .post, body")
            .expect("valid selector");

        if let Some(body) = html.select(&body_selector).next() {
            // Process all descendant elements in document order
            for node in body.descendants() {
                if let Some(el) = node.value().as_element() {
                    let tag = el.name();

                    // Skip non-content tags
                    if matches!(
                        tag,
                        "script" | "style" | "nav" | "header" | "footer" | "noscript" | "svg"
                    ) {
                        continue;
                    }

                    let text: String = node
                        .children()
                        .filter_map(|c| c.value().as_text().map(|t| t.to_string()))
                        .collect::<Vec<_>>()
                        .join("")
                        .trim()
                        .to_string();

                    if text.is_empty() {
                        continue;
                    }

                    // Map HTML tags to element kinds
                    let mut matched = false;
                    for (selector_tag, kind, depth) in &content_selectors {
                        if tag == *selector_tag {
                            elements.push(Element::new(kind.clone(), &text).with_depth(*depth));
                            matched = true;
                            break;
                        }
                    }

                    if !matched {
                        match tag {
                            "p" => {
                                elements.push(Element::new(ElementKind::Paragraph, &text));
                            }
                            "pre" | "code" => {
                                // Get full text including nested elements
                                let full_text: String = scraper::ElementRef::wrap(node)
                                    .map(|er| er.text().collect::<String>())
                                    .unwrap_or(text.clone());
                                elements.push(Element::new(ElementKind::Code, full_text.trim()));
                            }
                            "blockquote" => {
                                let full_text: String = scraper::ElementRef::wrap(node)
                                    .map(|er| er.text().collect::<String>())
                                    .unwrap_or(text.clone());
                                elements
                                    .push(Element::new(ElementKind::BlockQuote, full_text.trim()));
                            }
                            "li" => {
                                elements.push(Element::new(ElementKind::ListItem, &text));
                            }
                            "hr" => {
                                elements.push(Element::new(ElementKind::HorizontalRule, ""));
                            }
                            "table" => {
                                let full_text: String = scraper::ElementRef::wrap(node)
                                    .map(|er| render_html_table(er))
                                    .unwrap_or(text.clone());
                                elements.push(Element::new(ElementKind::Table, full_text.trim()));
                            }
                            "img" => {
                                let src = el.attr("src").unwrap_or("");
                                let alt = el.attr("alt").unwrap_or("");
                                let mut elem = Element::new(ElementKind::Image, src);
                                elem.attributes.insert("alt".to_string(), alt.to_string());
                                elements.push(elem);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(Document { metadata, elements })
    }
}

fn render_html_table(table: scraper::ElementRef) -> String {
    let mut out = String::new();
    let row_sel = Selector::parse("tr").expect("valid");
    let th_sel = Selector::parse("th").expect("valid");
    let td_sel = Selector::parse("td").expect("valid");

    let mut is_first_row = true;
    for row in table.select(&row_sel) {
        out.push('|');
        let cells: Vec<String> = row
            .select(&th_sel)
            .chain(row.select(&td_sel))
            .map(|c| c.text().collect::<String>().trim().to_string())
            .collect();

        for cell in &cells {
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');

        if is_first_row {
            out.push('|');
            for _ in &cells {
                out.push_str(" --- |");
            }
            out.push('\n');
            is_first_row = false;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_html(input: &str) -> Document {
        let parser = HtmlParser;
        let source = SourceInfo::new(FileFormat::Html).with_filename("test.html");
        parser.parse(input.as_bytes(), &source).unwrap()
    }

    #[test]
    fn test_basic_html() {
        let doc = parse_html(
            "<html><head><title>Test Page</title></head>\
             <body><h1>Hello</h1><p>World</p></body></html>",
        );
        assert_eq!(doc.metadata.title, Some("Test Page".to_string()));
        assert!(
            doc.elements
                .iter()
                .any(|e| e.kind == ElementKind::Title && e.text == "Hello")
        );
        assert!(
            doc.elements
                .iter()
                .any(|e| e.kind == ElementKind::Paragraph && e.text == "World")
        );
    }

    #[test]
    fn test_strips_script_style() {
        let doc = parse_html(
            "<html><body>\
             <script>alert('xss')</script>\
             <style>body { color: red; }</style>\
             <p>Content</p>\
             </body></html>",
        );
        assert!(
            !doc.elements
                .iter()
                .any(|e| e.text.contains("alert") || e.text.contains("color"))
        );
        assert!(
            doc.elements
                .iter()
                .any(|e| e.kind == ElementKind::Paragraph && e.text == "Content")
        );
    }

    #[test]
    fn test_heading_levels() {
        let doc = parse_html(
            "<html><body>\
             <h1>H1</h1><h2>H2</h2><h3>H3</h3>\
             </body></html>",
        );
        let headings: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Title || e.kind == ElementKind::Heading)
            .collect();
        assert_eq!(headings.len(), 3);
        assert_eq!(headings[0].depth, 0); // h1
        assert_eq!(headings[1].depth, 1); // h2
        assert_eq!(headings[2].depth, 2); // h3
    }
}
