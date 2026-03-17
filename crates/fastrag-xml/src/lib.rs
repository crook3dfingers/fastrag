use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};
use quick_xml::Reader;
use quick_xml::events::Event;

pub struct XmlParser;

impl Parser for XmlParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Xml]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(FileFormat::Xml);
        metadata.source_file = source.filename.clone();

        let mut elements = Vec::new();
        let mut reader = Reader::from_reader(input);
        let mut buf = Vec::new();
        let mut text_buf = String::new();
        let mut tag_stack: Vec<String> = Vec::new();
        let mut root_element: Option<String> = None;
        let mut in_table = false;
        let mut table_rows: Vec<Vec<String>> = Vec::new();
        let mut current_row: Vec<String> = Vec::new();
        let mut current_cell = String::new();
        let mut list_items: Vec<String> = Vec::new();
        let mut in_list = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if root_element.is_none() {
                        root_element = Some(tag.clone());
                    }

                    if matches!(tag.as_str(), "table") {
                        // Flush any pending text
                        flush_text(&mut text_buf, &tag_stack, &mut elements);
                        in_table = true;
                        table_rows.clear();
                    } else if matches!(tag.as_str(), "tr" | "row") && in_table {
                        current_row.clear();
                    } else if matches!(tag.as_str(), "td" | "th" | "cell") && in_table {
                        current_cell.clear();
                    } else if matches!(tag.as_str(), "ul" | "ol" | "list") {
                        flush_text(&mut text_buf, &tag_stack, &mut elements);
                        in_list = true;
                        list_items.clear();
                    } else if matches!(tag.as_str(), "li" | "item") && in_list {
                        text_buf.clear();
                    } else {
                        // Flush any accumulated text before a new structural tag
                        if is_structural_tag(&tag) {
                            flush_text(&mut text_buf, &tag_stack, &mut elements);
                        }
                    }

                    tag_stack.push(tag);
                }
                Ok(Event::End(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if matches!(tag.as_str(), "table") {
                        in_table = false;
                        if !table_rows.is_empty() {
                            let md = render_markdown_table(&table_rows);
                            elements.push(Element::new(ElementKind::Table, md));
                        }
                        table_rows.clear();
                    } else if matches!(tag.as_str(), "tr" | "row") && in_table {
                        if !current_row.is_empty() {
                            table_rows.push(current_row.clone());
                        }
                    } else if matches!(tag.as_str(), "td" | "th" | "cell") && in_table {
                        current_row.push(current_cell.trim().to_string());
                        current_cell.clear();
                    } else if matches!(tag.as_str(), "ul" | "ol" | "list") {
                        in_list = false;
                        for item in &list_items {
                            elements.push(Element::new(ElementKind::ListItem, item.as_str()));
                        }
                        list_items.clear();
                    } else if matches!(tag.as_str(), "li" | "item") && in_list {
                        let text = text_buf.trim().to_string();
                        if !text.is_empty() {
                            list_items.push(text);
                        }
                        text_buf.clear();
                    } else if is_structural_tag(&tag) {
                        flush_text(&mut text_buf, &tag_stack, &mut elements);
                    }

                    tag_stack.pop();
                }
                Ok(Event::Text(ref e)) => {
                    let text = e.unescape().unwrap_or_default().to_string();
                    if in_table
                        && tag_stack
                            .iter()
                            .any(|t| matches!(t.as_str(), "td" | "th" | "cell"))
                    {
                        current_cell.push_str(&text);
                    } else {
                        text_buf.push_str(&text);
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if matches!(tag.as_str(), "img" | "image") {
                        flush_text(&mut text_buf, &tag_stack, &mut elements);
                        let mut el = Element::new(ElementKind::Image, "");
                        for attr in e.attributes().flatten() {
                            let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                            let val = String::from_utf8_lossy(&attr.value).to_string();
                            if key == "src" || key == "href" {
                                el.text = val;
                            } else if key == "alt" {
                                el.attributes.insert("alt".to_string(), val);
                            }
                        }
                        elements.push(el);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(FastRagError::Parse {
                        format: FileFormat::Xml,
                        message: format!("XML parse error: {e}"),
                    });
                }
                _ => {}
            }
            buf.clear();
        }

        // Flush any remaining text
        flush_text(&mut text_buf, &tag_stack, &mut elements);

        if let Some(root) = root_element {
            metadata.custom.insert("root_element".to_string(), root);
        }

        Ok(Document { metadata, elements })
    }
}

fn is_structural_tag(tag: &str) -> bool {
    matches!(
        tag,
        "title"
            | "heading"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "p"
            | "para"
            | "code"
            | "pre"
            | "img"
            | "image"
    )
}

fn tag_to_element_kind(tag_stack: &[String]) -> (ElementKind, u8) {
    // Find the innermost structural tag
    for tag in tag_stack.iter().rev() {
        match tag.as_str() {
            "title" | "h1" => return (ElementKind::Title, 0),
            "heading" => return (ElementKind::Heading, 1),
            "h2" => return (ElementKind::Heading, 1),
            "h3" => return (ElementKind::Heading, 2),
            "h4" => return (ElementKind::Heading, 3),
            "h5" => return (ElementKind::Heading, 4),
            "h6" => return (ElementKind::Heading, 5),
            "p" | "para" => return (ElementKind::Paragraph, 0),
            "code" | "pre" => return (ElementKind::Code, 0),
            _ => {}
        }
    }
    (ElementKind::Paragraph, 0)
}

fn flush_text(text_buf: &mut String, tag_stack: &[String], elements: &mut Vec<Element>) {
    let trimmed = text_buf.trim().to_string();
    if !trimmed.is_empty() {
        let (kind, depth) = tag_to_element_kind(tag_stack);
        elements.push(Element::new(kind, trimmed).with_depth(depth));
    }
    text_buf.clear();
}

fn render_markdown_table(rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut out = String::new();

    // Header row
    if let Some(header) = rows.first() {
        out.push('|');
        for i in 0..col_count {
            let cell = header.get(i).map_or("", |s| s.as_str());
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');

        // Separator
        out.push('|');
        for _ in 0..col_count {
            out.push_str(" --- |");
        }
        out.push('\n');
    }

    // Data rows
    for row in rows.iter().skip(1) {
        out.push('|');
        for i in 0..col_count {
            let cell = row.get(i).map_or("", |s| s.as_str());
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
    }

    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_xml(xml: &str) -> Document {
        let parser = XmlParser;
        let source = SourceInfo::new(FileFormat::Xml).with_filename("test.xml");
        parser.parse(xml.as_bytes(), &source).unwrap()
    }

    #[test]
    fn supported_formats_returns_xml() {
        let parser = XmlParser;
        assert_eq!(parser.supported_formats(), &[FileFormat::Xml]);
    }

    #[test]
    fn basic_xml_produces_paragraphs() {
        let doc = parse_xml("<root><p>Hello world</p><p>Second paragraph</p></root>");
        assert_eq!(doc.elements.len(), 2);
        assert_eq!(doc.elements[0].kind, ElementKind::Paragraph);
        assert_eq!(doc.elements[0].text, "Hello world");
        assert_eq!(doc.elements[1].text, "Second paragraph");
    }

    #[test]
    fn xml_title_element_detected() {
        let doc = parse_xml("<root><title>My Document</title><p>Body text</p></root>");
        assert_eq!(doc.elements[0].kind, ElementKind::Title);
        assert_eq!(doc.elements[0].text, "My Document");
    }

    #[test]
    fn xml_heading_elements() {
        let doc = parse_xml("<root><h2>Section</h2><h3>Subsection</h3></root>");
        assert_eq!(doc.elements[0].kind, ElementKind::Heading);
        assert_eq!(doc.elements[0].depth, 1);
        assert_eq!(doc.elements[1].kind, ElementKind::Heading);
        assert_eq!(doc.elements[1].depth, 2);
    }

    #[test]
    fn xml_table_rendered_as_markdown() {
        let xml = "<root><table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table></root>";
        let doc = parse_xml(xml);
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].kind, ElementKind::Table);
        assert!(doc.elements[0].text.contains("| Name | Age |"));
        assert!(doc.elements[0].text.contains("| --- |"));
        assert!(doc.elements[0].text.contains("| Alice | 30 |"));
    }

    #[test]
    fn xml_code_element() {
        let doc = parse_xml("<root><code>let x = 1;</code></root>");
        assert_eq!(doc.elements[0].kind, ElementKind::Code);
        assert_eq!(doc.elements[0].text, "let x = 1;");
    }

    #[test]
    fn xml_list_items() {
        let doc = parse_xml("<root><ul><li>First</li><li>Second</li></ul></root>");
        assert_eq!(doc.elements.len(), 2);
        assert_eq!(doc.elements[0].kind, ElementKind::ListItem);
        assert_eq!(doc.elements[0].text, "First");
        assert_eq!(doc.elements[1].text, "Second");
    }

    #[test]
    fn invalid_xml_returns_parse_error() {
        let parser = XmlParser;
        let source = SourceInfo::new(FileFormat::Xml);
        let result = parser.parse(b"<a><b></a></b>", &source);
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Parse { format, .. } => assert_eq!(format, FileFormat::Xml),
            other => panic!("expected Parse error, got: {other}"),
        }
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_xml("<root><p>Hi</p></root>");
        assert_eq!(doc.metadata.source_file, Some("test.xml".to_string()));
    }

    #[test]
    fn root_element_stored_in_metadata() {
        let doc = parse_xml("<myroot><p>text</p></myroot>");
        assert_eq!(
            doc.metadata.custom.get("root_element"),
            Some(&"myroot".to_string())
        );
    }
}
