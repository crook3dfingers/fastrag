use std::io::{Cursor, Read};

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};
use quick_xml::Reader;
use quick_xml::events::Event;

pub struct DocxParser;

impl Parser for DocxParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Docx]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(FileFormat::Docx);
        metadata.source_file = source.filename.clone();

        let cursor = Cursor::new(input);
        let mut archive = zip::ZipArchive::new(cursor).map_err(|e| FastRagError::Parse {
            format: FileFormat::Docx,
            message: format!("Invalid ZIP archive: {e}"),
        })?;

        // Extract metadata from docProps/core.xml
        if let Ok(mut core_file) = archive.by_name("docProps/core.xml") {
            let mut core_xml = String::new();
            core_file.read_to_string(&mut core_xml).ok();
            parse_core_metadata(&core_xml, &mut metadata);
        }

        // Extract document body from word/document.xml
        let mut doc_xml = String::new();
        {
            let mut doc_file =
                archive
                    .by_name("word/document.xml")
                    .map_err(|e| FastRagError::Parse {
                        format: FileFormat::Docx,
                        message: format!("Missing word/document.xml: {e}"),
                    })?;
            doc_file
                .read_to_string(&mut doc_xml)
                .map_err(|e| FastRagError::Parse {
                    format: FileFormat::Docx,
                    message: format!("Failed to read document.xml: {e}"),
                })?;
        }

        let elements = parse_document_xml(&doc_xml)?;

        Ok(Document { metadata, elements })
    }
}

fn parse_core_metadata(xml: &str, metadata: &mut Metadata) {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut current_tag = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                current_tag = name;
            }
            Ok(Event::Text(ref e)) => {
                let text = e.unescape().unwrap_or_default().trim().to_string();
                if !text.is_empty() {
                    match current_tag.as_str() {
                        "dc:title" => metadata.title = Some(text),
                        "dc:creator" => metadata.author = Some(text),
                        "dcterms:created" => metadata.created_at = Some(text),
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
}

fn parse_document_xml(xml: &str) -> Result<Vec<Element>, FastRagError> {
    let mut elements = Vec::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut tag_stack: Vec<String> = Vec::new();
    let mut text_buf = String::new();
    let mut in_body = false;
    let mut current_style_id = String::new();
    let mut has_num_pr = false;
    let mut in_table = false;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_cell_text = String::new();
    let mut in_drawing = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                let tag = local.to_string();

                if tag == "body" {
                    in_body = true;
                }

                if in_body {
                    match tag.as_str() {
                        "p" => {
                            text_buf.clear();
                            current_style_id.clear();
                            has_num_pr = false;
                        }
                        "tbl" => {
                            in_table = true;
                            table_rows.clear();
                        }
                        "tr" => {
                            current_row.clear();
                        }
                        "tc" => {
                            current_cell_text.clear();
                        }
                        "drawing" => {
                            in_drawing = true;
                        }
                        "pStyle" => {
                            // Extract val attribute for style ID
                            for attr in e.attributes().flatten() {
                                let key = local_name(attr.key.as_ref());
                                if key == "val" {
                                    current_style_id =
                                        String::from_utf8_lossy(&attr.value).to_string();
                                }
                            }
                        }
                        "numPr" => {
                            has_num_pr = true;
                        }
                        _ => {}
                    }
                }

                tag_stack.push(tag);
            }
            Ok(Event::End(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                let tag = local.to_string();

                if in_body {
                    match tag.as_str() {
                        "p" => {
                            if in_table {
                                // Text in table cell paragraph
                                let trimmed = text_buf.trim().to_string();
                                if !trimmed.is_empty() {
                                    if !current_cell_text.is_empty() {
                                        current_cell_text.push(' ');
                                    }
                                    current_cell_text.push_str(&trimmed);
                                }
                            } else if in_drawing {
                                // Skip paragraphs inside drawings
                            } else {
                                let trimmed = text_buf.trim().to_string();
                                if !trimmed.is_empty() {
                                    let el =
                                        classify_paragraph(&trimmed, &current_style_id, has_num_pr);
                                    elements.push(el);
                                }
                            }
                            text_buf.clear();
                        }
                        "tc" => {
                            current_row.push(current_cell_text.trim().to_string());
                            current_cell_text.clear();
                        }
                        "tr" => {
                            if !current_row.is_empty() {
                                table_rows.push(current_row.clone());
                            }
                        }
                        "tbl" => {
                            in_table = false;
                            if !table_rows.is_empty() {
                                let md = render_markdown_table(&table_rows);
                                elements.push(Element::new(ElementKind::Table, md));
                            }
                            table_rows.clear();
                        }
                        "drawing" => {
                            in_drawing = false;
                            elements.push(Element::new(ElementKind::Image, ""));
                        }
                        "body" => {
                            in_body = false;
                        }
                        _ => {}
                    }
                }

                tag_stack.pop();
            }
            Ok(Event::Text(ref e)) => {
                if in_body {
                    // Only collect text from w:t elements
                    if tag_stack.last().is_some_and(|t| t == "t") {
                        let text = e.unescape().unwrap_or_default().to_string();
                        text_buf.push_str(&text);
                    }
                }
            }
            Ok(Event::Empty(ref e)) => {
                if in_body {
                    let name_bytes = e.name();
                    let local = local_name(name_bytes.as_ref());
                    match local {
                        "pStyle" => {
                            for attr in e.attributes().flatten() {
                                let key = local_name(attr.key.as_ref());
                                if key == "val" {
                                    current_style_id =
                                        String::from_utf8_lossy(&attr.value).to_string();
                                }
                            }
                        }
                        "numPr" => {
                            has_num_pr = true;
                        }
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(FastRagError::Parse {
                    format: FileFormat::Docx,
                    message: format!("XML parse error in document.xml: {e}"),
                });
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(elements)
}

fn classify_paragraph(text: &str, style_id: &str, has_num_pr: bool) -> Element {
    // Match on style ID (locale-independent)
    match style_id {
        "Title" => return Element::new(ElementKind::Title, text),
        s if s.starts_with("Heading") || s.starts_with("heading") => {
            let depth = s.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(1) as u8;
            // Heading1 → depth 0 (Title-level), Heading2 → depth 1, etc.
            let adjusted = depth.saturating_sub(1);
            return Element::new(ElementKind::Heading, text).with_depth(adjusted);
        }
        _ => {}
    }

    if has_num_pr || style_id.contains("List") {
        return Element::new(ElementKind::ListItem, text);
    }

    Element::new(ElementKind::Paragraph, text)
}

fn local_name(full: &[u8]) -> &str {
    let s = std::str::from_utf8(full).unwrap_or("");
    s.rsplit_once(':').map_or(s, |(_, local)| local)
}

fn render_markdown_table(rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut out = String::new();

    if let Some(header) = rows.first() {
        out.push('|');
        for i in 0..col_count {
            let cell = header.get(i).map_or("", |s| s.as_str());
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');

        out.push('|');
        for _ in 0..col_count {
            out.push_str(" --- |");
        }
        out.push('\n');
    }

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

    fn fixture_path() -> String {
        format!(
            "{}/../../tests/fixtures/sample.docx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn parse_fixture() -> Document {
        let data = std::fs::read(fixture_path()).expect("sample.docx fixture required");
        let parser = DocxParser;
        let source = SourceInfo::new(FileFormat::Docx).with_filename("sample.docx");
        parser.parse(&data, &source).unwrap()
    }

    #[test]
    fn supported_formats_returns_docx() {
        let parser = DocxParser;
        assert_eq!(parser.supported_formats(), &[FileFormat::Docx]);
    }

    #[test]
    fn basic_docx_extracts_paragraphs() {
        let doc = parse_fixture();
        let paragraphs: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Paragraph)
            .collect();
        assert!(
            paragraphs.len() >= 2,
            "expected at least 2 paragraphs, got {}",
            paragraphs.len()
        );
        assert!(
            paragraphs
                .iter()
                .any(|p| p.text.contains("first paragraph"))
        );
    }

    #[test]
    fn docx_title_style_becomes_title_element() {
        let doc = parse_fixture();
        let titles: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Title)
            .collect();
        assert!(!titles.is_empty(), "expected at least one Title element");
        assert!(titles[0].text.contains("Sample Document Title"));
    }

    #[test]
    fn docx_heading_styles_with_correct_depth() {
        let doc = parse_fixture();
        let headings: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Heading)
            .collect();
        assert!(headings.len() >= 2);
        // Heading2 → depth 1, Heading3 → depth 2
        assert!(
            headings
                .iter()
                .any(|h| h.text == "Section One" && h.depth == 1)
        );
        assert!(
            headings
                .iter()
                .any(|h| h.text == "Subsection" && h.depth == 2)
        );
    }

    #[test]
    fn docx_table_rendered_as_markdown() {
        let doc = parse_fixture();
        let tables: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Table)
            .collect();
        assert!(!tables.is_empty(), "expected at least one Table element");
        assert!(tables[0].text.contains("| Name | Value |"));
        assert!(tables[0].text.contains("| --- |"));
        assert!(tables[0].text.contains("| Alpha | 100 |"));
    }

    #[test]
    fn docx_metadata_extracts_title_author() {
        let doc = parse_fixture();
        assert_eq!(
            doc.metadata.title,
            Some("Sample Document Title".to_string())
        );
        assert_eq!(doc.metadata.author, Some("Test Author".to_string()));
    }

    #[test]
    fn docx_list_items_detected() {
        let doc = parse_fixture();
        let list_items: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::ListItem)
            .collect();
        assert!(
            list_items.len() >= 2,
            "expected at least 2 list items, got {}",
            list_items.len()
        );
    }

    #[test]
    fn invalid_zip_returns_parse_error() {
        let parser = DocxParser;
        let source = SourceInfo::new(FileFormat::Docx);
        let result = parser.parse(b"not a zip file", &source);
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Parse { format, .. } => assert_eq!(format, FileFormat::Docx),
            other => panic!("expected Parse error, got: {other}"),
        }
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_fixture();
        assert_eq!(doc.metadata.source_file, Some("sample.docx".to_string()));
    }
}
