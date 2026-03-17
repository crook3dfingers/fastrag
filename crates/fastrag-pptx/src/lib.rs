use std::io::{Cursor, Read};

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};
use quick_xml::Reader;
use quick_xml::events::Event;

pub struct PptxParser;

impl Parser for PptxParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Pptx]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(FileFormat::Pptx);
        metadata.source_file = source.filename.clone();

        let cursor = Cursor::new(input);
        let mut archive = zip::ZipArchive::new(cursor).map_err(|e| FastRagError::Parse {
            format: FileFormat::Pptx,
            message: format!("Invalid ZIP archive: {e}"),
        })?;

        // Extract metadata from docProps/core.xml
        if let Ok(mut core_file) = archive.by_name("docProps/core.xml") {
            let mut core_xml = String::new();
            core_file.read_to_string(&mut core_xml).ok();
            parse_core_metadata(&core_xml, &mut metadata);
        }

        // Find and sort slide files
        let mut slide_names: Vec<String> = archive
            .file_names()
            .filter(|n| {
                n.starts_with("ppt/slides/slide") && n.ends_with(".xml") && !n.contains("Layout")
            })
            .map(|s| s.to_string())
            .collect();
        slide_names.sort_by(|a, b| {
            let num_a = extract_slide_number(a);
            let num_b = extract_slide_number(b);
            num_a.cmp(&num_b)
        });

        metadata.page_count = Some(slide_names.len());

        let mut elements = Vec::new();

        for (i, slide_name) in slide_names.iter().enumerate() {
            let slide_num = i + 1;
            let mut slide_xml = String::new();
            {
                let mut file = archive
                    .by_name(slide_name)
                    .map_err(|e| FastRagError::Parse {
                        format: FileFormat::Pptx,
                        message: format!("Failed to read {slide_name}: {e}"),
                    })?;
                file.read_to_string(&mut slide_xml)
                    .map_err(|e| FastRagError::Parse {
                        format: FileFormat::Pptx,
                        message: format!("Failed to read {slide_name}: {e}"),
                    })?;
            }

            let slide_elements = parse_slide_xml(&slide_xml, slide_num)?;
            elements.extend(slide_elements);
        }

        Ok(Document { metadata, elements })
    }
}

fn extract_slide_number(name: &str) -> usize {
    // Extract number from "ppt/slides/slideN.xml"
    name.strip_prefix("ppt/slides/slide")
        .and_then(|s| s.strip_suffix(".xml"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
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

fn parse_slide_xml(xml: &str, slide_num: usize) -> Result<Vec<Element>, FastRagError> {
    let mut elements = Vec::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut tag_stack: Vec<String> = Vec::new();
    let mut text_buf = String::new();

    // Track current shape's placeholder type
    let mut current_ph_type: Option<String> = None;
    let mut in_sp = false; // inside p:sp (shape)
    let mut in_txbody = false;
    let mut in_table = false;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut current_cell_text = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                let tag = local.to_string();

                match tag.as_str() {
                    "sp" => {
                        in_sp = true;
                        current_ph_type = None;
                    }
                    "txBody" => {
                        in_txbody = true;
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
                    "p" if in_txbody || in_table => {
                        text_buf.clear();
                    }
                    _ => {}
                }

                tag_stack.push(tag);
            }
            Ok(Event::End(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());
                let tag = local.to_string();

                match tag.as_str() {
                    "sp" => {
                        in_sp = false;
                        current_ph_type = None;
                    }
                    "txBody" => {
                        in_txbody = false;
                    }
                    "p" if in_table => {
                        let trimmed = text_buf.trim().to_string();
                        if !trimmed.is_empty() {
                            if !current_cell_text.is_empty() {
                                current_cell_text.push(' ');
                            }
                            current_cell_text.push_str(&trimmed);
                        }
                        text_buf.clear();
                    }
                    "p" if in_txbody => {
                        let trimmed = text_buf.trim().to_string();
                        if !trimmed.is_empty() {
                            let el = classify_text(&trimmed, current_ph_type.as_deref(), slide_num);
                            elements.push(el);
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
                            elements
                                .push(Element::new(ElementKind::Table, md).with_page(slide_num));
                        }
                        table_rows.clear();
                    }
                    _ => {}
                }

                tag_stack.pop();
            }
            Ok(Event::Empty(ref e)) => {
                let name_bytes = e.name();
                let local = local_name(name_bytes.as_ref());

                if local == "ph" && in_sp {
                    // Extract placeholder type
                    for attr in e.attributes().flatten() {
                        let key = local_name(attr.key.as_ref());
                        if key == "type" {
                            current_ph_type =
                                Some(String::from_utf8_lossy(&attr.value).to_string());
                        }
                    }
                }
            }
            Ok(Event::Text(ref e)) => {
                // Collect text from a:t elements
                if tag_stack.last().is_some_and(|t| t == "t") && (in_txbody || in_table) {
                    let text = e.unescape().unwrap_or_default().to_string();
                    text_buf.push_str(&text);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(FastRagError::Parse {
                    format: FileFormat::Pptx,
                    message: format!("XML parse error in slide: {e}"),
                });
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(elements)
}

fn classify_text(text: &str, ph_type: Option<&str>, slide_num: usize) -> Element {
    match ph_type {
        Some("title") | Some("ctrTitle") => {
            if slide_num == 1 {
                Element::new(ElementKind::Title, text).with_page(slide_num)
            } else {
                Element::new(ElementKind::Heading, text)
                    .with_page(slide_num)
                    .with_depth(0)
            }
        }
        Some("subTitle") => Element::new(ElementKind::Paragraph, text).with_page(slide_num),
        _ => Element::new(ElementKind::Paragraph, text).with_page(slide_num),
    }
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
            "{}/../../tests/fixtures/sample.pptx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn parse_fixture() -> Document {
        let data = std::fs::read(fixture_path()).expect("sample.pptx fixture required");
        let parser = PptxParser;
        let source = SourceInfo::new(FileFormat::Pptx).with_filename("sample.pptx");
        parser.parse(&data, &source).unwrap()
    }

    #[test]
    fn supported_formats_returns_pptx() {
        let parser = PptxParser;
        assert_eq!(parser.supported_formats(), &[FileFormat::Pptx]);
    }

    #[test]
    fn basic_pptx_extracts_slide_text() {
        let doc = parse_fixture();
        assert!(
            !doc.elements.is_empty(),
            "expected elements from slide text"
        );
        let texts: Vec<&str> = doc.elements.iter().map(|e| e.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("Presentation Title")),
            "expected 'Presentation Title' in elements, got: {texts:?}"
        );
    }

    #[test]
    fn pptx_title_placeholder_becomes_title() {
        let doc = parse_fixture();
        let titles: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Title)
            .collect();
        assert!(!titles.is_empty(), "expected at least one Title element");
        assert!(titles[0].text.contains("Presentation Title"));
    }

    #[test]
    fn pptx_page_numbers_match_slides() {
        let doc = parse_fixture();
        // Should have elements from slide 1, 2, and 3
        let pages: Vec<usize> = doc.elements.iter().filter_map(|e| e.page).collect();
        assert!(pages.contains(&1));
        assert!(pages.contains(&2));
        assert!(pages.contains(&3));
    }

    #[test]
    fn pptx_metadata_page_count() {
        let doc = parse_fixture();
        assert_eq!(doc.metadata.page_count, Some(3));
    }

    #[test]
    fn pptx_table_rendered_as_markdown() {
        let doc = parse_fixture();
        let tables: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Table)
            .collect();
        assert!(!tables.is_empty(), "expected at least one Table element");
        assert!(tables[0].text.contains("| Item | Count |"));
        assert!(tables[0].text.contains("| --- |"));
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_fixture();
        assert_eq!(doc.metadata.source_file, Some("sample.pptx".to_string()));
    }
}
