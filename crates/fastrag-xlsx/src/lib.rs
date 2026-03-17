use std::io::Cursor;

use calamine::{Reader, open_workbook_auto_from_rs};
use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

pub struct XlsxParser;

impl Parser for XlsxParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Xlsx]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(FileFormat::Xlsx);
        metadata.source_file = source.filename.clone();

        let cursor = Cursor::new(input);
        let mut workbook = open_workbook_auto_from_rs(cursor).map_err(|e| FastRagError::Parse {
            format: FileFormat::Xlsx,
            message: format!("Failed to open workbook: {e}"),
        })?;

        let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
        metadata.page_count = Some(sheet_names.len());
        metadata
            .custom
            .insert("sheet_names".to_string(), sheet_names.join(", "));

        let mut elements = Vec::new();

        for name in &sheet_names {
            let range = workbook
                .worksheet_range(name)
                .map_err(|e| FastRagError::Parse {
                    format: FileFormat::Xlsx,
                    message: format!("Failed to read sheet '{name}': {e}"),
                })?;

            let rows: Vec<Vec<String>> = range
                .rows()
                .map(|row| row.iter().map(|cell| format!("{cell}")).collect())
                .collect();

            if rows.is_empty() {
                continue;
            }

            let md = render_markdown_table(&rows);
            elements.push(Element::new(ElementKind::Table, md).with_section(name.as_str()));
        }

        Ok(Document { metadata, elements })
    }
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

    // We need a real XLSX file for testing. We'll create a minimal one using the test fixture.
    fn fixture_path() -> String {
        format!(
            "{}/../../tests/fixtures/sample.xlsx",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn parse_fixture() -> Document {
        let data = std::fs::read(fixture_path()).expect("sample.xlsx fixture required");
        let parser = XlsxParser;
        let source = SourceInfo::new(FileFormat::Xlsx).with_filename("sample.xlsx");
        parser.parse(&data, &source).unwrap()
    }

    #[test]
    fn supported_formats_returns_xlsx() {
        let parser = XlsxParser;
        assert_eq!(parser.supported_formats(), &[FileFormat::Xlsx]);
    }

    #[test]
    fn basic_xlsx_produces_table() {
        let doc = parse_fixture();
        assert!(doc.elements.iter().any(|e| e.kind == ElementKind::Table));
    }

    #[test]
    fn multi_sheet_produces_multiple_tables() {
        let doc = parse_fixture();
        let tables: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Table)
            .collect();
        assert_eq!(tables.len(), 2);
    }

    #[test]
    fn sheet_name_in_section() {
        let doc = parse_fixture();
        let sections: Vec<_> = doc
            .elements
            .iter()
            .filter_map(|e| e.section.as_deref())
            .collect();
        assert!(sections.contains(&"People"));
        assert!(sections.contains(&"Scores"));
    }

    #[test]
    fn metadata_page_count_equals_sheet_count() {
        let doc = parse_fixture();
        assert_eq!(doc.metadata.page_count, Some(2));
    }

    #[test]
    fn table_text_is_markdown_format() {
        let doc = parse_fixture();
        let table = doc
            .elements
            .iter()
            .find(|e| e.kind == ElementKind::Table)
            .unwrap();
        assert!(table.text.contains("|"));
        assert!(table.text.contains("---"));
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_fixture();
        assert_eq!(doc.metadata.source_file, Some("sample.xlsx".to_string()));
    }

    #[test]
    fn sheet_names_in_metadata() {
        let doc = parse_fixture();
        let names = doc.metadata.custom.get("sheet_names").unwrap();
        assert!(names.contains("People"));
        assert!(names.contains("Scores"));
    }
}
