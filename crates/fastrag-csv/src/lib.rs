use std::collections::HashMap;

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// CSV parser. Converts CSV data into Table elements, chunked at `rows_per_chunk` rows.
pub struct CsvParser {
    pub rows_per_chunk: usize,
}

impl Default for CsvParser {
    fn default() -> Self {
        Self {
            rows_per_chunk: 100,
        }
    }
}

impl Parser for CsvParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Csv]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(input);

        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| FastRagError::Parse {
                format: FileFormat::Csv,
                message: e.to_string(),
            })?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut elements = Vec::new();
        let mut chunk_rows: Vec<Vec<String>> = Vec::new();
        let mut total_rows: usize = 0;

        for result in reader.records() {
            let record = result.map_err(|e| FastRagError::Parse {
                format: FileFormat::Csv,
                message: e.to_string(),
            })?;

            chunk_rows.push(record.iter().map(|f| f.to_string()).collect());
            total_rows += 1;

            if chunk_rows.len() >= self.rows_per_chunk {
                elements.push(make_table_element(&headers, &chunk_rows));
                chunk_rows.clear();
            }
        }

        if !chunk_rows.is_empty() {
            elements.push(make_table_element(&headers, &chunk_rows));
        }

        metadata
            .custom
            .insert("row_count".to_string(), total_rows.to_string());
        metadata
            .custom
            .insert("column_count".to_string(), headers.len().to_string());

        Ok(Document { metadata, elements })
    }
}

fn make_table_element(headers: &[String], rows: &[Vec<String>]) -> Element {
    let mut table = String::new();

    table.push('|');
    for h in headers {
        table.push_str(&format!(" {h} |"));
    }
    table.push('\n');

    table.push('|');
    for _ in headers {
        table.push_str(" --- |");
    }
    table.push('\n');

    for row in rows {
        table.push('|');
        for (i, cell) in row.iter().enumerate() {
            if i < headers.len() {
                table.push_str(&format!(" {cell} |"));
            }
        }
        table.push('\n');
    }

    let mut attrs = HashMap::new();
    attrs.insert("rows".to_string(), rows.len().to_string());
    attrs.insert("columns".to_string(), headers.len().to_string());

    Element {
        kind: ElementKind::Table,
        text: table,
        page: None,
        section: None,
        depth: 0,
        attributes: attrs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv(input: &str) -> Document {
        let parser = CsvParser::default();
        let source = SourceInfo::new(FileFormat::Csv).with_filename("test.csv");
        parser.parse(input.as_bytes(), &source).unwrap()
    }

    #[test]
    fn supported_formats_returns_csv() {
        assert_eq!(CsvParser::default().supported_formats(), &[FileFormat::Csv]);
    }

    #[test]
    fn basic_csv_produces_table() {
        let doc = parse_csv("name,age\nAlice,30\nBob,25\n");
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].kind, ElementKind::Table);
        assert!(doc.elements[0].text.contains("| name |"));
        assert!(doc.elements[0].text.contains("| --- |"));
        assert!(doc.elements[0].text.contains("| Alice | 30 |"));
    }

    #[test]
    fn metadata_row_and_column_count() {
        let doc = parse_csv("a,b\n1,2\n3,4\n");
        assert_eq!(doc.metadata.custom["row_count"], "2");
        assert_eq!(doc.metadata.custom["column_count"], "2");
    }

    #[test]
    fn chunking_200_rows() {
        let mut csv = String::from("id,value\n");
        for i in 0..200 {
            csv.push_str(&format!("{i},data{i}\n"));
        }
        let parser = CsvParser {
            rows_per_chunk: 100,
        };
        let source = SourceInfo::new(FileFormat::Csv);
        let doc = parser.parse(csv.as_bytes(), &source).unwrap();
        assert_eq!(doc.elements.len(), 2);
    }

    #[test]
    fn chunking_250_rows() {
        let mut csv = String::from("id,value\n");
        for i in 0..250 {
            csv.push_str(&format!("{i},data{i}\n"));
        }
        let parser = CsvParser {
            rows_per_chunk: 100,
        };
        let source = SourceInfo::new(FileFormat::Csv);
        let doc = parser.parse(csv.as_bytes(), &source).unwrap();
        assert_eq!(doc.elements.len(), 3);
    }

    #[test]
    fn table_element_has_rows_columns_attrs() {
        let doc = parse_csv("a,b,c\n1,2,3\n");
        assert_eq!(doc.elements[0].attributes["rows"], "1");
        assert_eq!(doc.elements[0].attributes["columns"], "3");
    }

    #[test]
    fn headers_only_no_data_rows() {
        let doc = parse_csv("name,age\n");
        assert!(doc.elements.is_empty());
    }

    #[test]
    fn single_column() {
        let doc = parse_csv("name\nAlice\nBob\n");
        assert_eq!(doc.elements.len(), 1);
        assert!(doc.elements[0].text.contains("| name |"));
        assert!(doc.elements[0].text.contains("| Alice |"));
    }

    #[test]
    fn source_file_propagated() {
        let doc = parse_csv("a\n1\n");
        assert_eq!(doc.metadata.source_file, Some("test.csv".to_string()));
    }

    #[test]
    fn default_rows_per_chunk() {
        assert_eq!(CsvParser::default().rows_per_chunk, 100);
    }
}
