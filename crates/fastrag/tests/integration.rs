use fastrag::{ElementKind, parse};

fn fixtures() -> String {
    format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn parse_sample_txt() {
    let doc = parse(format!("{}/sample.txt", fixtures())).unwrap();
    let first = &doc.elements[0];
    assert_eq!(first.kind, ElementKind::Title);
    assert_eq!(first.text, "INTRODUCTION");

    assert!(
        doc.elements
            .iter()
            .any(|e| e.kind == ElementKind::Heading && e.text == "Summary")
    );

    assert!(
        doc.elements
            .iter()
            .any(|e| e.kind == ElementKind::Title && e.text == "CONCLUSION")
    );
}

#[test]
fn parse_sample_csv() {
    let doc = parse(format!("{}/sample.csv", fixtures())).unwrap();
    let table = doc
        .elements
        .iter()
        .find(|e| e.kind == ElementKind::Table)
        .expect("should have a Table element");
    assert!(table.text.contains("| Alice | 30 | New York | Engineer |"));
    assert_eq!(doc.metadata.custom["row_count"], "4");
    assert_eq!(doc.metadata.custom["column_count"], "4");
}

#[test]
fn parse_sample_md() {
    let doc = parse(format!("{}/sample.md", fixtures())).unwrap();
    assert_eq!(
        doc.metadata.title,
        Some("FastRAG Sample Document".to_string())
    );

    let list_items: Vec<_> = doc
        .elements
        .iter()
        .filter(|e| e.kind == ElementKind::ListItem)
        .collect();
    assert_eq!(list_items.len(), 3);

    assert!(
        doc.elements
            .iter()
            .any(|e| e.kind == ElementKind::Code && e.text.contains("println!"))
    );
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

#[test]
fn parse_sample_html() {
    let doc = parse(format!("{}/sample.html", fixtures())).unwrap();
    assert_eq!(doc.metadata.title, Some("FastRAG Test Page".to_string()));
    assert!(
        doc.elements
            .iter()
            .any(|e| e.kind == ElementKind::Title && e.text == "Welcome to FastRAG")
    );
    assert!(
        doc.elements
            .iter()
            .any(|e| e.kind == ElementKind::Paragraph && e.text.contains("fast document parser"))
    );
    assert!(doc.elements.iter().any(|e| e.kind == ElementKind::ListItem));
}

#[test]
fn to_markdown_roundtrip_on_md_fixture() {
    let doc = parse(format!("{}/sample.md", fixtures())).unwrap();
    let md = doc.to_markdown();
    assert!(md.contains("# FastRAG Sample Document"));
}

#[test]
fn to_json_deserializes() {
    let doc = parse(format!("{}/sample.md", fixtures())).unwrap();
    let json = doc.to_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["elements"].is_array());
    assert!(!parsed["elements"].as_array().unwrap().is_empty());
}

#[test]
fn to_plain_text_no_formatting() {
    let doc = parse(format!("{}/sample.md", fixtures())).unwrap();
    let text = doc.to_plain_text();
    assert!(!text.contains("# "));
    assert!(!text.contains("```"));
}
