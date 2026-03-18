use crate::document::{Document, ElementKind};

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Markdown,
    Json,
    Jsonl,
    PlainText,
    Html,
}

impl Document {
    /// Render the document as structured markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();

        let first_is_title = self
            .elements
            .first()
            .is_some_and(|e| e.kind == ElementKind::Title);
        if !first_is_title && let Some(title) = &self.metadata.title {
            out.push_str(&format!("# {title}\n\n"));
        }

        for element in &self.elements {
            match &element.kind {
                ElementKind::Title => {
                    out.push_str(&format!("# {}\n\n", element.text));
                }
                ElementKind::Heading => {
                    let prefix = "#".repeat(element.depth.max(1) as usize + 1);
                    out.push_str(&format!("{prefix} {}\n\n", element.text));
                }
                ElementKind::Paragraph => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::Table => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::Code => {
                    let lang = element
                        .attributes
                        .get("language")
                        .map_or("", |s| s.as_str());
                    out.push_str(&format!("```{lang}\n{}\n```\n\n", element.text));
                }
                ElementKind::List => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
                ElementKind::ListItem => {
                    out.push_str(&format!("- {}\n", element.text));
                }
                ElementKind::Image => {
                    let alt = element.attributes.get("alt").map_or("", |s| s.as_str());
                    out.push_str(&format!("![{alt}]({})\n\n", element.text));
                }
                ElementKind::BlockQuote => {
                    for line in element.text.lines() {
                        out.push_str(&format!("> {line}\n"));
                    }
                    out.push('\n');
                }
                ElementKind::HorizontalRule => {
                    out.push_str("---\n\n");
                }
                ElementKind::FormField => {
                    let name = element
                        .attributes
                        .get("field_name")
                        .map_or("", |s| s.as_str());
                    let field_type = element
                        .attributes
                        .get("field_type")
                        .map_or("", |s| s.as_str());
                    if field_type.is_empty() {
                        out.push_str(&format!("**{name}**: {}\n\n", element.text));
                    } else {
                        out.push_str(&format!("**{name}** ({field_type}): {}\n\n", element.text));
                    }
                }
                ElementKind::Unknown => {
                    out.push_str(&element.text);
                    out.push_str("\n\n");
                }
            }
        }

        out.trim_end().to_string()
    }

    /// Render the document as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render the document as JSONL (one JSON object per element, one per line).
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for element in &self.elements {
            if let Ok(line) = serde_json::to_string(element) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out.trim_end().to_string()
    }

    /// Render the document as plain text (all element text concatenated).
    pub fn to_plain_text(&self) -> String {
        let mut out = String::new();

        for element in &self.elements {
            if !element.text.is_empty() {
                out.push_str(&element.text);
                out.push('\n');
            }
        }

        out.trim_end().to_string()
    }

    /// Render the document as HTML.
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        out.push_str("<!DOCTYPE html>\n<html>\n<body>\n");

        for element in &self.elements {
            match &element.kind {
                ElementKind::Title => {
                    out.push_str(&format!("<h1>{}</h1>\n", escape_html(&element.text)));
                }
                ElementKind::Heading => {
                    let level = (element.depth.max(1) + 1).min(6);
                    out.push_str(&format!(
                        "<h{level}>{}</h{level}>\n",
                        escape_html(&element.text)
                    ));
                }
                ElementKind::Paragraph => {
                    out.push_str(&format!("<p>{}</p>\n", escape_html(&element.text)));
                }
                ElementKind::Table => {
                    out.push_str(&markdown_table_to_html(&element.text));
                    out.push('\n');
                }
                ElementKind::Code => {
                    let lang = element
                        .attributes
                        .get("language")
                        .map_or("", |s| s.as_str());
                    if lang.is_empty() {
                        out.push_str(&format!(
                            "<pre><code>{}</code></pre>\n",
                            escape_html(&element.text)
                        ));
                    } else {
                        out.push_str(&format!(
                            "<pre><code class=\"{lang}\">{}</code></pre>\n",
                            escape_html(&element.text)
                        ));
                    }
                }
                ElementKind::List => {
                    out.push_str(&format!("<ul>{}</ul>\n", escape_html(&element.text)));
                }
                ElementKind::ListItem => {
                    out.push_str(&format!("<li>{}</li>\n", escape_html(&element.text)));
                }
                ElementKind::Image => {
                    let alt = element.attributes.get("alt").map_or("", |s| s.as_str());
                    out.push_str(&format!(
                        "<img src=\"{}\" alt=\"{alt}\">\n",
                        escape_html(&element.text)
                    ));
                }
                ElementKind::BlockQuote => {
                    out.push_str(&format!(
                        "<blockquote><p>{}</p></blockquote>\n",
                        escape_html(&element.text)
                    ));
                }
                ElementKind::HorizontalRule => {
                    out.push_str("<hr>\n");
                }
                ElementKind::FormField => {
                    let name = element
                        .attributes
                        .get("field_name")
                        .map_or("", |s| s.as_str());
                    let field_type = element
                        .attributes
                        .get("field_type")
                        .map_or("", |s| s.as_str());
                    out.push_str("<dl>");
                    if field_type.is_empty() {
                        out.push_str(&format!("<dt>{}</dt>", escape_html(name)));
                    } else {
                        out.push_str(&format!(
                            "<dt>{} ({})</dt>",
                            escape_html(name),
                            escape_html(field_type)
                        ));
                    }
                    out.push_str(&format!("<dd>{}</dd>", escape_html(&element.text)));
                    out.push_str("</dl>\n");
                }
                ElementKind::Unknown => {
                    out.push_str(&format!("<p>{}</p>\n", escape_html(&element.text)));
                }
            }
        }

        out.push_str("</body>\n</html>");
        out
    }
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Convert a markdown table to HTML table tags.
fn markdown_table_to_html(table_text: &str) -> String {
    let mut html = String::from("<table>\n");
    let mut is_first_row = true;

    for line in table_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Skip separator rows (e.g. |---|---|)
        if trimmed
            .chars()
            .all(|c| c == '|' || c == '-' || c == ':' || c == ' ')
        {
            continue;
        }
        html.push_str("<tr>");
        let cells: Vec<&str> = trimmed
            .split('|')
            .filter(|c| !c.is_empty())
            .map(|c| c.trim())
            .collect();
        for cell in &cells {
            let tag = if is_first_row { "th" } else { "td" };
            html.push_str(&format!("<{tag}>{}</{tag}>", escape_html(cell)));
        }
        html.push_str("</tr>\n");
        is_first_row = false;
    }

    html.push_str("</table>");
    html
}

#[cfg(test)]
mod tests {
    use crate::document::*;
    use crate::format::FileFormat;

    fn doc_with(elements: Vec<Element>) -> Document {
        Document {
            metadata: Metadata::new(FileFormat::Text),
            elements,
        }
    }

    fn doc_with_title_meta(title: &str, elements: Vec<Element>) -> Document {
        let mut m = Metadata::new(FileFormat::Text);
        m.title = Some(title.to_string());
        Document {
            metadata: m,
            elements,
        }
    }

    // --- to_markdown ---

    #[test]
    fn markdown_title_renders_h1() {
        let doc = doc_with(vec![Element::new(ElementKind::Title, "My Title")]);
        let md = doc.to_markdown();
        assert!(md.contains("# My Title"));
    }

    #[test]
    fn markdown_heading_depth1_renders_h2() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Heading, "Section").with_depth(1),
        ]);
        let md = doc.to_markdown();
        assert!(md.contains("## Section"));
    }

    #[test]
    fn markdown_heading_depth2_renders_h3() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Heading, "Sub").with_depth(2),
        ]);
        let md = doc.to_markdown();
        assert!(md.contains("### Sub"));
    }

    #[test]
    fn markdown_paragraph_followed_by_blank_line() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "Hello"),
            Element::new(ElementKind::Paragraph, "World"),
        ]);
        let md = doc.to_markdown();
        assert!(md.contains("Hello\n\nWorld"));
    }

    #[test]
    fn markdown_code_without_language() {
        let doc = doc_with(vec![Element::new(ElementKind::Code, "let x = 1;")]);
        let md = doc.to_markdown();
        assert!(md.contains("```\nlet x = 1;\n```"));
    }

    #[test]
    fn markdown_code_with_language() {
        let mut el = Element::new(ElementKind::Code, "fn main() {}");
        el.attributes
            .insert("language".to_string(), "rust".to_string());
        let doc = doc_with(vec![el]);
        let md = doc.to_markdown();
        assert!(md.contains("```rust\nfn main() {}\n```"));
    }

    #[test]
    fn markdown_blockquote_prefixed() {
        let doc = doc_with(vec![Element::new(ElementKind::BlockQuote, "A wise quote.")]);
        let md = doc.to_markdown();
        assert!(md.contains("> A wise quote."));
    }

    #[test]
    fn markdown_horizontal_rule() {
        let doc = doc_with(vec![Element::new(ElementKind::HorizontalRule, "")]);
        let md = doc.to_markdown();
        assert!(md.contains("---"));
    }

    #[test]
    fn markdown_metadata_title_suppressed_when_first_is_title() {
        let doc = doc_with_title_meta(
            "Meta Title",
            vec![Element::new(ElementKind::Title, "Element Title")],
        );
        let md = doc.to_markdown();
        // Should NOT contain meta title as separate heading
        assert_eq!(md.matches("# ").count(), 1);
        assert!(md.contains("# Element Title"));
    }

    #[test]
    fn markdown_metadata_title_emitted_when_first_is_not_title() {
        let doc = doc_with_title_meta(
            "Meta Title",
            vec![Element::new(ElementKind::Paragraph, "Some text")],
        );
        let md = doc.to_markdown();
        assert!(md.contains("# Meta Title"));
    }

    // --- to_json ---

    #[test]
    fn json_round_trip_deserialize() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "T"),
            Element::new(ElementKind::Paragraph, "P"),
        ]);
        let json = doc.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["elements"].is_array());
    }

    #[test]
    fn json_element_count_matches() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "T"),
            Element::new(ElementKind::Paragraph, "P"),
            Element::new(ElementKind::Code, "C"),
        ]);
        let json = doc.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["elements"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn json_none_fields_absent() {
        let doc = doc_with(vec![Element::new(ElementKind::Paragraph, "text")]);
        let json = doc.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        // page is None and skip_serializing_if, should be absent
        assert!(parsed["elements"][0].get("page").is_none());
    }

    #[test]
    fn json_format_field_present() {
        let doc = doc_with(vec![]);
        let json = doc.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["metadata"]["format"], "text");
    }

    // --- to_plain_text ---

    #[test]
    fn plain_text_concatenates() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Title"),
            Element::new(ElementKind::Paragraph, "Body"),
        ]);
        let text = doc.to_plain_text();
        assert!(text.contains("Title"));
        assert!(text.contains("Body"));
    }

    #[test]
    fn plain_text_no_formatting() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Title"),
            Element::new(ElementKind::HorizontalRule, ""),
            Element::new(ElementKind::BlockQuote, "Quote"),
        ]);
        let text = doc.to_plain_text();
        assert!(!text.contains("# "));
        assert!(!text.contains("---"));
        assert!(!text.contains("> "));
    }

    #[test]
    fn plain_text_skips_empty() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Title"),
            Element::new(ElementKind::HorizontalRule, ""),
            Element::new(ElementKind::Paragraph, "End"),
        ]);
        let text = doc.to_plain_text();
        // HorizontalRule has empty text, should be skipped
        assert_eq!(text, "Title\nEnd");
    }

    // --- to_html ---

    #[test]
    fn html_title_renders_h1() {
        let doc = doc_with(vec![Element::new(ElementKind::Title, "Title")]);
        let html = doc.to_html();
        assert!(html.contains("<h1>Title</h1>"), "got: {html}");
    }

    #[test]
    fn html_heading_depth_renders_correct_tag() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Heading, "Sec").with_depth(1),
            Element::new(ElementKind::Heading, "Sub").with_depth(2),
        ]);
        let html = doc.to_html();
        assert!(html.contains("<h2>Sec</h2>"), "got: {html}");
        assert!(html.contains("<h3>Sub</h3>"), "got: {html}");
    }

    #[test]
    fn html_paragraph_renders_p() {
        let doc = doc_with(vec![Element::new(ElementKind::Paragraph, "Hello world")]);
        let html = doc.to_html();
        assert!(html.contains("<p>Hello world</p>"), "got: {html}");
    }

    #[test]
    fn html_code_renders_pre_code() {
        let mut el = Element::new(ElementKind::Code, "let x = 1;");
        el.attributes
            .insert("language".to_string(), "rust".to_string());
        let doc = doc_with(vec![el]);
        let html = doc.to_html();
        assert!(
            html.contains("<pre><code class=\"rust\">let x = 1;</code></pre>"),
            "got: {html}"
        );
    }

    #[test]
    fn html_table_renders_table_tags() {
        let table_md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |";
        let doc = doc_with(vec![Element::new(ElementKind::Table, table_md)]);
        let html = doc.to_html();
        assert!(html.contains("<table>"), "got: {html}");
        assert!(html.contains("<th>Name</th>"), "got: {html}");
        assert!(html.contains("<td>Alice</td>"), "got: {html}");
        assert!(html.contains("</table>"), "got: {html}");
    }

    #[test]
    fn html_image_renders_img() {
        let mut el = Element::new(ElementKind::Image, "photo.jpg");
        el.attributes
            .insert("alt".to_string(), "A photo".to_string());
        let doc = doc_with(vec![el]);
        let html = doc.to_html();
        assert!(
            html.contains("<img src=\"photo.jpg\" alt=\"A photo\">"),
            "got: {html}"
        );
    }

    #[test]
    fn html_blockquote_renders_tag() {
        let doc = doc_with(vec![Element::new(ElementKind::BlockQuote, "A quote")]);
        let html = doc.to_html();
        assert!(
            html.contains("<blockquote><p>A quote</p></blockquote>"),
            "got: {html}"
        );
    }

    #[test]
    fn html_horizontal_rule_renders_hr() {
        let doc = doc_with(vec![Element::new(ElementKind::HorizontalRule, "")]);
        let html = doc.to_html();
        assert!(html.contains("<hr>"), "got: {html}");
    }

    #[test]
    fn html_wraps_in_document_structure() {
        let doc = doc_with(vec![Element::new(ElementKind::Paragraph, "text")]);
        let html = doc.to_html();
        assert!(html.contains("<!DOCTYPE html>"), "got: {html}");
        assert!(html.contains("<html>"), "got: {html}");
        assert!(html.contains("<body>"), "got: {html}");
        assert!(html.contains("</body>"), "got: {html}");
        assert!(html.contains("</html>"), "got: {html}");
    }

    // --- FormField rendering ---

    #[test]
    fn markdown_form_field_renders() {
        let mut el = Element::new(ElementKind::FormField, "John Doe");
        el.attributes
            .insert("field_name".to_string(), "Full Name".to_string());
        el.attributes
            .insert("field_type".to_string(), "Text".to_string());
        let doc = doc_with(vec![el]);
        let md = doc.to_markdown();
        assert!(md.contains("**Full Name** (Text): John Doe"), "got: {md}");
    }

    #[test]
    fn html_form_field_renders() {
        let mut el = Element::new(ElementKind::FormField, "John Doe");
        el.attributes
            .insert("field_name".to_string(), "Full Name".to_string());
        el.attributes
            .insert("field_type".to_string(), "Text".to_string());
        let doc = doc_with(vec![el]);
        let html = doc.to_html();
        assert!(html.contains("<dl>"), "got: {html}");
        assert!(html.contains("<dt>Full Name (Text)</dt>"), "got: {html}");
        assert!(html.contains("<dd>John Doe</dd>"), "got: {html}");
        assert!(html.contains("</dl>"), "got: {html}");
    }

    #[test]
    fn form_field_element_kind_exists() {
        let el = Element::new(ElementKind::FormField, "value");
        assert_eq!(el.kind, ElementKind::FormField);
        assert_eq!(el.text, "value");
    }

    // --- to_jsonl ---

    #[test]
    fn jsonl_one_line_per_element() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "T"),
            Element::new(ElementKind::Paragraph, "P"),
            Element::new(ElementKind::Code, "C"),
        ]);
        let jsonl = doc.to_jsonl();
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed["kind"].is_string());
            assert!(parsed["text"].is_string());
        }
    }

    #[test]
    fn jsonl_empty_doc() {
        let doc = doc_with(vec![]);
        let jsonl = doc.to_jsonl();
        assert_eq!(jsonl, "");
    }
}
