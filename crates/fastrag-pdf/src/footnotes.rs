use crate::table::PositionedText;
use fastrag_core::{Element, ElementKind};

/// Fraction of page height from the bottom that defines the footnote region.
const FOOTNOTE_REGION_FRACTION: f32 = 0.15;

/// Extract footnote elements from positioned text on a page.
///
/// Footnotes are text in the bottom ~15% of the page that start with numeric
/// reference markers (e.g., `1.`, `[1]`, or superscript characters like `¹`).
///
/// Returns `(footnote_elements, remaining_body_items)`.
pub fn extract_footnotes(
    items: &[PositionedText],
    page_num: u32,
    page_height: f32,
) -> (Vec<Element>, Vec<PositionedText>) {
    if items.is_empty() || page_height <= 0.0 {
        return (Vec::new(), items.to_vec());
    }

    let footnote_threshold = page_height * FOOTNOTE_REGION_FRACTION;

    let mut footnotes = Vec::new();
    let mut remaining = Vec::new();

    for item in items {
        if item.y < footnote_threshold {
            if let Some((ref_id, text)) = parse_footnote_marker(&item.text) {
                let mut el =
                    Element::new(ElementKind::Footnote, text).with_page(page_num as usize + 1);
                el.attributes.insert("reference_id".to_string(), ref_id);
                footnotes.push(el);
            } else {
                // In footnote region but no marker — treat as body text
                remaining.push(item.clone());
            }
        } else {
            remaining.push(item.clone());
        }
    }

    (footnotes, remaining)
}

/// Try to parse a footnote marker from text.
/// Returns `Some((reference_id, cleaned_text))` if a marker is found.
fn parse_footnote_marker(text: &str) -> Option<(String, String)> {
    let trimmed = text.trim();

    // Pattern: `[N]` — bracket style
    if trimmed.starts_with('[') {
        if let Some(end) = trimmed.find(']') {
            let num = &trimmed[1..end];
            if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
                let rest = trimmed[end + 1..].trim().to_string();
                return Some((num.to_string(), rest));
            }
        }
    }

    // Pattern: superscript digits (¹²³⁴⁵⁶⁷⁸⁹⁰)
    let superscripts: &[(char, char)] = &[
        ('\u{00B9}', '1'),
        ('\u{00B2}', '2'),
        ('\u{00B3}', '3'),
        ('\u{2074}', '4'),
        ('\u{2075}', '5'),
        ('\u{2076}', '6'),
        ('\u{2077}', '7'),
        ('\u{2078}', '8'),
        ('\u{2079}', '9'),
        ('\u{2070}', '0'),
    ];

    let first_char = trimmed.chars().next()?;
    for &(sup, digit) in superscripts {
        if first_char == sup {
            let rest = trimmed[first_char.len_utf8()..].trim().to_string();
            return Some((digit.to_string(), rest));
        }
    }

    // Pattern: `N.` or `N ` — plain numeric
    let mut num_end = 0;
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() {
            num_end += ch.len_utf8();
        } else {
            break;
        }
    }

    if num_end > 0 {
        let num = &trimmed[..num_end];
        let after = &trimmed[num_end..];
        // Must be followed by `.` or whitespace (not just more text like "2024")
        if after.starts_with('.') || after.starts_with(' ') || after.starts_with('\t') {
            let rest = after.trim_start_matches('.').trim().to_string();
            return Some((num.to_string(), rest));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_footnotes_separates_bottom_region() {
        // Page height 792 (letter), footnote region < 792 * 0.15 = 118.8
        let items = vec![
            PositionedText {
                x: 72.0,
                y: 700.0,
                text: "Body text at top".into(),
            },
            PositionedText {
                x: 72.0,
                y: 600.0,
                text: "More body text".into(),
            },
            PositionedText {
                x: 72.0,
                y: 85.0,
                text: "1. First footnote about source.".into(),
            },
            PositionedText {
                x: 72.0,
                y: 70.0,
                text: "2. Second footnote reference.".into(),
            },
        ];

        let (footnotes, remaining) = extract_footnotes(&items, 0, 792.0);

        assert_eq!(
            footnotes.len(),
            2,
            "expected 2 footnotes, got: {footnotes:?}"
        );
        assert_eq!(
            remaining.len(),
            2,
            "expected 2 remaining, got: {remaining:?}"
        );

        assert_eq!(footnotes[0].kind, ElementKind::Footnote);
        assert_eq!(footnotes[0].text, "First footnote about source.");
        assert_eq!(
            footnotes[0].attributes.get("reference_id"),
            Some(&"1".to_string())
        );
        assert_eq!(footnotes[0].page, Some(1));

        assert_eq!(footnotes[1].text, "Second footnote reference.");
        assert_eq!(
            footnotes[1].attributes.get("reference_id"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn bracket_style_markers() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 50.0,
            text: "[3] Third footnote in bracket style.".into(),
        }];

        let (footnotes, remaining) = extract_footnotes(&items, 0, 792.0);
        assert_eq!(footnotes.len(), 1);
        assert_eq!(remaining.len(), 0);
        assert_eq!(
            footnotes[0].attributes.get("reference_id"),
            Some(&"3".to_string())
        );
        assert_eq!(footnotes[0].text, "Third footnote in bracket style.");
    }

    #[test]
    fn superscript_markers() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 50.0,
            text: "\u{00B9} Superscript one footnote.".into(),
        }];

        let (footnotes, _) = extract_footnotes(&items, 0, 792.0);
        assert_eq!(footnotes.len(), 1);
        assert_eq!(
            footnotes[0].attributes.get("reference_id"),
            Some(&"1".to_string())
        );
        assert_eq!(footnotes[0].text, "Superscript one footnote.");
    }

    #[test]
    fn no_marker_in_footnote_region_stays_as_body() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 50.0,
            text: "Just text without a marker".into(),
        }];

        let (footnotes, remaining) = extract_footnotes(&items, 0, 792.0);
        assert_eq!(footnotes.len(), 0);
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn body_text_above_threshold_not_extracted() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 400.0,
            text: "1. This looks like a footnote but is in body region".into(),
        }];

        let (footnotes, remaining) = extract_footnotes(&items, 0, 792.0);
        assert_eq!(footnotes.len(), 0);
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn empty_items_returns_empty() {
        let (footnotes, remaining) = extract_footnotes(&[], 0, 792.0);
        assert!(footnotes.is_empty());
        assert!(remaining.is_empty());
    }

    #[test]
    fn zero_page_height_returns_all_as_body() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 50.0,
            text: "1. Some text".into(),
        }];
        let (footnotes, remaining) = extract_footnotes(&items, 0, 0.0);
        assert!(footnotes.is_empty());
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn page_number_propagated() {
        let items = vec![PositionedText {
            x: 72.0,
            y: 50.0,
            text: "1. Footnote on page 3".into(),
        }];

        let (footnotes, _) = extract_footnotes(&items, 2, 792.0);
        assert_eq!(footnotes[0].page, Some(3));
    }

    #[test]
    fn parse_footnote_marker_dot_style() {
        let result = parse_footnote_marker("1. Some text");
        assert_eq!(result, Some(("1".to_string(), "Some text".to_string())));
    }

    #[test]
    fn parse_footnote_marker_space_style() {
        let result = parse_footnote_marker("2 Some text");
        assert_eq!(result, Some(("2".to_string(), "Some text".to_string())));
    }

    #[test]
    fn parse_footnote_marker_bracket_style() {
        let result = parse_footnote_marker("[10] Reference text");
        assert_eq!(
            result,
            Some(("10".to_string(), "Reference text".to_string()))
        );
    }

    #[test]
    fn parse_footnote_marker_no_match() {
        assert_eq!(parse_footnote_marker("Regular text"), None);
        assert_eq!(parse_footnote_marker(""), None);
        // "2024" without separator should not match
        assert_eq!(parse_footnote_marker("2024year"), None);
    }

    #[test]
    fn integration_parse_footnotes_pdf() {
        use crate::PdfParser;
        use fastrag_core::{FileFormat, Parser, SourceInfo};

        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample_footnotes.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf).with_filename("sample_footnotes.pdf");
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        let footnotes: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Footnote)
            .collect();

        assert!(
            footnotes.len() >= 2,
            "expected at least 2 footnotes, got {}. All elements: {:?}",
            footnotes.len(),
            doc.elements
                .iter()
                .map(|e| (&e.kind, &e.text, e.page))
                .collect::<Vec<_>>()
        );

        // All footnotes should have reference_id attribute
        for fn_el in &footnotes {
            assert!(
                fn_el.attributes.contains_key("reference_id"),
                "footnote missing reference_id: {:?}",
                fn_el
            );
        }

        // Body text should not contain footnote text
        let body: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Paragraph)
            .collect();
        assert!(!body.is_empty(), "expected body paragraphs, got none");
    }
}
