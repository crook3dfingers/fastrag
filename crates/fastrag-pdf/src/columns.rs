use crate::table::PositionedText;

/// Minimum horizontal gap (in points) between clusters to be considered separate columns.
const MIN_COLUMN_GAP: f32 = 40.0;

/// A detected column region on a page.
#[derive(Debug, Clone)]
pub struct Column {
    pub x_min: f32,
    pub x_max: f32,
    pub items: Vec<PositionedText>,
}

/// Detect columns in positioned text and return them sorted left-to-right.
///
/// Columns are detected by finding wide horizontal gaps (> `MIN_COLUMN_GAP`)
/// in the x-coordinates of text items. Items are clustered into columns by
/// their x-position, and columns are sorted left-to-right.
pub fn detect_columns(items: &[PositionedText]) -> Vec<Column> {
    if items.is_empty() {
        return Vec::new();
    }

    // Collect all unique x-coordinates and sort them
    let mut x_coords: Vec<f32> = items.iter().map(|t| t.x).collect();
    x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    x_coords.dedup_by(|a, b| (*a - *b).abs() < 1.0);

    if x_coords.len() < 2 {
        // Single column — return as-is
        return vec![Column {
            x_min: x_coords[0],
            x_max: x_coords[0],
            items: items.to_vec(),
        }];
    }

    // Find column boundaries by detecting large gaps
    let mut boundaries: Vec<f32> = Vec::new(); // gap midpoints
    for w in x_coords.windows(2) {
        if w[1] - w[0] > MIN_COLUMN_GAP {
            boundaries.push((w[0] + w[1]) / 2.0);
        }
    }

    if boundaries.is_empty() {
        // No significant gaps — single column
        return vec![Column {
            x_min: x_coords[0],
            x_max: *x_coords.last().unwrap(),
            items: items.to_vec(),
        }];
    }

    // Assign items to columns based on boundaries
    let num_cols = boundaries.len() + 1;
    let mut columns: Vec<Column> = (0..num_cols)
        .map(|_| Column {
            x_min: f32::MAX,
            x_max: f32::MIN,
            items: Vec::new(),
        })
        .collect();

    for item in items {
        let col_idx = boundaries.iter().filter(|&&b| item.x >= b).count();
        columns[col_idx].items.push(item.clone());
        if item.x < columns[col_idx].x_min {
            columns[col_idx].x_min = item.x;
        }
        if item.x > columns[col_idx].x_max {
            columns[col_idx].x_max = item.x;
        }
    }

    // Remove empty columns
    columns.retain(|c| !c.items.is_empty());

    // Sort columns left-to-right by x_min
    columns.sort_by(|a, b| {
        a.x_min
            .partial_cmp(&b.x_min)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    columns
}

/// Reorder positioned text items by column reading order.
///
/// Detects columns, sorts columns left-to-right, and within each column
/// sorts items top-to-bottom (y descending in PDF coordinates).
/// For single-column content, returns items in their original top-to-bottom order.
pub fn reorder_by_columns(items: Vec<PositionedText>) -> Vec<PositionedText> {
    let columns = detect_columns(&items);

    if columns.len() <= 1 {
        // Single column — just sort top-to-bottom
        let mut sorted = items;
        sorted.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));
        return sorted;
    }

    let mut result = Vec::with_capacity(items.len());
    for mut col in columns {
        // Sort within column: top-to-bottom (y descending)
        col.items
            .sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));
        result.extend(col.items);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f32, y: f32, text: &str) -> PositionedText {
        PositionedText {
            x,
            y,
            text: text.to_string(),
        }
    }

    #[test]
    fn two_columns_detected() {
        let items = vec![
            pt(72.0, 700.0, "Left 1"),
            pt(72.0, 680.0, "Left 2"),
            pt(330.0, 700.0, "Right 1"),
            pt(330.0, 680.0, "Right 2"),
        ];

        let columns = detect_columns(&items);
        assert_eq!(columns.len(), 2, "expected 2 columns, got: {columns:?}");
        assert!(columns[0].x_min < columns[1].x_min);
        assert_eq!(columns[0].items.len(), 2);
        assert_eq!(columns[1].items.len(), 2);
    }

    #[test]
    fn three_columns_detected() {
        let items = vec![
            pt(72.0, 700.0, "Col1"),
            pt(250.0, 700.0, "Col2"),
            pt(430.0, 700.0, "Col3"),
        ];

        let columns = detect_columns(&items);
        assert_eq!(columns.len(), 3, "expected 3 columns, got: {columns:?}");
    }

    #[test]
    fn single_column_no_change() {
        let items = vec![
            pt(72.0, 700.0, "Line 1"),
            pt(72.0, 680.0, "Line 2"),
            pt(80.0, 660.0, "Line 3"),
        ];

        let columns = detect_columns(&items);
        assert_eq!(columns.len(), 1, "expected 1 column, got: {columns:?}");
    }

    #[test]
    fn reorder_two_columns_left_before_right() {
        // Interleaved: Right 1, Left 1, Right 2, Left 2
        let items = vec![
            pt(330.0, 700.0, "Right 1"),
            pt(72.0, 700.0, "Left 1"),
            pt(330.0, 680.0, "Right 2"),
            pt(72.0, 680.0, "Left 2"),
        ];

        let reordered = reorder_by_columns(items);
        let texts: Vec<&str> = reordered.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["Left 1", "Left 2", "Right 1", "Right 2"],
            "left column should come before right column"
        );
    }

    #[test]
    fn reorder_single_column_sorted_top_to_bottom() {
        let items = vec![
            pt(72.0, 500.0, "Bottom"),
            pt(72.0, 700.0, "Top"),
            pt(72.0, 600.0, "Middle"),
        ];

        let reordered = reorder_by_columns(items);
        let texts: Vec<&str> = reordered.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["Top", "Middle", "Bottom"]);
    }

    #[test]
    fn empty_input() {
        let columns = detect_columns(&[]);
        assert!(columns.is_empty());

        let reordered = reorder_by_columns(Vec::new());
        assert!(reordered.is_empty());
    }

    #[test]
    fn within_column_sorted_by_y_descending() {
        let items = vec![
            pt(72.0, 600.0, "Left Bottom"),
            pt(72.0, 700.0, "Left Top"),
            pt(330.0, 600.0, "Right Bottom"),
            pt(330.0, 700.0, "Right Top"),
        ];

        let reordered = reorder_by_columns(items);
        let texts: Vec<&str> = reordered.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["Left Top", "Left Bottom", "Right Top", "Right Bottom"]
        );
    }

    #[test]
    fn integration_reorder_twocol_pdf_positioned_text() {
        use crate::table::collect_positioned_text;

        let pdf_bytes: &[u8] = include_bytes!("../../../tests/fixtures/sample_twocol.pdf");
        let pdf = pdf::file::FileOptions::cached().load(pdf_bytes).unwrap();
        let resolver = pdf.resolver();
        let page = pdf.get_page(0).unwrap();
        let ops = page
            .contents
            .as_ref()
            .and_then(|c: &pdf::content::Content| c.operations(&resolver).ok())
            .unwrap_or_default();

        let positioned = collect_positioned_text(&ops);
        assert!(positioned.len() >= 4, "expected text items from PDF");

        let reordered = reorder_by_columns(positioned);
        let texts: Vec<&str> = reordered.iter().map(|t| t.text.as_str()).collect();

        // Find first left and first right column text
        let left_idx = texts
            .iter()
            .position(|t| t.contains("Left"))
            .expect("left column text not found");
        let right_idx = texts
            .iter()
            .position(|t| t.contains("Right"))
            .expect("right column text not found");

        assert!(
            left_idx < right_idx,
            "after reorder, left column (idx {left_idx}) should come before right (idx {right_idx}). texts: {texts:?}"
        );
    }
}
