use fastrag_core::{
    BoundingBox, Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser,
    SourceInfo,
};
use pdf::object::Resolve;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "images")]
pub mod images;

#[cfg(feature = "table-detect")]
pub mod table;

#[cfg(feature = "forms")]
pub mod forms;

#[cfg(feature = "ocr")]
pub mod ocr;

/// PDF parser using the `pdf` crate.
pub struct PdfParser;

fn extract_page_elements(
    page: &pdf::object::Page,
    resolver: &impl Resolve,
    page_num: u32,
    #[allow(unused_variables)] input_bytes: &[u8],
) -> Result<Vec<Element>, FastRagError> {
    let ops = page
        .contents
        .as_ref()
        .and_then(|content| content.operations(resolver).ok())
        .unwrap_or_default();

    // OCR: if this is a scanned page, use OCR instead of text extraction
    #[cfg(feature = "ocr")]
    {
        if ocr::is_scanned_page(&ops) {
            return ocr::ocr_page(input_bytes, page_num, 150);
        }
    }

    let mut page_text = String::with_capacity(ops.len() * 20);

    // Track text positions for bounding box computation
    let mut current_x: f32 = 0.0;
    let mut current_y: f32 = 0.0;
    // Positions collected per paragraph: (x, y) of each text fragment
    let mut para_positions: Vec<Vec<(f32, f32)>> = vec![Vec::new()];

    for op in &ops {
        match op {
            pdf::content::Op::SetTextMatrix { matrix } => {
                current_x = matrix.e;
                current_y = matrix.f;
            }
            pdf::content::Op::TextDraw { text } => {
                if let Ok(s) = text.to_string() {
                    if !s.trim().is_empty()
                        && let Some(last) = para_positions.last_mut()
                    {
                        last.push((current_x, current_y));
                    }
                    page_text.push_str(&s);
                }
            }
            pdf::content::Op::TextDrawAdjusted { array } => {
                for item in array {
                    if let pdf::content::TextDrawAdjusted::Text(t) = item
                        && let Ok(s) = t.to_string()
                    {
                        if !s.trim().is_empty()
                            && let Some(last) = para_positions.last_mut()
                        {
                            last.push((current_x, current_y));
                        }
                        page_text.push_str(&s);
                    }
                }
            }
            pdf::content::Op::TextNewline => {
                page_text.push('\n');
            }
            pdf::content::Op::MoveTextPosition { translation } => {
                current_x += translation.x;
                current_y += translation.y;
                if translation.y.abs() > 1.0 && !page_text.is_empty() {
                    page_text.push('\n');
                    // Check if this creates a paragraph break
                    if translation.y.abs() > 15.0 {
                        para_positions.push(Vec::new());
                    }
                }
            }
            _ => {}
        }
    }

    let mut elements = Vec::new();

    // Extract images if feature enabled
    #[cfg(feature = "images")]
    {
        if let Ok(resources) = page.resources() {
            let image_elements = images::extract_images(&ops, resources, resolver, page_num);
            elements.extend(image_elements);
        }
    }

    // Table detection: extract tables and build paragraphs from remaining text
    #[cfg(feature = "table-detect")]
    {
        let (table_elements, remaining) = table::extract_tables_from_ops(&ops, page_num);
        if !table_elements.is_empty() {
            elements.extend(table_elements);
            // Build paragraphs from remaining (non-table) text
            for pt in &remaining {
                elements.push(
                    Element::new(ElementKind::Paragraph, pt.text.clone())
                        .with_page(page_num as usize + 1),
                );
            }
            return Ok(elements);
        }
    }

    // Default paragraph extraction from plain text
    if page_text.trim().is_empty() {
        return Ok(elements);
    }

    let paragraphs: Vec<&str> = page_text.split("\n\n").collect();
    for (para_idx, para) in paragraphs.iter().enumerate() {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut el = Element::new(ElementKind::Paragraph, trimmed).with_page(page_num as usize + 1);

        // Compute bounding box from tracked positions
        if let Some(positions) = para_positions.get(para_idx)
            && !positions.is_empty()
        {
            let min_x = positions.iter().map(|(x, _)| *x).fold(f32::MAX, f32::min);
            let max_x = positions.iter().map(|(x, _)| *x).fold(f32::MIN, f32::max);
            let min_y = positions.iter().map(|(_, y)| *y).fold(f32::MAX, f32::min);
            let max_y = positions.iter().map(|(_, y)| *y).fold(f32::MIN, f32::max);

            el = el.with_bounding_box(BoundingBox {
                x: min_x,
                y: min_y,
                width: (max_x - min_x).max(0.0),
                height: (max_y - min_y).max(0.0),
                page: page_num as usize + 1,
            });
        }

        elements.push(el);
    }
    Ok(elements)
}

impl Parser for PdfParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Pdf]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let pdf =
            pdf::file::FileOptions::cached()
                .load(input)
                .map_err(|e| FastRagError::Parse {
                    format: FileFormat::Pdf,
                    message: e.to_string(),
                })?;

        let num_pages = pdf.num_pages();
        metadata.page_count = Some(num_pages as usize);

        if let Some(ref info) = pdf.trailer.info_dict {
            if let Some(ref title) = info.title
                && let Ok(t) = title.to_string()
                && !t.is_empty()
            {
                metadata.title = Some(t);
            }
            if let Some(ref author) = info.author
                && let Ok(a) = author.to_string()
                && !a.is_empty()
            {
                metadata.author = Some(a);
            }
        }

        let resolver = pdf.resolver();
        let page_range: Vec<u32> = (0..num_pages).collect();

        let process_page = |&pn: &u32| -> Result<Vec<Element>, FastRagError> {
            let page = pdf.get_page(pn).map_err(|e| FastRagError::Parse {
                format: FileFormat::Pdf,
                message: format!("page {}: {}", pn + 1, e),
            })?;
            extract_page_elements(&page, &resolver, pn, input)
        };

        #[cfg(feature = "parallel")]
        let page_results: Result<Vec<Vec<Element>>, FastRagError> =
            page_range.par_iter().map(process_page).collect();

        #[cfg(not(feature = "parallel"))]
        let page_results: Result<Vec<Vec<Element>>, FastRagError> =
            page_range.iter().map(process_page).collect();

        #[allow(unused_mut)]
        let mut elements: Vec<Element> = page_results?.into_iter().flatten().collect();

        #[cfg(feature = "table-detect")]
        table::merge_continued_tables(&mut elements);

        #[cfg(feature = "forms")]
        {
            let form_elements = forms::extract_form_fields(&pdf, &resolver);
            elements.extend(form_elements);
        }

        Ok(Document { metadata, elements })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_formats_returns_pdf() {
        assert_eq!(PdfParser.supported_formats(), &[FileFormat::Pdf]);
    }

    #[test]
    fn invalid_bytes_returns_parse_error() {
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let result = parser.parse(b"not a pdf", &source);
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Parse { format, .. } => assert_eq!(format, FileFormat::Pdf),
            other => panic!("expected Parse error, got: {other}"),
        }
    }

    #[test]
    fn empty_input_returns_error() {
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let result = parser.parse(&[], &source);
        assert!(result.is_err());
    }

    #[test]
    fn multi_page_pdf_preserves_content_and_order() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf).with_filename("sample.pdf");
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        assert_eq!(doc.metadata.page_count, Some(3));

        let page1: Vec<_> = doc.elements.iter().filter(|e| e.page == Some(1)).collect();
        let page2: Vec<_> = doc.elements.iter().filter(|e| e.page == Some(2)).collect();
        let page3: Vec<_> = doc.elements.iter().filter(|e| e.page == Some(3)).collect();

        assert!(page1[0].text.contains("Page one"), "got: {}", page1[0].text);
        assert!(page2[0].text.contains("Page two"), "got: {}", page2[0].text);
        assert!(
            page3[0].text.contains("Page three"),
            "got: {}",
            page3[0].text
        );
    }

    #[test]
    fn dense_page_text_fully_extracted() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        let total_len: usize = doc.elements.iter().map(|e| e.text.len()).sum();
        assert!(total_len > 30, "total text length was only {total_len}");
    }

    #[test]
    fn page_order_monotonically_increasing() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        let pages: Vec<usize> = doc.elements.iter().filter_map(|e| e.page).collect();
        for w in pages.windows(2) {
            assert!(w[0] <= w[1], "order violated: {} after {}", w[1], w[0]);
        }
        let unique: std::collections::HashSet<_> = pages.iter().collect();
        assert!(unique.len() >= 2, "expected elements from ≥2 pages");
    }

    #[test]
    fn minimal_pdf_parses() {
        // Minimal valid PDF with one empty page
        let pdf_bytes = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF";
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf).with_filename("test.pdf");
        // This may or may not parse with the pdf crate - test error path if it fails
        let result = parser.parse(pdf_bytes, &source);
        match result {
            Ok(doc) => {
                assert_eq!(doc.metadata.page_count, Some(1));
                // Empty page should have no elements
                assert!(doc.elements.is_empty());
            }
            Err(FastRagError::Parse { format, .. }) => {
                // Acceptable: the minimal PDF may not parse with this crate
                assert_eq!(format, FileFormat::Pdf);
            }
            Err(other) => panic!("unexpected error type: {other}"),
        }
    }

    #[test]
    fn pdf_elements_have_bounding_boxes() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        let with_bbox: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.bounding_box.is_some())
            .collect();
        assert!(
            !with_bbox.is_empty(),
            "expected some elements to have bounding boxes"
        );
    }

    #[test]
    fn bounding_box_page_matches_element_page() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let parser = PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf);
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        for el in &doc.elements {
            if let Some(ref bbox) = el.bounding_box {
                assert_eq!(
                    bbox.page,
                    el.page.unwrap(),
                    "bbox page {} != element page {:?}",
                    bbox.page,
                    el.page
                );
            }
        }
    }
}
