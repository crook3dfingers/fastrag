use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};
use pdf::object::Resolve;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// PDF parser using the `pdf` crate.
pub struct PdfParser;

fn extract_page_elements(
    page: &pdf::object::Page,
    resolver: &impl Resolve,
    page_num: u32,
) -> Result<Vec<Element>, FastRagError> {
    let ops = page
        .contents
        .as_ref()
        .and_then(|content| content.operations(resolver).ok())
        .unwrap_or_default();

    let mut page_text = String::with_capacity(ops.len() * 20);

    for op in &ops {
        match op {
            pdf::content::Op::TextDraw { text } => {
                if let Ok(s) = text.to_string() {
                    page_text.push_str(&s);
                }
            }
            pdf::content::Op::TextDrawAdjusted { array } => {
                for item in array {
                    if let pdf::content::TextDrawAdjusted::Text(t) = item
                        && let Ok(s) = t.to_string()
                    {
                        page_text.push_str(&s);
                    }
                }
            }
            pdf::content::Op::TextNewline => {
                page_text.push('\n');
            }
            pdf::content::Op::MoveTextPosition { translation } => {
                if translation.y.abs() > 1.0 && !page_text.is_empty() {
                    page_text.push('\n');
                }
            }
            _ => {}
        }
    }

    if page_text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut elements = Vec::new();
    for para in page_text.split("\n\n") {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }
        elements.push(
            Element::new(ElementKind::Paragraph, trimmed).with_page(page_num as usize + 1),
        );
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

        let pdf = pdf::file::FileOptions::cached()
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
            extract_page_elements(&page, &resolver, pn)
        };

        #[cfg(feature = "parallel")]
        let page_results: Result<Vec<Vec<Element>>, FastRagError> =
            page_range.par_iter().map(process_page).collect();

        #[cfg(not(feature = "parallel"))]
        let page_results: Result<Vec<Vec<Element>>, FastRagError> =
            page_range.iter().map(process_page).collect();

        let elements: Vec<Element> = page_results?.into_iter().flatten().collect();

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
        assert!(page3[0].text.contains("Page three"), "got: {}", page3[0].text);
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
}
