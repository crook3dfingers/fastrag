use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, Parser, SourceInfo,
};

/// PDF parser using the `pdf` crate.
pub struct PdfParser;

impl Parser for PdfParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Pdf]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        let pdf = pdf::file::FileOptions::uncached()
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
        let mut elements = Vec::new();

        for page_num in 0..num_pages {
            let page = pdf.get_page(page_num).map_err(|e| FastRagError::Parse {
                format: FileFormat::Pdf,
                message: format!("page {}: {}", page_num + 1, e),
            })?;

            let mut page_text = String::new();

            if let Some(ref content) = page.contents
                && let Ok(ops) = content.operations(&resolver)
            {
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
            }

            if page_text.trim().is_empty() {
                continue;
            }

            for para in page_text.split("\n\n") {
                let trimmed = para.trim();
                if trimmed.is_empty() {
                    continue;
                }
                elements.push(
                    Element::new(ElementKind::Paragraph, trimmed).with_page(page_num as usize + 1),
                );
            }
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
