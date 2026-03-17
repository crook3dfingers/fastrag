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

        // Extract document info if available
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
                            // Large Y offset likely means new line
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

            // Split page text into paragraphs on blank lines
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
