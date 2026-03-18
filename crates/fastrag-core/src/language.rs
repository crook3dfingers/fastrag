use crate::document::{Document, ElementKind};

impl Document {
    /// Detect the primary language of the document and store it in metadata.
    /// Stores ISO 639-1 code in `custom["language"]` and confidence in `custom["language_confidence"]`.
    pub fn detect_language(&mut self) {
        let sample = self.collect_text_sample(1000);
        if sample.is_empty() {
            return;
        }

        if let Some(info) = whatlang::detect(&sample) {
            let code = lang_to_iso639_1(info.lang());
            self.metadata
                .custom
                .insert("language".to_string(), code.to_string());
            self.metadata.custom.insert(
                "language_confidence".to_string(),
                format!("{:.4}", info.confidence()),
            );
        }
    }

    /// Detect language per element and store in element attributes.
    /// Only processes text-heavy element kinds (Paragraph, Title, Heading, BlockQuote, ListItem)
    /// with at least 20 characters.
    pub fn detect_element_languages(&mut self) {
        for element in &mut self.elements {
            if !matches!(
                element.kind,
                ElementKind::Paragraph
                    | ElementKind::Title
                    | ElementKind::Heading
                    | ElementKind::BlockQuote
                    | ElementKind::ListItem
            ) {
                continue;
            }
            if element.text.len() < 20 {
                continue;
            }
            if let Some(info) = whatlang::detect(&element.text) {
                let code = lang_to_iso639_1(info.lang());
                element
                    .attributes
                    .insert("language".to_string(), code.to_string());
                element.attributes.insert(
                    "language_confidence".to_string(),
                    format!("{:.4}", info.confidence()),
                );
            }
        }
    }

    /// Collect text from Paragraph, Title, and Heading elements (skip Code and Table).
    fn collect_text_sample(&self, max_chars: usize) -> String {
        let mut sample = String::new();
        for element in &self.elements {
            match element.kind {
                ElementKind::Paragraph | ElementKind::Title | ElementKind::Heading => {
                    if sample.len() >= max_chars {
                        break;
                    }
                    if !sample.is_empty() {
                        sample.push(' ');
                    }
                    sample.push_str(&element.text);
                }
                _ => {}
            }
        }
        sample.truncate(max_chars);
        sample
    }
}

fn lang_to_iso639_1(lang: whatlang::Lang) -> &'static str {
    match lang {
        whatlang::Lang::Eng => "en",
        whatlang::Lang::Deu => "de",
        whatlang::Lang::Fra => "fr",
        whatlang::Lang::Spa => "es",
        whatlang::Lang::Ita => "it",
        whatlang::Lang::Por => "pt",
        whatlang::Lang::Nld => "nl",
        whatlang::Lang::Rus => "ru",
        whatlang::Lang::Cmn => "zh",
        whatlang::Lang::Jpn => "ja",
        whatlang::Lang::Kor => "ko",
        whatlang::Lang::Ara => "ar",
        whatlang::Lang::Hin => "hi",
        whatlang::Lang::Tur => "tr",
        whatlang::Lang::Pol => "pl",
        whatlang::Lang::Swe => "sv",
        whatlang::Lang::Dan => "da",
        whatlang::Lang::Fin => "fi",
        whatlang::Lang::Nob => "no",
        whatlang::Lang::Ukr => "uk",
        whatlang::Lang::Heb => "he",
        whatlang::Lang::Ces => "cs",
        whatlang::Lang::Ron => "ro",
        whatlang::Lang::Hun => "hu",
        whatlang::Lang::Ell => "el",
        whatlang::Lang::Tha => "th",
        whatlang::Lang::Vie => "vi",
        whatlang::Lang::Ind => "id",
        _ => "und",
    }
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

    #[test]
    fn detect_english_text() {
        let mut doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "The quick brown fox jumps over the lazy dog. This is a typical English sentence used for testing purposes.",
        )]);
        doc.detect_language();
        assert_eq!(doc.metadata.custom.get("language"), Some(&"en".to_string()));
    }

    #[test]
    fn detect_german_text() {
        let mut doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "Dies ist ein deutscher Satz. Die deutsche Sprache ist eine westgermanische Sprache, die vor allem in Mitteleuropa gesprochen wird.",
        )]);
        doc.detect_language();
        assert_eq!(doc.metadata.custom.get("language"), Some(&"de".to_string()));
    }

    #[test]
    fn detect_language_empty_doc() {
        let mut doc = doc_with(vec![]);
        doc.detect_language();
        assert!(doc.metadata.custom.get("language").is_none());
    }

    #[test]
    fn detect_language_code_only_doc() {
        let mut doc = doc_with(vec![
            Element::new(ElementKind::Code, "fn main() { println!(\"Hello\"); }"),
            Element::new(ElementKind::Table, "| A | B |\n| --- |\n| 1 | 2 |"),
        ]);
        doc.detect_language();
        // No text sampled from Code/Table, so no detection
        assert!(doc.metadata.custom.get("language").is_none());
    }

    #[test]
    fn confidence_stored_as_float_string() {
        let mut doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "This is a sample English text that should be detected with high confidence.",
        )]);
        doc.detect_language();
        let conf = doc.metadata.custom.get("language_confidence").unwrap();
        let val: f64 = conf
            .parse()
            .expect("confidence should be parseable as float");
        assert!(val > 0.0 && val <= 1.0);
    }

    // --- detect_element_languages ---

    #[test]
    fn detect_element_languages_english() {
        let mut doc = doc_with(vec![
            Element::new(
                ElementKind::Paragraph,
                "The quick brown fox jumps over the lazy dog and runs across the field.",
            ),
            Element::new(
                ElementKind::Title,
                "An Introduction to English Language Testing Methods",
            ),
        ]);
        doc.detect_element_languages();
        assert_eq!(
            doc.elements[0].attributes.get("language"),
            Some(&"en".to_string())
        );
        assert_eq!(
            doc.elements[1].attributes.get("language"),
            Some(&"en".to_string())
        );
    }

    #[test]
    fn detect_element_languages_skips_short() {
        let mut doc = doc_with(vec![Element::new(ElementKind::Paragraph, "Short text.")]);
        doc.detect_element_languages();
        assert!(doc.elements[0].attributes.get("language").is_none());
    }

    #[test]
    fn detect_element_languages_skips_code_and_table() {
        let mut doc = doc_with(vec![
            Element::new(
                ElementKind::Code,
                "fn main() { println!(\"Hello world this is a long test string\"); }",
            ),
            Element::new(
                ElementKind::Table,
                "| Column A | Column B | Column C | Column D |",
            ),
        ]);
        doc.detect_element_languages();
        assert!(doc.elements[0].attributes.get("language").is_none());
        assert!(doc.elements[1].attributes.get("language").is_none());
    }

    #[test]
    fn detect_element_languages_mixed() {
        let mut doc = doc_with(vec![
            Element::new(
                ElementKind::Paragraph,
                "The quick brown fox jumps over the lazy dog and runs across the open field.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Dies ist ein deutscher Satz. Die deutsche Sprache ist eine westgermanische Sprache.",
            ),
        ]);
        doc.detect_element_languages();
        assert_eq!(
            doc.elements[0].attributes.get("language"),
            Some(&"en".to_string())
        );
        assert_eq!(
            doc.elements[1].attributes.get("language"),
            Some(&"de".to_string())
        );
    }

    #[test]
    fn collect_text_sample_skips_tables_and_code() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "English text here"),
            Element::new(ElementKind::Code, "let x = 42;"),
            Element::new(ElementKind::Table, "| col |"),
            Element::new(ElementKind::Paragraph, "more English text"),
        ]);
        let sample = doc.collect_text_sample(1000);
        assert!(sample.contains("English text here"));
        assert!(sample.contains("more English text"));
        assert!(!sample.contains("let x = 42"));
        assert!(!sample.contains("| col |"));
    }
}
