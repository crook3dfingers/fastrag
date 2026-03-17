use crate::document::{Document, Element, ElementKind};

/// A chunk of a document, containing one or more elements.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub elements: Vec<Element>,
    pub text: String,
    pub char_count: usize,
    pub section: Option<String>,
    pub index: usize,
}

/// Strategy for splitting a document into chunks.
pub enum ChunkingStrategy {
    /// Split by accumulating elements up to `max_characters`.
    Basic { max_characters: usize },
    /// Split on Title/Heading boundaries, with sub-chunking for large sections.
    ByTitle { max_characters: usize },
}

impl Document {
    /// Split this document into chunks according to the given strategy.
    pub fn chunk(&self, strategy: &ChunkingStrategy) -> Vec<Chunk> {
        match strategy {
            ChunkingStrategy::Basic { max_characters } => {
                basic_chunk(&self.elements, *max_characters)
            }
            ChunkingStrategy::ByTitle { max_characters } => {
                by_title_chunk(&self.elements, *max_characters)
            }
        }
    }
}

fn basic_chunk(elements: &[Element], max_characters: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_elements: Vec<Element> = Vec::new();
    let mut current_len = 0usize;

    for element in elements {
        let el_len = element.text.len();

        // If adding this element would exceed the limit and we have content, flush
        if current_len + el_len > max_characters && !current_elements.is_empty() {
            chunks.push(build_chunk(
                std::mem::take(&mut current_elements),
                chunks.len(),
                None,
            ));
            current_len = 0;
        }

        current_elements.push(element.clone());
        current_len += el_len;
    }

    if !current_elements.is_empty() {
        chunks.push(build_chunk(current_elements, chunks.len(), None));
    }

    chunks
}

fn by_title_chunk(elements: &[Element], max_characters: usize) -> Vec<Chunk> {
    // Split elements into sections based on Title/Heading boundaries
    let mut sections: Vec<(Option<String>, Vec<Element>)> = Vec::new();
    let mut current_section: Option<String> = None;
    let mut current_elements: Vec<Element> = Vec::new();

    for element in elements {
        if matches!(element.kind, ElementKind::Title | ElementKind::Heading) {
            // Flush previous section
            if !current_elements.is_empty() {
                sections.push((
                    current_section.clone(),
                    std::mem::take(&mut current_elements),
                ));
            }
            current_section = Some(element.text.clone());
        }
        current_elements.push(element.clone());
    }

    if !current_elements.is_empty() {
        sections.push((current_section, current_elements));
    }

    let mut chunks = Vec::new();

    for (section_name, section_elements) in sections {
        let total_chars: usize = section_elements.iter().map(|e| e.text.len()).sum();

        if total_chars <= max_characters {
            chunks.push(build_chunk(section_elements, chunks.len(), section_name));
        } else {
            // Sub-chunk via basic strategy
            let sub_chunks = basic_chunk(&section_elements, max_characters);
            for sub in sub_chunks {
                let chunk = build_chunk(sub.elements, chunks.len(), section_name.clone());
                chunks.push(chunk);
            }
        }
    }

    chunks
}

fn build_chunk(elements: Vec<Element>, index: usize, section: Option<String>) -> Chunk {
    let text: String = elements
        .iter()
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let char_count = text.len();
    Chunk {
        elements,
        text,
        char_count,
        section,
        index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Metadata;
    use crate::format::FileFormat;

    fn doc_with(elements: Vec<Element>) -> Document {
        Document {
            metadata: Metadata::new(FileFormat::Text),
            elements,
        }
    }

    #[test]
    fn basic_chunk_single_element_fits() {
        let doc = doc_with(vec![Element::new(ElementKind::Paragraph, "Hello world")]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 100,
        });
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
        assert_eq!(chunks[0].char_count, 11);
    }

    #[test]
    fn basic_chunk_splits_at_boundary() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "AAAAAAAAAA"), // 10 chars
            Element::new(ElementKind::Paragraph, "BBBBBBBBBB"), // 10 chars
            Element::new(ElementKind::Paragraph, "CCCCCCCCCC"), // 10 chars
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic { max_characters: 15 });
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].elements.len(), 1);
        assert_eq!(chunks[0].text, "AAAAAAAAAA");
        assert_eq!(chunks[1].text, "BBBBBBBBBB");
        assert_eq!(chunks[2].text, "CCCCCCCCCC");
    }

    #[test]
    fn basic_chunk_large_element_own_chunk() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "short"),
            Element::new(
                ElementKind::Paragraph,
                "this is a very long element that exceeds the limit",
            ),
            Element::new(ElementKind::Paragraph, "end"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic { max_characters: 10 });
        assert_eq!(chunks.len(), 3);
        assert!(chunks[1].text.contains("very long element"));
    }

    #[test]
    fn basic_chunk_empty_document() {
        let doc = doc_with(vec![]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 100,
        });
        assert!(chunks.is_empty());
    }

    #[test]
    fn by_title_splits_on_headings() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Introduction"),
            Element::new(ElementKind::Paragraph, "Intro text"),
            Element::new(ElementKind::Heading, "Chapter 1"),
            Element::new(ElementKind::Paragraph, "Chapter 1 text"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
        });
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section, Some("Introduction".to_string()));
        assert_eq!(chunks[1].section, Some("Chapter 1".to_string()));
    }

    #[test]
    fn by_title_preamble_chunk() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "Preamble text before any heading"),
            Element::new(ElementKind::Title, "First Title"),
            Element::new(ElementKind::Paragraph, "Body"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
        });
        assert_eq!(chunks.len(), 2);
        // Preamble has no section name
        assert_eq!(chunks[0].section, None);
        assert!(chunks[0].text.contains("Preamble"));
    }

    #[test]
    fn by_title_section_name_set() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Heading, "My Section"),
            Element::new(ElementKind::Paragraph, "Content"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
        });
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section, Some("My Section".to_string()));
    }

    #[test]
    fn by_title_large_section_sub_chunked() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Big Section"),
            Element::new(ElementKind::Paragraph, "AAAAAAAAAA"), // 10
            Element::new(ElementKind::Paragraph, "BBBBBBBBBB"), // 10
            Element::new(ElementKind::Paragraph, "CCCCCCCCCC"), // 10
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle { max_characters: 20 });
        // Section total > 20, so it gets sub-chunked
        assert!(chunks.len() >= 2);
        // All sub-chunks should carry the section name
        for chunk in &chunks {
            assert_eq!(chunk.section, Some("Big Section".to_string()));
        }
    }

    #[test]
    fn chunk_index_sequential() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "A"),
            Element::new(ElementKind::Paragraph, "text"),
            Element::new(ElementKind::Heading, "B"),
            Element::new(ElementKind::Paragraph, "more text"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
        });
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn chunk_char_count_matches_text() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "Hello"),
            Element::new(ElementKind::Paragraph, "World"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 1000,
        });
        for chunk in &chunks {
            assert_eq!(chunk.char_count, chunk.text.len());
        }
    }
}
