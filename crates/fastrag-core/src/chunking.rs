use serde::{Deserialize, Serialize};

use crate::document::{Document, Element, ElementKind};

/// A chunk of a document, containing one or more elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    Basic {
        max_characters: usize,
        overlap: usize,
    },
    /// Split on Title/Heading boundaries, with sub-chunking for large sections.
    ByTitle {
        max_characters: usize,
        overlap: usize,
    },
    /// Recursive character splitting: tries separators in order, falling back to the next.
    RecursiveCharacter {
        max_characters: usize,
        overlap: usize,
        separators: Vec<String>,
    },
}

/// Default separators for recursive character splitting (most to least specific).
pub fn default_separators() -> Vec<String> {
    vec![
        "\n\n".to_string(),
        "\n".to_string(),
        ". ".to_string(),
        " ".to_string(),
        String::new(),
    ]
}

impl Document {
    /// Split this document into chunks according to the given strategy.
    pub fn chunk(&self, strategy: &ChunkingStrategy) -> Vec<Chunk> {
        match strategy {
            ChunkingStrategy::Basic {
                max_characters,
                overlap,
            } => basic_chunk(&self.elements, *max_characters, *overlap),
            ChunkingStrategy::ByTitle {
                max_characters,
                overlap,
            } => by_title_chunk(&self.elements, *max_characters, *overlap),
            ChunkingStrategy::RecursiveCharacter {
                max_characters,
                overlap,
                separators,
            } => recursive_character_chunk(&self.elements, *max_characters, *overlap, separators),
        }
    }
}

fn basic_chunk(elements: &[Element], max_characters: usize, overlap: usize) -> Vec<Chunk> {
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

    apply_overlap(&mut chunks, overlap);
    chunks
}

fn by_title_chunk(elements: &[Element], max_characters: usize, overlap: usize) -> Vec<Chunk> {
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
            let sub_chunks = basic_chunk(&section_elements, max_characters, 0);
            for sub in sub_chunks {
                let chunk = build_chunk(sub.elements, chunks.len(), section_name.clone());
                chunks.push(chunk);
            }
        }
    }

    apply_overlap(&mut chunks, overlap);
    chunks
}

fn apply_overlap(chunks: &mut [Chunk], overlap: usize) {
    if overlap == 0 || chunks.len() < 2 {
        return;
    }
    for i in 1..chunks.len() {
        let prev_text = chunks[i - 1].text.clone();
        let overlap_len = overlap.min(prev_text.len());
        let overlap_text = &prev_text[prev_text.len() - overlap_len..];
        chunks[i].text = format!("{}{}", overlap_text, chunks[i].text);
        chunks[i].char_count = chunks[i].text.len();
    }
}

fn recursive_character_chunk(
    elements: &[Element],
    max_characters: usize,
    overlap: usize,
    separators: &[String],
) -> Vec<Chunk> {
    let full_text: String = elements
        .iter()
        .map(|e| e.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let pieces = recursive_split(&full_text, separators, max_characters);

    let mut chunks: Vec<Chunk> = pieces
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let char_count = text.len();
            Chunk {
                elements: vec![Element::new(ElementKind::Paragraph, &text)],
                text,
                char_count,
                section: None,
                index: i,
            }
        })
        .collect();

    apply_overlap(&mut chunks, overlap);
    // Fix indices after overlap (they don't change, but be explicit)
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
    chunks
}

fn recursive_split(text: &str, separators: &[String], max_characters: usize) -> Vec<String> {
    if text.len() <= max_characters {
        return if text.is_empty() {
            vec![]
        } else {
            vec![text.to_string()]
        };
    }

    if separators.is_empty() {
        // Hard character-level split
        return text
            .as_bytes()
            .chunks(max_characters)
            .map(|chunk| String::from_utf8_lossy(chunk).to_string())
            .collect();
    }

    let separator = &separators[0];
    let remaining_separators = &separators[1..];

    if separator.is_empty() {
        // Empty separator = hard character-level split
        return text
            .as_bytes()
            .chunks(max_characters)
            .map(|chunk| String::from_utf8_lossy(chunk).to_string())
            .collect();
    }

    // Split text keeping separator on the left side
    let pieces = split_keeping_separator(text, separator);
    let mut result = Vec::new();
    let mut current = String::new();

    for piece in pieces {
        if current.len() + piece.len() <= max_characters {
            current.push_str(&piece);
        } else {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            if piece.len() > max_characters {
                // Piece still too large — recurse with remaining separators
                let sub = recursive_split(&piece, remaining_separators, max_characters);
                result.extend(sub);
            } else {
                current = piece;
            }
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

/// Split text on `separator`, keeping the separator at the end of the left piece.
fn split_keeping_separator(text: &str, separator: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut start = 0;

    while let Some(pos) = text[start..].find(separator) {
        let end = start + pos + separator.len();
        pieces.push(text[start..end].to_string());
        start = end;
    }

    if start < text.len() {
        pieces.push(text[start..].to_string());
    }

    pieces
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

    #[test]
    fn chunk_serializes_to_json() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Intro"),
            Element::new(ElementKind::Paragraph, "Body text"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        });
        let json = serde_json::to_string(&chunks[0]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["text"], "Intro\nBody text");
        assert_eq!(parsed["char_count"], 15);
        assert_eq!(parsed["index"], 0);
        assert!(parsed["section"].is_null());
    }

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
            overlap: 0,
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
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 15,
            overlap: 0,
        });
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
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 10,
            overlap: 0,
        });
        assert_eq!(chunks.len(), 3);
        assert!(chunks[1].text.contains("very long element"));
    }

    #[test]
    fn basic_chunk_empty_document() {
        let doc = doc_with(vec![]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 100,
            overlap: 0,
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
            overlap: 0,
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
            overlap: 0,
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
            overlap: 0,
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
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 20,
            overlap: 0,
        });
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
            overlap: 0,
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
            overlap: 0,
        });
        for chunk in &chunks {
            assert_eq!(chunk.char_count, chunk.text.len());
        }
    }

    // --- Overlap tests ---

    #[test]
    fn basic_chunk_overlap_shares_characters() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "AAAAAAAAAA"), // 10 chars
            Element::new(ElementKind::Paragraph, "BBBBBBBBBB"), // 10 chars
            Element::new(ElementKind::Paragraph, "CCCCCCCCCC"), // 10 chars
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 15,
            overlap: 5,
        });
        assert_eq!(chunks.len(), 3);
        // chunk[1] should start with last 5 chars of chunk[0]
        let chunk0_last5 = &chunks[0].text[chunks[0].text.len() - 5..];
        assert!(
            chunks[1].text.starts_with(chunk0_last5),
            "chunk[1] should start with '{}' but was '{}'",
            chunk0_last5,
            &chunks[1].text[..5.min(chunks[1].text.len())]
        );
    }

    #[test]
    fn basic_chunk_overlap_zero_unchanged() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "AAAAAAAAAA"),
            Element::new(ElementKind::Paragraph, "BBBBBBBBBB"),
            Element::new(ElementKind::Paragraph, "CCCCCCCCCC"),
        ]);
        let chunks_no_overlap = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 15,
            overlap: 0,
        });
        assert_eq!(chunks_no_overlap.len(), 3);
        assert_eq!(chunks_no_overlap[0].text, "AAAAAAAAAA");
        assert_eq!(chunks_no_overlap[1].text, "BBBBBBBBBB");
        assert_eq!(chunks_no_overlap[2].text, "CCCCCCCCCC");
    }

    #[test]
    fn by_title_chunk_overlap_works() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "Section A"),
            Element::new(ElementKind::Paragraph, "Content of section A here"),
            Element::new(ElementKind::Title, "Section B"),
            Element::new(ElementKind::Paragraph, "Content of section B here"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
            overlap: 5,
        });
        assert_eq!(chunks.len(), 2);
        let chunk0_last5 = &chunks[0].text[chunks[0].text.len() - 5..];
        assert!(chunks[1].text.starts_with(chunk0_last5));
    }

    #[test]
    fn overlap_char_count_updated() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "AAAAAAAAAA"),
            Element::new(ElementKind::Paragraph, "BBBBBBBBBB"),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 15,
            overlap: 5,
        });
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].char_count, chunks[1].text.len());
        // chunk[1] text should be 5 overlap + 10 original = 15
        assert_eq!(chunks[1].text.len(), 15);
    }

    // --- Recursive character chunking tests ---

    #[test]
    fn recursive_split_by_double_newline() {
        let doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "Paragraph one here.\n\nParagraph two here.\n\nParagraph three here.",
        )]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 30,
            overlap: 0,
            separators: default_separators(),
        });
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= 30,
                "chunk too large: {} chars",
                chunk.text.len()
            );
        }
    }

    #[test]
    fn recursive_split_falls_back_to_sentence() {
        // Single paragraph (no \n\n), but has sentence boundaries
        let doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "First sentence here. Second sentence here. Third sentence here.",
        )]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 30,
            overlap: 0,
            separators: default_separators(),
        });
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                chunk.text.len() <= 30,
                "chunk too large: {} chars, text: '{}'",
                chunk.text.len(),
                chunk.text
            );
        }
    }

    #[test]
    fn recursive_split_hard_break() {
        // No separators match — must hard-split
        let doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 10,
            overlap: 0,
            separators: default_separators(),
        });
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "ABCDEFGHIJ");
        assert_eq!(chunks[1].text, "KLMNOPQRST");
        assert_eq!(chunks[2].text, "UVWXYZ");
    }

    #[test]
    fn recursive_character_with_overlap() {
        let doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 10,
            overlap: 3,
            separators: default_separators(),
        });
        assert!(chunks.len() >= 3);
        // chunk[1] should start with last 3 chars of chunk[0]
        let chunk0_last3 = &chunks[0].text[chunks[0].text.len() - 3..];
        assert!(
            chunks[1].text.starts_with(chunk0_last3),
            "expected chunk[1] to start with '{}', got '{}'",
            chunk0_last3,
            &chunks[1].text
        );
    }

    #[test]
    fn recursive_character_via_document() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Title, "My Title"),
            Element::new(ElementKind::Paragraph, "Some paragraph text."),
            Element::new(ElementKind::Paragraph, "Another paragraph here."),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 25,
            overlap: 0,
            separators: default_separators(),
        });
        assert!(!chunks.is_empty());
        // All text from all elements should be present across chunks
        let all_text: String = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(all_text.contains("My Title"));
        assert!(all_text.contains("Some paragraph text."));
    }

    #[test]
    fn recursive_split_preserves_separator() {
        // Separator should stay on the left side
        let doc = doc_with(vec![Element::new(
            ElementKind::Paragraph,
            "Hello world. Goodbye world.",
        )]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 15,
            overlap: 0,
            separators: vec![". ".to_string(), String::new()],
        });
        assert_eq!(chunks.len(), 2);
        // First chunk should end with ". "
        assert!(
            chunks[0].text.ends_with(". "),
            "first chunk should end with separator, got: '{}'",
            chunks[0].text
        );
    }
}
