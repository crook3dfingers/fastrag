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
    /// Raw chunk text with an LLM-generated context prefix prepended by the
    /// contextualization stage (see `fastrag-context`). `None` when no
    /// contextualizer ran for this chunk or when contextualization failed in
    /// non-strict mode; downstream consumers fall back to [`Self::text`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contextualized_text: Option<String>,
}

/// Configuration for injecting document context into chunk text.
#[derive(Debug, Clone)]
pub struct ContextInjection {
    pub template: String,
}

impl Default for ContextInjection {
    fn default() -> Self {
        Self {
            template: "{document_title} > {section}\n\n{chunk_text}".to_string(),
        }
    }
}

impl Document {
    /// Inject document context into each chunk's text using the given template.
    /// Available placeholders: `{document_title}`, `{section}`, `{chunk_text}`.
    pub fn inject_context(&self, chunks: &mut [Chunk], injection: &ContextInjection) {
        let title = self.metadata.title.clone().unwrap_or_default();

        for chunk in chunks.iter_mut() {
            let section = chunk.section.clone().unwrap_or_default();

            let mut result = injection.template.clone();
            result = result.replace("{document_title}", &title);
            result = result.replace("{section}", &section);
            result = result.replace("{chunk_text}", &chunk.text);

            chunk.text = result;
            chunk.char_count = chunk.text.len();
        }
    }
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
    /// Semantic chunking: split on embedding similarity boundaries.
    Semantic {
        max_characters: usize,
        similarity_threshold: Option<f32>,
        percentile_threshold: Option<f32>,
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
            ChunkingStrategy::Semantic {
                max_characters,
                similarity_threshold,
                percentile_threshold,
            } => semantic_chunk(
                &self.elements,
                *max_characters,
                *similarity_threshold,
                *percentile_threshold,
                &default_embedder,
            ),
        }
    }

    /// Split this document into chunks using semantic strategy with a custom embedder.
    pub fn chunk_with_embedder(
        &self,
        strategy: &ChunkingStrategy,
        embedder: &dyn Fn(&str) -> Vec<f32>,
    ) -> Vec<Chunk> {
        match strategy {
            ChunkingStrategy::Semantic {
                max_characters,
                similarity_threshold,
                percentile_threshold,
            } => semantic_chunk(
                &self.elements,
                *max_characters,
                *similarity_threshold,
                *percentile_threshold,
                embedder,
            ),
            _ => self.chunk(strategy),
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
        // Walk backward to a char boundary so we don't slice through a
        // multi-byte UTF-8 codepoint (e.g. em-dashes in VIPER playbooks).
        let mut start = prev_text.len() - overlap_len;
        while start > 0 && !prev_text.is_char_boundary(start) {
            start -= 1;
        }
        let overlap_text = &prev_text[start..];
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
                contextualized_text: None,
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

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Default bag-of-words embedder using character trigram feature hashing.
/// Uses a fixed-size vector (256 dimensions) for consistent comparison.
pub fn default_embedder(text: &str) -> Vec<f32> {
    const DIM: usize = 256;
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut vec = vec![0.0f32; DIM];

    if chars.len() < 3 {
        for (i, &c) in chars.iter().enumerate() {
            vec[c as usize % DIM] += (i + 1) as f32;
        }
        return vec;
    }

    for w in chars.windows(3) {
        // Simple hash: combine the three char codes
        let hash = (w[0] as usize)
            .wrapping_mul(31)
            .wrapping_add(w[1] as usize)
            .wrapping_mul(31)
            .wrapping_add(w[2] as usize);
        vec[hash % DIM] += 1.0;
    }

    vec
}

fn semantic_chunk(
    elements: &[Element],
    max_characters: usize,
    similarity_threshold: Option<f32>,
    percentile_threshold: Option<f32>,
    embedder: &dyn Fn(&str) -> Vec<f32>,
) -> Vec<Chunk> {
    if elements.is_empty() {
        return Vec::new();
    }

    // Split elements into sentence-level units
    let mut sentences: Vec<(String, Vec<usize>)> = Vec::new(); // (text, source element indices)
    for (i, el) in elements.iter().enumerate() {
        let text = el.text.trim();
        if text.is_empty() {
            continue;
        }
        // Split on sentence boundaries
        let mut start = 0;
        for (j, _) in text.match_indices(". ") {
            let end = j + 2;
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push((sentence.to_string(), vec![i]));
            }
            start = end;
        }
        let remainder = text[start..].trim();
        if !remainder.is_empty() {
            sentences.push((remainder.to_string(), vec![i]));
        }
    }

    if sentences.len() <= 1 {
        return vec![build_chunk(elements.to_vec(), 0, None)];
    }

    // Embed each sentence
    let embeddings: Vec<Vec<f32>> = sentences.iter().map(|(text, _)| embedder(text)).collect();

    // Compute consecutive cosine similarities
    let mut similarities: Vec<f32> = Vec::new();
    for i in 0..embeddings.len() - 1 {
        // Pad shorter vectors with zeros for comparison
        let a = &embeddings[i];
        let b = &embeddings[i + 1];
        let max_len = a.len().max(b.len());
        let mut a_padded = a.clone();
        let mut b_padded = b.clone();
        a_padded.resize(max_len, 0.0);
        b_padded.resize(max_len, 0.0);
        similarities.push(cosine_similarity(&a_padded, &b_padded));
    }

    // Determine threshold
    let threshold = if let Some(t) = similarity_threshold {
        t
    } else if let Some(p) = percentile_threshold {
        // percentile-based: sort similarities, find the value at the given percentile
        let mut sorted = similarities.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (sorted.len() as f32 - 1.0)).round().max(0.0) as usize;
        sorted[idx.min(sorted.len() - 1)]
    } else {
        0.5 // default
    };

    // Find split points where similarity drops below threshold
    let mut split_points: Vec<usize> = Vec::new();
    for (i, sim) in similarities.iter().enumerate() {
        if *sim < threshold {
            split_points.push(i + 1); // split after sentence i
        }
    }

    // Group sentences into chunks
    let mut chunks = Vec::new();
    let mut start = 0;
    for &split in &split_points {
        let group: Vec<&(String, Vec<usize>)> = sentences[start..split].iter().collect();
        if !group.is_empty() {
            let chunk = build_sentence_chunk(&group, &chunks, elements, max_characters);
            chunks.extend(chunk);
        }
        start = split;
    }
    // Remaining sentences
    let group: Vec<&(String, Vec<usize>)> = sentences[start..].iter().collect();
    if !group.is_empty() {
        let chunk = build_sentence_chunk(&group, &chunks, elements, max_characters);
        chunks.extend(chunk);
    }

    // Fix indices
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
    chunks
}

fn build_sentence_chunk(
    group: &[&(String, Vec<usize>)],
    existing_chunks: &[Chunk],
    elements: &[Element],
    max_characters: usize,
) -> Vec<Chunk> {
    let text: String = group
        .iter()
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if text.len() <= max_characters {
        // Collect source elements
        let mut el_indices: Vec<usize> = group
            .iter()
            .flat_map(|(_, idxs)| idxs.iter().copied())
            .collect();
        el_indices.sort();
        el_indices.dedup();
        let chunk_elements: Vec<Element> =
            el_indices.iter().map(|&i| elements[i].clone()).collect();
        let char_count = text.len();
        vec![Chunk {
            elements: chunk_elements,
            text,
            char_count,
            section: None,
            index: existing_chunks.len(),
            contextualized_text: None,
        }]
    } else {
        // Sub-chunk if text exceeds max_characters
        let mut sub_chunks = Vec::new();
        let mut current_text = String::new();
        let mut current_els: Vec<usize> = Vec::new();

        for (t, idxs) in group.iter() {
            if !current_text.is_empty() && current_text.len() + 1 + t.len() > max_characters {
                let el_list: Vec<Element> = {
                    let mut sorted = current_els.clone();
                    sorted.sort();
                    sorted.dedup();
                    sorted.iter().map(|&i| elements[i].clone()).collect()
                };
                let char_count = current_text.len();
                sub_chunks.push(Chunk {
                    elements: el_list,
                    text: std::mem::take(&mut current_text),
                    char_count,
                    section: None,
                    index: existing_chunks.len() + sub_chunks.len(),
                    contextualized_text: None,
                });
                current_els.clear();
            }
            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(t);
            current_els.extend(idxs);
        }

        if !current_text.is_empty() {
            let el_list: Vec<Element> = {
                let mut sorted = current_els;
                sorted.sort();
                sorted.dedup();
                sorted.iter().map(|&i| elements[i].clone()).collect()
            };
            let char_count = current_text.len();
            sub_chunks.push(Chunk {
                elements: el_list,
                text: current_text,
                char_count,
                section: None,
                index: existing_chunks.len() + sub_chunks.len(),
                contextualized_text: None,
            });
        }

        sub_chunks
    }
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
        contextualized_text: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Metadata;
    use crate::format::FileFormat;

    #[test]
    fn chunk_contextualized_text_defaults_to_none() {
        let chunk = Chunk {
            elements: vec![],
            text: "Raw text.".to_string(),
            char_count: 9,
            section: None,
            index: 0,
            contextualized_text: None,
        };
        assert!(chunk.contextualized_text.is_none());
        assert_eq!(chunk.text, "Raw text.");
    }

    #[test]
    fn chunk_contextualized_text_holds_prefix_plus_raw() {
        let chunk = Chunk {
            elements: vec![],
            text: "Raw text.".to_string(),
            char_count: 9,
            section: None,
            index: 0,
            contextualized_text: Some("Context. Raw text.".to_string()),
        };
        assert_eq!(
            chunk.contextualized_text.as_deref(),
            Some("Context. Raw text.")
        );
        // Raw text field must remain unchanged — it's what display/exact-match uses.
        assert_eq!(chunk.text, "Raw text.");
    }

    #[test]
    fn chunk_contextualized_text_skipped_when_none_in_json() {
        let chunk = Chunk {
            elements: vec![],
            text: "Raw.".to_string(),
            char_count: 4,
            section: None,
            index: 0,
            contextualized_text: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed.get("contextualized_text").is_none(),
            "None should be skipped in JSON output, got: {json}"
        );
    }

    #[test]
    fn chunk_deserializes_without_contextualized_text_field() {
        let json = r#"{"elements":[],"text":"Raw.","char_count":4,"section":null,"index":0}"#;
        let chunk: Chunk = serde_json::from_str(json).unwrap();
        assert!(chunk.contextualized_text.is_none());
        assert_eq!(chunk.text, "Raw.");
    }

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
    fn overlap_does_not_panic_on_multibyte_utf8() {
        // Em-dash (─, 3 bytes) at the would-be slice boundary used to panic
        // with "byte index N is not a char boundary". Regression for
        // VIPER Assist corpus indexing.
        let text = "abcdefghij─klmnopqrstuv─wxyz1234567890─end";
        let doc = doc_with(vec![Element::new(ElementKind::Paragraph, text)]);
        let chunks = doc.chunk(&ChunkingStrategy::RecursiveCharacter {
            max_characters: 12,
            overlap: 5,
            separators: default_separators(),
        });
        assert!(chunks.len() >= 2, "need overlap to exercise the path");
        for c in &chunks {
            assert!(c.text.is_char_boundary(0));
            assert!(c.text.is_char_boundary(c.text.len()));
        }
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

    // --- Semantic chunking tests ---

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "expected 0.0, got {sim}");
    }

    #[test]
    fn semantic_chunk_splits_dissimilar_topics() {
        let doc = doc_with(vec![
            Element::new(
                ElementKind::Paragraph,
                "Rust is a systems programming language focused on safety and performance.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "The Rust compiler enforces memory safety without a garbage collector.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Chocolate cake is made with flour, sugar, cocoa powder, and eggs.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Baking requires precise measurements and oven temperature control.",
            ),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Semantic {
            max_characters: 500,
            similarity_threshold: Some(0.3),
            percentile_threshold: None,
        });
        assert!(
            chunks.len() >= 2,
            "expected ≥2 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn semantic_chunk_keeps_similar_together() {
        let doc = doc_with(vec![
            Element::new(
                ElementKind::Paragraph,
                "Rust is a systems programming language.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Rust programming focuses on memory safety.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "The Rust language has a strong type system.",
            ),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Semantic {
            max_characters: 500,
            similarity_threshold: Some(0.1),
            percentile_threshold: None,
        });
        assert_eq!(chunks.len(), 1, "expected 1 chunk, got {}", chunks.len());
    }

    #[test]
    fn semantic_chunk_with_custom_embedder() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "Hello world"),
            Element::new(ElementKind::Paragraph, "Goodbye world"),
        ]);
        // Custom embedder that returns length-based vector
        let embedder = |text: &str| -> Vec<f32> { vec![text.len() as f32] };
        let chunks = doc.chunk_with_embedder(
            &ChunkingStrategy::Semantic {
                max_characters: 500,
                similarity_threshold: Some(0.5),
                percentile_threshold: None,
            },
            &embedder,
        );
        assert!(!chunks.is_empty());
    }

    #[test]
    fn semantic_chunk_percentile_threshold() {
        let doc = doc_with(vec![
            Element::new(ElementKind::Paragraph, "Rust programming language is fast."),
            Element::new(
                ElementKind::Paragraph,
                "Rust ensures memory safety at compile time.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Cooking pasta requires boiling water and salt.",
            ),
            Element::new(
                ElementKind::Paragraph,
                "Italian cuisine uses fresh ingredients and olive oil.",
            ),
        ]);
        let chunks = doc.chunk(&ChunkingStrategy::Semantic {
            max_characters: 500,
            similarity_threshold: None,
            percentile_threshold: Some(50.0),
        });
        assert!(
            chunks.len() >= 2,
            "expected ≥2 chunks, got {}",
            chunks.len()
        );
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

    // --- Context injection tests ---

    #[test]
    fn context_injection_prepends_title_and_section() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![
                Element::new(ElementKind::Title, "My Doc"),
                Element::new(ElementKind::Paragraph, "Content here"),
            ],
        };
        doc.metadata.title = Some("My Doc".to_string());
        let mut chunks = doc.chunk(&ChunkingStrategy::ByTitle {
            max_characters: 1000,
            overlap: 0,
        });
        let injection = ContextInjection::default();
        doc.inject_context(&mut chunks, &injection);
        assert!(
            chunks[0].text.contains("My Doc"),
            "chunk should contain title"
        );
        assert!(
            chunks[0].text.contains("Content here"),
            "chunk should contain original text"
        );
    }

    #[test]
    fn context_injection_custom_template() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![Element::new(ElementKind::Paragraph, "Hello")],
        };
        doc.metadata.title = Some("Title".to_string());
        let mut chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        });
        chunks[0].section = Some("Intro".to_string());
        let injection = ContextInjection {
            template: "[{document_title}|{section}] {chunk_text}".to_string(),
        };
        doc.inject_context(&mut chunks, &injection);
        assert_eq!(chunks[0].text, "[Title|Intro] Hello");
    }

    #[test]
    fn context_injection_missing_section() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![Element::new(ElementKind::Paragraph, "Text")],
        };
        doc.metadata.title = Some("Doc".to_string());
        let mut chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        });
        // section is None
        let injection = ContextInjection::default();
        doc.inject_context(&mut chunks, &injection);
        // Should not panic, section replaced with empty string
        assert!(chunks[0].text.contains("Doc"));
        assert!(chunks[0].text.contains("Text"));
    }

    #[test]
    fn context_injection_updates_char_count() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![Element::new(ElementKind::Paragraph, "Short")],
        };
        doc.metadata.title = Some("Title".to_string());
        let mut chunks = doc.chunk(&ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        });
        let original_len = chunks[0].char_count;
        let injection = ContextInjection::default();
        doc.inject_context(&mut chunks, &injection);
        assert!(
            chunks[0].char_count > original_len,
            "char_count should increase"
        );
        assert_eq!(chunks[0].char_count, chunks[0].text.len());
    }
}
