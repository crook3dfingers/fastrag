//! Ingest-time hygiene filter chain for security corpora.
//!
//! Filters compose via [`HygieneChain`] and run between chunking and
//! contextualization in `index_path_with_metadata`. The chain applies:
//! 1. Document-level reject filters (MetadataRejectFilter)
//! 2. Chunk-text strip filters (BoilerplateStripper)
//! 3. Language filters (LanguageFilter)
//! 4. Metadata enrichers (KevTemporalTagger)

use std::collections::BTreeMap;

use fastrag_core::Chunk;

pub mod boilerplate;
pub mod kev;
pub mod language;
pub mod reject;

pub use boilerplate::BoilerplateStripper;
pub use reject::{DocFilter, MetadataRejectFilter};

/// Per-run hygiene statistics surfaced to the CLI summary line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HygieneStats {
    pub docs_rejected: usize,
    pub chunks_stripped: usize,
    pub chunks_lang_dropped: usize,
    pub chunks_kev_tagged: usize,
}

/// A filter applied to individual chunk text. Returns the (possibly modified)
/// text. Returning an empty string signals the chunk should be dropped.
pub trait ChunkFilter: Send + Sync {
    fn apply(&self, text: &str, metadata: &BTreeMap<String, String>) -> String;
}

/// Composable hygiene filter chain.
pub struct HygieneChain {
    doc_filters: Vec<Box<dyn DocFilter>>,
    chunk_filters: Vec<Box<dyn ChunkFilter>>,
    /// Metadata enrichers that mutate the metadata map in place.
    enrichers: Vec<Box<dyn MetadataEnricher>>,
}

/// A mutating metadata enricher (e.g., KEV tagger). Runs after chunk filters.
pub trait MetadataEnricher: Send + Sync {
    fn enrich(&self, metadata: &mut BTreeMap<String, String>);
}

impl HygieneChain {
    pub fn new() -> Self {
        Self {
            doc_filters: vec![],
            chunk_filters: vec![],
            enrichers: vec![],
        }
    }

    pub fn with_doc_filter(mut self, f: Box<dyn DocFilter>) -> Self {
        self.doc_filters.push(f);
        self
    }

    pub fn with_chunk_filter(mut self, f: Box<dyn ChunkFilter>) -> Self {
        self.chunk_filters.push(f);
        self
    }

    pub fn with_enricher(mut self, e: Box<dyn MetadataEnricher>) -> Self {
        self.enrichers.push(e);
        self
    }

    /// Apply the full chain to a document's chunks and metadata.
    ///
    /// Returns `None` if a doc-level filter rejects the document, or
    /// `Some((filtered_chunks, stats))` otherwise.
    pub fn apply(
        &self,
        chunks: Vec<Chunk>,
        metadata: &mut BTreeMap<String, String>,
    ) -> Option<(Vec<Chunk>, HygieneStats)> {
        let mut stats = HygieneStats::default();

        // Doc-level reject pass.
        for filter in &self.doc_filters {
            if !filter.keep(&chunks, metadata) {
                return None;
            }
        }

        // Chunk-text strip pass.
        let mut surviving: Vec<Chunk> = Vec::with_capacity(chunks.len());
        for mut chunk in chunks {
            let original_len = chunk.text.len();
            let mut text = chunk.text.clone();
            for filter in &self.chunk_filters {
                text = filter.apply(&text, metadata);
            }
            if text.trim().is_empty() {
                stats.chunks_lang_dropped += 1;
                continue;
            }
            if text.len() != original_len {
                stats.chunks_stripped += 1;
            }
            chunk.text = text.trim().to_string();
            chunk.char_count = chunk.text.len();
            surviving.push(chunk);
        }

        // Metadata enrichment pass.
        for enricher in &self.enrichers {
            enricher.enrich(metadata);
        }
        if metadata.get("kev_flag").map(String::as_str) == Some("true") {
            stats.chunks_kev_tagged += surviving.len();
        }

        Some((surviving, stats))
    }
}

impl Default for HygieneChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::{Chunk, Element, ElementKind};

    fn make_chunk(text: &str) -> Chunk {
        Chunk {
            elements: vec![Element::new(ElementKind::Paragraph, text)],
            text: text.to_string(),
            char_count: text.len(),
            section: None,
            index: 0,
            contextualized_text: None,
        }
    }

    fn rejected_meta() -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("vuln_status".to_string(), "Rejected".to_string());
        m
    }

    #[test]
    fn chain_rejects_rejected_document() {
        let chain = HygieneChain::new().with_doc_filter(Box::new(MetadataRejectFilter::default()));
        let chunks = vec![make_chunk("some text")];
        let mut meta = rejected_meta();
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_none(), "Rejected doc must be dropped");
    }

    #[test]
    fn chain_keeps_analyzed_document() {
        let chain = HygieneChain::new().with_doc_filter(Box::new(MetadataRejectFilter::default()));
        let chunks = vec![make_chunk("CVE analyzed content")];
        let mut meta = BTreeMap::new();
        meta.insert("vuln_status".to_string(), "Analyzed".to_string());
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_some());
        let (out_chunks, _) = result.unwrap();
        assert_eq!(out_chunks.len(), 1);
        assert_eq!(out_chunks[0].text, "CVE analyzed content");
    }

    #[test]
    fn empty_chain_is_passthrough() {
        let chain = HygieneChain::new();
        let chunks = vec![make_chunk("hello")];
        let mut meta = BTreeMap::new();
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_some());
        let (out, _stats) = result.unwrap();
        assert_eq!(out.len(), 1);
    }
}
