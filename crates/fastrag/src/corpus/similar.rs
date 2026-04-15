//! Similarity-threshold retrieval.
//!
//! Owns `similarity_search`: embed once, fan out per corpus, adaptive overfetch
//! until above-threshold rows are exhausted (or a server cap is hit), merge and
//! sort by raw cosine, truncate to `max_results`.
//!
//! Narrow by design: no hybrid, no temporal decay, no CWE expansion, no rerank.

use std::path::PathBuf;

use serde::Serialize;

use crate::corpus::{CorpusError, LatencyBreakdown, SearchHitDto};
use crate::filter::FilterExpr;
use fastrag_embed::DynEmbedderTrait;

/// Input to `similarity_search`. Caller resolves corpus names to paths and
/// stamps them back on each hit.
#[derive(Debug, Clone)]
pub struct SimilarityRequest {
    pub text: String,
    pub threshold: f32,
    pub max_results: usize,
    /// Resolved `(name, path)` pairs. Non-empty; caller validates.
    pub targets: Vec<(String, PathBuf)>,
    pub filter: Option<FilterExpr>,
    pub snippet_len: usize,
    /// Hard cap on per-corpus overfetch. The adaptive loop stops at this count.
    pub overfetch_cap: usize,
}

/// One hit in the merged result set.
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityHit {
    pub cosine_similarity: f32,
    pub corpus: String,
    #[serde(flatten)]
    pub dto: SearchHitDto,
}

/// Per-corpus diagnostics surfaced in the response.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PerCorpusStats {
    pub candidates_examined: usize,
    pub above_threshold: usize,
}

/// Aggregate diagnostics surfaced in the response.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SimilarityStats {
    pub candidates_examined: usize,
    pub above_threshold: usize,
    pub returned: usize,
    /// Populated only when the request targeted multiple corpora.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub per_corpus: std::collections::BTreeMap<String, PerCorpusStats>,
}

/// Full response body.
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityResponse {
    pub hits: Vec<SimilarityHit>,
    pub truncated: bool,
    pub stats: SimilarityStats,
    pub latency: LatencyBreakdown,
}

/// Run similarity search. Embeds `request.text` once, fans out per target
/// corpus, merges, sorts, truncates to `max_results`.
pub fn similarity_search(
    _embedder: &dyn DynEmbedderTrait,
    _request: &SimilarityRequest,
) -> Result<SimilarityResponse, CorpusError> {
    // Implemented in Task 3. Keep the function reachable so downstream tasks
    // have a stable symbol to dispatch against.
    Err(CorpusError::Other(
        "similarity_search: not implemented".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_compile() {
        // This test exists to lock the public type shape: every public field
        // is referenced below. Removing or renaming any field breaks compile.
        let stats = SimilarityStats::default();
        let _ = stats.candidates_examined;
        let _ = stats.above_threshold;
        let _ = stats.returned;
        let _ = stats.per_corpus;
        let pc = PerCorpusStats::default();
        let _ = pc.candidates_examined;
        let _ = pc.above_threshold;
    }
}
