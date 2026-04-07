//! Reranker abstraction for FastRAG.
//!
//! A reranker takes a query and a list of first-stage retrieval hits, assigns
//! relevance scores using a cross-encoder, and returns the list sorted by
//! descending score. Used as an opt-in pipeline stage after HNSW retrieval.

use fastrag_index::SearchHit;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("rerank model error: {0}")]
    Model(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &'static str;

    /// Score each hit against the query and return hits sorted by descending score.
    /// The implementation overwrites `hit.score` with the cross-encoder score.
    fn rerank(&self, query: &str, hits: Vec<SearchHit>) -> Result<Vec<SearchHit>, RerankError>;
}

#[cfg(any(feature = "test-utils", test))]
pub mod test_utils {
    use super::*;

    /// Deterministic mock reranker that scores by lexical overlap between query
    /// and chunk text. Good enough to verify ordering changes in tests.
    pub struct MockReranker;

    impl Reranker for MockReranker {
        fn model_id(&self) -> &'static str {
            "mock-reranker"
        }

        fn rerank(
            &self,
            query: &str,
            mut hits: Vec<SearchHit>,
        ) -> Result<Vec<SearchHit>, RerankError> {
            let q_tokens: Vec<String> = query
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            for hit in &mut hits {
                let text = hit.entry.chunk_text.to_lowercase();
                let overlap = q_tokens
                    .iter()
                    .filter(|t| text.contains(t.as_str()))
                    .count() as f32;
                hit.score = overlap;
            }
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(hits)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::MockReranker;
    use super::*;
    use fastrag_index::{ElementKind, IndexEntry};
    use std::path::PathBuf;

    fn hit(id: u64, text: &str, score: f32) -> SearchHit {
        SearchHit {
            entry: IndexEntry {
                id,
                vector: vec![],
                chunk_text: text.to_string(),
                source_path: PathBuf::from(format!("/tmp/doc{id}.txt")),
                chunk_index: 0,
                section: None,
                element_kinds: vec![ElementKind::Paragraph],
                pages: vec![],
                language: None,
                metadata: std::collections::BTreeMap::new(),
            },
            score,
        }
    }

    #[test]
    fn mock_reranker_reorders_by_lexical_overlap() {
        // Initial order (by first-stage HNSW score) puts irrelevant hit first.
        let hits = vec![
            hit(1, "the quick brown fox", 0.9),
            hit(2, "rust async runtime concurrency", 0.8),
            hit(3, "tokio async rust tasks", 0.7),
        ];
        let reranked = MockReranker.rerank("rust async runtime", hits).unwrap();

        // Hit 2 contains all three query terms → should be first.
        assert_eq!(reranked[0].entry.id, 2);
        assert_eq!(reranked[0].score, 3.0);
        // Hit 3 contains two terms.
        assert_eq!(reranked[1].entry.id, 3);
        assert_eq!(reranked[1].score, 2.0);
        // Hit 1 contains none.
        assert_eq!(reranked[2].entry.id, 1);
        assert_eq!(reranked[2].score, 0.0);
    }

    #[test]
    fn mock_reranker_empty_hits_returns_empty() {
        let out = MockReranker.rerank("anything", vec![]).unwrap();
        assert!(out.is_empty());
    }
}
