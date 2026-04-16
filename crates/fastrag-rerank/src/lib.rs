//! Reranker abstraction for FastRAG.
//!
//! A reranker takes a query and a list of first-stage retrieval hits, assigns
//! relevance scores using a cross-encoder, and returns the list sorted by
//! descending score. Used as an opt-in pipeline stage after HNSW retrieval.

use thiserror::Error;

#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;
#[cfg(feature = "onnx")]
pub mod onnx;

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("rerank model error: {0}")]
    Model(String),
    #[error("HTTP: {0}")]
    Http(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Lightweight hit for reranking — carries only what the cross-encoder needs.
#[derive(Debug, Clone)]
pub struct RerankHit {
    pub id: u64,
    pub chunk_text: String,
    pub score: f32,
}

pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &'static str;

    /// Score each hit against the query and return hits sorted by descending score.
    /// The implementation overwrites `hit.score` with the cross-encoder score.
    fn rerank(&self, query: &str, hits: Vec<RerankHit>) -> Result<Vec<RerankHit>, RerankError>;

    /// Liveness probe for `/ready`. In-process rerankers are always ready
    /// once constructed; HTTP-backed ones can override to check the remote
    /// peer. Must be cheap — `/ready` is polled frequently.
    fn is_ready(&self) -> bool {
        true
    }
}

#[cfg(any(feature = "test-utils", test))]
pub mod test_utils {
    use super::*;

    /// Build a `RerankHit` for testing.
    pub fn test_hit(id: u64, text: &str, score: f32) -> RerankHit {
        RerankHit {
            id,
            chunk_text: text.to_string(),
            score,
        }
    }

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
            mut hits: Vec<RerankHit>,
        ) -> Result<Vec<RerankHit>, RerankError> {
            let q_tokens: Vec<String> = query
                .to_lowercase()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            for hit in &mut hits {
                let text = hit.chunk_text.to_lowercase();
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
    use super::test_utils::{MockReranker, test_hit};
    use super::*;

    #[test]
    fn mock_reranker_reorders_by_lexical_overlap() {
        let hits = vec![
            test_hit(1, "the quick brown fox", 0.9),
            test_hit(2, "rust async runtime concurrency", 0.8),
            test_hit(3, "tokio async rust tasks", 0.7),
        ];
        let reranked = MockReranker.rerank("rust async runtime", hits).unwrap();

        assert_eq!(reranked[0].id, 2);
        assert_eq!(reranked[0].score, 3.0);
        assert_eq!(reranked[1].id, 3);
        assert_eq!(reranked[1].score, 2.0);
        assert_eq!(reranked[2].id, 1);
        assert_eq!(reranked[2].score, 0.0);
    }

    #[test]
    fn mock_reranker_empty_hits_returns_empty() {
        let out = MockReranker.rerank("anything", vec![]).unwrap();
        assert!(out.is_empty());
    }
}
