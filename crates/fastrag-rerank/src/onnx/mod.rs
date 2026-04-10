//! ONNX-based reranker using gte-reranker-modernbert-base.
//!
//! The model is a cross-encoder: it takes (query, passage) pairs as input and
//! outputs a single relevance logit per pair. We apply sigmoid to convert to a
//! [0, 1] score.

pub mod model_source;

use std::path::Path;
use std::sync::Mutex;

use fastrag_index::SearchHit;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

use crate::RerankError;
use crate::Reranker;
use model_source::{
    HfHubOnnxDownloader, MODEL_FILE, OnnxModelSource, TOKENIZER_FILE, resolve_model_dir_default,
};

/// gte-reranker-modernbert-base (149M params, Apache 2.0).
///
/// The `Mutex` around `Session` is needed because `ort::Session::run()` takes
/// `&mut self` while the `Reranker` trait requires `&self` (for `Send + Sync`).
pub struct GteModernBertReranker {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
}

impl GteModernBertReranker {
    pub const MODEL_ID: &'static str = "gte-reranker-modernbert-base";

    pub fn model_source() -> OnnxModelSource {
        OnnxModelSource {
            repo: "Alibaba-NLP/gte-reranker-modernbert-base",
            dir_name: "gte-reranker-modernbert-base",
            model_path_in_repo: "onnx/model.onnx",
            tokenizer_path_in_repo: "tokenizer.json",
        }
    }

    /// Load from a local directory containing `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self, RerankError> {
        let tokenizer_path = model_dir.join(TOKENIZER_FILE);
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RerankError::Model(format!("load tokenizer: {e}")))?;

        let model_path = model_dir.join(MODEL_FILE);
        // Disable ORT's BFCArena CPU allocator to avoid monotonic memory growth.
        // See reference_ort_memory_lessons.md — safe for small-batch reranking,
        // but the arena never frees and would leak on bulk workloads.
        let session = Session::builder()
            .map_err(|e| RerankError::Model(format!("ORT session builder: {e}")))?
            .with_execution_providers([ort::ep::CPU::default().build()])
            .map_err(|e| RerankError::Model(format!("ORT CPU provider: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::Model(format!("ORT optimization level: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| RerankError::Model(format!("ORT load model: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    /// Download the model if needed and load it.
    pub fn load_default() -> Result<Self, RerankError> {
        let model_dir = resolve_model_dir_default(&Self::model_source(), &HfHubOnnxDownloader)?;
        Self::load(&model_dir)
    }

    /// Score a batch of (query, passage) pairs. Returns one score per pair.
    fn score_pairs(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, RerankError> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }

        // Build tokenizer input: each pair is (query, passage)
        let pairs: Vec<_> = passages
            .iter()
            .map(|passage| {
                tokenizers::EncodeInput::Dual(
                    tokenizers::InputSequence::Raw(query.into()),
                    tokenizers::InputSequence::Raw((*passage).into()),
                )
            })
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| RerankError::Model(format!("tokenize: {e}")))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build padded input_ids and attention_mask tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let offset = i * max_len;
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[offset + j] = id as i64;
                attention_mask[offset + j] = m as i64;
            }
        }

        let shape = [batch_size, max_len];

        let input_ids_array = ndarray::Array2::from_shape_vec(shape, input_ids)
            .map_err(|e| RerankError::Model(format!("shape input_ids: {e}")))?;
        let attention_mask_array = ndarray::Array2::from_shape_vec(shape, attention_mask)
            .map_err(|e| RerankError::Model(format!("shape attention_mask: {e}")))?;

        let ids_tensor = TensorRef::from_array_view(&input_ids_array)
            .map_err(|e| RerankError::Model(format!("build input_ids tensor: {e}")))?;
        let mask_tensor = TensorRef::from_array_view(&attention_mask_array)
            .map_err(|e| RerankError::Model(format!("build attention_mask tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| RerankError::Model(format!("session lock poisoned: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| RerankError::Model(format!("ORT run: {e}")))?;

        // Extract logits — shape is [batch_size, 1] or [batch_size]
        let logits_value = &outputs[0];
        let (_shape, logits_data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| RerankError::Model(format!("extract logits: {e}")))?;

        let scores: Vec<f32> = logits_data
            .iter()
            .take(batch_size)
            .map(|&logit| sigmoid(logit))
            .collect();

        Ok(scores)
    }
}

impl Reranker for GteModernBertReranker {
    fn model_id(&self) -> &'static str {
        Self::MODEL_ID
    }

    fn rerank(&self, query: &str, mut hits: Vec<SearchHit>) -> Result<Vec<SearchHit>, RerankError> {
        if hits.is_empty() {
            return Ok(hits);
        }

        let passages: Vec<&str> = hits.iter().map(|h| h.entry.chunk_text.as_str()).collect();
        let scores = self.score_pairs(query, &passages)?;

        if scores.len() != hits.len() {
            return Err(RerankError::Model(format!(
                "score count mismatch: expected {}, got {}",
                hits.len(),
                scores.len()
            )));
        }

        for (hit, score) in hits.iter_mut().zip(scores.iter()) {
            hit.score = *score;
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(hits)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn model_id_is_correct() {
        assert_eq!(
            GteModernBertReranker::MODEL_ID,
            "gte-reranker-modernbert-base"
        );
    }

    #[test]
    fn model_source_coordinates() {
        let src = GteModernBertReranker::model_source();
        assert_eq!(src.repo, "Alibaba-NLP/gte-reranker-modernbert-base");
        assert_eq!(src.dir_name, "gte-reranker-modernbert-base");
    }

    /// Full inference test — requires the ONNX model to be downloaded.
    /// Run with: FASTRAG_RERANK_TEST=1 cargo test -p fastrag-rerank --features onnx -- --ignored
    #[test]
    #[ignore]
    fn rerank_with_real_model() {
        if std::env::var("FASTRAG_RERANK_TEST").is_err() {
            return;
        }

        let reranker =
            GteModernBertReranker::load_default().expect("load gte-reranker-modernbert-base");

        let hits = vec![
            crate::test_utils::test_hit(1, "The capital of France is Paris", 0.5),
            crate::test_utils::test_hit(2, "Rust is a systems programming language", 0.8),
            crate::test_utils::test_hit(3, "Paris is known for the Eiffel Tower", 0.3),
        ];

        let reranked = reranker
            .rerank("What is the capital of France?", hits)
            .expect("rerank");

        assert_eq!(reranked.len(), 3);
        // All scores should be in [0, 1]
        for hit in &reranked {
            assert!(
                hit.score >= 0.0 && hit.score <= 1.0,
                "score {} out of range",
                hit.score
            );
        }
        // The Paris-related hits should score higher than the Rust hit
        let rust_hit = reranked.iter().find(|h| h.entry.id == 2).unwrap();
        let capital_hit = reranked.iter().find(|h| h.entry.id == 1).unwrap();
        assert!(
            capital_hit.score > rust_hit.score,
            "capital hit ({}) should score higher than rust hit ({})",
            capital_hit.score,
            rust_hit.score
        );
    }
}
