//! llama-cpp-based reranker using bge-reranker-v2-m3 GGUF Q8_0.
//!
//! Spawns a second `llama-server` subprocess in reranking mode
//! (`--embedding --pooling rank`) and talks to it via `/v1/rerank`.

pub mod client;

use fastrag_embed::llama_cpp::{LlamaServerConfig, LlamaServerHandle, ModelSource};
use fastrag_index::SearchHit;

use crate::RerankError;
use crate::Reranker;
use client::{LlamaCppRerankClient, RerankResult};

/// bge-reranker-v2-m3 at Q8_0 quantization, served via llama-server.
pub struct BgeRerankerV2M3Llama {
    // Rust drops fields in declaration order: client drops first (closing HTTP
    // connections), then handle drops (sending SIGKILL to llama-server).
    client: LlamaCppRerankClient,
    #[allow(dead_code)]
    handle: LlamaServerHandle,
}

impl BgeRerankerV2M3Llama {
    pub const HF_REPO: &'static str = "klnstpr/bge-reranker-v2-m3-Q8_0-GGUF";
    pub const GGUF_FILE: &'static str = "bge-reranker-v2-m3-q8_0.gguf";

    pub fn model_source() -> ModelSource {
        ModelSource::HfHub {
            repo: Self::HF_REPO,
            file: Self::GGUF_FILE,
        }
    }

    /// Spawn a llama-server in reranking mode and build a ready-to-use preset.
    ///
    /// The `LlamaServerConfig.extra_args` must include `--model <path>`,
    /// `--embedding`, and `--pooling rank`.
    pub fn load(server: LlamaServerConfig) -> Result<Self, RerankError> {
        let model_name = server
            .extra_args
            .windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .unwrap_or_default();

        let handle = LlamaServerHandle::spawn(server)
            .map_err(|e| RerankError::Model(format!("llama-server spawn: {e}")))?;

        let client = LlamaCppRerankClient::new(handle.base_url().to_string(), model_name)
            .map_err(|e| RerankError::Model(format!("build rerank client: {e}")))?;

        Ok(Self { client, handle })
    }

    pub fn base_url(&self) -> &str {
        self.handle.base_url()
    }
}

impl Reranker for BgeRerankerV2M3Llama {
    fn model_id(&self) -> &'static str {
        "bge-reranker-v2-m3-Q8_0-GGUF"
    }

    fn rerank(&self, query: &str, mut hits: Vec<SearchHit>) -> Result<Vec<SearchHit>, RerankError> {
        if hits.is_empty() {
            return Ok(hits);
        }

        let documents: Vec<&str> = hits.iter().map(|h| h.entry.chunk_text.as_str()).collect();
        let results: Vec<RerankResult> = self.client.rerank(query, &documents, documents.len())?;

        // Map scores back to hits by index
        for result in &results {
            if result.index < hits.len() {
                hits[result.index].score = result.score;
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_is_correct() {
        // Can't instantiate without a running server, so test the constant.
        assert_eq!(
            BgeRerankerV2M3Llama::HF_REPO,
            "klnstpr/bge-reranker-v2-m3-Q8_0-GGUF"
        );
        assert_eq!(
            BgeRerankerV2M3Llama::GGUF_FILE,
            "bge-reranker-v2-m3-q8_0.gguf"
        );
    }

    #[test]
    fn model_source_coordinates() {
        let src = BgeRerankerV2M3Llama::model_source();
        match src {
            ModelSource::HfHub { repo, file } => {
                assert_eq!(repo, "klnstpr/bge-reranker-v2-m3-Q8_0-GGUF");
                assert_eq!(file, "bge-reranker-v2-m3-q8_0.gguf");
            }
            _ => panic!("expected HfHub variant"),
        }
    }
}
