//! Completion model preset + chat client for contextualization.
//!
//! Consumed by the `fastrag-context` crate. The preset is a data-file
//! choice (GGUF path + friendly model id) and can be swapped from the CLI
//! without a code change via `--context-model`. The chat client is a thin
//! blocking HTTP wrapper over `llama-server`'s `/v1/chat/completions`
//! endpoint — parallel to [`crate::llama_cpp::LlamaCppClient`] which hits
//! the `/v1/embeddings` endpoint.

use crate::llama_cpp::model_source::ModelSource;

/// Default completion preset used for contextualization.
///
/// **Research pass (2026-04-10):**
/// - HF repo:  `unsloth/Qwen3-4B-Instruct-2507-GGUF`
/// - GGUF file: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
/// - Context window: 32768 tokens (base; YaRN extends further)
/// - Quantization: Q4_K_M — ~95–98% of full-precision quality at ~4× the
///   CPU throughput of Q8.
/// - Rationale: Qwen3-4B-Instruct-2507 (released 2025-07) is within the
///   12-month recency window, runs well under llama-server on CPU, and
///   benchmarks above Llama-3.2-3B / Qwen2.5-3B on instruction following
///   for short, structured outputs — which is exactly what a 50–100 token
///   context prefix is. 40k-chunk ingest on a mid-range CPU is the
///   ~8-hour target from the spec; with llama-server continuous batching
///   this is tight but feasible. Swap via `--context-model` if the local
///   CPU budget does not hit the target.
pub struct DefaultCompletionPreset;

impl DefaultCompletionPreset {
    pub const MODEL_ID: &'static str = "qwen3-4b-instruct-2507-q4_k_m";
    pub const HF_REPO: &'static str = "unsloth/Qwen3-4B-Instruct-2507-GGUF";
    pub const GGUF_FILE: &'static str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    pub const CONTEXT_WINDOW: usize = 32768;

    pub fn model_source() -> ModelSource {
        ModelSource::HfHub {
            repo: Self::HF_REPO,
            file: Self::GGUF_FILE,
        }
    }
}

/// Minimal HTTP client for `llama-server`'s `/v1/chat/completions` endpoint.
///
/// Synchronous, `reqwest::blocking`-based to match the rest of fastrag-embed.
/// The client does not spawn the server — pair it with a
/// [`crate::llama_cpp::LlamaServerHandle`] that exposes `base_url()`.
pub struct LlamaCppChatClient {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

impl LlamaCppChatClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("reqwest client"),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    /// Send a single-turn chat completion and return the trimmed content.
    ///
    /// Error mapping:
    /// - HTTP layer errors → [`CompletionError::Http`]
    /// - Non-2xx responses → [`CompletionError::BadStatus`] (body preserved)
    /// - Valid 200 but empty / whitespace content → [`CompletionError::EmptyCompletion`]
    /// - Malformed JSON or missing expected fields → [`CompletionError::ParseError`]
    pub fn complete(&self, prompt: &str) -> Result<String, CompletionError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "max_tokens": 200,
            "temperature": 0.0,
            "stream": false,
        });

        let resp = self.client.post(&url).json(&body).send()?;
        let status = resp.status();
        let text = resp.text()?;

        if !status.is_success() {
            return Err(CompletionError::BadStatus {
                status: status.as_u16(),
                body: text,
            });
        }

        let parsed: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CompletionError::ParseError(format!("{e}: {text}")))?;

        let content = parsed
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| {
                CompletionError::ParseError(format!("missing choices[0].message.content: {text}"))
            })?;

        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Err(CompletionError::EmptyCompletion);
        }
        Ok(trimmed.to_string())
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Errors returned by [`LlamaCppChatClient::complete`].
#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("non-200: status={status}, body={body}")]
    BadStatus { status: u16, body: String },
    #[error("parse: {0}")]
    ParseError(String),
    #[error("empty completion")]
    EmptyCompletion,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: ensure the research-pass values were substituted for
    /// the original `REPLACE` placeholders. Swap the preset freely; do not
    /// reintroduce literal `REPLACE` tokens without updating this test.
    #[test]
    fn preset_constants_are_nonempty() {
        assert!(!DefaultCompletionPreset::HF_REPO.contains("REPLACE"));
        assert!(!DefaultCompletionPreset::GGUF_FILE.contains("REPLACE"));
        assert!(!DefaultCompletionPreset::MODEL_ID.contains("REPLACE"));
        assert!(!DefaultCompletionPreset::HF_REPO.is_empty());
        assert!(!DefaultCompletionPreset::GGUF_FILE.is_empty());
        assert!(!DefaultCompletionPreset::MODEL_ID.is_empty());
        const { assert!(DefaultCompletionPreset::CONTEXT_WINDOW >= 2048) };
    }

    #[test]
    fn preset_model_source_is_hf_hub() {
        let src = DefaultCompletionPreset::model_source();
        match src {
            ModelSource::HfHub { repo, file } => {
                assert_eq!(repo, DefaultCompletionPreset::HF_REPO);
                assert_eq!(file, DefaultCompletionPreset::GGUF_FILE);
            }
            _ => panic!("expected HfHub model source"),
        }
    }
}
