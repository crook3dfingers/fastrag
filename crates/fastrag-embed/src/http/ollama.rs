//! Ollama embedder backend.
//!
//! Ollama's /api/embeddings endpoint takes a single `prompt` at a time and
//! hosts arbitrary user-pulled models, so we probe dimension on construction.

use std::env;

use serde::Deserialize;
use serde_json::json;

use crate::http::{build_client, ensure_success, send_with_retry};
use crate::{DynEmbedderTrait, EmbedError, EmbedderIdentity, PassageText, PrefixScheme, QueryText};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug)]
pub struct OllamaEmbedder {
    base_url: String,
    model: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

#[derive(Deserialize)]
struct Resp {
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    pub fn new(model: String) -> Result<Self, EmbedError> {
        let base_url = env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let client = build_client()?;
        let dim = Self::probe_dim(&client, &base_url, &model)?;
        Ok(Self {
            base_url,
            model,
            dim,
            client,
        })
    }

    /// Test helper — skip dim probe.
    pub fn from_parts(base_url: String, model: String, dim: usize) -> Self {
        Self {
            base_url,
            model,
            dim,
            client: build_client().expect("reqwest client"),
        }
    }

    pub fn runtime_identity(&self) -> EmbedderIdentity {
        EmbedderIdentity {
            model_id: format!("ollama:{}", self.model),
            dim: self.dim,
            prefix_scheme_hash: PrefixScheme::NONE.hash(),
        }
    }

    fn probe_dim(
        client: &reqwest::blocking::Client,
        base_url: &str,
        model: &str,
    ) -> Result<usize, EmbedError> {
        let url = format!("{}/api/embeddings", base_url);
        let body = json!({ "model": model, "prompt": "a" });
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| EmbedError::DimensionProbeFailed(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(EmbedError::DimensionProbeFailed(format!(
                "status {}",
                resp.status().as_u16()
            )));
        }
        let parsed: Resp = resp
            .json()
            .map_err(|e| EmbedError::DimensionProbeFailed(e.to_string()))?;
        if parsed.embedding.is_empty() {
            return Err(EmbedError::DimensionProbeFailed(
                "empty embedding vector".into(),
            ));
        }
        Ok(parsed.embedding.len())
    }

    fn call(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embeddings", self.base_url);
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let body = json!({ "model": &self.model, "prompt": text });
            let resp = send_with_retry(|| self.client.post(&url).json(&body))?;
            let resp = ensure_success(resp)?;
            let parsed: Resp = resp.json().map_err(|e| EmbedError::Http(e.to_string()))?;
            if parsed.embedding.len() != self.dim {
                return Err(EmbedError::UnexpectedDim {
                    expected: self.dim,
                    got: parsed.embedding.len(),
                });
            }
            out.push(parsed.embedding);
        }
        Ok(out)
    }
}

impl DynEmbedderTrait for OllamaEmbedder {
    fn model_id(&self) -> &'static str {
        // Ollama model ids are runtime strings; return a stable placeholder.
        "ollama:<runtime>"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn prefix_scheme(&self) -> PrefixScheme {
        PrefixScheme::NONE
    }

    fn prefix_scheme_hash(&self) -> u64 {
        PrefixScheme::NONE.hash()
    }

    /// Override default — model_id() is a placeholder so we build from runtime state.
    fn identity(&self) -> EmbedderIdentity {
        self.runtime_identity()
    }

    fn default_batch_size(&self) -> usize {
        16
    }

    fn embed_query_dyn(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(QueryText::as_str).collect();
        self.call(&refs)
    }

    fn embed_passage_dyn(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(PassageText::as_str).collect();
        self.call(&refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    async fn mount_probe_and_embed(server: &MockServer, dim: usize) {
        let body = json!({ "embedding": vec![0.25_f32; dim] });
        Mock::given(method("POST"))
            .and(path("/api/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(server)
            .await;
    }

    #[test]
    fn happy_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (uri, _g) = rt.block_on(async {
            let server = MockServer::start().await;
            mount_probe_and_embed(&server, 4).await;
            (server.uri(), server)
        });
        unsafe { std::env::set_var("OLLAMA_HOST", &uri) };
        let e = OllamaEmbedder::new("nomic-embed-text".into()).unwrap();
        assert_eq!(e.dim(), 4);
        let out = e
            .embed_query_dyn(&[QueryText::new("hello"), QueryText::new("world")])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn probe_failure_on_refused() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("OLLAMA_HOST", "http://127.0.0.1:1") };
        let err = OllamaEmbedder::new("nomic-embed-text".into()).unwrap_err();
        assert!(matches!(err, EmbedError::DimensionProbeFailed(_)));
    }

    #[test]
    fn missing_model_404() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (uri, _g) = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/api/embeddings"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(json!({ "embedding": vec![0.0_f32; 4] })),
                )
                .up_to_n_times(1)
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .and(path("/api/embeddings"))
                .respond_with(ResponseTemplate::new(404).set_body_string("model not found"))
                .mount(&server)
                .await;
            (server.uri(), server)
        });
        unsafe { std::env::set_var("OLLAMA_HOST", &uri) };
        let e = OllamaEmbedder::new("nomic-embed-text".into()).unwrap();
        let err = e.embed_query_dyn(&[QueryText::new("hello")]).unwrap_err();
        match err {
            EmbedError::Api { status: 404, .. } => {}
            other => panic!("expected Api 404, got {other:?}"),
        }
    }

    #[test]
    fn identity_is_namespaced_with_runtime_model() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (uri, _g) = rt.block_on(async {
            let server = MockServer::start().await;
            mount_probe_and_embed(&server, 8).await;
            (server.uri(), server)
        });
        unsafe { std::env::set_var("OLLAMA_HOST", &uri) };
        let e = OllamaEmbedder::new("nomic-embed-text".into()).unwrap();
        let id = e.identity();
        assert_eq!(id.model_id, "ollama:nomic-embed-text");
        assert_eq!(id.dim, 8);
        assert_eq!(e.default_batch_size(), 16);
    }
}

#[cfg(test)]
mod ollama_runtime_identity {
    use super::*;
    use crate::{EmbedderIdentity, PrefixScheme};

    #[test]
    fn runtime_identity_encodes_model_name_and_probed_dim() {
        let e = OllamaEmbedder::from_parts(
            "http://localhost:11434".into(),
            "nomic-embed-text".into(),
            768,
        );
        let id: EmbedderIdentity = e.runtime_identity();
        assert_eq!(id.model_id, "ollama:nomic-embed-text");
        assert_eq!(id.dim, 768);
        assert_eq!(id.prefix_scheme_hash, PrefixScheme::NONE.hash());
    }
}
