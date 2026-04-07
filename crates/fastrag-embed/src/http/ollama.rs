//! Ollama embedder backend.
//!
//! Ollama's /api/embeddings endpoint takes a single `prompt` at a time and
//! hosts arbitrary user-pulled models, so we probe dimension on construction.

use std::env;

use serde::Deserialize;
use serde_json::json;

use crate::http::{build_client, ensure_success, send_with_retry};
use crate::{EmbedError, Embedder};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug)]
pub struct OllamaEmbedder {
    model: String,
    base_url: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

#[derive(Deserialize)]
struct Resp {
    embedding: Vec<f32>,
}

impl OllamaEmbedder {
    pub fn new(model: impl Into<String>) -> Result<Self, EmbedError> {
        let model = model.into();
        let base_url = env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Self::construct(model, base_url)
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        if let Ok(d) = probe_dim(&self.client, &self.base_url, &self.model) {
            self.dim = d;
        }
        self
    }

    fn construct(model: String, base_url: String) -> Result<Self, EmbedError> {
        let client = build_client()?;
        let dim = probe_dim(&client, &base_url, &model)?;
        Ok(Self {
            model,
            base_url,
            dim,
            client,
        })
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

impl Embedder for OllamaEmbedder {
    fn model_id(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn default_batch_size(&self) -> usize {
        1
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
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
        let e = OllamaEmbedder::new("nomic-embed-text").unwrap();
        assert_eq!(e.dim(), 4);
        let out = e.embed(&["hello", "world"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn probe_failure_on_refused() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("OLLAMA_HOST", "http://127.0.0.1:1") };
        let err = OllamaEmbedder::new("nomic-embed-text").unwrap_err();
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
        let e = OllamaEmbedder::new("nomic-embed-text").unwrap();
        let err = e.embed(&["hello"]).unwrap_err();
        match err {
            EmbedError::Api { status: 404, .. } => {}
            other => panic!("expected Api 404, got {other:?}"),
        }
    }

    #[test]
    fn model_id_is_namespaced() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (uri, _g) = rt.block_on(async {
            let server = MockServer::start().await;
            mount_probe_and_embed(&server, 8).await;
            (server.uri(), server)
        });
        unsafe { std::env::set_var("OLLAMA_HOST", &uri) };
        let e = OllamaEmbedder::new("nomic-embed-text").unwrap();
        assert_eq!(e.model_id(), "ollama:nomic-embed-text");
        assert_eq!(e.default_batch_size(), 1);
    }
}
