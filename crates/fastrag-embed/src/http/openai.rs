//! OpenAI embedder backend.
//!
//! Blocking HTTP client. Static dim table — no silent probing. Supports
//! OpenAI's native batch input in one request.

use std::env;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::http::{build_client, ensure_success, send_with_retry};
use crate::{EmbedError, Embedder};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

fn dim_for(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        _ => None,
    }
}

#[derive(Debug)]
pub struct OpenAIEmbedder {
    model: String,
    api_key: String,
    base_url: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl OpenAIEmbedder {
    pub fn new(model: impl Into<String>) -> Result<Self, EmbedError> {
        let model = model.into();
        let dim = dim_for(&model).ok_or_else(|| EmbedError::UnknownModel {
            backend: "openai",
            model: model.clone(),
        })?;
        let api_key =
            env::var("OPENAI_API_KEY").map_err(|_| EmbedError::MissingEnv("OPENAI_API_KEY"))?;
        Ok(Self {
            model,
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            dim,
            client: build_client()?,
        })
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[allow(dead_code)]
#[derive(Serialize)]
struct Req<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct Resp {
    data: Vec<RespItem>,
}

#[derive(Deserialize)]
struct RespItem {
    embedding: Vec<f32>,
}

impl Embedder for OpenAIEmbedder {
    fn model_id(&self) -> String {
        format!("openai:{}", self.model)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn default_batch_size(&self) -> usize {
        512
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url);
        let body = json!({ "model": &self.model, "input": texts });
        let resp = send_with_retry(|| {
            self.client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
        })?;
        let resp = ensure_success(resp)?;
        let parsed: Resp = resp.json().map_err(|e| EmbedError::Http(e.to_string()))?;
        if parsed.data.len() != texts.len() {
            return Err(EmbedError::UnexpectedDim {
                expected: texts.len(),
                got: parsed.data.len(),
            });
        }
        let vecs: Vec<Vec<f32>> = parsed.data.into_iter().map(|r| r.embedding).collect();
        if let Some(first) = vecs.first()
            && first.len() != self.dim
        {
            return Err(EmbedError::UnexpectedDim {
                expected: self.dim,
                got: first.len(),
            });
        }
        Ok(vecs)
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

    fn make_embedder(base: &str) -> OpenAIEmbedder {
        unsafe { std::env::set_var("OPENAI_API_KEY", "test-key") };
        OpenAIEmbedder::new("text-embedding-3-small")
            .unwrap()
            .with_base_url(base.to_string())
    }

    #[test]
    fn happy_path_round_trip() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (server_uri, _guard) = rt.block_on(async {
            let server = MockServer::start().await;
            let body = json!({
                "data": [
                    { "embedding": vec![0.1_f32; 1536] },
                    { "embedding": vec![0.2_f32; 1536] },
                ]
            });
            Mock::given(method("POST"))
                .and(path("/embeddings"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body))
                .mount(&server)
                .await;
            (server.uri(), server)
        });
        let e = make_embedder(&server_uri);
        let out = e.embed(&["a", "b"]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 1536);
        assert!((out[0][0] - 0.1).abs() < 1e-6);
        assert!((out[1][0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn api_error_401() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (server_uri, _guard) = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/embeddings"))
                .respond_with(ResponseTemplate::new(401).set_body_string(r#"{"error":"bad key"}"#))
                .mount(&server)
                .await;
            (server.uri(), server)
        });
        let e = make_embedder(&server_uri);
        let err = e.embed(&["a"]).unwrap_err();
        match err {
            EmbedError::Api { status, message } => {
                assert_eq!(status, 401);
                assert!(message.contains("bad key"));
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn length_mismatch() {
        let _guard = ENV_LOCK.lock().unwrap();
        let rt = rt();
        let (server_uri, _guard) = rt.block_on(async {
            let server = MockServer::start().await;
            let body = json!({ "data": [ { "embedding": vec![0.0_f32; 1536] } ] });
            Mock::given(method("POST"))
                .and(path("/embeddings"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body))
                .mount(&server)
                .await;
            (server.uri(), server)
        });
        let e = make_embedder(&server_uri);
        let err = e.embed(&["a", "b"]).unwrap_err();
        assert!(matches!(
            err,
            EmbedError::UnexpectedDim {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn unknown_model_is_rejected() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("OPENAI_API_KEY", "k") };
        let err = OpenAIEmbedder::new("text-embedding-9001").unwrap_err();
        match err {
            EmbedError::UnknownModel { backend, model } => {
                assert_eq!(backend, "openai");
                assert_eq!(model, "text-embedding-9001");
            }
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }

    #[test]
    fn model_id_is_namespaced() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("OPENAI_API_KEY", "k") };
        let e = OpenAIEmbedder::new("text-embedding-3-large").unwrap();
        assert_eq!(e.model_id(), "openai:text-embedding-3-large");
        assert_eq!(e.dim(), 3072);
    }
}
