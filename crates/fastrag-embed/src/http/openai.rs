//! OpenAI embedder backend.
//!
//! Blocking HTTP client. Static dim table — no silent probing. Supports
//! OpenAI's native batch input in one request.

use std::env;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::http::{build_client, ensure_success, send_with_retry};
use crate::{EmbedError, Embedder, PassageText, PrefixScheme, QueryText};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug)]
pub struct OpenAiEmbedder<const DIM: usize> {
    base_url: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

pub type OpenAiSmall = OpenAiEmbedder<1536>;
pub type OpenAiLarge = OpenAiEmbedder<3072>;

impl<const DIM: usize> OpenAiEmbedder<DIM> {
    pub fn new() -> Result<Self, EmbedError> {
        let api_key =
            env::var("OPENAI_API_KEY").map_err(|_| EmbedError::MissingEnv("OPENAI_API_KEY"))?;
        Ok(Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key,
            client: build_client()?,
        })
    }

    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    fn model_name(model_id: &'static str) -> &'static str {
        model_id.strip_prefix("openai:").unwrap_or(model_id)
    }

    fn call(&self, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url);
        let body = json!({ "model": model, "input": texts });
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
            && first.len() != DIM
        {
            return Err(EmbedError::UnexpectedDim {
                expected: DIM,
                got: first.len(),
            });
        }
        Ok(vecs)
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

impl Embedder for OpenAiSmall {
    const DIM: usize = 1536;
    const MODEL_ID: &'static str = "openai:text-embedding-3-small";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(QueryText::as_str).collect();
        self.call(Self::model_name(Self::MODEL_ID), &refs)
    }

    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(PassageText::as_str).collect();
        self.call(Self::model_name(Self::MODEL_ID), &refs)
    }

    fn default_batch_size(&self) -> usize {
        512
    }
}

impl Embedder for OpenAiLarge {
    const DIM: usize = 3072;
    const MODEL_ID: &'static str = "openai:text-embedding-3-large";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(QueryText::as_str).collect();
        self.call(Self::model_name(Self::MODEL_ID), &refs)
    }

    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(PassageText::as_str).collect();
        self.call(Self::model_name(Self::MODEL_ID), &refs)
    }

    fn default_batch_size(&self) -> usize {
        512
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

    fn make_small(base: &str) -> OpenAiSmall {
        unsafe { std::env::set_var("OPENAI_API_KEY", "test-key") };
        OpenAiSmall::new().unwrap().with_base_url(base.to_string())
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
        let e = make_small(&server_uri);
        let out = e
            .embed_query(&[QueryText::new("a"), QueryText::new("b")])
            .unwrap();
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
        let e = make_small(&server_uri);
        let err = e.embed_query(&[QueryText::new("a")]).unwrap_err();
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
        let e = make_small(&server_uri);
        let err = e
            .embed_query(&[QueryText::new("a"), QueryText::new("b")])
            .unwrap_err();
        assert!(matches!(
            err,
            EmbedError::UnexpectedDim {
                expected: 2,
                got: 1
            }
        ));
    }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;
    use crate::Embedder;

    #[test]
    fn openai_small_consts() {
        assert_eq!(OpenAiSmall::DIM, 1536);
        assert_eq!(OpenAiSmall::MODEL_ID, "openai:text-embedding-3-small");
    }

    #[test]
    fn openai_large_consts() {
        assert_eq!(OpenAiLarge::DIM, 3072);
        assert_eq!(OpenAiLarge::MODEL_ID, "openai:text-embedding-3-large");
    }
}
