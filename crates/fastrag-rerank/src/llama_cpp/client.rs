//! HTTP client for llama-server's `/v1/rerank` endpoint.
//!
//! Request shape:
//! ```json
//! {"model": "<path>", "query": "...", "documents": ["...", "..."], "top_n": 50}
//! ```
//!
//! Response shape:
//! ```json
//! {"results": [{"index": 0, "relevance_score": 0.87}, ...]}
//! ```

use serde::Deserialize;
use serde_json::json;

use crate::RerankError;
use fastrag_embed::EmbedError;
use fastrag_embed::http::{build_client, ensure_success, send_with_retry};

/// A single result from the `/v1/rerank` endpoint.
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

/// HTTP client that talks to a llama-server `/v1/rerank` endpoint.
///
/// Uses `fastrag_embed::http` helpers (and its reqwest version) for HTTP
/// transport, retry, and error handling.
pub struct LlamaCppRerankClient {
    base_url: String,
    model_name: String,
    http: fastrag_embed::BlockingClient,
}

impl LlamaCppRerankClient {
    pub fn new(
        base_url: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Result<Self, RerankError> {
        Ok(Self {
            base_url: base_url.into(),
            model_name: model_name.into(),
            http: build_client().map_err(embed_to_rerank)?,
        })
    }

    pub fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> Result<Vec<RerankResult>, RerankError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/v1/rerank", self.base_url);
        let body = json!({
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        });

        let resp = send_with_retry(|| self.http.post(&url).json(&body)).map_err(embed_to_rerank)?;
        let resp = ensure_success(resp).map_err(embed_to_rerank)?;

        let parsed: RerankResp = resp
            .json()
            .map_err(|e| RerankError::Http(format!("parse response: {e}")))?;

        Ok(parsed
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                score: r.relevance_score,
            })
            .collect())
    }
}

fn embed_to_rerank(e: EmbedError) -> RerankError {
    RerankError::Http(e.to_string())
}

#[derive(Deserialize)]
struct RerankResp {
    results: Vec<RerankRespItem>,
}

#[derive(Deserialize)]
struct RerankRespItem {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn rerank_round_trip() {
        let rt = rt();
        let (uri, _guard) = rt.block_on(async {
            let server = MockServer::start().await;
            let body = json!({
                "results": [
                    { "index": 1, "relevance_score": 0.95 },
                    { "index": 0, "relevance_score": 0.42 },
                ]
            });
            Mock::given(method("POST"))
                .and(path("/v1/rerank"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body))
                .mount(&server)
                .await;
            (server.uri(), server)
        });

        let c = LlamaCppRerankClient::new(uri, "test-model").unwrap();
        let out = c.rerank("test query", &["doc a", "doc b"], 2).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 1);
        assert!((out[0].score - 0.95).abs() < 1e-6);
        assert_eq!(out[1].index, 0);
        assert!((out[1].score - 0.42).abs() < 1e-6);
    }

    #[test]
    fn server_500_returns_http_error() {
        let rt = rt();
        let (uri, _guard) = rt.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/rerank"))
                .respond_with(ResponseTemplate::new(500).set_body_string("rerank failed"))
                .mount(&server)
                .await;
            (server.uri(), server)
        });
        let c = LlamaCppRerankClient::new(uri, "test-model").unwrap();
        let err = c.rerank("query", &["a"], 1).unwrap_err();
        assert!(
            matches!(err, RerankError::Http(_)),
            "expected Http error, got {err:?}"
        );
    }

    #[test]
    fn connection_refused_is_http_error() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let c =
            LlamaCppRerankClient::new(format!("http://127.0.0.1:{port}"), "test-model").unwrap();
        let err = c.rerank("query", &["a"], 1).unwrap_err();
        assert!(
            matches!(err, RerankError::Http(_)),
            "expected Http error, got {err:?}"
        );
    }

    #[test]
    fn empty_documents_returns_empty() {
        let c = LlamaCppRerankClient::new("http://127.0.0.1:1", "test-model").unwrap();
        let out = c.rerank("query", &[], 0).unwrap();
        assert!(out.is_empty());
    }
}
