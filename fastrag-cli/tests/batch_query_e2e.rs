//! Integration tests for POST /batch-query.

use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_embedder};
use fastrag_embed::test_utils::MockEmbedder;
use std::sync::Arc;

fn toy_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let doc_path = dir.path().join("doc.txt");
    std::fs::write(&doc_path, "SQL injection vulnerability in login form").unwrap();
    fastrag::corpus::index_path(
        &doc_path,
        dir.path(),
        &fastrag::ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        },
        &MockEmbedder,
    )
    .unwrap();
    dir
}

async fn start_server(
    corpus_dir: std::path::PathBuf,
    token: Option<String>,
) -> std::net::SocketAddr {
    let e = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_http_with_embedder(
            corpus_dir,
            listener,
            e,
            token,
            false,
            HttpRerankerConfig::default(),
            100,
        )
        .await
        .unwrap();
    });
    addr
}

#[tokio::test]
async fn batch_query_returns_results_for_valid_queries() {
    let dir = toy_corpus();
    let addr = start_server(dir.path().to_path_buf(), None).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "queries": [
            {"q": "SQL injection", "top_k": 3},
            {"q": "memory corruption", "top_k": 3}
        ]
    });

    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "should have one result per query");
    assert!(
        results[0]["hits"].is_array(),
        "query 0 should have hits array"
    );
    assert!(
        results[0].get("error").is_none(),
        "query 0 should not have error"
    );
    assert!(
        results[1]["hits"].is_array(),
        "query 1 should have hits array"
    );
}

#[tokio::test]
async fn batch_query_partial_failure_bad_filter() {
    let dir = toy_corpus();
    let addr = start_server(dir.path().to_path_buf(), None).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "queries": [
            {"q": "SQL injection", "top_k": 3},
            {"q": "bad filter query", "top_k": 3, "filter": "INVALID FILTER !!!"},
            {"q": "memory corruption", "top_k": 3}
        ]
    });

    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200, "partial failure returns HTTP 200");
    let json: serde_json::Value = resp.json().await.unwrap();
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    // index 0: success
    assert!(results[0]["hits"].is_array());
    assert!(results[0].get("error").is_none());

    // index 1: bad filter -> error
    assert!(
        results[1]["error"].is_string(),
        "bad filter should produce error"
    );
    assert!(results[1].get("hits").is_none());

    // index 2: success
    assert!(results[2]["hits"].is_array());
}

#[tokio::test]
async fn batch_query_rejects_over_limit() {
    let dir = toy_corpus();
    let addr = start_server(dir.path().to_path_buf(), None).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let queries: Vec<serde_json::Value> = (0..101)
        .map(|i| serde_json::json!({"q": format!("query {i}"), "top_k": 1}))
        .collect();
    let body = serde_json::json!({ "queries": queries });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn batch_query_requires_auth_when_token_set() {
    let dir = toy_corpus();
    let addr = start_server(dir.path().to_path_buf(), Some("secret".to_string())).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({"queries": [{"q": "test", "top_k": 1}]});

    // No token -> 401
    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // With token -> 200
    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .header("x-fastrag-token", "secret")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}
