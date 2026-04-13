//! Integration tests for snippet generation and field selection.

use std::sync::Arc;

use fastrag::corpus::CorpusRegistry;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;

async fn spawn_server(registry: CorpusRegistry) -> std::net::SocketAddr {
    let embedder: fastrag::DynEmbedder = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
            52_428_800,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    addr
}

async fn ingest_test_records(addr: std::net::SocketAddr) {
    let client = reqwest::Client::new();
    let body = concat!(
        r#"{"id":"v1","body":"SQL injection vulnerability allows remote code execution","severity":"HIGH","cvss":9.8}"#,
        "\n",
        r#"{"id":"v2","body":"Buffer overflow in kernel network stack","severity":"CRITICAL","cvss":9.1}"#,
        "\n",
    );
    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body&metadata_fields=severity,cvss&metadata_types=cvss=numeric",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn source_exposed_in_response() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/query?q=SQL+injection&top_k=3", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty(), "expected hits");
    let source = &hits[0]["source"];
    assert!(source.is_object(), "source should be an object: {source}");
    assert!(source["id"].is_string(), "source should have id field");
}

#[tokio::test]
async fn snippet_present_in_response() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&snippet_len=200",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    let snippet = &hits[0]["snippet"];
    assert!(snippet.is_string(), "snippet should be a string: {snippet}");
    assert!(
        !snippet.as_str().unwrap().is_empty(),
        "snippet should not be empty"
    );
}

#[tokio::test]
async fn snippet_disabled_when_zero() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&snippet_len=0",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits[0].get("snippet").is_none(),
        "snippet should be absent when snippet_len=0"
    );
}

#[tokio::test]
async fn field_selection_include() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&fields=score,snippet",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    let obj = hits[0].as_object().unwrap();
    assert!(obj.contains_key("score"), "score should be present");
    assert!(
        !obj.contains_key("chunk_text"),
        "chunk_text should be excluded by field selection"
    );
    assert!(
        !obj.contains_key("source_path"),
        "source_path should be excluded"
    );
}

#[tokio::test]
async fn field_selection_exclude() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&fields=-chunk_text,-source",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    let obj = hits[0].as_object().unwrap();
    assert!(
        !obj.contains_key("chunk_text"),
        "chunk_text should be excluded"
    );
    assert!(!obj.contains_key("source"), "source should be excluded");
    assert!(obj.contains_key("score"), "score should remain");
}
