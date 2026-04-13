//! Integration tests for POST /ingest.

use std::sync::Arc;

use fastrag::corpus::CorpusRegistry;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;

#[tokio::test]
async fn ingest_creates_queryable_records() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

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

    let client = reqwest::Client::new();

    // POST two NDJSON records
    let body = concat!(
        r#"{"id":"v1","body":"SQL injection vuln","severity":"HIGH"}"#,
        "\n",
        r#"{"id":"v2","body":"buffer overflow in kernel","severity":"CRITICAL"}"#,
        "\n",
    );

    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body&metadata_fields=severity",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let first_json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        first_json["records_new"].as_u64().unwrap(),
        2,
        "first ingest should create 2 new records: {first_json}"
    );

    // Re-send identical payload — should be fully deduplicated
    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body&metadata_fields=severity",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let second_json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        second_json["records_unchanged"].as_u64().unwrap(),
        2,
        "second ingest should show 2 unchanged records: {second_json}"
    );
    assert_eq!(
        second_json["records_new"].as_u64().unwrap(),
        0,
        "second ingest should show 0 new records: {second_json}"
    );

    // Query the corpus — should find at least one hit for "SQL injection"
    let resp = client
        .get(format!("http://{}/query?q=SQL+injection&top_k=3", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(
        !hits.is_empty(),
        "expected at least 1 hit for SQL injection"
    );
}

#[tokio::test]
async fn ingest_rejects_unknown_corpus() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("mydata", corpus_dir.path().to_path_buf());

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

    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "http://{}/ingest?corpus=nonexistent&id_field=id&text_fields=body",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(r#"{"id":"v1","body":"test"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn ingest_rejects_oversized_body() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

    let embedder: fastrag::DynEmbedder = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Set a tiny max body (256 bytes)
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
            256,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let big_body = "x".repeat(512);
    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(big_body)
        .send()
        .await
        .unwrap();
    // Should be 413 Payload Too Large (axum layer) or our handler check
    assert!(
        resp.status() == 413 || resp.status() == 400,
        "expected 413 or 400, got {}",
        resp.status()
    );
}
