//! Integration tests for multi-corpus federation.

use fastrag::corpus::CorpusRegistry;
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Build a Store-backed toy corpus via JSONL ingest.
///
/// The single document has NO `engagement_id` metadata field, so a tenant
/// filter `engagement_id = <anything>` returns 0 hits — exercising the
/// semantic filter path that `query_corpus_with_filter` applies only when a
/// `schema.json` (Store) is present.
fn toy_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let jsonl_path = dir.path().join("docs.jsonl");
    // One record: id + body only, no engagement_id.
    std::fs::write(
        &jsonl_path,
        r#"{"id":"doc-1","body":"SQL injection vulnerability"}"#,
    )
    .unwrap();

    let corpus_dir = tempfile::tempdir().unwrap();
    let config = JsonlIngestConfig {
        text_fields: vec!["body".to_string()],
        id_field: "id".to_string(),
        metadata_fields: vec![],
        metadata_types: BTreeMap::new(),
        array_fields: vec![],
    };
    index_jsonl(
        &jsonl_path,
        corpus_dir.path(),
        &fastrag::ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        },
        &MockEmbedder,
        &config,
    )
    .unwrap();
    corpus_dir
}

#[tokio::test]
async fn named_corpus_query() {
    let dir = toy_corpus();
    let e = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = CorpusRegistry::new();
    registry.register("docs", dir.path().to_path_buf());

    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // Query named corpus -> 200
    let resp = client
        .get(format!("http://{}/query?q=SQL&corpus=docs&top_k=3", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.is_array(), "response should be a hits array");

    // Query unknown corpus -> 404
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL&corpus=unknown&top_k=3",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn get_corpora_lists_registry() {
    let dir = toy_corpus();
    let e = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = CorpusRegistry::new();
    registry.register("nvd", dir.path().to_path_buf());

    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/corpora", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let corpora = json["corpora"].as_array().unwrap();
    assert_eq!(corpora.len(), 1);
    assert_eq!(corpora[0]["name"].as_str().unwrap(), "nvd");
    assert_eq!(corpora[0]["status"].as_str().unwrap(), "unloaded");
    assert!(
        !corpora[0]["path"].as_str().unwrap_or("").is_empty(),
        "path should be non-empty"
    );
}

#[tokio::test]
async fn default_corpus_used_when_no_corpus_param() {
    let dir = toy_corpus();
    let e = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = CorpusRegistry::new();
    registry.register("default", dir.path().to_path_buf());

    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    // No corpus= param -> uses "default"
    let resp = client
        .get(format!("http://{}/query?q=SQL&top_k=3", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.is_array(), "response should be a hits array");
}

#[tokio::test]
async fn tenant_enforcement_rejects_missing_header() {
    let dir = toy_corpus();
    let e = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = CorpusRegistry::new();
    registry.register("default", dir.path().to_path_buf());

    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            Some("engagement_id".to_string()), // tenant enforcement ON
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // No tenant header -> 401.
    let resp = client
        .get(format!("http://{}/query?q=SQL&top_k=1", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // With tenant header -> 200.
    let resp = client
        .get(format!("http://{}/query?q=SQL&top_k=1", addr))
        .header("x-fastrag-tenant", "engagement-abc")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: serde_json::Value = resp.json().await.unwrap();
    let hits_arr = hits.as_array().expect("response should be an array");
    // Tenant filter `engagement_id = engagement-abc` is applied; the toy corpus
    // has no engagement_id metadata, so the filter returns 0 hits.
    // If the filter were not applied, the "SQL injection vulnerability" doc would appear.
    assert_eq!(
        hits_arr.len(),
        0,
        "tenant filter should exclude all toy corpus docs"
    );
}
