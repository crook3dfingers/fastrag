//! End-to-end tests for POST /similar.
#![cfg(feature = "retrieval")]

use std::collections::BTreeMap;
use std::sync::Arc;

use fastrag::ChunkingStrategy;
use fastrag::corpus::CorpusRegistry;
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::json;

fn build_toy_corpus(docs: &[(&str, &str)]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let jsonl = tmp.path().join("docs.jsonl");
    let lines: Vec<String> = docs
        .iter()
        .map(|(id, body)| json!({"id": id, "body": body}).to_string())
        .collect();
    std::fs::write(&jsonl, lines.join("\n")).unwrap();
    let corpus = tmp.path().join("corpus");
    let cfg = JsonlIngestConfig {
        text_fields: vec!["body".into()],
        id_field: "id".into(),
        metadata_fields: vec![],
        metadata_types: BTreeMap::new(),
        array_fields: vec![],
        cwe_field: None,
    };
    index_jsonl(
        &jsonl,
        &corpus,
        &ChunkingStrategy::Basic {
            max_characters: 500,
            overlap: 0,
        },
        &MockEmbedder as &dyn fastrag::DynEmbedderTrait,
        &cfg,
    )
    .unwrap();
    // Return a TempDir pointing at the corpus subdir by shifting: the corpus
    // lives inside tmp. Return the outer handle so tmp survives.
    let holder = tempfile::tempdir().unwrap();
    // Move the corpus into holder so the TempDir drop guards it.
    let dest = holder.path().join("corpus");
    std::fs::rename(&corpus, &dest).unwrap();
    // tmp is dropped (cleans the jsonl); holder survives with the corpus.
    holder
}

async fn spawn_server(registry: CorpusRegistry) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
            52_428_800,
            10_000,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    format!("http://{}", addr)
}

#[tokio::test]
async fn post_similar_happy_path() {
    let corpus = build_toy_corpus(&[("a", "alpha"), ("b", "xyzzy plover"), ("c", "quux frob")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let client = Client::new();
    let resp = client
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.95,
            "max_results": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["cosine_similarity"].as_f64().unwrap() >= 0.95);
    assert_eq!(hits[0]["corpus"].as_str().unwrap(), "default");
    assert!(!body["truncated"].as_bool().unwrap());
    assert_eq!(body["stats"]["returned"].as_u64().unwrap(), 1);
    assert!(body["latency"]["embed_us"].as_u64().is_some());
}

// --- Task 6: HTTP validation paths ---

#[tokio::test]
async fn post_similar_rejects_hybrid_params() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "hybrid": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("/query"), "error should point to /query: {body}");
}

#[tokio::test]
async fn post_similar_rejects_rerank_param() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "rerank": "on"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("reranking"));
}

#[tokio::test]
async fn post_similar_rejects_both_corpus_and_corpora() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("one", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpus": "one",
            "corpora": ["one"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("exactly one"));
}

#[tokio::test]
async fn post_similar_rejects_empty_corpora() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpora": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("corpora") && body.contains("non-empty"));
}

#[tokio::test]
async fn post_similar_rejects_threshold_out_of_range() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 1.5,
            "max_results": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("threshold"));
}

#[tokio::test]
async fn post_similar_rejects_max_results_out_of_range() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_similar_rejects_empty_text() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "",
            "threshold": 0.5,
            "max_results": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("text"));
}

#[tokio::test]
async fn post_similar_corpus_not_found() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpus": "missing"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = resp.text().await.unwrap();
    assert!(body.contains("missing"));
}

// --- Task 7: Multi-corpus fan-out, tenant filter, truncation ---

#[tokio::test]
async fn post_similar_fan_out_merges_across_corpora() {
    let c1 = build_toy_corpus(&[("a1", "alpha"), ("b1", "zzz1")]);
    let c2 = build_toy_corpus(&[("a2", "alpha"), ("b2", "zzz2")]);
    let registry = CorpusRegistry::new();
    registry.register("one", c1.path().join("corpus"));
    registry.register("two", c2.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.95,
            "max_results": 10,
            "corpora": ["one", "two"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    let corpora: std::collections::BTreeSet<&str> = hits
        .iter()
        .map(|h| h["corpus"].as_str().unwrap())
        .collect();
    assert!(corpora.contains("one"));
    assert!(corpora.contains("two"));
    assert!(body["stats"]["per_corpus"]["one"].is_object());
    assert!(body["stats"]["per_corpus"]["two"].is_object());
}

#[cfg(feature = "store")]
#[tokio::test]
async fn post_similar_tenant_filter_applied() {
    use fastrag_store::schema::TypedKind;

    // Seed two tenants into a single corpus.
    let tmp = tempfile::tempdir().unwrap();
    let jsonl = tmp.path().join("docs.jsonl");
    std::fs::write(
        &jsonl,
        concat!(
            r#"{"id":"a","body":"alpha","tenant":"acme"}"#,
            "\n",
            r#"{"id":"b","body":"alpha","tenant":"widgetco"}"#,
        ),
    )
    .unwrap();
    let corpus = tmp.path().join("corpus");
    let cfg = fastrag::ingest::jsonl::JsonlIngestConfig {
        text_fields: vec!["body".into()],
        id_field: "id".into(),
        metadata_fields: vec!["tenant".into()],
        metadata_types: BTreeMap::from([("tenant".into(), TypedKind::String)]),
        array_fields: vec![],
        cwe_field: None,
    };
    fastrag::ingest::engine::index_jsonl(
        &jsonl,
        &corpus,
        &ChunkingStrategy::Basic {
            max_characters: 500,
            overlap: 0,
        },
        &MockEmbedder as &dyn fastrag::DynEmbedderTrait,
        &cfg,
    )
    .unwrap();

    // Spawn server with tenant_field = "tenant".
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.clone());
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            Some("tenant".into()),
            52_428_800,
            10_000,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://{addr}/similar"))
        .header("x-fastrag-tenant", "acme")
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "tenant filter should limit to acme");
    let tenant = hits[0]["source"]["tenant"]
        .as_str()
        .or_else(|| hits[0]["metadata"]["tenant"].as_str());
    assert_eq!(tenant, Some("acme"));
}

#[tokio::test]
async fn post_similar_truncated_flag() {
    // 20 matching docs, tiny overfetch cap -> truncated=true.
    let docs: Vec<(String, String)> =
        (0..20).map(|i| (format!("d{i}"), "alpha".to_string())).collect();
    let docs_ref: Vec<(&str, &str)> =
        docs.iter().map(|(i, b)| (i.as_str(), b.as_str())).collect();
    let corpus = build_toy_corpus(&docs_ref);

    // Spawn with overfetch_cap = 5.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
            52_428_800,
            5, // similar_overfetch_cap
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let resp = Client::new()
        .post(format!("http://{addr}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 100
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["truncated"].as_bool().unwrap());
    let hits = body["hits"].as_array().unwrap();
    assert!(hits.len() <= 5);
}
