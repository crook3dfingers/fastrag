//! HTTP e2e: POST /query with `temporal_policy` + `date_fields` in the body.
#![cfg(all(feature = "retrieval", feature = "store"))]

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use chrono::{Duration, Utc};
use fastrag::ChunkingStrategy;
use fastrag::corpus::CorpusRegistry;
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::DynEmbedderTrait;
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_store::schema::TypedKind;

async fn spawn(registry: CorpusRegistry) -> std::net::SocketAddr {
    let embedder: fastrag::DynEmbedder = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,  // no token
            false, // dense_only
            false, // cwe_expand_default
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
    addr
}

fn build_dated_corpus(path: &std::path::Path) {
    let fresh = (Utc::now() - Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();
    let jsonl = path.join("docs.jsonl");
    std::fs::write(
        &jsonl,
        format!(
            r#"{{"id":"STALE","text":"log4j remote code execution vulnerability","published_date":"2016-01-01"}}
{{"id":"FRESH","text":"log4j remote code execution vulnerability","published_date":"{fresh}"}}"#
        ),
    )
    .unwrap();

    let corpus = path.join("corpus");
    let embedder = MockEmbedder;
    let cfg = JsonlIngestConfig {
        text_fields: vec!["text".into()],
        id_field: "id".into(),
        metadata_fields: vec!["published_date".into()],
        metadata_types: BTreeMap::from([("published_date".into(), TypedKind::Date)]),
        array_fields: vec![],
        cwe_field: None,
    };
    let chunking = ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    };
    index_jsonl(
        &jsonl,
        &corpus,
        &chunking,
        &embedder as &dyn DynEmbedderTrait,
        &cfg,
    )
    .unwrap();
}

fn hit_ids(hits: &[serde_json::Value]) -> HashSet<String> {
    hits.iter()
        .filter_map(|h| {
            h.get("source")
                .and_then(|s| s.get("id"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect()
}

fn top_id(hits: &[serde_json::Value]) -> Option<String> {
    hits.first()
        .and_then(|h| h.get("source"))
        .and_then(|s| s.get("id"))
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// POST /query with `temporal_policy: {favor_recent: "medium"}` promotes fresh docs.
#[tokio::test]
async fn post_query_with_temporal_policy_favor_recent_medium() {
    let tmp = tempfile::tempdir().unwrap();
    build_dated_corpus(tmp.path());

    let registry = CorpusRegistry::new();
    registry.register("default", tmp.path().join("corpus"));
    let addr = spawn(registry).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "q": "latest log4j advisory",
        "top_k": 2,
        "date_fields": ["published_date"],
        "temporal_policy": {"favor_recent": "medium"}
    });
    let resp = client
        .post(format!("http://{addr}/query"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "expected 200");
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(hits.len() >= 2, "expected >=2 hits, got {}", hits.len());
    let ids = hit_ids(&hits);
    assert!(ids.contains("FRESH"), "missing FRESH: {ids:?}");
    assert!(ids.contains("STALE"), "missing STALE: {ids:?}");
    assert_eq!(
        top_id(&hits).as_deref(),
        Some("FRESH"),
        "decay should promote FRESH to top; hits={hits:#?}"
    );
}

/// POST /query with `temporal_policy: "off"` returns both docs without decay ordering enforced.
#[tokio::test]
async fn post_query_with_temporal_policy_off() {
    let tmp = tempfile::tempdir().unwrap();
    build_dated_corpus(tmp.path());

    let registry = CorpusRegistry::new();
    registry.register("default", tmp.path().join("corpus"));
    let addr = spawn(registry).await;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "q": "log4j remote code execution",
        "top_k": 2,
        "date_fields": ["published_date"],
        "temporal_policy": "off"
    });
    let resp = client
        .post(format!("http://{addr}/query"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "expected 200 for off policy");
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    // Both docs exist; off policy means no decay — just verify the request succeeds
    // and returns results (ordering is non-deterministic without decay).
    assert!(
        !hits.is_empty(),
        "expected results with off policy; hits={hits:#?}"
    );
    let ids = hit_ids(&hits);
    assert!(
        ids.contains("FRESH") || ids.contains("STALE"),
        "expected at least one known doc; ids={ids:?}"
    );
}

/// Serde round-trip: verify body struct deserializes both JSON shapes correctly.
#[test]
fn temporal_policy_serde_shapes() {
    use fastrag::corpus::temporal::{Strength, TemporalPolicy};

    let auto: TemporalPolicy = serde_json::from_str(r#""auto""#).unwrap();
    assert_eq!(auto, TemporalPolicy::Auto);

    let off: TemporalPolicy = serde_json::from_str(r#""off""#).unwrap();
    assert_eq!(off, TemporalPolicy::Off);

    let favor: TemporalPolicy = serde_json::from_str(r#"{"favor_recent":"medium"}"#).unwrap();
    assert_eq!(favor, TemporalPolicy::FavorRecent(Strength::Medium));
}
