//! Integration tests for GET /stats.

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

#[tokio::test]
async fn stats_after_ingest() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

    let addr = spawn_server(registry).await;
    let client = reqwest::Client::new();

    // Ingest 2 NDJSON records with severity (text) + cvss (numeric)
    let body = concat!(
        r#"{"id":"v1","body":"SQL injection vuln","severity":"HIGH","cvss":9.8}"#,
        "\n",
        r#"{"id":"v2","body":"buffer overflow","severity":"LOW","cvss":3.1}"#,
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
    assert_eq!(resp.status(), 200, "ingest should succeed");

    // GET /stats
    let resp = client
        .get(format!("http://{}/stats", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "stats should return 200");

    let json: serde_json::Value = resp.json().await.unwrap();

    assert_eq!(json["corpus"], "default");
    assert_eq!(json["entries"]["live"].as_u64().unwrap(), 2);
    assert_eq!(json["entries"]["tombstoned"].as_u64().unwrap(), 0);
    assert!(
        json["disk_bytes"].as_u64().unwrap() > 0,
        "disk_bytes should be > 0"
    );
    assert!(
        json["embedding"]["dimensions"].as_u64().unwrap() > 0,
        "embedding.dimensions should be > 0"
    );
    assert!(
        json["timestamps"]["created_unix"].as_u64().is_some(),
        "timestamps.created_unix should be present"
    );

    // Verify field stats
    let fields = json["fields"]
        .as_array()
        .expect("fields should be an array");

    let severity = fields
        .iter()
        .find(|f| f["name"] == "severity")
        .expect("should have severity field");
    assert_eq!(severity["type"], "text");
    assert!(
        severity["cardinality"].as_u64().unwrap() > 0,
        "severity cardinality should be > 0"
    );

    let cvss = fields
        .iter()
        .find(|f| f["name"] == "cvss")
        .expect("should have cvss field");
    assert_eq!(cvss["type"], "numeric");
    assert!(cvss["min"].as_f64().is_some(), "cvss.min should be present");
    assert!(cvss["max"].as_f64().is_some(), "cvss.max should be present");
}

#[tokio::test]
async fn stats_unknown_corpus_returns_404() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

    let addr = spawn_server(registry).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("http://{}/stats?corpus=nonexistent", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    let body = resp.text().await.unwrap();
    assert!(
        body.contains("corpus not found"),
        "expected 'corpus not found' in body, got: {body}"
    );
}

#[tokio::test]
async fn stats_reflects_delete() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

    let addr = spawn_server(registry).await;
    let client = reqwest::Client::new();

    // Ingest 2 records
    let body = concat!(
        r#"{"id":"v1","body":"SQL injection vuln"}"#,
        "\n",
        r#"{"id":"v2","body":"buffer overflow"}"#,
        "\n",
    );

    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Stats before delete
    let resp = client
        .get(format!("http://{}/stats", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["entries"]["live"].as_u64().unwrap(), 2);
    assert_eq!(json["entries"]["tombstoned"].as_u64().unwrap(), 0);

    // Delete one record
    let resp = client
        .delete(format!("http://{}/ingest/v1?corpus=default", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // Stats after delete
    let resp = client
        .get(format!("http://{}/stats", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["entries"]["live"].as_u64().unwrap(), 1);
    assert_eq!(json["entries"]["tombstoned"].as_u64().unwrap(), 1);
}

#[tokio::test]
async fn stats_uninitialized_corpus_returns_500() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());

    let addr = spawn_server(registry).await;
    let client = reqwest::Client::new();

    // No ingest — corpus dir has no manifest.json
    let resp = client
        .get(format!("http://{}/stats", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        500,
        "stats on uninitialized corpus should return 500"
    );
}
