use std::fs;
use std::sync::Arc;

use fastrag::ChunkingStrategy;
use fastrag::ops;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_embedder};
use fastrag_embed::test_utils::MockEmbedder;
use reqwest::Client;
use reqwest::StatusCode;
use tokio::net::TcpListener;

fn temp_corpus_dir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn sample_input_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("alpha.txt"),
        "ALPHA\n\nalpha beta gamma delta.",
    )
    .unwrap();
    fs::write(
        dir.path().join("beta.txt"),
        "BETA\n\nbeta gamma delta epsilon.",
    )
    .unwrap();
    dir
}

#[tokio::test]
async fn http_query_and_health_end_to_end() {
    let input = sample_input_dir();
    let corpus = temp_corpus_dir();
    let stats = ops::index_path(
        input.path(),
        corpus.path(),
        &ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        },
        &MockEmbedder,
    )
    .unwrap();
    assert_eq!(stats.chunk_count, 2);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn({
        let corpus_dir = corpus.path().to_path_buf();
        let embedder = Arc::new(MockEmbedder);
        async move {
            let _ = serve_http_with_embedder(
                corpus_dir,
                listener,
                embedder,
                None,
                false,
                HttpRerankerConfig::default(),
                100,
            )
            .await;
        }
    });

    let client = Client::new();
    let health = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let health_body: serde_json::Value = health.json().await.unwrap();
    assert_eq!(health_body["status"], "ok");

    let response = client
        .get(format!(
            "http://{addr}/query?q=alpha%20beta%20gamma%20delta.&top_k=2"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let hits: serde_json::Value = response.json().await.unwrap();
    let arr = hits.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    // VectorEntry has no text/path — just verify we got scored results.
    assert!(arr[0]["score"].as_f64().unwrap() > 0.0);
    assert!(arr[0]["score"].as_f64().unwrap() >= arr[1]["score"].as_f64().unwrap());

    let metrics = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(metrics.status(), StatusCode::OK);
    let body = metrics.text().await.unwrap();
    assert!(
        body.contains("fastrag_query_total"),
        "missing fastrag_query_total in:\n{body}"
    );
    assert!(
        body.contains("fastrag_query_duration_seconds"),
        "missing fastrag_query_duration_seconds in:\n{body}"
    );
    assert!(
        body.contains("fastrag_index_entries"),
        "missing fastrag_index_entries in:\n{body}"
    );
    let lines = prometheus_parse::Scrape::parse(body.lines().map(|l| Ok(l.to_string()))).unwrap();
    let total: f64 = lines
        .samples
        .iter()
        .find(|s| s.metric == "fastrag_query_total")
        .map(|s| match s.value {
            prometheus_parse::Value::Counter(v) => v,
            prometheus_parse::Value::Untyped(v) => v,
            _ => panic!("unexpected metric kind for fastrag_query_total"),
        })
        .expect("fastrag_query_total sample present");
    assert!(
        total >= 1.0,
        "expected at least one query recorded, got {total}"
    );

    server.abort();
}

async fn spawn_server_with_token(
    token: Option<String>,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    tempfile::TempDir,
    tempfile::TempDir,
) {
    let input = sample_input_dir();
    let corpus = temp_corpus_dir();
    ops::index_path(
        input.path(),
        corpus.path(),
        &ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        },
        &MockEmbedder,
    )
    .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn({
        let corpus_dir = corpus.path().to_path_buf();
        let embedder = Arc::new(MockEmbedder);
        let token = token.clone();
        async move {
            let _ = serve_http_with_embedder(
                corpus_dir,
                listener,
                embedder,
                token,
                false,
                HttpRerankerConfig::default(),
                100,
            )
            .await;
        }
    });
    (addr, server, input, corpus)
}

#[tokio::test]
async fn auth_rejects_missing_token() {
    let (addr, server, _input, _corpus) = spawn_server_with_token(Some("s3cret".into())).await;
    let client = Client::new();
    let resp = client
        .get(format!("http://{addr}/query?q=alpha&top_k=1"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // /health still open
    let health = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    server.abort();
}

#[tokio::test]
async fn auth_rejects_wrong_token() {
    let (addr, server, _input, _corpus) = spawn_server_with_token(Some("s3cret".into())).await;
    let client = Client::new();
    let resp = client
        .get(format!("http://{addr}/query?q=alpha&top_k=1"))
        .header("X-Fastrag-Token", "wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    // Length mismatch must also 401, not leak via early return.
    let resp2 = client
        .get(format!("http://{addr}/query?q=alpha&top_k=1"))
        .header("X-Fastrag-Token", "a")
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::UNAUTHORIZED);
    server.abort();
}

#[tokio::test]
async fn auth_accepts_correct_token_header_and_bearer() {
    let (addr, server, _input, _corpus) = spawn_server_with_token(Some("s3cret".into())).await;
    let client = Client::new();
    let ok1 = client
        .get(format!("http://{addr}/query?q=alpha%20beta&top_k=1"))
        .header("X-Fastrag-Token", "s3cret")
        .send()
        .await
        .unwrap();
    assert_eq!(ok1.status(), StatusCode::OK);

    let ok2 = client
        .get(format!("http://{addr}/query?q=alpha%20beta&top_k=1"))
        .header("Authorization", "Bearer s3cret")
        .send()
        .await
        .unwrap();
    assert_eq!(ok2.status(), StatusCode::OK);
    server.abort();
}

#[tokio::test]
async fn no_token_configured_accepts_anonymous() {
    let (addr, server, _input, _corpus) = spawn_server_with_token(None).await;
    let client = Client::new();
    let ok = client
        .get(format!("http://{addr}/query?q=alpha&top_k=1"))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), StatusCode::OK);
    server.abort();
}
