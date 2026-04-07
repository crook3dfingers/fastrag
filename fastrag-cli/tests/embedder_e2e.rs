#![cfg(feature = "retrieval")]

use assert_cmd::Command;
use serde_json::json;
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

async fn mount_openai(server: &MockServer, dim: usize) {
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [ { "embedding": vec![0.1_f32; dim] } ]
        })))
        .mount(server)
        .await;
}

#[test]
fn index_and_query_with_openai_backend() {
    let rt = rt();
    let (uri, _g) = rt.block_on(async {
        let s = MockServer::start().await;
        mount_openai(&s, 1536).await;
        (s.uri(), s)
    });

    let docs = tempdir().unwrap();
    std::fs::write(docs.path().join("a.txt"), "hello world").unwrap();
    let corpus = tempdir().unwrap();

    Command::cargo_bin("fastrag")
        .unwrap()
        .env("OPENAI_API_KEY", "test")
        .args([
            "index",
            docs.path().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--embedder",
            "openai",
            "--openai-base-url",
            &uri,
        ])
        .assert()
        .success();

    Command::cargo_bin("fastrag")
        .unwrap()
        .env("OPENAI_API_KEY", "test")
        .args([
            "query",
            "hello",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--openai-base-url",
            &uri,
        ])
        .assert()
        .success();

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(corpus.path().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest["embedding_model_id"].as_str().unwrap(),
        "openai:text-embedding-3-small"
    );
}

#[test]
fn query_with_mismatched_embedder_flag_fails() {
    let rt = rt();
    let (uri, _g) = rt.block_on(async {
        let s = MockServer::start().await;
        mount_openai(&s, 1536).await;
        (s.uri(), s)
    });

    let docs = tempdir().unwrap();
    std::fs::write(docs.path().join("a.txt"), "hello").unwrap();
    let corpus = tempdir().unwrap();

    Command::cargo_bin("fastrag")
        .unwrap()
        .env("OPENAI_API_KEY", "test")
        .args([
            "index",
            docs.path().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--embedder",
            "openai",
            "--openai-base-url",
            &uri,
        ])
        .assert()
        .success();

    let out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "hello",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--embedder",
            "bge",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("openai:text-embedding-3-small"),
        "stderr should mention existing model_id, got: {stderr}"
    );
}
