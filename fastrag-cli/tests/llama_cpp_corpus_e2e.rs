//! End-to-end corpus round-trip test with the Qwen3 llama-cpp backend.
//!
//! Requires a real `llama-server` in PATH and the Qwen3-Embedding-0.6B GGUF
//! model (auto-downloaded on first run). Gated behind `FASTRAG_LLAMA_TEST=1`
//! and `#[ignore]` so `cargo test --workspace` skips it.

#![cfg(feature = "retrieval")]

mod support;

use assert_cmd::Command;
use tempfile::tempdir;

/// Index two tiny documents with `--backend qwen3-q8`, then query and verify
/// the top-1 hit is semantically correct + manifest fields are right.
#[test]
#[ignore]
fn qwen3_index_query_round_trip() {
    if std::env::var("FASTRAG_LLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_LLAMA_TEST=1 to run");
        return;
    }
    let Some(model_path) = support::llama_cpp_embed_model_path() else {
        eprintln!(
            "skipping: set FASTRAG_LLAMA_EMBED_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf"
        );
        return;
    };

    let input = tempdir().unwrap();
    std::fs::write(
        input.path().join("rust.txt"),
        "Rust is a systems programming language focused on safety and performance",
    )
    .unwrap();
    std::fs::write(
        input.path().join("python.txt"),
        "Python is an interpreted language popular for data science and scripting",
    )
    .unwrap();

    let corpus = tempdir().unwrap();
    let cfg = tempdir().unwrap();
    let config_path = support::write_llama_cpp_config(cfg.path(), "qwen3", &model_path);

    // --- Index ---
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            input.path().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // --- Verify manifest ---
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(corpus.path().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["version"].as_u64().unwrap(), 5);
    assert_eq!(
        manifest["identity"]["model_id"].as_str().unwrap(),
        format!("llama-cpp:{}", model_path.display())
    );
    assert_eq!(manifest["identity"]["dim"].as_u64().unwrap(), 1024);
    assert!(
        manifest["chunk_count"].as_u64().unwrap() >= 2,
        "expected at least 2 chunks"
    );

    // --- Query: systems programming → expect rust.txt top-1 ---
    let output = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "systems programming memory safety",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--top-k",
            "2",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "query exited non-zero");

    let hits: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert!(!hits.is_empty(), "expected at least one hit");
    let top_path = hits[0]["source_path"].as_str().unwrap();
    assert!(
        top_path.contains("rust"),
        "expected top-1 to be rust.txt, got: {top_path}"
    );

    // --- Query: data science → expect python.txt top-1 ---
    let output = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "data science machine learning",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--top-k",
            "2",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "query exited non-zero");

    let hits: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert!(!hits.is_empty(), "expected at least one hit");
    let top_path = hits[0]["source_path"].as_str().unwrap();
    assert!(
        top_path.contains("python"),
        "expected top-1 to be python.txt, got: {top_path}"
    );
}
