//! End-to-end reranking test with the llama-cpp backend (bge-reranker-v2-m3).
//!
//! Requires a real `llama-server` in PATH and the bge-reranker GGUF model
//! (auto-downloaded on first run). Gated behind `FASTRAG_LLAMA_TEST=1`
//! and `#[ignore]` so `cargo test --workspace` skips it.

#![cfg(feature = "rerank")]

use assert_cmd::Command;
use tempfile::tempdir;

/// Index two documents, query with `--rerank=llama-cpp`, verify reranked output.
#[test]
#[ignore]
fn llama_cpp_rerank_query_round_trip() {
    if std::env::var("FASTRAG_LLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_LLAMA_TEST=1 to run");
        return;
    }

    let input = tempdir().unwrap();
    std::fs::write(
        input.path().join("capital.txt"),
        "The capital of France is Paris, known for the Eiffel Tower.",
    )
    .unwrap();
    std::fs::write(
        input.path().join("rust.txt"),
        "Rust is a systems programming language focused on safety and performance.",
    )
    .unwrap();

    let corpus = tempdir().unwrap();

    // Index
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            input.path().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Query with llama-cpp reranking
    let reranked = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "What is the capital of France?",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--top-k",
            "2",
            "--format",
            "json",
            "--rerank=llama-cpp",
        ])
        .output()
        .expect("failed to run reranked query");

    assert!(
        reranked.status.success(),
        "reranked query failed: {}",
        String::from_utf8_lossy(&reranked.stderr)
    );

    let hits: Vec<serde_json::Value> =
        serde_json::from_slice(&reranked.stdout).expect("parse JSON output");
    assert_eq!(hits.len(), 2);

    // The capital-related document should be ranked higher than the Rust doc
    let first_text = hits[0]["chunk_text"].as_str().unwrap();
    assert!(
        first_text.contains("capital") || first_text.contains("Paris"),
        "expected capital/Paris hit first, got: {first_text}"
    );
}
