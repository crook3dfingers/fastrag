//! End-to-end reranking test with the ONNX backend (gte-reranker-modernbert-base).
//!
//! Requires the ONNX model files (auto-downloaded on first run).
//! Gated behind `FASTRAG_RERANK_TEST=1` and `#[ignore]` so `cargo test --workspace`
//! skips it.

#![cfg(feature = "rerank")]

use assert_cmd::Command;
use tempfile::tempdir;

/// Index two documents, query with `--rerank=onnx`, then query with `--no-rerank`,
/// and verify the reranked scores are in [0, 1] and ordering may differ.
#[test]
#[ignore]
fn onnx_rerank_query_round_trip() {
    if std::env::var("FASTRAG_RERANK_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_RERANK_TEST=1 to run");
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
    std::fs::write(
        input.path().join("paris.txt"),
        "Paris is a beautiful city in France with many museums.",
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

    // Query with reranking
    let reranked = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "What is the capital of France?",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--top-k",
            "3",
            "--format",
            "json",
            "--rerank=onnx",
        ])
        .output()
        .expect("failed to run reranked query");

    assert!(
        reranked.status.success(),
        "reranked query failed: {}",
        String::from_utf8_lossy(&reranked.stderr)
    );

    let reranked_hits: Vec<serde_json::Value> =
        serde_json::from_slice(&reranked.stdout).expect("parse reranked JSON output");
    assert_eq!(reranked_hits.len(), 3);

    // All reranked scores should be in [0, 1] (sigmoid output)
    for hit in &reranked_hits {
        let score = hit["score"].as_f64().unwrap();
        assert!(
            (0.0..=1.0).contains(&score),
            "reranked score {score} out of [0, 1]"
        );
    }

    // Query without reranking
    let plain = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "What is the capital of France?",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--top-k",
            "3",
            "--format",
            "json",
            "--no-rerank",
        ])
        .output()
        .expect("failed to run plain query");

    assert!(
        plain.status.success(),
        "plain query failed: {}",
        String::from_utf8_lossy(&plain.stderr)
    );

    let plain_hits: Vec<serde_json::Value> =
        serde_json::from_slice(&plain.stdout).expect("parse plain JSON output");
    assert_eq!(plain_hits.len(), 3);

    // Verify the two result sets have different scores (reranker uses sigmoid,
    // HNSW uses cosine similarity — the scales differ).
    let reranked_scores: Vec<f64> = reranked_hits
        .iter()
        .map(|h| h["score"].as_f64().unwrap())
        .collect();
    let plain_scores: Vec<f64> = plain_hits
        .iter()
        .map(|h| h["score"].as_f64().unwrap())
        .collect();
    assert_ne!(
        reranked_scores, plain_scores,
        "reranked and plain scores should differ"
    );
}
