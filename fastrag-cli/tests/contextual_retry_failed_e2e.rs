//! End-to-end test for `--retry-failed` after injected transient failures.
//!
//! Ingests a fixture corpus with `FASTRAG_TEST_INJECT_FAILURES=2` so that
//! exactly two chunks land as `failed` rows in the SQLite cache. Re-runs
//! `index --retry-failed` against a healthy contextualizer (no injection)
//! and asserts that all five rows become `ok` and that the dense index has
//! been rebuilt so the repaired chunks are findable via query.
//!
//! Requires a real `llama-server` and both GGUFs (auto-downloaded). Gated
//! behind `FASTRAG_LLAMA_TEST=1` and `#[ignore]`.

#![cfg(all(feature = "contextual", feature = "contextual-llama"))]

mod support;

use std::path::PathBuf;

use assert_cmd::Command;
use tempfile::tempdir;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/contextual_corpus")
}

#[test]
#[ignore]
fn retry_failed_repairs_all_transient_failures() {
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

    let corpus = tempdir().unwrap();
    let cfg = tempdir().unwrap();
    let config_path = support::write_llama_cpp_config(cfg.path(), "qwen3", &model_path);

    // 1. Ingest with two injected failures.
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_dir().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--contextualize",
        ])
        .env("FASTRAG_TEST_INJECT_FAILURES", "2")
        .assert()
        .success();

    // 2. Confirm 3 ok / 2 failed via corpus-info.
    let out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "corpus-info",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8(out.stdout).unwrap();
    assert!(
        info.contains("ok:    3"),
        "expected 'ok:    3' after injected failures, got: {info}"
    );
    assert!(
        info.contains("failed: 2"),
        "expected 'failed: 2' after injected failures, got: {info}"
    );

    // 3. Run --retry-failed without the injection env var.
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_dir().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--contextualize",
            "--retry-failed",
        ])
        .env_remove("FASTRAG_TEST_INJECT_FAILURES")
        .assert()
        .success();

    // 4. All five rows should now be ok.
    let out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "corpus-info",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8(out.stdout).unwrap();
    assert!(
        info.contains("ok:    5"),
        "expected 'ok:    5' after retry, got: {info}"
    );
    assert!(
        info.contains("failed: 0"),
        "expected 'failed: 0' after retry, got: {info}"
    );

    // 5. Query the corpus and confirm the libfoo chunk is findable post-repair.
    let out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            "Is there an RCE in libfoo?",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--top-k",
            "1",
            "--no-rerank",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let result = String::from_utf8(out.stdout).unwrap();
    assert!(
        result.contains("01-libfoo-advisory.md"),
        "repaired corpus should find the libfoo advisory chunk, got: {result}"
    );
}
