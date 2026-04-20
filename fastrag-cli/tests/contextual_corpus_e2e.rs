//! End-to-end test for Contextual Retrieval over a fixture corpus.
//!
//! Indexes the same fixture corpus twice — once with `--contextualize` and
//! once without — then issues a query that requires the contextualized
//! prefix to match. Without contextualization the target chunk has no
//! lexical or semantic link to the query terms; with contextualization the
//! generated prefix supplies the missing keywords (the doc title carries
//! "libfoo" and "RCE", the chunk body only says "the vulnerability").
//!
//! Requires a real `llama-server` in PATH and both the embedder GGUF and
//! the completion GGUF (auto-downloaded on first run). Gated behind
//! `FASTRAG_LLAMA_TEST=1` and `#[ignore]` so `cargo test --workspace` skips
//! it.

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
fn contextualization_enables_pronoun_resolution() {
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

    let raw_corpus = tempdir().unwrap();
    let ctx_corpus = tempdir().unwrap();
    let cfg = tempdir().unwrap();
    let config_path = support::write_llama_cpp_config(cfg.path(), "qwen3", &model_path);

    // 1. Index without contextualization.
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_dir().to_str().unwrap(),
            "--corpus",
            raw_corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // 2. Index with contextualization.
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_dir().to_str().unwrap(),
            "--corpus",
            ctx_corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--contextualize",
        ])
        .assert()
        .success();

    // 3. corpus-info on the ctx corpus reports contextualized state.
    let info_out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "corpus-info",
            "--corpus",
            ctx_corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(info_out.status.success());
    let info = String::from_utf8(info_out.stdout).unwrap();
    assert!(
        info.contains("contextualized: true"),
        "expected 'contextualized: true' in stdout, got: {info}"
    );
    assert!(
        info.contains("ok:    5"),
        "expected 'ok:    5' (5 contextualized chunks) in stdout, got: {info}"
    );
    assert!(
        info.contains("failed: 0"),
        "expected 'failed: 0' in stdout, got: {info}"
    );

    // 4. Query that requires context to find the libfoo RCE chunk. The
    //    chunk body refers to "the vulnerability" without naming the
    //    library; only the title carries "libfoo".
    let query = "Is there an RCE in libfoo?";

    // 4a. Raw corpus — should NOT find the libfoo vulnerability chunk.
    let raw_out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            query,
            "--corpus",
            raw_corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--top-k",
            "1",
            "--no-rerank",
        ])
        .output()
        .unwrap();
    assert!(raw_out.status.success());
    let raw_result = String::from_utf8(raw_out.stdout).unwrap();

    // 4b. Contextualized corpus — should find the libfoo vulnerability chunk.
    let ctx_out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "query",
            query,
            "--corpus",
            ctx_corpus.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "--top-k",
            "1",
            "--no-rerank",
        ])
        .output()
        .unwrap();
    assert!(ctx_out.status.success());
    let ctx_result = String::from_utf8(ctx_out.stdout).unwrap();

    assert!(
        ctx_result.contains("01-libfoo-advisory.md"),
        "contextualized top-1 should be the libfoo advisory, got: {ctx_result}"
    );
    assert!(
        !raw_result.contains("01-libfoo-advisory.md"),
        "raw top-1 should NOT be the libfoo advisory (no lexical or semantic link to 'libfoo' in chunk body), got: {raw_result}"
    );
}
