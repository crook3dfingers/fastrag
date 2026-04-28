//! VIPER Assist profile retrieval-quality smoke test.
//!
//! Indexes a curated 11-row subset of the live VIPER corpus
//! (`tests/fixtures/viper_assist/smoke_corpus.jsonl`) using the `viper-assist`
//! preset + `nomic-ai/nomic-embed-text-v1.5` GGUF, runs the issue-#74 smoke
//! prompts, and asserts each query's top-k contains at least one allowlisted
//! VIPER page id.
//!
//! Gated like the other llama-cpp e2e tests: `FASTRAG_LLAMA_TEST=1` plus a
//! GGUF model path. Defaults to `/var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf`;
//! override with `VIPER_NOMIC_GGUF`.

#![cfg(all(feature = "retrieval", feature = "store"))]

use std::path::PathBuf;

use assert_cmd::Command;
use serde::Deserialize;
use tempfile::tempdir;

const FIXTURE_CORPUS: &str = include_str!("fixtures/viper_assist/smoke_corpus.jsonl");
const FIXTURE_QUERIES: &str = include_str!("fixtures/viper_assist/smoke_queries.json");

#[derive(Deserialize)]
struct SmokeQuerySet {
    top_k: usize,
    queries: Vec<SmokeQuery>,
}

#[derive(Deserialize)]
struct SmokeQuery {
    name: String,
    query: String,
    expected_ids_any_of: Vec<String>,
}

fn nomic_gguf_path() -> Option<PathBuf> {
    let candidate = std::env::var_os("VIPER_NOMIC_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf"));
    candidate.exists().then_some(candidate)
}

#[test]
#[ignore]
fn viper_assist_smoke_queries_return_relevant_pages() {
    if std::env::var("FASTRAG_LLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_LLAMA_TEST=1 to run");
        return;
    }
    let Some(model_path) = nomic_gguf_path() else {
        eprintln!(
            "skipping: place nomic-embed-text-v1.5.Q5_K_M.gguf at \
             /var/lib/fastrag/models/ or set VIPER_NOMIC_GGUF=/abs/path.gguf"
        );
        return;
    };

    let work = tempdir().unwrap();
    let corpus_path = work.path().join("smoke_corpus.jsonl");
    std::fs::write(&corpus_path, FIXTURE_CORPUS).unwrap();

    let bundle = tempdir().unwrap();
    let cfg_dir = tempdir().unwrap();
    let cfg_path = cfg_dir.path().join("fastrag.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[embedder]\n\
             default_profile = \"viper-assist\"\n\n\
             [embedder.profiles.viper-assist]\n\
             backend = \"llama-cpp\"\n\
             model = \"{}\"\n\
             use_catalog_defaults = true\n",
            model_path.display()
        ),
    )
    .unwrap();

    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            corpus_path.to_str().unwrap(),
            "--corpus",
            bundle.path().to_str().unwrap(),
            "--config",
            cfg_path.to_str().unwrap(),
            "--embedder-profile",
            "viper-assist",
            "--preset",
            "viper-assist",
        ])
        .assert()
        .success();

    let smoke: SmokeQuerySet = serde_json::from_str(FIXTURE_QUERIES).unwrap();
    let top_k_str = smoke.top_k.to_string();
    let mut failures = Vec::new();

    for q in &smoke.queries {
        // --no-rerank: smoke exercises embedder+preset+filter quality only;
        // reranker tuning is out of scope for #74 and would couple the
        // smoke to HuggingFace network access.
        let out = Command::cargo_bin("fastrag")
            .unwrap()
            .args([
                "query",
                &q.query,
                "--corpus",
                bundle.path().to_str().unwrap(),
                "--config",
                cfg_path.to_str().unwrap(),
                "--embedder-profile",
                "viper-assist",
                "--top-k",
                &top_k_str,
                "--format",
                "json",
                "--no-rerank",
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "query `{}` exited non-zero: {}",
            q.name,
            String::from_utf8_lossy(&out.stderr)
        );

        let hits: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
            panic!(
                "query `{}` produced non-JSON output: {e}\nstdout: {}",
                q.name,
                String::from_utf8_lossy(&out.stdout)
            )
        });

        let top_ids: Vec<String> = hits
            .iter()
            .filter_map(|h| {
                h.get("source_path")
                    .or_else(|| h.get("external_id"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect();

        let matched = q
            .expected_ids_any_of
            .iter()
            .any(|expected| top_ids.iter().any(|id| id == expected));

        if !matched {
            failures.push(format!(
                "query `{}` top-{}: {top_ids:?}\n  expected any of: {:?}",
                q.name, smoke.top_k, q.expected_ids_any_of
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} smoke queries failed:\n{}",
        failures.len(),
        smoke.queries.len(),
        failures.join("\n")
    );
}
