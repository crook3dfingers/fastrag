//! End-to-end: CLI --time-decay-* flags. Covers the error path (decay params
//! without --time-decay-field) and the happy path (fresh doc promoted to top).
#![cfg(all(feature = "retrieval", feature = "store"))]

use std::collections::HashSet;
use std::fs;
use std::process::Command;

use chrono::Utc;

fn bin() -> String {
    env!("CARGO_BIN_EXE_fastrag").to_string()
}

#[test]
fn decay_flags_without_field_error() {
    // build_hybrid_opts runs AFTER the embedder loads, so we need a valid
    // corpus on disk first. Ingest a tiny document, then query with the
    // offending flag combination.
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("seed.jsonl");
    fs::write(
        &jsonl,
        r#"{"id":"X","text":"seed"}
"#,
    )
    .unwrap();
    let status = Command::new(bin())
        .args([
            "index",
            jsonl.to_str().unwrap(),
            "--corpus",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--text-fields",
            "text",
            "--id-field",
            "id",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "seed ingest failed");

    let out = Command::new(bin())
        .args([
            "query",
            "x",
            "--corpus",
            corpus.to_str().unwrap(),
            "--time-decay-halflife",
            "7d",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "should have errored");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--time-decay-field"),
        "stderr should reference --time-decay-field; got {stderr}"
    );
}

#[test]
fn decay_promotes_fresh_over_stale_via_cli() {
    // Fresh date = yesterday (always well within the 30-day halflife window).
    let fresh = (Utc::now() - chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();

    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("f.jsonl");
    fs::write(
        &jsonl,
        format!(
            r#"{{"id":"STALE","text":"openssl heap overflow","published_date":"2016-01-01"}}
{{"id":"FRESH","text":"openssl heap overflow","published_date":"{fresh}"}}
"#
        ),
    )
    .unwrap();

    // Ingest with typed date metadata.
    let status = Command::new(bin())
        .args([
            "index",
            jsonl.to_str().unwrap(),
            "--corpus",
            corpus.to_str().unwrap(),
            "--format",
            "jsonl",
            "--text-fields",
            "text",
            "--id-field",
            "id",
            "--metadata-fields",
            "published_date",
            "--metadata-types",
            "published_date=date",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "ingest failed");

    // Query with decay on published_date. --time-decay-field implies --hybrid.
    let out = Command::new(bin())
        .args([
            "query",
            "openssl heap overflow",
            "--corpus",
            corpus.to_str().unwrap(),
            "--top-k",
            "2",
            "--time-decay-field",
            "published_date",
            "--time-decay-halflife",
            "30d",
            "--no-rerank",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "query failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let top_id = top_hit_id(&stdout).unwrap_or_else(|| panic!("no top hit id in {stdout}"));
    assert_eq!(
        top_id, "FRESH",
        "decay should promote FRESH over STALE; got top_id={top_id}; stdout={stdout}"
    );
    // Sanity: both ids present in the result set.
    let ids = extract_ids(&stdout);
    assert!(ids.contains("FRESH"), "missing FRESH: {stdout}");
    assert!(ids.contains("STALE"), "missing STALE: {stdout}");
}

fn top_hit_id(json_text: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json_text).ok()?;
    v.as_array()?
        .first()?
        .get("source")?
        .get("id")?
        .as_str()
        .map(String::from)
}

fn extract_ids(json_text: &str) -> HashSet<String> {
    let v: serde_json::Value = serde_json::from_str(json_text).unwrap_or(serde_json::Value::Null);
    v.as_array()
        .map(|hits| {
            hits.iter()
                .filter_map(|h| {
                    h.get("source")
                        .and_then(|s| s.get("id"))
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default()
}
