//! End-to-end: CLI `--temporal-policy` flag and deprecation warnings for legacy
//! `--time-decay-*` flags.
#![cfg(all(feature = "retrieval", feature = "store"))]

use std::fs;
use std::process::Command;

use chrono::Utc;

fn bin() -> String {
    env!("CARGO_BIN_EXE_fastrag").to_string()
}

/// Ingest two docs — one stale (2016), one fresh (yesterday) — then return the
/// corpus path inside `tmp`.
fn setup_corpus(tmp: &tempfile::TempDir) -> std::path::PathBuf {
    let fresh = (Utc::now() - chrono::Duration::days(1))
        .format("%Y-%m-%d")
        .to_string();

    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("docs.jsonl");
    fs::write(
        &jsonl,
        format!(
            r#"{{"id":"OLD","text":"openssl heap overflow advisory","published_date":"2016-01-01"}}
{{"id":"NEW","text":"openssl heap overflow advisory","published_date":"{fresh}"}}
"#
        ),
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
            "--metadata-fields",
            "published_date",
            "--metadata-types",
            "published_date=date",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "corpus ingest failed");
    corpus
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

/// `--temporal-policy favor-recent-medium` should promote the fresh doc.
#[test]
fn query_with_temporal_policy_favor_recent_medium() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = setup_corpus(&tmp);

    let out = Command::new(bin())
        .args([
            "query",
            "openssl heap overflow advisory",
            "--corpus",
            corpus.to_str().unwrap(),
            "--temporal-policy",
            "favor-recent-medium",
            "--time-decay-field",
            "published_date",
            "--top-k",
            "3",
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
    let top = top_hit_id(&stdout).unwrap_or_else(|| panic!("no top hit id in stdout: {stdout}"));
    assert_eq!(
        top, "NEW",
        "--temporal-policy favor-recent-medium should promote NEW over OLD; stdout={stdout}"
    );
}

/// Legacy `--time-decay-halflife` without `--temporal-policy` (defaults to
/// `auto`) should emit a deprecation warning on stderr.
#[test]
fn deprecated_halflife_flag_emits_stderr_warning() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = setup_corpus(&tmp);

    let out = Command::new(bin())
        .args([
            "query",
            "openssl heap overflow",
            "--corpus",
            corpus.to_str().unwrap(),
            "--time-decay-halflife",
            "90d",
            "--time-decay-field",
            "published_date",
            "--top-k",
            "3",
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
    let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(
        stderr.contains("deprecated"),
        "stderr should contain 'deprecated'; got: {stderr}"
    );
}

/// `--temporal-policy auto` should abstain on a historical/descriptive query
/// (no recency intent), so the stale doc that semantically matches "heartbleed"
/// is NOT penalised.  We only assert exit=0 and no crash; ranking correctness
/// on a two-doc stub corpus with a mock embedder is non-deterministic but the
/// flag path itself must not error.
#[test]
fn temporal_policy_auto_does_not_error() {
    let tmp = tempfile::tempdir().unwrap();

    // Build a corpus with a doc from 2014 (Heartbleed era).
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("docs.jsonl");
    fs::write(
        &jsonl,
        r#"{"id":"HEARTBLEED","text":"CVE-2014-0160 OpenSSL heartbleed","published_date":"2014-04-07"}
{"id":"RECENT","text":"CVE-2014-0160 OpenSSL heartbleed analysis","published_date":"2024-01-01"}
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
            "--metadata-fields",
            "published_date",
            "--metadata-types",
            "published_date=date",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "ingest failed");

    let out = Command::new(bin())
        .args([
            "query",
            "describe CVE-2014-0160",
            "--corpus",
            corpus.to_str().unwrap(),
            "--temporal-policy",
            "auto",
            "--time-decay-field",
            "published_date",
            "--top-k",
            "3",
            "--no-rerank",
            "--format",
            "json",
        ])
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "query with --temporal-policy auto failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Minimal: we get a JSON array back (not an error).
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("stdout should be valid JSON");
    assert!(parsed.is_array(), "expected JSON array; got {stdout}");
}
