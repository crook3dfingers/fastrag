#![cfg(feature = "eval")]

use std::io::Write;

use assert_cmd::Command;
use tempfile::NamedTempFile;

/// Verifies that `fastrag eval --config-matrix --gold-set <bad>` exits non-zero
/// and surfaces a useful error when the gold set has an entry with an empty question.
///
/// This test does not require any model servers — it fails purely on validation.
#[test]
fn config_matrix_rejects_gold_set_with_empty_question() {
    let mut f = NamedTempFile::new().expect("tempfile");
    write!(
        f,
        r#"{{
            "version": 1,
            "entries": [
                {{
                    "id": "bad_one",
                    "question": "",
                    "must_contain_cve_ids": ["CVE-2024-1"],
                    "must_contain_terms": []
                }}
            ]
        }}"#
    )
    .unwrap();
    f.flush().unwrap();

    let output = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "eval",
            "--config-matrix",
            "--gold-set",
            f.path().to_str().unwrap(),
            "--corpus",
            "/tmp/nonexistent_corpus_for_test",
            "--corpus-no-contextual",
            "/tmp/nonexistent_corpus_raw_for_test",
            "--report",
            "/tmp/nonexistent_report.json",
        ])
        .output()
        .expect("fastrag binary must run");

    assert!(
        !output.status.success(),
        "expected non-zero exit for invalid gold set, got success"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("bad_one"),
        "stderr must name the offending entry id 'bad_one', got: {stderr}"
    );
    assert!(
        stderr.contains("empty question"),
        "stderr must contain 'empty question', got: {stderr}"
    );
}
