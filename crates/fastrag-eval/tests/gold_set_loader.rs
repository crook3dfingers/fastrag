//! Integration test: gold_set::load validation branches against on-disk fixtures.

use std::path::PathBuf;

use fastrag_eval::gold_set::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn valid_fixture_loads() {
    let gs = load(&fixture("gold_valid.json")).expect("valid fixture should load");
    assert_eq!(gs.entries.len(), 2);
}

#[test]
fn empty_question_rejected() {
    let err = load(&fixture("gold_invalid_empty_q.json")).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("q_bad"), "{msg}");
    assert!(msg.contains("empty question"), "{msg}");
}

#[test]
fn duplicate_id_rejected() {
    let err = load(&fixture("gold_invalid_dup_id.json")).unwrap_err();
    assert!(format!("{err}").contains("duplicate"));
}

#[test]
fn malformed_cve_rejected() {
    let err = load(&fixture("gold_invalid_malformed_cve.json")).unwrap_err();
    assert!(format!("{err}").contains("CVE-24-1"));
}

#[test]
fn zero_assertions_rejected() {
    let err = load(&fixture("gold_invalid_zero_assertions.json")).unwrap_err();
    assert!(format!("{err}").contains("no must_contain"));
}
