//! Integration test: gold_set::score_entry over synthetic chunk shapes.
//!
//! Exercises the union-of-top-k semantics: multi-chunk unions, case-insensitive
//! term matches, pronoun-resolution miss, and the honest-miss scoring path.

use fastrag_eval::gold_set::{score_entry, GoldSetEntry};

fn entry(cve: &[&str], terms: &[&str]) -> GoldSetEntry {
    GoldSetEntry {
        id: "qtest".into(),
        question: "x?".into(),
        must_contain_cve_ids: cve.iter().map(|s| s.to_string()).collect(),
        must_contain_terms: terms.iter().map(|s| s.to_string()).collect(),
        notes: None,
    }
}

#[test]
fn two_chunk_union_satisfies_at_k_2() {
    let e = entry(&["CVE-2024-1", "CVE-2024-2"], &[]);
    let chunks = vec!["only CVE-2024-1 here", "only CVE-2024-2 here"];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert!(s.hit_at_5);
    assert_eq!(s.reciprocal_rank, 0.5);
}

#[test]
fn pronoun_resolution_miss_when_title_not_in_chunks() {
    let e = entry(&["CVE-2024-12345"], &["libfoo"]);
    let chunks = vec![
        "the vulnerability affects all deployments",
        "impact is rated critical",
    ];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_10);
    assert_eq!(s.reciprocal_rank, 0.0);
    assert!(s.missing_cve_ids.contains(&"CVE-2024-12345".to_string()));
    assert!(s.missing_terms.contains(&"libfoo".to_string()));
}

#[test]
fn case_insensitive_cve_matching() {
    let e = entry(&["CVE-2024-12345"], &[]);
    let chunks = vec!["see cve-2024-12345 for details"];
    let s = score_entry(&e, &chunks);
    assert!(s.hit_at_1);
}

#[test]
fn empty_top_k_is_a_miss_not_a_crash() {
    let e = entry(&["CVE-2024-1"], &[]);
    let chunks: Vec<&str> = vec![];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert_eq!(s.reciprocal_rank, 0.0);
}
