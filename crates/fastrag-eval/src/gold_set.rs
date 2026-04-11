//! Gold set schema loader + union-of-top-k scorer.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::EvalError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldSet {
    pub version: u32,
    pub entries: Vec<GoldSetEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldSetEntry {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub must_contain_cve_ids: Vec<String>,
    #[serde(default)]
    pub must_contain_terms: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryScore {
    pub hit_at_1: bool,
    pub hit_at_5: bool,
    pub hit_at_10: bool,
    pub reciprocal_rank: f64,
    pub missing_cve_ids: Vec<String>,
    pub missing_terms: Vec<String>,
}

static CVE_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^CVE-\d{4}-\d+$").unwrap());

pub fn load(path: &Path) -> Result<GoldSet, EvalError> {
    let bytes = std::fs::read(path).map_err(EvalError::from)?;
    let gs: GoldSet = serde_json::from_slice(&bytes).map_err(|e| EvalError::GoldSetParse {
        path: path.to_path_buf(),
        source: e,
    })?;
    validate(&gs)?;
    Ok(gs)
}

fn validate(gs: &GoldSet) -> Result<(), EvalError> {
    if gs.version == 0 {
        return Err(EvalError::GoldSetInvalid("version must be >= 1".into()));
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for entry in &gs.entries {
        if entry.id.is_empty() {
            return Err(EvalError::GoldSetInvalid(
                "entry with empty id is not allowed".into(),
            ));
        }
        if !seen.insert(entry.id.as_str()) {
            return Err(EvalError::GoldSetInvalid(format!(
                "duplicate entry id '{}'",
                entry.id
            )));
        }
        if entry.question.trim().is_empty() {
            return Err(EvalError::GoldSetInvalid(format!(
                "entry '{}' has empty question",
                entry.id
            )));
        }
        if entry.must_contain_cve_ids.is_empty() && entry.must_contain_terms.is_empty() {
            return Err(EvalError::GoldSetInvalid(format!(
                "entry '{}' has no must_contain_cve_ids and no must_contain_terms",
                entry.id
            )));
        }
        for cve in &entry.must_contain_cve_ids {
            if !CVE_ID_RE.is_match(cve) {
                return Err(EvalError::GoldSetInvalid(format!(
                    "entry '{}' must_contain_cve_ids contains malformed id '{}'",
                    entry.id, cve
                )));
            }
        }
    }
    Ok(())
}

static CVE_FIND_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)CVE-\d{4}-\d+").unwrap());

pub fn score_entry(entry: &GoldSetEntry, top_k_chunks: &[&str]) -> EntryScore {
    let mut hit_at_1 = false;
    let mut hit_at_5 = false;
    let mut hit_at_10 = false;
    let mut reciprocal_rank = 0.0;

    let mut final_missing_cve_ids: Vec<String> = entry.must_contain_cve_ids.clone();
    let mut final_missing_terms: Vec<String> = entry.must_contain_terms.clone();

    for k in 1..=top_k_chunks.len().min(10) {
        let buffer: String = top_k_chunks[..k].join("\n\n");
        let buffer_lower = buffer.to_lowercase();

        let found_cves: HashSet<String> = CVE_FIND_RE
            .find_iter(&buffer)
            .map(|m| m.as_str().to_uppercase())
            .collect();

        let missing_cves: Vec<String> = entry
            .must_contain_cve_ids
            .iter()
            .filter(|c| !found_cves.contains(&c.to_uppercase()))
            .cloned()
            .collect();

        let missing_terms: Vec<String> = entry
            .must_contain_terms
            .iter()
            .filter(|t| !buffer_lower.contains(&t.to_lowercase()))
            .cloned()
            .collect();

        let satisfied = missing_cves.is_empty() && missing_terms.is_empty();

        if satisfied && reciprocal_rank == 0.0 {
            reciprocal_rank = 1.0 / (k as f64);
        }

        if k == 1 && satisfied {
            hit_at_1 = true;
        }
        if k <= 5 && satisfied {
            hit_at_5 = true;
        }
        if k <= 10 && satisfied {
            hit_at_10 = true;
        }

        if k == top_k_chunks.len().min(10) {
            final_missing_cve_ids = missing_cves;
            final_missing_terms = missing_terms;
        }
    }

    EntryScore {
        hit_at_1,
        hit_at_5,
        hit_at_10,
        reciprocal_rank,
        missing_cve_ids: final_missing_cve_ids,
        missing_terms: final_missing_terms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_fixture(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn gold_set_round_trips_through_json() {
        let gs = GoldSet {
            version: 1,
            entries: vec![GoldSetEntry {
                id: "q001".into(),
                question: "Is there an RCE in libfoo?".into(),
                must_contain_cve_ids: vec!["CVE-2024-12345".into()],
                must_contain_terms: vec!["libfoo".into()],
                notes: None,
            }],
        };
        let json = serde_json::to_string(&gs).unwrap();
        let back: GoldSet = serde_json::from_str(&json).unwrap();
        assert_eq!(gs, back);
    }

    #[test]
    fn load_accepts_well_formed_gold_set() {
        let f = write_fixture(r#"{
            "version": 1,
            "entries": [
                {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
            ]
        }"#);
        let gs = load(f.path()).expect("valid gold set should load");
        assert_eq!(gs.entries.len(), 1);
        assert_eq!(gs.entries[0].id, "q001");
    }

    #[test]
    fn load_rejects_empty_question() {
        let f = write_fixture(r#"{
            "version": 1,
            "entries": [
                {"id": "q001", "question": "", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
            ]
        }"#);
        let err = load(f.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("q001"), "error must name offending id, got: {msg}");
        assert!(msg.contains("empty question"), "error must say 'empty question', got: {msg}");
    }

    #[test]
    fn load_rejects_duplicate_id() {
        let f = write_fixture(r#"{
            "version": 1,
            "entries": [
                {"id": "q001", "question": "a?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []},
                {"id": "q001", "question": "b?", "must_contain_cve_ids": ["CVE-2024-2"], "must_contain_terms": []}
            ]
        }"#);
        let err = load(f.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "got: {msg}");
        assert!(msg.contains("q001"), "got: {msg}");
    }

    #[test]
    fn load_rejects_malformed_cve_id() {
        let f = write_fixture(r#"{
            "version": 1,
            "entries": [
                {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-24-1"], "must_contain_terms": []}
            ]
        }"#);
        let err = load(f.path()).unwrap_err();
        assert!(format!("{err}").contains("CVE-24-1"));
    }

    #[test]
    fn load_rejects_zero_assertions() {
        let f = write_fixture(r#"{
            "version": 1,
            "entries": [
                {"id": "q001", "question": "x?", "must_contain_cve_ids": [], "must_contain_terms": []}
            ]
        }"#);
        let err = load(f.path()).unwrap_err();
        assert!(format!("{err}").contains("no must_contain"));
    }

    fn entry(cve: &[&str], terms: &[&str]) -> GoldSetEntry {
        GoldSetEntry {
            id: "q001".into(),
            question: "x?".into(),
            must_contain_cve_ids: cve.iter().map(|s| s.to_string()).collect(),
            must_contain_terms: terms.iter().map(|s| s.to_string()).collect(),
            notes: None,
        }
    }

    #[test]
    fn score_entry_hit_at_1_when_first_chunk_satisfies() {
        let e = entry(&["CVE-2024-1"], &["libfoo"]);
        let chunks = vec![
            "advisory for libfoo mentions CVE-2024-1",
            "unrelated",
        ];
        let s = score_entry(&e, &chunks);
        assert!(s.hit_at_1);
        assert!(s.hit_at_5);
        assert!(s.hit_at_10);
        assert_eq!(s.reciprocal_rank, 1.0);
        assert!(s.missing_cve_ids.is_empty());
        assert!(s.missing_terms.is_empty());
    }

    #[test]
    fn score_entry_union_hit_at_3_across_chunks() {
        let e = entry(&["CVE-2024-1", "CVE-2024-2"], &[]);
        let chunks = vec![
            "mentions CVE-2024-1 only",
            "nothing here",
            "CVE-2024-2 found here",
        ];
        let s = score_entry(&e, &chunks);
        assert!(!s.hit_at_1);
        assert!(s.hit_at_5);
        assert_eq!(s.reciprocal_rank, 1.0 / 3.0);
    }

    #[test]
    fn score_entry_case_insensitive_term_match() {
        let e = entry(&[], &["SSRF"]);
        let chunks = vec!["the server was vulnerable to ssrf attacks"];
        let s = score_entry(&e, &chunks);
        assert!(s.hit_at_1);
    }

    #[test]
    fn score_entry_miss_when_nothing_satisfies() {
        let e = entry(&["CVE-2024-99999"], &[]);
        let chunks = vec!["irrelevant content", "also irrelevant"];
        let s = score_entry(&e, &chunks);
        assert!(!s.hit_at_1);
        assert!(!s.hit_at_5);
        assert!(!s.hit_at_10);
        assert_eq!(s.reciprocal_rank, 0.0);
        assert_eq!(s.missing_cve_ids, vec!["CVE-2024-99999".to_string()]);
    }

    #[test]
    fn score_entry_is_pure() {
        let e = entry(&["CVE-2024-1"], &["libfoo"]);
        let chunks = vec!["CVE-2024-1 in libfoo"];
        let s1 = score_entry(&e, &chunks);
        let s2 = score_entry(&e, &chunks);
        assert_eq!(s1, s2);
    }
}
