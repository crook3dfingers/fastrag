//! End-to-end: ingest a tiny JSONL corpus that tags documents with parent and
//! child CWEs, query for the parent CWE with cwe_expand on, and assert the
//! child-tagged doc appears in the results.
#![cfg(feature = "store")]

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use fastrag::ChunkingStrategy;
use fastrag::corpus::{LatencyBreakdown, QueryOpts, query_corpus_with_filter_opts};
use fastrag::filter::FilterExpr;
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag::corpus::SearchHitDto;
use fastrag_embed::DynEmbedderTrait;
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_store::schema::{TypedKind, TypedValue};

fn extract_ids(hits: &[SearchHitDto]) -> HashSet<String> {
    hits.iter()
        .filter_map(|h| {
            h.source
                .as_ref()
                .and_then(|v| v.get("id"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect()
}

// Write a JSONL file with three finding records:
//   - id "A", cwe_id 89  (parent: SQL Injection)
//   - id "B", cwe_id 564 (child of 89: Hibernate Injection)
//   - id "C", cwe_id 79  (unrelated: XSS)
fn write_fixture(path: &Path) {
    let lines = [
        r#"{"id":"A","title":"sqli in login","description":"SQL injection","cwe_id":89}"#,
        r#"{"id":"B","title":"hibernate injection","description":"hql sqli","cwe_id":564}"#,
        r#"{"id":"C","title":"xss","description":"stored xss","cwe_id":79}"#,
    ];
    fs::write(path, lines.join("\n")).unwrap();
}

fn cfg() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec!["title".into(), "description".into()],
        id_field: "id".into(),
        metadata_fields: vec!["cwe_id".into()],
        metadata_types: BTreeMap::from([("cwe_id".into(), TypedKind::Numeric)]),
        array_fields: vec![],
        cwe_field: Some("cwe_id".into()),
    }
}

fn chunking() -> ChunkingStrategy {
    ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    }
}

#[test]
fn cwe_expansion_returns_child_tagged_docs() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("findings.jsonl");
    write_fixture(&jsonl);

    let embedder = MockEmbedder;
    index_jsonl(
        &jsonl,
        &corpus,
        &chunking(),
        &embedder as &dyn DynEmbedderTrait,
        &cfg(),
    )
    .unwrap();

    // Sanity: manifest records cwe_field + taxonomy version.
    let mbytes = fs::read(corpus.join("manifest.json")).unwrap();
    let manifest: fastrag_index::CorpusManifest = serde_json::from_slice(&mbytes).unwrap();
    assert_eq!(manifest.cwe_field.as_deref(), Some("cwe_id"));
    assert!(manifest.cwe_taxonomy_version.is_some());

    // Query with filter cwe_id = 89, expansion ON.
    let filter = FilterExpr::Eq {
        field: "cwe_id".into(),
        value: TypedValue::Numeric(89.0),
    };
    let opts = QueryOpts { cwe_expand: true };
    let mut b = LatencyBreakdown::default();
    let hits_expanded = query_corpus_with_filter_opts(
        &corpus,
        "query",
        10,
        &embedder as &dyn DynEmbedderTrait,
        Some(&filter),
        &opts,
        &mut b,
        0,
    )
    .unwrap();
    let ids_expanded = extract_ids(&hits_expanded);
    assert!(
        ids_expanded.contains("A"),
        "parent doc A missing: {ids_expanded:?}"
    );
    assert!(
        ids_expanded.contains("B"),
        "child-CWE doc B missing: {ids_expanded:?}"
    );
    assert!(
        !ids_expanded.contains("C"),
        "unrelated doc C should not match: {ids_expanded:?}"
    );

    // Query with expansion OFF: only A.
    let opts_off = QueryOpts { cwe_expand: false };
    let mut b = LatencyBreakdown::default();
    let hits_plain = query_corpus_with_filter_opts(
        &corpus,
        "query",
        10,
        &embedder as &dyn DynEmbedderTrait,
        Some(&filter),
        &opts_off,
        &mut b,
        0,
    )
    .unwrap();
    let ids_plain = extract_ids(&hits_plain);
    assert!(ids_plain.contains("A"));
    assert!(
        !ids_plain.contains("B"),
        "child-CWE doc must NOT match without expansion"
    );
}

#[test]
fn free_text_trigger_synthesizes_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("findings.jsonl");
    write_fixture(&jsonl);
    let embedder = MockEmbedder;
    index_jsonl(
        &jsonl,
        &corpus,
        &chunking(),
        &embedder as &dyn DynEmbedderTrait,
        &cfg(),
    )
    .unwrap();

    // Free-text query mentioning CWE-89 with NO explicit filter. Expansion on.
    let opts = QueryOpts { cwe_expand: true };
    let mut b = LatencyBreakdown::default();
    let hits = query_corpus_with_filter_opts(
        &corpus,
        "vulnerability CWE-89 in login form",
        10,
        &embedder as &dyn DynEmbedderTrait,
        None,
        &opts,
        &mut b,
        0,
    )
    .unwrap();
    let ids = extract_ids(&hits);
    assert!(ids.contains("A"), "parent doc A missing: {ids:?}");
    assert!(ids.contains("B"), "child-CWE doc B missing: {ids:?}");
    assert!(
        !ids.contains("C"),
        "unrelated XSS should not match when query mentions CWE-89: {ids:?}"
    );
}
