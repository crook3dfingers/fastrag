//! End-to-end: ingest JSONL -> query -> upsert -> delete -> compact.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use fastrag::ChunkingStrategy;
use fastrag::filter::{self, FilterExpr};
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_embed::{DynEmbedderTrait, Embedder};
use fastrag_store::schema::TypedKind;

fn write_jsonl(path: &Path, lines: &[&str]) {
    let mut f = std::fs::File::create(path).unwrap();
    for line in lines {
        writeln!(f, "{line}").unwrap();
    }
}

fn make_config() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec!["title".into(), "description".into()],
        id_field: "id".into(),
        metadata_fields: vec!["severity".into()],
        metadata_types: BTreeMap::new(),
        array_fields: vec![],
    }
}

#[test]
fn jsonl_ingest_query_delete_compact() {
    let tmp = tempfile::tempdir().unwrap();
    let jsonl_path = tmp.path().join("findings.jsonl");
    let corpus_dir = tmp.path().join("corpus");

    let embedder = MockEmbedder;
    let chunking = ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    };
    let config = make_config();

    // ── 1. Ingest 3 JSONL records ────────────────────────────────────────────
    let lines = [
        r#"{"id":"f1","title":"SQL Injection","description":"Critical SQLi in login form","severity":"critical"}"#,
        r#"{"id":"f2","title":"XSS","description":"Reflected XSS in search parameter","severity":"high"}"#,
        r#"{"id":"f3","title":"Info Disclosure","description":"Server version exposed in headers","severity":"low"}"#,
    ];
    write_jsonl(&jsonl_path, &lines);

    let stats = index_jsonl(&jsonl_path, &corpus_dir, &chunking, &embedder, &config).unwrap();
    assert_eq!(stats.records_total, 3, "should see 3 total records");
    assert_eq!(stats.records_new, 3, "all 3 should be new");
    assert_eq!(stats.records_skipped, 0, "none skipped on first ingest");

    // ── 2. Query returns results with _source ────────────────────────────────
    {
        let store =
            fastrag_store::Store::open(&corpus_dir, &embedder as &dyn DynEmbedderTrait).unwrap();

        // Use the embedder's own output as the query vector for a known string
        let query_vec = embedder
            .embed_query(&[fastrag_embed::QueryText::new("SQL Injection")])
            .unwrap()
            .pop()
            .unwrap();

        let scored = store.query_dense(&query_vec, 10).unwrap();
        assert!(!scored.is_empty(), "dense query should return hits");

        let hits = store.fetch_hits(&scored).unwrap();
        assert!(!hits.is_empty(), "fetch_hits should return search results");

        let has_source = hits.iter().any(|h| h.source.is_some());
        assert!(has_source, "at least one hit should have populated source");
    }

    // ── 3. Re-ingest is idempotent ───────────────────────────────────────────
    let stats2 = index_jsonl(&jsonl_path, &corpus_dir, &chunking, &embedder, &config).unwrap();
    assert_eq!(stats2.records_total, 3);
    assert_eq!(
        stats2.records_skipped, 3,
        "all records should be skipped on re-ingest"
    );
    assert_eq!(stats2.records_new, 0, "no new records on re-ingest");

    // ── 4. Upsert on change ──────────────────────────────────────────────────
    let modified_lines = [
        r#"{"id":"f1","title":"SQL Injection","description":"UPDATED: blind SQLi via UNION in login","severity":"critical"}"#,
        r#"{"id":"f2","title":"XSS","description":"Reflected XSS in search parameter","severity":"high"}"#,
        r#"{"id":"f3","title":"Info Disclosure","description":"Server version exposed in headers","severity":"low"}"#,
    ];
    write_jsonl(&jsonl_path, &modified_lines);

    let stats3 = index_jsonl(&jsonl_path, &corpus_dir, &chunking, &embedder, &config).unwrap();
    assert_eq!(stats3.records_total, 3);
    assert_eq!(stats3.records_upserted, 1, "f1 changed, should be upserted");
    assert_eq!(stats3.records_skipped, 2, "f2 and f3 unchanged");

    // ── 5. Delete ────────────────────────────────────────────────────────────
    {
        let mut store =
            fastrag_store::Store::open(&corpus_dir, &embedder as &dyn DynEmbedderTrait).unwrap();

        let deleted_ids = store.delete_by_external_id("f2").unwrap();
        assert!(
            !deleted_ids.is_empty(),
            "deleting f2 should remove at least one chunk"
        );

        // ── 6. Compact ──────────────────────────────────────────────────────
        assert!(
            store.tombstone_count() > 0,
            "should have tombstones after delete"
        );
        store.compact();
        assert_eq!(
            store.tombstone_count(),
            0,
            "compact should purge all tombstones"
        );

        // ── 7. Save ─────────────────────────────────────────────────────────
        store.save().unwrap();
    }

    // Verify the corpus is still usable after save
    {
        let store =
            fastrag_store::Store::open(&corpus_dir, &embedder as &dyn DynEmbedderTrait).unwrap();
        assert!(
            store.live_count() > 0,
            "corpus should still have live entries after reopen"
        );
    }
}

fn make_filter_config() -> JsonlIngestConfig {
    let mut types = BTreeMap::new();
    types.insert("cvss_score".into(), TypedKind::Numeric);
    JsonlIngestConfig {
        text_fields: vec!["title".into(), "description".into()],
        id_field: "id".into(),
        metadata_fields: vec!["severity".into(), "cvss_score".into(), "tags".into()],
        metadata_types: types,
        array_fields: vec!["tags".into()],
    }
}

/// Ingest JSONL with typed metadata, then verify filter evaluation works
/// end-to-end through the Store → fetch_metadata → matches pipeline.
#[test]
fn jsonl_ingest_and_filter_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let jsonl_path = tmp.path().join("findings.jsonl");
    let corpus_dir = tmp.path().join("corpus");

    let embedder = MockEmbedder;
    let chunking = ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    };
    let config = make_filter_config();

    let lines = [
        r#"{"id":"f1","title":"SQL Injection","description":"Critical SQLi","severity":"critical","cvss_score":9.8,"tags":["sqli","rce"]}"#,
        r#"{"id":"f2","title":"XSS","description":"Reflected XSS","severity":"high","cvss_score":6.1,"tags":["xss"]}"#,
        r#"{"id":"f3","title":"Info Disclosure","description":"Version exposed","severity":"low","cvss_score":3.1,"tags":["info"]}"#,
    ];
    write_jsonl(&jsonl_path, &lines);

    let stats = index_jsonl(&jsonl_path, &corpus_dir, &chunking, &embedder, &config).unwrap();
    assert_eq!(stats.records_total, 3);

    // Open Store and fetch all metadata
    let store =
        fastrag_store::Store::open(&corpus_dir, &embedder as &dyn DynEmbedderTrait).unwrap();

    let query_vec = embedder
        .embed_query(&[fastrag_embed::QueryText::new("SQL Injection")])
        .unwrap()
        .pop()
        .unwrap();

    let scored = store.query_dense(&query_vec, 10).unwrap();
    assert!(scored.len() >= 3, "should return all 3 records");

    let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
    let metadata_rows = store.fetch_metadata(&ids).unwrap();
    assert!(!metadata_rows.is_empty(), "metadata should be populated");

    // ── String equality filter: severity = critical ─────────────────────
    let expr = filter::parse("severity = critical").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "only f1 has severity=critical");

    // ── Numeric comparison: cvss_score >= 7.0 ───────────────────────────
    let expr = filter::parse("cvss_score >= 7.0").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "only f1 has cvss_score >= 7.0");

    // ── CONTAINS on array field: tags CONTAINS sqli ─────────────────────
    let expr = filter::parse("tags CONTAINS sqli").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "only f1 has 'sqli' in tags");

    // ── Complex: severity = critical OR cvss_score >= 6.0 ───────────────
    let expr = filter::parse("severity = critical OR cvss_score >= 6.0").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 2, "f1 (critical, 9.8) and f2 (high, 6.1)");

    // ── AND precedence: severity = high AND cvss_score < 7.0 ────────────
    let expr = filter::parse("severity = high AND cvss_score < 7.0").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "only f2 matches both conditions");

    // ── NOT filter: NOT severity = low ──────────────────────────────────
    let expr = filter::parse("NOT severity = low").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 2, "f1 and f2 are not low");

    // ── IN filter: severity IN (critical, high) ─────────────────────────
    let expr = filter::parse("severity IN (critical, high)").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 2, "f1 and f2 match IN");

    // ── Legacy k=v,k2=v2 compat ─────────────────────────────────────────
    let expr = filter::parse("severity=critical").unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "legacy format should still work");

    // ── JSON AST round-trip ─────────────────────────────────────────────
    let json_ast = r#"{"eq":{"field":"severity","value":"critical"}}"#;
    let expr: FilterExpr = serde_json::from_str(json_ast).unwrap();
    let passing: Vec<_> = metadata_rows
        .iter()
        .filter(|(_, fields)| filter::matches(&expr, fields))
        .collect();
    assert_eq!(passing.len(), 1, "JSON AST filter should work");
}
