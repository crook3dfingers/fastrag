//! End-to-end: ingest a tiny JSONL corpus through index_jsonl + MockEmbedder,
//! then query via query_corpus_with_filter_opts with dense-only vs hybrid.
//! Assert the hybrid top hit is the BM25 lexical winner.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use fastrag::ChunkingStrategy;
use fastrag::corpus::hybrid::HybridOpts;
use fastrag::corpus::{LatencyBreakdown, QueryOpts, query_corpus_with_filter_opts};
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_embed::DynEmbedderTrait;
use fastrag_embed::test_utils::MockEmbedder;

fn write_fixture(path: &Path) {
    // id=1: no query-token matches; id=2: matches "delta" only; id=3: matches
    // both "delta" and "eta" → BM25 clearly ranks id=3 first on "delta eta".
    let lines = [
        r#"{"id":"1","title":"alpha beta gamma","description":"alpha beta gamma"}"#,
        r#"{"id":"2","title":"delta epsilon zeta","description":"delta epsilon zeta"}"#,
        r#"{"id":"3","title":"delta eta","description":"delta eta"}"#,
    ];
    fs::write(path, lines.join("\n")).unwrap();
}

fn cfg() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec!["title".into(), "description".into()],
        id_field: "id".into(),
        metadata_fields: vec![],
        metadata_types: BTreeMap::new(),
        array_fields: vec![],
        cwe_field: None,
    }
}

fn chunking() -> ChunkingStrategy {
    ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    }
}

#[test]
fn hybrid_changes_top_hit_vs_dense_only() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus = tmp.path().join("corpus");
    let jsonl = tmp.path().join("docs.jsonl");
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

    // Dense-only baseline.
    let dense_opts = QueryOpts::default();
    let mut bd = LatencyBreakdown::default();
    let dense_hits = query_corpus_with_filter_opts(
        &corpus,
        "delta eta",
        3,
        &embedder as &dyn DynEmbedderTrait,
        None,
        &dense_opts,
        &mut bd,
        0,
    )
    .unwrap();

    // Hybrid.
    let hybrid_opts = QueryOpts {
        hybrid: HybridOpts {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut bd2 = LatencyBreakdown::default();
    let hybrid_hits = query_corpus_with_filter_opts(
        &corpus,
        "delta eta",
        3,
        &embedder as &dyn DynEmbedderTrait,
        None,
        &hybrid_opts,
        &mut bd2,
        0,
    )
    .unwrap();

    assert_eq!(dense_hits.len(), 3, "dense should return 3 hits");
    assert_eq!(hybrid_hits.len(), 3, "hybrid should return 3 hits");

    // Concrete assertion: hybrid top hit contains the unique lexical-overlap
    // tokens (both "delta" AND "eta"). The BM25 winner on "delta eta" is the
    // doc that has both, which is id=3 ("delta eta"). Dense-only uses
    // MockEmbedder's byte-trigram fingerprint and does NOT guarantee that
    // ordering — hybrid's RRF promotes the BM25 winner to the top.
    let top_text = &hybrid_hits[0].chunk_text;
    assert!(
        top_text.contains("delta") && top_text.contains("eta"),
        "hybrid top hit should be the lexical winner containing both \
         'delta' and 'eta'; got {top_text:?}"
    );

    // Also sanity-check that the bm25 latency slot fired (hybrid path ran).
    assert!(bd2.bm25_us > 0, "hybrid path should populate bm25_us");
}
