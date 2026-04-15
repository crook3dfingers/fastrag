//! End-to-end: directory ingest reads markdown YAML frontmatter, promotes
//! named fields to `TypedValue` via the JSONL typing helpers, and writes the
//! result to the Store so it becomes available as `user_fields` on every
//! chunk of the document.
#![cfg(all(feature = "store", feature = "retrieval"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use fastrag::ChunkingStrategy;
use fastrag::corpus::index_path_with_metadata_typed;
use fastrag_embed::DynEmbedderTrait;
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_store::Store;
use fastrag_store::schema::{TypedKind, TypedValue};

#[test]
fn frontmatter_dates_land_as_typed_date_in_user_fields() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus_root: PathBuf = tmp.path().join("corpus");
    let input_dir: PathBuf = tmp.path().join("docs");
    fs::create_dir_all(&input_dir).unwrap();
    let doc = input_dir.join("log4shell.md");
    fs::write(
        &doc,
        "---\npublished_date: 2021-12-10\nseverity: high\n---\n# CVE-2021-44228\n\nLog4Shell advisory.\n",
    )
    .unwrap();

    let mut types: BTreeMap<String, TypedKind> = BTreeMap::new();
    types.insert("published_date".to_string(), TypedKind::Date);

    let fields = vec!["published_date".to_string(), "severity".to_string()];

    let embedder = MockEmbedder;
    let chunking = ChunkingStrategy::Basic {
        max_characters: 500,
        overlap: 0,
    };

    index_path_with_metadata_typed(
        &input_dir,
        &corpus_root,
        &chunking,
        &embedder as &dyn DynEmbedderTrait,
        &BTreeMap::new(),
        &fields,
        &types,
        #[cfg(feature = "contextual")]
        None,
        #[cfg(feature = "hygiene")]
        None,
    )
    .expect("index_path_with_metadata_typed should succeed on a well-formed fixture");

    // Re-open the Store via the manifest-passthrough embedder (no live model
    // required) and inspect the single chunk's user_fields.
    let store = Store::open_no_embedder(&corpus_root).expect("Store::open_no_embedder");

    // Drive a BM25 query wide enough to return our single document.
    let ids_scored = store
        .query_bm25("log4shell advisory cve-2021-44228", 10)
        .expect("bm25 query");
    assert!(
        !ids_scored.is_empty(),
        "ingest should produce at least one chunk and bm25 should see it"
    );

    let ids: Vec<u64> = ids_scored.iter().map(|(id, _)| *id).collect();
    let metadata = store.fetch_metadata(&ids).expect("fetch_metadata");
    assert!(!metadata.is_empty(), "metadata should be fetchable");

    // Find a chunk carrying the Date-typed published_date.
    let mut saw_typed_date = false;
    for (_id, fields_vec) in &metadata {
        for (k, v) in fields_vec {
            if k == "published_date" {
                match v {
                    TypedValue::Date(d) => {
                        assert_eq!(d.to_string(), "2021-12-10");
                        saw_typed_date = true;
                    }
                    other => panic!("expected TypedValue::Date, got {other:?}"),
                }
            }
        }
    }
    assert!(
        saw_typed_date,
        "no chunk carried a TypedValue::Date for published_date; got rows = {metadata:?}"
    );
}
