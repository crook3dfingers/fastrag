//! End-to-end: post-fusion multiplicative decay promotes fresh docs over
//! equally-relevant stale docs, and a dateless prior of 0.5 lands the
//! undated doc between them.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use chrono::{TimeZone, Utc};
use fastrag::ChunkingStrategy;
use fastrag::corpus::hybrid::{BlendMode, HybridOpts, TemporalOpts};
use fastrag::corpus::{LatencyBreakdown, QueryOpts, query_corpus_with_filter_opts};
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_embed::DynEmbedderTrait;
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_store::schema::TypedKind;

fn write_fixture(path: &Path) {
    // All three docs contain the query tokens "openssl heap overflow" so
    // BM25 and dense both rank them closely. The trailing discriminator
    // word (legacy/recent/dateless) lets us identify each hit by chunk_text.
    let lines = [
        r#"{"id":"A","text":"openssl heap overflow legacy","published_date":"2016-01-01"}"#,
        r#"{"id":"B","text":"openssl heap overflow recent","published_date":"2026-04-01"}"#,
        r#"{"id":"C","text":"openssl heap overflow dateless"}"#,
    ];
    fs::write(path, lines.join("\n")).unwrap();
}

fn cfg() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec!["text".into()],
        id_field: "id".into(),
        metadata_fields: vec!["published_date".into()],
        metadata_types: BTreeMap::from([("published_date".into(), TypedKind::Date)]),
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
fn decay_promotes_fresh_over_equally_relevant_stale() {
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

    // Hybrid on, multiplicative decay anchored at 2026-04-14 with 30d halflife.
    // Age(B) = 13d → factor ≈ 0.3 + 0.7 * 2^(-13/30) ≈ 0.82.
    // Age(A) = 3756d → factor saturates to floor 0.3.
    // C (dateless) → prior 0.5.
    let opts = QueryOpts {
        hybrid: HybridOpts {
            enabled: true,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: Some(TemporalOpts {
                date_fields: vec!["published_date".into()],
                halflife: Duration::from_secs(30 * 86_400),
                weight_floor: 0.3,
                dateless_prior: 0.5,
                blend: BlendMode::Multiplicative,
                now: Utc.with_ymd_and_hms(2026, 4, 14, 0, 0, 0).unwrap(),
            }),
        },
        ..Default::default()
    };

    let mut bd = LatencyBreakdown::default();
    let hits = query_corpus_with_filter_opts(
        &corpus,
        "openssl heap overflow",
        3,
        &embedder as &dyn DynEmbedderTrait,
        None,
        &opts,
        &mut bd,
        0,
    )
    .unwrap();

    assert_eq!(hits.len(), 3, "expected 3 hits, got {}", hits.len());

    // Strong assertion: fresh (id=B, "recent") outranks dateless (id=C,
    // "dateless") outranks stale (id=A, "legacy").
    let texts: Vec<&str> = hits.iter().map(|h| h.chunk_text.as_str()).collect();
    assert!(
        texts[0].contains("recent"),
        "hits[0] should be fresh ('recent'); got {texts:?}"
    );
    assert!(
        texts[1].contains("dateless"),
        "hits[1] should be dateless ('dateless'); got {texts:?}"
    );
    assert!(
        texts[2].contains("legacy"),
        "hits[2] should be stale ('legacy'); got {texts:?}"
    );
}
