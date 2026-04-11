//! E2E test for hybrid retrieval (BM25 + dense + RRF).
//!
//! Indexes a small corpus with the hybrid feature enabled, then verifies:
//! 1. CVE exact lookup is prepended as the first result.
//! 2. BM25 keyword matching contributes to ranking.
//! 3. `--dense-only` bypasses Tantivy and still returns results.
#![cfg(feature = "hybrid")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use fastrag::corpus::SearchHitDto;
use fastrag::corpus::hybrid::HybridIndex;
use fastrag::ops;
use fastrag::{CorpusManifest, DynEmbedderTrait, IndexEntry, ManifestChunkingStrategy};
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_embedder};
use fastrag_embed::test_utils::MockEmbedder;
use fastrag_embed::{CANARY_TEXT, Canary, Embedder, PassageText};
use reqwest::Client;
use tempfile::tempdir;

fn mock_manifest() -> CorpusManifest {
    let embedder = MockEmbedder;
    let canary_vec = embedder
        .embed_passage(&[PassageText::new(CANARY_TEXT)])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    CorpusManifest {
        version: 4,
        identity: embedder.identity(),
        canary: Canary {
            text_version: 1,
            vector: canary_vec,
        },
        created_at_unix_seconds: 0,
        chunk_count: 0,
        chunking_strategy: ManifestChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        },
        roots: vec![],
        files: vec![],
        contextualizer: None,
    }
}

fn test_entry(id: u64, text: &str, meta: BTreeMap<String, String>) -> IndexEntry {
    let embedder = MockEmbedder;
    let vector = embedder
        .embed_passage(&[PassageText::new(text)])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    IndexEntry {
        id,
        vector,
        chunk_text: text.to_string(),
        source_path: PathBuf::from(format!("doc_{id}.txt")),
        chunk_index: 0,
        section: None,
        element_kinds: vec![],
        pages: vec![],
        language: None,
        metadata: meta,
        display_text: None,
    }
}

fn cve_meta(cve: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("cve_id".to_string(), cve.to_string());
    m
}

/// Build a small hybrid corpus in a temp dir.
fn build_test_corpus(dir: &std::path::Path) {
    let manifest = mock_manifest();
    let mut hybrid = HybridIndex::create(dir, manifest).unwrap();
    hybrid
        .add(vec![
            test_entry(
                1,
                "A critical buffer overflow vulnerability in OpenSSL",
                cve_meta("CVE-2024-1234"),
            ),
            test_entry(
                2,
                "Rust Rust Rust systems programming language performance safety concurrency",
                BTreeMap::new(),
            ),
            test_entry(
                3,
                "Python is popular for data science and machine learning applications",
                BTreeMap::new(),
            ),
        ])
        .unwrap();
    hybrid.save(dir).unwrap();
}

#[test]
fn hybrid_query_prepends_cve_exact_hit() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let embedder = MockEmbedder;
    let filter = BTreeMap::new();
    let mut breakdown = fastrag::corpus::LatencyBreakdown::default();
    let hits = ops::query_corpus_hybrid(
        dir.path(),
        "CVE-2024-1234",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &filter,
        &mut breakdown,
    )
    .unwrap();

    assert!(!hits.is_empty());
    assert_eq!(
        hits[0].entry.id, 1,
        "exact CVE match should be the first result"
    );
}

#[test]
fn hybrid_query_bm25_boosts_keyword_match() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let embedder = MockEmbedder;
    let filter = BTreeMap::new();
    let hits = ops::query_corpus_hybrid(
        dir.path(),
        "Rust programming",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &filter,
        &mut fastrag::corpus::LatencyBreakdown::default(),
    )
    .unwrap();

    assert!(!hits.is_empty());
    let has_rust = hits.iter().any(|h| h.entry.id == 2);
    assert!(has_rust, "BM25 should surface the Rust-heavy document");
}

#[test]
fn latency_breakdown_threaded_through_query_corpus_hybrid() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let embedder = MockEmbedder;
    let filter = BTreeMap::new();
    let mut breakdown = fastrag::corpus::LatencyBreakdown::default();
    let hits = ops::query_corpus_hybrid(
        dir.path(),
        "Rust programming",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &filter,
        &mut breakdown,
    )
    .unwrap();

    assert!(!hits.is_empty());
    // Embed stage must have fired
    assert!(breakdown.embed_us > 0, "embed_us should be non-zero");
    // BM25 and HNSW must have fired (hybrid path, not dense-only fallback)
    assert!(breakdown.bm25_us > 0, "bm25_us should be non-zero");
    assert!(breakdown.hnsw_us > 0, "hnsw_us should be non-zero");
    // total_us must equal the sum of all stages (finalize semantics)
    assert_eq!(
        breakdown.total_us,
        breakdown.embed_us
            + breakdown.bm25_us
            + breakdown.hnsw_us
            + breakdown.rerank_us
            + breakdown.fuse_us,
        "total_us must equal sum of per-stage fields"
    );
    assert!(breakdown.total_us > 0, "total_us should be non-zero");
}

#[test]
fn dense_only_fallback_works() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    // Use the dense-only query path (no hybrid)
    let embedder = MockEmbedder;
    let filter = BTreeMap::new();
    let hits = ops::query_corpus_with_filter(
        dir.path(),
        "anything",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &filter,
        &mut fastrag::corpus::LatencyBreakdown::default(),
    )
    .unwrap();

    assert!(
        !hits.is_empty(),
        "dense-only path should still return results"
    );
}

#[tokio::test]
async fn http_hybrid_query() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn({
        let corpus_dir = dir.path().to_path_buf();
        let embedder = Arc::new(MockEmbedder);
        async move {
            let _ = serve_http_with_embedder(
                corpus_dir,
                listener,
                embedder,
                None,
                false,
                HttpRerankerConfig::default(),
            )
            .await;
        }
    });

    let client = Client::new();

    // Hybrid query (default mode)
    let resp = client
        .get(format!("http://{addr}/query?q=CVE-2024-1234&top_k=5"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<SearchHitDto> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].chunk_index, 0);

    // Dense-only mode via query param
    let resp = client
        .get(format!(
            "http://{addr}/query?q=anything&top_k=5&mode=dense-only"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<SearchHitDto> = resp.json().await.unwrap();
    assert!(!hits.is_empty());

    server.abort();
}

#[tokio::test]
async fn http_dense_only_server_flag() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn({
        let corpus_dir = dir.path().to_path_buf();
        let embedder = Arc::new(MockEmbedder);
        async move {
            let _ = serve_http_with_embedder(
                corpus_dir,
                listener,
                embedder,
                None,
                true, // dense_only=true
                HttpRerankerConfig::default(),
            )
            .await;
        }
    });

    let client = Client::new();
    let resp = client
        .get(format!("http://{addr}/query?q=Rust&top_k=5"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<SearchHitDto> = resp.json().await.unwrap();
    assert!(
        !hits.is_empty(),
        "dense-only server should still return results"
    );

    server.abort();
}

#[test]
fn latency_breakdown_threaded_through_query_corpus() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let embedder = MockEmbedder;
    let mut breakdown = fastrag::corpus::LatencyBreakdown::default();
    let hits = ops::query_corpus(
        dir.path(),
        "Rust programming",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &mut breakdown,
    )
    .unwrap();

    assert!(!hits.is_empty());
    assert!(breakdown.embed_us > 0, "embed_us should be non-zero");
    assert!(breakdown.hnsw_us > 0, "hnsw_us should be non-zero");
    assert_eq!(
        breakdown.bm25_us, 0,
        "dense-only path should not set bm25_us"
    );
    assert_eq!(
        breakdown.fuse_us, 0,
        "dense-only path should not set fuse_us"
    );
    assert_eq!(
        breakdown.rerank_us, 0,
        "no reranker — rerank_us should be 0"
    );
    assert_eq!(
        breakdown.total_us,
        breakdown.embed_us
            + breakdown.bm25_us
            + breakdown.hnsw_us
            + breakdown.rerank_us
            + breakdown.fuse_us,
        "total_us must equal sum of per-stage fields"
    );
    assert!(breakdown.total_us > 0, "total_us should be non-zero");
}

#[test]
fn latency_breakdown_threaded_through_query_corpus_with_filter() {
    let dir = tempdir().unwrap();
    build_test_corpus(dir.path());

    let embedder = MockEmbedder;
    let filter = BTreeMap::new();
    let mut breakdown = fastrag::corpus::LatencyBreakdown::default();
    let hits = ops::query_corpus_with_filter(
        dir.path(),
        "Rust programming",
        5,
        &embedder as &dyn DynEmbedderTrait,
        &filter,
        &mut breakdown,
    )
    .unwrap();

    assert!(!hits.is_empty());
    assert!(breakdown.embed_us > 0, "embed_us should be non-zero");
    assert!(breakdown.hnsw_us > 0, "hnsw_us should be non-zero");
    assert_eq!(
        breakdown.bm25_us, 0,
        "dense-only path should not set bm25_us"
    );
    assert_eq!(
        breakdown.fuse_us, 0,
        "dense-only path should not set fuse_us"
    );
    assert_eq!(
        breakdown.rerank_us, 0,
        "no reranker — rerank_us should be 0"
    );
    assert_eq!(
        breakdown.total_us,
        breakdown.embed_us
            + breakdown.bm25_us
            + breakdown.hnsw_us
            + breakdown.rerank_us
            + breakdown.fuse_us,
        "total_us must equal sum of per-stage fields"
    );
    assert!(breakdown.total_us > 0, "total_us should be non-zero");
}
