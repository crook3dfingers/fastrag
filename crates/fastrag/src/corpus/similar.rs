//! Similarity-threshold retrieval.
//!
//! Owns `similarity_search`: embed once, fan out per corpus, adaptive overfetch
//! until above-threshold rows are exhausted (or a server cap is hit), merge and
//! sort by raw cosine, truncate to `max_results`.
//!
//! Narrow by design: no hybrid, no temporal decay, no CWE expansion, no rerank.

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use crate::corpus::{CorpusError, LatencyBreakdown, SearchHitDto, scored_ids_to_dtos};
use crate::filter::FilterExpr;
use fastrag_embed::{DynEmbedderTrait, QueryText};

/// Input to `similarity_search`. Caller resolves corpus names to paths and
/// stamps them back on each hit.
#[derive(Debug, Clone)]
pub struct SimilarityRequest {
    pub text: String,
    pub threshold: f32,
    pub max_results: usize,
    /// Resolved `(name, path)` pairs. Non-empty; caller validates.
    pub targets: Vec<(String, PathBuf)>,
    pub filter: Option<FilterExpr>,
    pub snippet_len: usize,
    /// Hard cap on per-corpus overfetch. The adaptive loop stops at this count.
    pub overfetch_cap: usize,
}

/// One hit in the merged result set.
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityHit {
    pub cosine_similarity: f32,
    pub corpus: String,
    #[serde(flatten)]
    pub dto: SearchHitDto,
}

/// Per-corpus diagnostics surfaced in the response.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PerCorpusStats {
    pub candidates_examined: usize,
    pub above_threshold: usize,
}

/// Aggregate diagnostics surfaced in the response.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SimilarityStats {
    pub candidates_examined: usize,
    pub above_threshold: usize,
    pub returned: usize,
    /// Populated only when the request targeted multiple corpora.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub per_corpus: std::collections::BTreeMap<String, PerCorpusStats>,
}

/// Full response body.
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityResponse {
    pub hits: Vec<SimilarityHit>,
    pub truncated: bool,
    pub stats: SimilarityStats,
    pub latency: LatencyBreakdown,
}

/// Run similarity search. Embeds `request.text` once, fans out per target
/// corpus, merges, sorts, truncates to `max_results`.
pub fn similarity_search(
    embedder: &dyn DynEmbedderTrait,
    request: &SimilarityRequest,
) -> Result<SimilarityResponse, CorpusError> {
    let total_start = Instant::now();
    let mut latency = LatencyBreakdown::default();

    // Embed the query ONCE.
    let embed_start = Instant::now();
    let vector = embedder
        .embed_query_dyn(&[QueryText::new(&request.text)])
        .map_err(|e| CorpusError::Embed(e.to_string()))?
        .into_iter()
        .next()
        .ok_or(CorpusError::EmptyEmbeddingOutput)?;
    latency.embed_us = embed_start.elapsed().as_micros() as u64;

    // Fan out per corpus in parallel — embed happened once above.
    use rayon::prelude::*;

    let per: Vec<(String, Result<PerCorpusOutcome, CorpusError>)> = request
        .targets
        .par_iter()
        .map(|(name, path)| {
            let outcome = similarity_search_one(&vector, path, request, embedder);
            (name.clone(), outcome)
        })
        .collect();

    let mut per_corpus: std::collections::BTreeMap<String, PerCorpusStats> =
        std::collections::BTreeMap::new();
    let mut merged_raw: Vec<(String, u64, f32)> = Vec::new();
    let mut any_truncated = false;

    for (name, result) in per {
        let outcome = result?;
        latency.hnsw_us = latency.hnsw_us.saturating_add(outcome.hnsw_us);
        per_corpus.insert(
            name.clone(),
            PerCorpusStats {
                candidates_examined: outcome.candidates_examined,
                above_threshold: outcome.above.len(),
            },
        );
        any_truncated |= outcome.truncated;
        for (id, score) in outcome.above {
            merged_raw.push((name.clone(), id, score));
        }
    }

    // Sort descending by cosine; tie-break on (corpus, id) for determinism.
    merged_raw.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    let above_threshold_total = merged_raw.len();
    merged_raw.truncate(request.max_results);

    // Hydrate survivors via the Store(s) they came from.
    // We re-open each Store once per group to batch-hydrate.
    let mut hits: Vec<SimilarityHit> = Vec::with_capacity(merged_raw.len());
    let mut by_corpus: std::collections::BTreeMap<&String, Vec<(u64, f32)>> =
        std::collections::BTreeMap::new();
    for (name, id, score) in &merged_raw {
        by_corpus.entry(name).or_default().push((*id, *score));
    }
    for (name, scored) in by_corpus {
        let path = request
            .targets
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.clone())
            .expect("resolved corpus must be present in targets");
        let store = fastrag_store::Store::open(&path, embedder)?;
        let snippet_query = if request.snippet_len > 0 {
            Some(request.text.as_str())
        } else {
            None
        };
        let dtos = scored_ids_to_dtos(&store, &scored, snippet_query, request.snippet_len)?;
        for dto in dtos {
            hits.push(SimilarityHit {
                cosine_similarity: dto.score,
                corpus: name.clone(),
                dto,
            });
        }
    }
    // Re-sort after hydration so the output order matches the merged order.
    hits.sort_by(|a, b| {
        b.cosine_similarity
            .partial_cmp(&a.cosine_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.corpus.cmp(&b.corpus))
            .then_with(|| a.dto.chunk_index.cmp(&b.dto.chunk_index))
    });

    let returned = hits.len();
    latency.total_us = total_start.elapsed().as_micros() as u64;

    let stats = SimilarityStats {
        candidates_examined: per_corpus.values().map(|p| p.candidates_examined).sum(),
        above_threshold: above_threshold_total,
        returned,
        per_corpus: if request.targets.len() > 1 {
            per_corpus
        } else {
            std::collections::BTreeMap::new()
        },
    };

    Ok(SimilarityResponse {
        hits,
        truncated: any_truncated,
        stats,
        latency,
    })
}

struct PerCorpusOutcome {
    above: Vec<(u64, f32)>,
    candidates_examined: usize,
    truncated: bool,
    hnsw_us: u64,
}

fn similarity_search_one(
    vector: &[f32],
    corpus_path: &std::path::Path,
    request: &SimilarityRequest,
    embedder: &dyn DynEmbedderTrait,
) -> Result<PerCorpusOutcome, CorpusError> {
    let store = fastrag_store::Store::open(corpus_path, embedder)?;

    let mut fetch_count = request.max_results.saturating_mul(10).max(1);
    let cap = request.overfetch_cap.max(1);
    let mut total_hnsw_us: u64 = 0;

    loop {
        let n = fetch_count.min(cap);
        let hnsw_start = Instant::now();
        let candidates = store.query_dense(vector, n)?;
        total_hnsw_us = total_hnsw_us.saturating_add(hnsw_start.elapsed().as_micros() as u64);
        let candidates_examined = candidates.len();

        let filtered: Vec<(u64, f32)> = match &request.filter {
            Some(expr) => crate::corpus::filter_scored_ids(&store, &candidates, expr)?,
            None => candidates.clone(),
        };

        let above: Vec<(u64, f32)> = filtered
            .iter()
            .filter(|(_, s)| *s >= request.threshold)
            .copied()
            .collect();

        if above.len() >= request.max_results {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: false,
                hnsw_us: total_hnsw_us,
            });
        }
        // Treat "HNSW returned fewer rows than requested" as tail-exhausted:
        // the index has no more candidates to offer, so we are done.
        let exhausted = candidates.len() < n
            || candidates
                .last()
                .is_some_and(|(_, s)| *s < request.threshold);
        if exhausted || candidates.is_empty() {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: false,
                hnsw_us: total_hnsw_us,
            });
        }
        if n >= cap {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: true,
                hnsw_us: total_hnsw_us,
            });
        }
        fetch_count = fetch_count.saturating_mul(2).min(cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_compile() {
        // This test exists to lock the public type shape: every public field
        // is referenced below. Removing or renaming any field breaks compile.
        let stats = SimilarityStats::default();
        let _ = stats.candidates_examined;
        let _ = stats.above_threshold;
        let _ = stats.returned;
        let _ = stats.per_corpus;
        let pc = PerCorpusStats::default();
        let _ = pc.candidates_examined;
        let _ = pc.above_threshold;
    }

    #[cfg(feature = "store")]
    mod fan_out_tests {
        use super::super::*;
        use crate::ChunkingStrategy;
        use crate::ingest::engine::index_jsonl;
        use crate::ingest::jsonl::JsonlIngestConfig;
        use fastrag_embed::test_utils::MockEmbedder;
        use std::collections::BTreeMap;

        fn build_corpus_v2(docs: &[(&str, &str)]) -> (tempfile::TempDir, std::path::PathBuf) {
            let tmp = tempfile::tempdir().unwrap();
            let jsonl = tmp.path().join("docs.jsonl");
            let lines: Vec<String> = docs
                .iter()
                .map(|(id, body)| serde_json::json!({ "id": id, "body": body }).to_string())
                .collect();
            std::fs::write(&jsonl, lines.join("\n")).unwrap();
            let corpus = tmp.path().join("corpus");
            let cfg = JsonlIngestConfig {
                text_fields: vec!["body".into()],
                id_field: "id".into(),
                metadata_fields: vec![],
                metadata_types: BTreeMap::new(),
                array_fields: vec![],
                cwe_field: None,
            };
            index_jsonl(
                &jsonl,
                &corpus,
                &ChunkingStrategy::Basic {
                    max_characters: 500,
                    overlap: 0,
                },
                &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait,
                &cfg,
            )
            .unwrap();
            (tmp, corpus)
        }

        #[test]
        fn fan_out_merges_and_stamps_corpus_on_each_hit() {
            let (_t1, c1) = build_corpus_v2(&[("a1", "alpha"), ("b1", "beta")]);
            let (_t2, c2) = build_corpus_v2(&[("a2", "alpha"), ("b2", "gamma")]);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.95,
                max_results: 10,
                targets: vec![("one".into(), c1), ("two".into(), c2)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 2, "both corpora contribute one hit");
            let corpora: std::collections::BTreeSet<&str> =
                resp.hits.iter().map(|h| h.corpus.as_str()).collect();
            assert!(corpora.contains("one"));
            assert!(corpora.contains("two"));
            // per_corpus populated when targets.len() > 1
            assert_eq!(resp.stats.per_corpus.len(), 2);
            assert!(resp.stats.per_corpus.contains_key("one"));
            assert!(resp.stats.per_corpus.contains_key("two"));
        }

        #[test]
        fn fan_out_embeds_once() {
            use fastrag_embed::{DynEmbedderTrait, EmbedError, PassageText, QueryText};
            use std::sync::Arc;
            use std::sync::atomic::{AtomicUsize, Ordering};

            struct Counting {
                inner: MockEmbedder,
                query_calls: Arc<AtomicUsize>,
            }
            impl DynEmbedderTrait for Counting {
                fn embed_query_dyn(&self, t: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
                    self.query_calls.fetch_add(1, Ordering::SeqCst);
                    self.inner.embed_query_dyn(t)
                }
                fn embed_passage_dyn(
                    &self,
                    t: &[PassageText],
                ) -> Result<Vec<Vec<f32>>, EmbedError> {
                    self.inner.embed_passage_dyn(t)
                }
                fn dim(&self) -> usize {
                    self.inner.dim()
                }
                fn model_id(&self) -> &'static str {
                    self.inner.model_id()
                }
                fn prefix_scheme(&self) -> fastrag_embed::PrefixScheme {
                    self.inner.prefix_scheme()
                }
                fn prefix_scheme_hash(&self) -> u64 {
                    self.inner.prefix_scheme_hash()
                }
                fn default_batch_size(&self) -> usize {
                    self.inner.default_batch_size()
                }
            }

            let (_t1, c1) = build_corpus_v2(&[("a1", "alpha")]);
            let (_t2, c2) = build_corpus_v2(&[("a2", "alpha")]);
            let (_t3, c3) = build_corpus_v2(&[("a3", "alpha")]);
            let query_calls = Arc::new(AtomicUsize::new(0));
            let embedder = Counting {
                inner: MockEmbedder,
                query_calls: query_calls.clone(),
            };
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.5,
                max_results: 10,
                targets: vec![("one".into(), c1), ("two".into(), c2), ("three".into(), c3)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            similarity_search(&embedder, &req).unwrap();
            assert_eq!(
                query_calls.load(Ordering::SeqCst),
                1,
                "similarity_search must embed the query exactly once"
            );
        }

        #[test]
        fn ties_broken_deterministically() {
            // Two corpora, each with doc "alpha". Identical cosine -> tie.
            // Lexicographic tie-break on corpus name: "aaa" before "zzz".
            let (_t1, c1) = build_corpus_v2(&[("x", "alpha")]);
            let (_t2, c2) = build_corpus_v2(&[("x", "alpha")]);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.95,
                max_results: 10,
                targets: vec![("zzz".into(), c2), ("aaa".into(), c1)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 2);
            assert_eq!(resp.hits[0].corpus, "aaa");
            assert_eq!(resp.hits[1].corpus, "zzz");
        }
    }

    #[cfg(feature = "store")]
    mod single_corpus_tests {
        use super::*;
        use crate::ChunkingStrategy;
        use crate::ingest::engine::index_jsonl;
        use crate::ingest::jsonl::JsonlIngestConfig;
        use fastrag_embed::test_utils::MockEmbedder;
        use std::collections::BTreeMap;

        fn build_corpus(docs: &[(&str, &str)]) -> (tempfile::TempDir, std::path::PathBuf) {
            let tmp = tempfile::tempdir().unwrap();
            let jsonl = tmp.path().join("docs.jsonl");
            let lines: Vec<String> = docs
                .iter()
                .map(|(id, body)| serde_json::json!({ "id": id, "body": body }).to_string())
                .collect();
            std::fs::write(&jsonl, lines.join("\n")).unwrap();
            let corpus = tmp.path().join("corpus");
            let cfg = JsonlIngestConfig {
                text_fields: vec!["body".into()],
                id_field: "id".into(),
                metadata_fields: vec![],
                metadata_types: BTreeMap::new(),
                array_fields: vec![],
                cwe_field: None,
            };
            index_jsonl(
                &jsonl,
                &corpus,
                &ChunkingStrategy::Basic {
                    max_characters: 500,
                    overlap: 0,
                },
                &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait,
                &cfg,
            )
            .unwrap();
            (tmp, corpus)
        }

        #[test]
        fn threshold_filters_below_cutoff() {
            // Query matches doc "alpha" exactly -> cosine ~ 1.0.
            // Unrelated docs have much lower cosine under MockEmbedder.
            let (_t, corpus) =
                build_corpus(&[("a", "alpha"), ("b", "xyzzy plover"), ("c", "quux frob")]);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.95,
                max_results: 10,
                targets: vec![("default".into(), corpus)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 1, "only 'a' should pass 0.95 threshold");
            assert_eq!(resp.hits[0].corpus, "default");
            assert!(resp.hits[0].cosine_similarity >= 0.95);
            assert_eq!(resp.stats.above_threshold, 1);
            assert_eq!(resp.stats.returned, 1);
            assert!(!resp.truncated);
        }

        #[test]
        fn max_results_caps_above_threshold() {
            // 10 identical docs -> all cosine ~ 1.0 -> all above any sensible
            // threshold. max_results=3 caps output.
            let docs: Vec<(String, String)> = (0..10)
                .map(|i| (format!("d{i}"), "alpha".to_string()))
                .collect();
            let docs_ref: Vec<(&str, &str)> =
                docs.iter().map(|(i, b)| (i.as_str(), b.as_str())).collect();
            let (_t, corpus) = build_corpus(&docs_ref);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.5,
                max_results: 3,
                targets: vec![("default".into(), corpus)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 3);
            assert_eq!(resp.stats.returned, 3);
            assert!(resp.stats.above_threshold >= 3);
            assert!(!resp.truncated);
        }

        #[test]
        fn adaptive_overfetch_returns_all_above_threshold_when_tail_exhausted() {
            // 5 matches + 20 misses, max_results=10. Initial overfetch = 10*10 = 100
            // already pulls everything (but the cap-hit/tail-below logic must not trip).
            // We want to verify "above.len() < max_results" AND "tail below" fires,
            // returning all 5 without truncation.
            let mut docs: Vec<(String, String)> = Vec::new();
            for i in 0..5 {
                docs.push((format!("hit{i}"), "alpha".to_string()));
            }
            for i in 0..20 {
                docs.push((format!("miss{i}"), format!("zzz_{i}_unrelated")));
            }
            let refs: Vec<(&str, &str)> =
                docs.iter().map(|(i, b)| (i.as_str(), b.as_str())).collect();
            let (_t, corpus) = build_corpus(&refs);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.95,
                max_results: 10, // ask for more than exist
                targets: vec![("default".into(), corpus)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            // Must return exactly 5 matches, not truncated.
            assert_eq!(resp.hits.len(), 5, "all 5 matching docs should be returned");
            assert_eq!(resp.stats.above_threshold, 5);
            assert!(!resp.truncated);
        }

        #[test]
        fn small_corpus_fully_above_threshold_is_not_truncated() {
            // 3 matching docs, max_results=10, cap large. The loop must detect
            // HNSW-returned-fewer-than-requested and bail with truncated=false.
            let (_t, corpus) = build_corpus(&[("a", "alpha"), ("b", "alpha"), ("c", "alpha")]);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.5,
                max_results: 10,
                targets: vec![("default".into(), corpus)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 3);
            assert_eq!(resp.stats.returned, 3);
            assert!(
                !resp.truncated,
                "exhausted corpus must not trigger truncated"
            );
        }

        #[test]
        fn truncated_flag_set_when_cap_hit() {
            // 20 identical-matching docs, tiny cap -> every fetch keeps returning
            // above-threshold tail. Cap terminates the loop -> truncated=true.
            let docs: Vec<(String, String)> = (0..20)
                .map(|i| (format!("d{i}"), "alpha".to_string()))
                .collect();
            let refs: Vec<(&str, &str)> =
                docs.iter().map(|(i, b)| (i.as_str(), b.as_str())).collect();
            let (_t, corpus) = build_corpus(&refs);
            let req = SimilarityRequest {
                text: "alpha".into(),
                threshold: 0.5,
                max_results: 100, // asks for 100; only 20 exist; adaptive loop chases
                targets: vec![("default".into(), corpus)],
                filter: None,
                snippet_len: 0,
                overfetch_cap: 5, // cap below corpus size
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert!(resp.truncated, "cap should trip the truncated flag");
            assert!(resp.hits.len() <= 5, "at most cap rows ever fetched");
        }

        #[test]
        fn filter_applied_before_threshold() {
            // 2 high-scoring docs tagged "drop" (body matches query exactly),
            // 2 lower-scoring docs tagged "keep" (body partially matches query).
            // With max_results=2 and overfetch_cap=4 we fetch all 4 candidates.
            // A threshold-first-then-filter implementation would pick the top 2
            // (both "drop") and the filter would evict them, yielding 0 results.
            // The correct filter-before-threshold implementation keeps both "keep"
            // rows because they still clear the 0.3 threshold.
            let tmp = tempfile::tempdir().unwrap();
            let jsonl = tmp.path().join("docs.jsonl");
            std::fs::write(
                &jsonl,
                concat!(
                    // High-scoring "drop" rows — body identical to query "alpha".
                    r#"{"id":"a","body":"alpha","tag":"drop"}"#,
                    "\n",
                    r#"{"id":"b","body":"alpha","tag":"drop"}"#,
                    "\n",
                    // Lower-scoring "keep" rows — body shares "alpha" trigrams but
                    // adds unrelated tokens, reducing cosine similarity.
                    r#"{"id":"c","body":"alpha quux zork","tag":"keep"}"#,
                    "\n",
                    r#"{"id":"d","body":"alpha quux zork","tag":"keep"}"#,
                ),
            )
            .unwrap();
            let corpus = tmp.path().join("corpus");
            let cfg = JsonlIngestConfig {
                text_fields: vec!["body".into()],
                id_field: "id".into(),
                metadata_fields: vec!["tag".into()],
                metadata_types: BTreeMap::from([(
                    "tag".into(),
                    fastrag_store::schema::TypedKind::String,
                )]),
                array_fields: vec![],
                cwe_field: None,
            };
            index_jsonl(
                &jsonl,
                &corpus,
                &ChunkingStrategy::Basic {
                    max_characters: 500,
                    overlap: 0,
                },
                &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait,
                &cfg,
            )
            .unwrap();

            let req = SimilarityRequest {
                text: "alpha".into(),
                // 0.3 is loose enough to admit "alpha quux zork" rows but tight
                // enough to exclude truly unrelated content.
                threshold: 0.3,
                // max_results=2: a wrong threshold-first impl would fill this
                // with the 2 "drop" rows before the filter can evict them.
                max_results: 2,
                targets: vec![("default".into(), corpus)],
                filter: Some(FilterExpr::Eq {
                    field: "tag".into(),
                    value: fastrag_store::schema::TypedValue::String("keep".into()),
                }),
                snippet_len: 0,
                overfetch_cap: 10_000,
            };
            let resp = similarity_search(&MockEmbedder, &req).unwrap();
            assert_eq!(resp.hits.len(), 2, "only tag=keep rows should survive");
            for hit in &resp.hits {
                let tag = hit.dto.metadata.get("tag");
                assert!(
                    matches!(tag, Some(fastrag_store::schema::TypedValue::String(s)) if s == "keep"),
                    "hit metadata should carry tag=keep: {:?}",
                    tag,
                );
            }
        }
    }
}
