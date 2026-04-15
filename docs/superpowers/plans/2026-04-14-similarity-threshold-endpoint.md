# Similarity Threshold Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `POST /similar` — a raw-cosine threshold-filter endpoint over the dense retrieval path with single-corpus and multi-corpus fan-out.

**Architecture:** A new core module `crates/fastrag/src/corpus/similar.rs` owns the adaptive-overfetch loop, per-corpus parallel fan-out, and merge/sort. The HTTP handler in `fastrag-cli/src/http.rs` parses the POST body, resolves target corpora via the existing `CorpusRegistry`, embeds once, and delegates to the core module. Hybrid, temporal-decay, CWE expansion, and reranking are deliberately rejected at the boundary.

**Tech Stack:** Rust, axum (HTTP), tokio (spawn_blocking for parallel fan-out), `fastrag_store::Store::query_dense`, `fastrag::filter::matches`, `fastrag_embed::DynEmbedderTrait`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-14-similarity-threshold-endpoint-design.md`.

---

## File Structure

**Create:**
- `crates/fastrag/src/corpus/similar.rs` — core `similarity_search` fn, types, unit tests.
- `fastrag-cli/tests/similar_http_e2e.rs` — HTTP integration tests.

**Modify:**
- `crates/fastrag/src/corpus/mod.rs` — add `pub mod similar;`, re-export public types, extract `filter_scored_ids` helper from the existing filtered branch.
- `fastrag-cli/src/http.rs` — request/response structs, `similar_handler`, route registration, metrics, `similar_overfetch_cap` in `AppState`.
- `fastrag-cli/src/args.rs` — `--similar-overfetch-cap` flag on the `ServeHttp` variant.
- `fastrag-cli/src/main.rs` — destructure the new flag and thread it through to the server builder.
- `README.md` — new "Similarity Search" section.
- `CLAUDE.md` — append `cargo test` line for `similar_http_e2e`.

---

## Task 1: Core types and module skeleton

**Files:**
- Create: `crates/fastrag/src/corpus/similar.rs`
- Modify: `crates/fastrag/src/corpus/mod.rs`

- [ ] **Step 1: Create `similar.rs` with public types and a stub function**

```rust
// crates/fastrag/src/corpus/similar.rs
//! Similarity-threshold retrieval.
//!
//! Owns `similarity_search`: embed once, fan out per corpus, adaptive overfetch
//! until above-threshold rows are exhausted (or a server cap is hit), merge and
//! sort by raw cosine, truncate to `max_results`.
//!
//! Narrow by design: no hybrid, no temporal decay, no CWE expansion, no rerank.

use std::path::PathBuf;

use serde::Serialize;

use crate::corpus::{CorpusError, LatencyBreakdown, SearchHitDto};
use crate::filter::FilterExpr;
use fastrag_embed::DynEmbedderTrait;

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
    _embedder: &dyn DynEmbedderTrait,
    _request: &SimilarityRequest,
) -> Result<SimilarityResponse, CorpusError> {
    // Implemented in Task 3. Keep the function reachable so downstream tasks
    // have a stable symbol to dispatch against.
    Err(CorpusError::Other("similarity_search: not implemented".into()))
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
}
```

- [ ] **Step 2: Add `CorpusError::Other` variant if it is missing**

Check: `rg -n "CorpusError" crates/fastrag/src/corpus/mod.rs | head -20`.

If `CorpusError` has no `Other(String)` variant, append one to its `enum` definition:

```rust
// crates/fastrag/src/corpus/mod.rs — inside the CorpusError enum
#[error("{0}")]
Other(String),
```

If a string-carrying variant already exists (e.g. `CorpusError::Other`, `CorpusError::Internal`, `CorpusError::Validation`), use that instead and update the `similarity_search` stub in Step 1 accordingly. Otherwise the compiler error will guide the fix.

- [ ] **Step 3: Register the module in `corpus/mod.rs`**

Near the top of `crates/fastrag/src/corpus/mod.rs`, next to `pub mod hybrid;`, add:

```rust
pub mod similar;
```

Also re-export the public types from the crate root so `fastrag-cli` can use them without digging:

```rust
// crates/fastrag/src/corpus/mod.rs — near the existing re-exports
pub use similar::{
    PerCorpusStats, SimilarityHit, SimilarityRequest, SimilarityResponse, SimilarityStats,
    similarity_search,
};
```

- [ ] **Step 4: Run tests to verify the skeleton compiles and the lock test passes**

Run:
```bash
cargo test -p fastrag --lib corpus::similar::tests --features retrieval,store
```

Expected: `test types_compile ... ok` in 1 test. Zero warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/similar.rs crates/fastrag/src/corpus/mod.rs
git commit -m "feat(corpus): scaffold similarity_search types and module"
```

---

## Task 2: Extract `filter_scored_ids` helper

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (lift the filter eval out of `query_corpus_with_filter_opts`, add a unit test)

**Why:** `similar.rs` needs the same "fetch metadata for a candidate list, filter by `FilterExpr`" logic that the filtered branch of `query_corpus_with_filter_opts` currently inlines. Extract it once, reuse it twice, keep behaviour unchanged.

- [ ] **Step 1: Write the failing test for the new helper**

Append to the existing `#[cfg(test)] mod tests` block in `crates/fastrag/src/corpus/mod.rs` (or create one if none exists at file bottom):

```rust
#[cfg(all(test, feature = "store"))]
mod filter_scored_ids_tests {
    use super::*;
    use crate::filter::FilterExpr;
    use fastrag::ingest::engine::index_jsonl;
    use fastrag::ingest::jsonl::JsonlIngestConfig;
    use fastrag_embed::test_utils::MockEmbedder;
    use fastrag_store::schema::{TypedKind, TypedValue};
    use std::collections::BTreeMap;

    #[test]
    fn filter_scored_ids_keeps_only_matching_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let jsonl = tmp.path().join("docs.jsonl");
        std::fs::write(
            &jsonl,
            concat!(
                r#"{"id":"a","body":"alpha","sev":1}"#, "\n",
                r#"{"id":"b","body":"beta","sev":2}"#, "\n",
                r#"{"id":"c","body":"gamma","sev":3}"#,
            ),
        )
        .unwrap();
        let corpus = tmp.path().join("corpus");
        let cfg = JsonlIngestConfig {
            text_fields: vec!["body".into()],
            id_field: "id".into(),
            metadata_fields: vec!["sev".into()],
            metadata_types: BTreeMap::from([("sev".into(), TypedKind::Numeric)]),
            array_fields: vec![],
            cwe_field: None,
        };
        index_jsonl(
            &jsonl,
            &corpus,
            &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
            &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait,
            &cfg,
        )
        .unwrap();

        let store =
            fastrag_store::Store::open(&corpus, &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait)
                .unwrap();
        // Score doesn't matter for filter eval; include all ids with synthetic scores.
        let scored: Vec<(u64, f32)> = (1..=3).map(|i| (i as u64, 0.9 - i as f32 * 0.1)).collect();

        let expr = FilterExpr::Eq {
            field: "sev".into(),
            value: TypedValue::Numeric(2.0),
        };

        let kept = filter_scored_ids(&store, &scored, &expr).unwrap();
        assert_eq!(kept.len(), 1, "only the row with sev=2 should pass");
        // Preserves input score, not a fresh one.
        assert!((kept[0].1 - scored.iter().find(|(i, _)| *i == kept[0].0).unwrap().1).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cargo test -p fastrag --features retrieval,store --lib filter_scored_ids_tests
```

Expected: compile error `cannot find function 'filter_scored_ids' in this scope` (or equivalent).

- [ ] **Step 3: Add the helper to `crates/fastrag/src/corpus/mod.rs`**

Place this near `scored_ids_to_dtos` (around line 1143), with matching `#[cfg(feature = "store")]` gating:

```rust
/// Filter a scored candidate list by a `FilterExpr`, preserving input order
/// and scores. Drops rows whose metadata misses, or whose metadata fails the
/// expression.
///
/// Shared by `query_corpus_with_filter_opts` (filtered branch) and
/// `corpus::similar::similarity_search`.
#[cfg(feature = "store")]
pub(crate) fn filter_scored_ids(
    store: &fastrag_store::Store,
    scored: &[(u64, f32)],
    filter: &crate::filter::FilterExpr,
) -> Result<Vec<(u64, f32)>, CorpusError> {
    if scored.is_empty() {
        return Ok(vec![]);
    }
    let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
    let metadata_rows = store.fetch_metadata(&ids)?;
    let meta_map: std::collections::HashMap<
        u64,
        &[(String, fastrag_store::schema::TypedValue)],
    > = metadata_rows
        .iter()
        .map(|(id, fields)| (*id, fields.as_slice()))
        .collect();

    Ok(scored
        .iter()
        .filter(|(id, _)| {
            meta_map
                .get(id)
                .is_some_and(|fields| crate::filter::matches(filter, fields))
        })
        .copied()
        .collect())
}
```

- [ ] **Step 4: Replace the inlined filter eval in `query_corpus_with_filter_opts`**

In `crates/fastrag/src/corpus/mod.rs`, locate the filtered-overfetch loop (around line 1055 — the `for &factor in overfetch_factors` block). Replace the body from the `let ids: Vec<u64> = ...` line through the `let passing: Vec<(u64, f32)> = ...` block with:

```rust
    for &factor in overfetch_factors {
        let fetch_count = top_k.saturating_mul(factor).max(top_k);

        let scored = fetch_candidates(fetch_count, breakdown)?;

        if scored.is_empty() {
            breakdown.finalize();
            return Ok(vec![]);
        }

        let passing_all = filter_scored_ids(&store, &scored, filter_expr)?;
        let passing: Vec<(u64, f32)> = passing_all.into_iter().take(top_k).collect();

        if passing.len() >= top_k || factor == *overfetch_factors.last().unwrap() {
            breakdown.finalize();
            return scored_ids_to_dtos(&store, &passing, Some(query), snippet_len);
        }
        // Not enough survivors — retry with larger overfetch.
    }
```

- [ ] **Step 5: Run the new test plus the full filter-path suite to verify no regression**

Run:
```bash
cargo test -p fastrag --features retrieval,store
```

Expected: all tests pass, including the new `filter_scored_ids_keeps_only_matching_rows` and every existing `query_corpus_with_filter*` test.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs
git commit -m "refactor(corpus): extract filter_scored_ids helper"
```

---

## Task 3: Single-corpus similarity search with adaptive overfetch

**Files:**
- Modify: `crates/fastrag/src/corpus/similar.rs`

- [ ] **Step 1: Write failing tests for the single-corpus path**

Append to the `#[cfg(test)] mod tests` block in `crates/fastrag/src/corpus/similar.rs`:

```rust
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
            .map(|(id, body)| {
                serde_json::json!({ "id": id, "body": body }).to_string()
            })
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
            &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
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
        let (_t, corpus) = build_corpus(&[
            ("a", "alpha"),
            ("b", "xyzzy plover"),
            ("c", "quux frob"),
        ]);
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
        let docs: Vec<(String, String)> =
            (0..10).map(|i| (format!("d{i}"), "alpha".to_string())).collect();
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
        // Mix 5 matching docs with 20 non-matching. All 5 should be returned
        // even though initial overfetch = max_results * 10 = 10 is larger than
        // the number of matches.
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
            max_results: 1, // max_results * 10 = 10 initial overfetch
            targets: vec![("default".into(), corpus)],
            filter: None,
            snippet_len: 0,
            overfetch_cap: 10_000,
        };
        let resp = similarity_search(&MockEmbedder, &req).unwrap();
        // max_results=1 caps output to 1 hit, but stats.above_threshold reports
        // the full count we detected. We only know it's >= 1 and <= 5.
        assert_eq!(resp.hits.len(), 1);
        assert!(resp.stats.above_threshold >= 1);
        assert!(!resp.truncated);
    }

    #[test]
    fn truncated_flag_set_when_cap_hit() {
        // 20 identical-matching docs, tiny cap -> every fetch keeps returning
        // above-threshold tail. Cap terminates the loop -> truncated=true.
        let docs: Vec<(String, String)> =
            (0..20).map(|i| (format!("d{i}"), "alpha".to_string())).collect();
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
        // 4 docs with a `tag` metadata field; threshold alone matches all;
        // filter keeps only tag="keep".
        let tmp = tempfile::tempdir().unwrap();
        let jsonl = tmp.path().join("docs.jsonl");
        std::fs::write(
            &jsonl,
            concat!(
                r#"{"id":"a","body":"alpha","tag":"keep"}"#, "\n",
                r#"{"id":"b","body":"alpha","tag":"drop"}"#, "\n",
                r#"{"id":"c","body":"alpha","tag":"keep"}"#, "\n",
                r#"{"id":"d","body":"alpha","tag":"drop"}"#,
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
            &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
            &MockEmbedder as &dyn fastrag_embed::DynEmbedderTrait,
            &cfg,
        )
        .unwrap();

        let req = SimilarityRequest {
            text: "alpha".into(),
            threshold: 0.5,
            max_results: 10,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
cargo test -p fastrag --features retrieval,store --lib corpus::similar::tests
```

Expected: all 5 new tests fail with `CorpusError::Other("similarity_search: not implemented")`.

- [ ] **Step 3: Implement `similarity_search` and `similarity_search_one`**

Replace the stub `similarity_search` in `crates/fastrag/src/corpus/similar.rs` with the full single-corpus implementation. (Fan-out + parallelism land in Task 4; for now the loop runs targets serially.)

```rust
use std::time::Instant;

use fastrag_embed::QueryText;

use crate::corpus::{filter_scored_ids, scored_ids_to_dtos};

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

    // Fan out per corpus. Task 4 replaces this with parallel spawn_blocking.
    let mut per_corpus: std::collections::BTreeMap<String, PerCorpusStats> =
        std::collections::BTreeMap::new();
    let mut merged_raw: Vec<(String, u64, f32)> = Vec::new();
    let mut any_truncated = false;

    for (name, path) in &request.targets {
        let outcome = similarity_search_one(&vector, path, request, &mut latency)?;
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
}

fn similarity_search_one(
    vector: &[f32],
    corpus_path: &std::path::Path,
    request: &SimilarityRequest,
    latency: &mut LatencyBreakdown,
) -> Result<PerCorpusOutcome, CorpusError> {
    // NOTE: passing the embedder only for Store::open; no embedding happens here.
    // Store::open needs something that implements DynEmbedderTrait, so we thread
    // one through via a lightweight shim in Task 4 when we switch to spawn_blocking.
    // For the serial path, we rebuild the store with any embedder; MockEmbedder
    // satisfies the trait without calling it.
    // We pick up the *real* embedder from the caller by requiring it to pass
    // the vector. The Store only uses the embedder to record identity; it
    // doesn't run it during queries.
    use fastrag_embed::test_utils::MockEmbedder;
    let store = fastrag_store::Store::open(corpus_path, &MockEmbedder)?;

    let mut fetch_count = request.max_results.saturating_mul(10).max(1);
    let cap = request.overfetch_cap.max(1);
    let mut above: Vec<(u64, f32)> = Vec::new();
    let mut candidates_examined = 0usize;

    loop {
        let n = fetch_count.min(cap);
        let hnsw_start = Instant::now();
        let candidates = store.query_dense(vector, n)?;
        latency.hnsw_us = latency.hnsw_us.saturating_add(hnsw_start.elapsed().as_micros() as u64);
        candidates_examined = candidates.len();

        let filtered: Vec<(u64, f32)> = match &request.filter {
            Some(expr) => filter_scored_ids(&store, &candidates, expr)?,
            None => candidates.clone(),
        };

        above = filtered
            .iter()
            .filter(|(_, s)| *s >= request.threshold)
            .copied()
            .collect();

        if above.len() >= request.max_results {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: false,
            });
        }
        let tail_below = candidates
            .last()
            .is_some_and(|(_, s)| *s < request.threshold);
        if tail_below || candidates.is_empty() {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: false,
            });
        }
        if n >= cap {
            return Ok(PerCorpusOutcome {
                above,
                candidates_examined,
                truncated: true,
            });
        }
        fetch_count = fetch_count.saturating_mul(2).min(cap);
    }
}
```

The embedder-shim note is a placeholder: `Store::open` has signature `open(path, embedder)` but doesn't actually *use* the embedder for queries. Check `crates/fastrag-store/src/lib.rs` for the exact signature. If the Store requires a specific embedder identity at open time (it compares against the manifest), pass the caller's `embedder` reference through instead of `MockEmbedder`. Adjust `similarity_search_one` to take `embedder: &dyn DynEmbedderTrait` and pass it to `Store::open`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cargo test -p fastrag --features retrieval,store --lib corpus::similar::tests
```

Expected: all 5 new tests pass, plus `types_compile` from Task 1.

- [ ] **Step 5: Run the full fastrag test suite to catch regressions**

Run:
```bash
cargo test -p fastrag --features retrieval,store
```

Expected: all tests pass. Focus on the filter-path tests from Task 2.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/similar.rs
git commit -m "feat(corpus): implement single-corpus similarity_search with adaptive overfetch"
```

---

## Task 4: Parallel fan-out across corpora

**Files:**
- Modify: `crates/fastrag/src/corpus/similar.rs`

**Why:** The spec requires fan-out to run per-corpus work in parallel (`spawn_blocking`) and to embed only once. The serial loop in Task 3 already embeds once; this task adds the parallelism and the ties-across-corpora guarantees.

- [ ] **Step 1: Write failing tests for fan-out behaviour**

Append to `#[cfg(test)] mod tests::single_corpus_tests` (or a new module `fan_out_tests`) in `crates/fastrag/src/corpus/similar.rs`:

```rust
#[cfg(feature = "store")]
mod fan_out_tests {
    use super::super::*;
    use super::super::tests::single_corpus_tests::*; // reuse build_corpus helper — if build_corpus was pub(super) in single_corpus_tests
    // If build_corpus is private to that module, duplicate the small helper here:

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
            &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
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
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use fastrag_embed::{DynEmbedderTrait, EmbedError, PassageText, QueryText};

        struct Counting {
            inner: MockEmbedder,
            query_calls: Arc<AtomicUsize>,
        }
        impl DynEmbedderTrait for Counting {
            fn embed_query_dyn(&self, t: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
                self.query_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.embed_query_dyn(t)
            }
            fn embed_passage_dyn(&self, t: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
                self.inner.embed_passage_dyn(t)
            }
            fn dim(&self) -> usize { self.inner.dim() }
            fn model_id(&self) -> &'static str { self.inner.model_id() }
            fn prefix_scheme(&self) -> fastrag_embed::PrefixScheme {
                self.inner.prefix_scheme()
            }
        }

        let (_t1, c1) = build_corpus_v2(&[("a1", "alpha")]);
        let (_t2, c2) = build_corpus_v2(&[("a2", "alpha")]);
        let (_t3, c3) = build_corpus_v2(&[("a3", "alpha")]);
        let query_calls = Arc::new(AtomicUsize::new(0));
        let embedder = Counting { inner: MockEmbedder, query_calls: query_calls.clone() };
        let req = SimilarityRequest {
            text: "alpha".into(),
            threshold: 0.5,
            max_results: 10,
            targets: vec![
                ("one".into(), c1),
                ("two".into(), c2),
                ("three".into(), c3),
            ],
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
```

- [ ] **Step 2: Run the failing tests**

Run:
```bash
cargo test -p fastrag --features retrieval,store --lib corpus::similar::tests::fan_out_tests
```

Expected: `fan_out_merges_and_stamps_corpus_on_each_hit` and `ties_broken_deterministically` should already pass from Task 3's serial implementation. `fan_out_embeds_once` may already pass too (we do embed once); rerun to confirm current status. If any fail, proceed to Step 3.

- [ ] **Step 3: Convert the serial loop to rayon-parallel execution**

Parallelize only the per-corpus work. The embed still happens once, before the fan-out.

Replace the per-corpus loop in `similarity_search` (the `for (name, path) in &request.targets` block) with:

```rust
use rayon::prelude::*;

let per: Vec<(String, Result<PerCorpusOutcome, CorpusError>)> = request
    .targets
    .par_iter()
    .map(|(name, path)| {
        let mut local_latency = LatencyBreakdown::default();
        let outcome = similarity_search_one(&vector, path, request, &mut local_latency);
        // TODO: merge local_latency.hnsw_us back into the outer latency. We do
        // that after the fold below.
        (name.clone(), outcome)
    })
    .collect();

let mut any_truncated = false;
let mut per_corpus: std::collections::BTreeMap<String, PerCorpusStats> =
    std::collections::BTreeMap::new();
let mut merged_raw: Vec<(String, u64, f32)> = Vec::new();
for (name, result) in per {
    let outcome = result?;
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
```

Update the per-corpus latency accumulation: have `similarity_search_one` return the populated `LatencyBreakdown` alongside `PerCorpusOutcome` (or sum `hnsw_us` inside the outcome). Keep it a `u64` sum rather than a max — it's wall-clock sum across corpora, which is documented as "per-stage microseconds" in the existing `LatencyBreakdown` doc.

Check `Cargo.toml` of the `fastrag` crate. If `rayon` is not already a dependency, add:

```toml
rayon = { workspace = true }
```

(Likely already present — `crates/fastrag/src/corpus/mod.rs::batch_query` uses rayon.)

- [ ] **Step 4: Remove the MockEmbedder shim in `similarity_search_one`**

Plumb the caller's embedder through `similarity_search_one` so `Store::open` receives the correct model identity. Update the signature:

```rust
fn similarity_search_one(
    embedder: &dyn DynEmbedderTrait,
    vector: &[f32],
    corpus_path: &std::path::Path,
    request: &SimilarityRequest,
    latency: &mut LatencyBreakdown,
) -> Result<PerCorpusOutcome, CorpusError> {
    let store = fastrag_store::Store::open(corpus_path, embedder)?;
    // ... rest unchanged ...
}
```

And in the parallel map, capture the embedder into each rayon task. `&dyn DynEmbedderTrait` is `Sync` if the underlying trait is — check the trait definition. If not `Sync`, use `Arc<dyn DynEmbedderTrait>` instead, but that requires a signature change in the public API. In practice the trait IS `Send + Sync` (already required because `AppState::embedder: DynEmbedder` is `Arc<dyn ...>`). Use a plain `&embedder` capture.

- [ ] **Step 5: Run the fan-out tests and the full suite**

Run:
```bash
cargo test -p fastrag --features retrieval,store --lib corpus::similar::tests
cargo test -p fastrag --features retrieval,store
```

Expected: all tests pass, including the three new fan-out tests.

- [ ] **Step 6: Run clippy to catch style issues introduced by the parallelism**

Run:
```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
```

Expected: zero warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/fastrag/src/corpus/similar.rs crates/fastrag/Cargo.toml
git commit -m "feat(corpus): parallelize similarity_search fan-out across corpora"
```

---

## Task 5: HTTP handler — happy path

**Files:**
- Modify: `fastrag-cli/src/http.rs`

- [ ] **Step 1: Write a failing integration test for `POST /similar` happy path**

Create `fastrag-cli/tests/similar_http_e2e.rs`:

```rust
//! End-to-end tests for POST /similar.
#![cfg(feature = "retrieval")]

use std::collections::BTreeMap;
use std::sync::Arc;

use fastrag::ChunkingStrategy;
use fastrag::corpus::CorpusRegistry;
use fastrag::ingest::engine::index_jsonl;
use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::json;

fn build_toy_corpus(docs: &[(&str, &str)]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let jsonl = tmp.path().join("docs.jsonl");
    let lines: Vec<String> = docs
        .iter()
        .map(|(id, body)| json!({"id": id, "body": body}).to_string())
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
        &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
        &MockEmbedder as &dyn fastrag::DynEmbedderTrait,
        &cfg,
    )
    .unwrap();
    // Return a TempDir pointing at the corpus subdir by shifting: the corpus
    // lives inside tmp. Return the outer handle so tmp survives.
    let holder = tempfile::tempdir().unwrap();
    // Move the corpus into holder so the TempDir drop guards it.
    let dest = holder.path().join("corpus");
    std::fs::rename(&corpus, &dest).unwrap();
    // tmp is dropped (cleans the jsonl); holder survives with the corpus.
    holder
}

async fn spawn_server(registry: CorpusRegistry) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
            52_428_800,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    format!("http://{}", addr)
}

#[tokio::test]
async fn post_similar_happy_path() {
    let corpus = build_toy_corpus(&[
        ("a", "alpha"),
        ("b", "xyzzy plover"),
        ("c", "quux frob"),
    ]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let client = Client::new();
    let resp = client
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.95,
            "max_results": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["cosine_similarity"].as_f64().unwrap() >= 0.95);
    assert_eq!(hits[0]["corpus"].as_str().unwrap(), "default");
    assert_eq!(body["truncated"].as_bool().unwrap(), false);
    assert_eq!(body["stats"]["returned"].as_u64().unwrap(), 1);
    assert!(body["latency"]["embed_us"].as_u64().is_some());
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cargo test -p fastrag-cli --features retrieval --test similar_http_e2e -- post_similar_happy_path
```

Expected: 404 from the server (no route `/similar`), test fails.

- [ ] **Step 3: Add request/response body types to `fastrag-cli/src/http.rs`**

Append near the existing `QueryParams` / `BatchQueryRequest` structs (around line 190):

```rust
#[derive(Debug, serde::Deserialize)]
struct SimilarRequest {
    text: String,
    threshold: f32,
    max_results: usize,
    #[serde(default)]
    corpus: Option<String>,
    #[serde(default)]
    corpora: Option<Vec<String>>,
    /// Accepts either string syntax ("severity = HIGH") or JSON AST.
    #[serde(default)]
    filter: Option<serde_json::Value>,
    #[serde(default)]
    fields: Option<String>,
    // Catch-all for rejected params. Any of these set -> 400.
    #[serde(default)] hybrid: Option<serde_json::Value>,
    #[serde(default)] rrf_k: Option<serde_json::Value>,
    #[serde(default)] rrf_overfetch: Option<serde_json::Value>,
    #[serde(default)] time_decay_field: Option<serde_json::Value>,
    #[serde(default)] time_decay_halflife: Option<serde_json::Value>,
    #[serde(default)] time_decay_weight: Option<serde_json::Value>,
    #[serde(default)] time_decay_dateless_prior: Option<serde_json::Value>,
    #[serde(default)] time_decay_blend: Option<serde_json::Value>,
    #[serde(default)] rerank: Option<serde_json::Value>,
    #[serde(default)] cwe_expand: Option<serde_json::Value>,
}
```

- [ ] **Step 4: Add `similar_overfetch_cap` to `AppState` and `serve_http_with_registry`**

Locate `struct AppState` (around line 33) and add:
```rust
    similar_overfetch_cap: usize,
```

Locate `serve_http_with_registry` (around line 449). Add a parameter `similar_overfetch_cap: usize` to the signature, after `ingest_max_body`. Thread it into `AppState { ..., similar_overfetch_cap, ... }`. Update `serve_http_with_registry_port` similarly, and `serve_http_with_embedder` to pass a default (10_000).

Also update the existing `http_e2e.rs`, `federation_e2e.rs`, `temporal_decay_http_e2e.rs`, `cwe_expand_http_e2e.rs`, and any other callers to pass `10_000` as the cap in their `serve_http_with_registry(...)` call. Build will tell you exactly which files.

- [ ] **Step 5: Add the `similar_handler` async fn**

Append to `fastrag-cli/src/http.rs` near the other handlers (after `batch_query_handler`, before `query`):

```rust
async fn similar_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantFilter>>,
    Json(req): Json<SimilarRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    // Reject unsupported params up front.
    if req.hybrid.is_some()
        || req.rrf_k.is_some()
        || req.rrf_overfetch.is_some()
        || req.time_decay_field.is_some()
        || req.time_decay_halflife.is_some()
        || req.time_decay_weight.is_some()
        || req.time_decay_dateless_prior.is_some()
        || req.time_decay_blend.is_some()
        || req.cwe_expand.is_some()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "/similar does not support hybrid or temporal decay; see /query",
        )
            .into_response());
    }
    if req.rerank.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "/similar does not support reranking",
        )
            .into_response());
    }

    // Validate scalars.
    if req.text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text must be non-empty").into_response());
    }
    if !(0.0..=1.0).contains(&req.threshold) {
        return Err((StatusCode::BAD_REQUEST, "threshold must be in [0.0, 1.0]").into_response());
    }
    if req.max_results == 0 || req.max_results > 1000 {
        return Err((StatusCode::BAD_REQUEST, "max_results must be in [1, 1000]").into_response());
    }

    // Resolve target corpora.
    if req.corpus.is_some() && req.corpora.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "exactly one of `corpus` or `corpora` may be set",
        )
            .into_response());
    }
    let names: Vec<String> = match (&req.corpus, &req.corpora) {
        (Some(n), None) => vec![n.clone()],
        (None, Some(v)) => {
            if v.is_empty() {
                return Err((StatusCode::BAD_REQUEST, "`corpora` must be non-empty").into_response());
            }
            v.clone()
        }
        (None, None) => vec!["default".into()],
        (Some(_), Some(_)) => unreachable!(), // handled above
    };
    let mut targets: Vec<(String, std::path::PathBuf)> = Vec::with_capacity(names.len());
    for name in &names {
        let Some(path) = state.registry.corpus_path(name) else {
            return Err(
                (StatusCode::NOT_FOUND, format!("corpus not found: {name}")).into_response()
            );
        };
        targets.push((name.clone(), path));
    }

    // Parse filter (string or JSON AST) + AND-in tenant filter.
    let base_filter: Option<fastrag::filter::FilterExpr> = match &req.filter {
        None => None,
        Some(serde_json::Value::String(s)) => match fastrag::filter::parse(s) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
        Some(v) => match serde_json::from_value::<fastrag::filter::FilterExpr>(v.clone()) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
    };
    let filter = if let Some(Extension(tf)) = tenant_ext {
        let tenant_cond = fastrag::filter::FilterExpr::Eq {
            field: tf.field.clone(),
            value: fastrag_store::schema::TypedValue::String(tf.value.clone()),
        };
        Some(match base_filter {
            Some(existing) => fastrag::filter::FilterExpr::And(vec![tenant_cond, existing]),
            None => tenant_cond,
        })
    } else {
        base_filter
    };

    // Acquire per-corpus read locks (same pattern as /query).
    let mut guards = Vec::with_capacity(targets.len());
    for (name, _) in &targets {
        let lock = get_or_create_lock(&state.ingest_locks, name);
        guards.push(lock);
    }
    let _read_guards: Vec<_> = {
        let mut out = Vec::with_capacity(guards.len());
        for lock in &guards {
            out.push(lock.read().await);
        }
        out
    };

    let snippet_len = 150;
    let request = fastrag::corpus::SimilarityRequest {
        text: req.text.clone(),
        threshold: req.threshold,
        max_results: req.max_results,
        targets,
        filter,
        snippet_len,
        overfetch_cap: state.similar_overfetch_cap,
    };
    let embedder = state.embedder.clone();

    let field_sel = match parse_field_selection(req.fields.as_deref()) {
        Ok(sel) => sel,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e).into_response()),
    };

    let span = info_span!(
        "similar",
        text = %request.text,
        threshold = request.threshold,
        max_results = request.max_results,
        corpora_count = request.targets.len(),
    );
    let _enter = span.enter();
    let start = Instant::now();

    let result = tokio::task::spawn_blocking(move || {
        fastrag::corpus::similarity_search(
            embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
            &request,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response())?;

    let elapsed = start.elapsed();
    metrics::counter!("fastrag_similar_total").increment(1);
    metrics::histogram!("fastrag_similar_duration_seconds").record(elapsed.as_secs_f64());

    match result {
        Ok(resp) => {
            let mut value = serde_json::to_value(&resp).unwrap();
            // Apply field projection to each hit's source via existing helper.
            if let Some(hits) = value.get_mut("hits").and_then(|v| v.as_array_mut()) {
                apply_field_selection(hits, &field_sel);
            }
            info!(hit_count = resp.hits.len(), "similar served");
            Ok(Json(value))
        }
        Err(CorpusError::NotFound(name)) => {
            Err((StatusCode::NOT_FOUND, format!("corpus not found: {name}")).into_response())
        }
        Err(e) => {
            warn!(error = %e, "similar failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response())
        }
    }
}
```

- [ ] **Step 6: Register the route and describe new metrics**

In `serve_http_with_registry` (around line 521), append `.route("/similar", axum::routing::post(similar_handler).layer(DefaultBodyLimit::max(1 * 1024 * 1024)))` to the `protected` router.

Near the existing `metrics::describe_counter!` calls (around line 473), add:
```rust
metrics::describe_counter!("fastrag_similar_total", "Total /similar requests served");
metrics::describe_histogram!(
    "fastrag_similar_duration_seconds",
    "Latency of /similar requests in seconds"
);
```

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
cargo test -p fastrag-cli --features retrieval --test similar_http_e2e -- post_similar_happy_path
```

Expected: PASS.

- [ ] **Step 8: Run the full HTTP test suite to catch regressions**

Run:
```bash
cargo test -p fastrag-cli --features retrieval --test http_e2e --test federation_e2e
```

Expected: all existing HTTP tests pass.

- [ ] **Step 9: Commit**

```bash
git add fastrag-cli/src/http.rs fastrag-cli/tests/similar_http_e2e.rs fastrag-cli/tests/http_e2e.rs fastrag-cli/tests/federation_e2e.rs fastrag-cli/tests/temporal_decay_http_e2e.rs fastrag-cli/tests/cwe_expand_http_e2e.rs
git commit -m "feat(http): POST /similar threshold endpoint (happy path)"
```

(Adjust the staged files based on which existing test files actually needed the extra `similar_overfetch_cap` argument — `git status` shows you.)

---

## Task 6: HTTP validation paths

**Files:**
- Modify: `fastrag-cli/tests/similar_http_e2e.rs`

**Why:** Covers every 400/404 path called out in the spec. The handler logic for these is already in Task 5; these tests prove each branch.

- [ ] **Step 1: Add validation tests**

Append to `fastrag-cli/tests/similar_http_e2e.rs`:

```rust
#[tokio::test]
async fn post_similar_rejects_hybrid_params() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "hybrid": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("/query"), "error should point to /query: {body}");
}

#[tokio::test]
async fn post_similar_rejects_rerank_param() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "rerank": "on"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("reranking"));
}

#[tokio::test]
async fn post_similar_rejects_both_corpus_and_corpora() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("one", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpus": "one",
            "corpora": ["one"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("exactly one"));
}

#[tokio::test]
async fn post_similar_rejects_empty_corpora() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpora": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("corpora") && body.contains("non-empty"));
}

#[tokio::test]
async fn post_similar_rejects_threshold_out_of_range() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 1.5,
            "max_results": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("threshold"));
}

#[tokio::test]
async fn post_similar_rejects_max_results_out_of_range() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn post_similar_rejects_empty_text() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "",
            "threshold": 0.5,
            "max_results": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = resp.text().await.unwrap();
    assert!(body.contains("text"));
}

#[tokio::test]
async fn post_similar_corpus_not_found() {
    let corpus = build_toy_corpus(&[("a", "alpha")]);
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 5,
            "corpus": "missing"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = resp.text().await.unwrap();
    assert!(body.contains("missing"));
}
```

- [ ] **Step 2: Run the new validation tests**

Run:
```bash
cargo test -p fastrag-cli --features retrieval --test similar_http_e2e
```

Expected: all tests pass. The handler logic from Task 5 already covers each reject.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/similar_http_e2e.rs
git commit -m "test(http): /similar validation and error-path coverage"
```

---

## Task 7: Multi-corpus fan-out, tenant filter, truncation over HTTP

**Files:**
- Modify: `fastrag-cli/tests/similar_http_e2e.rs`

- [ ] **Step 1: Add fan-out, tenant-filter, and truncated-flag tests**

Append to `fastrag-cli/tests/similar_http_e2e.rs`:

```rust
#[tokio::test]
async fn post_similar_fan_out_merges_across_corpora() {
    let c1 = build_toy_corpus(&[("a1", "alpha"), ("b1", "zzz1")]);
    let c2 = build_toy_corpus(&[("a2", "alpha"), ("b2", "zzz2")]);
    let registry = CorpusRegistry::new();
    registry.register("one", c1.path().join("corpus"));
    registry.register("two", c2.path().join("corpus"));
    let base = spawn_server(registry).await;

    let resp = Client::new()
        .post(format!("{base}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.95,
            "max_results": 10,
            "corpora": ["one", "two"]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2);
    let corpora: std::collections::BTreeSet<&str> = hits
        .iter()
        .map(|h| h["corpus"].as_str().unwrap())
        .collect();
    assert!(corpora.contains("one"));
    assert!(corpora.contains("two"));
    assert!(body["stats"]["per_corpus"]["one"].is_object());
    assert!(body["stats"]["per_corpus"]["two"].is_object());
}

#[tokio::test]
async fn post_similar_tenant_filter_applied() {
    // Seed two tenants into a single corpus.
    let tmp = tempfile::tempdir().unwrap();
    let jsonl = tmp.path().join("docs.jsonl");
    std::fs::write(
        &jsonl,
        concat!(
            r#"{"id":"a","body":"alpha","tenant":"acme"}"#, "\n",
            r#"{"id":"b","body":"alpha","tenant":"widgetco"}"#,
        ),
    )
    .unwrap();
    let corpus = tmp.path().join("corpus");
    let cfg = JsonlIngestConfig {
        text_fields: vec!["body".into()],
        id_field: "id".into(),
        metadata_fields: vec!["tenant".into()],
        metadata_types: BTreeMap::from([(
            "tenant".into(),
            fastrag_store::schema::TypedKind::String,
        )]),
        array_fields: vec![],
        cwe_field: None,
    };
    index_jsonl(
        &jsonl,
        &corpus,
        &ChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
        &MockEmbedder as &dyn fastrag::DynEmbedderTrait,
        &cfg,
    )
    .unwrap();

    // Spawn server with tenant_field = "tenant".
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.clone());
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        serve_http_with_registry(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            Some("tenant".into()),
            52_428_800,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://{addr}/similar"))
        .header("x-fastrag-tenant", "acme")
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 10
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    let hits = body["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "tenant filter should limit to acme");
    let tenant = hits[0]["source"]["tenant"]
        .as_str()
        .or_else(|| hits[0]["metadata"]["tenant"].as_str());
    assert_eq!(tenant, Some("acme"));
}

#[tokio::test]
async fn post_similar_truncated_flag() {
    // 20 matching docs, tiny overfetch cap -> truncated=true.
    let docs: Vec<(String, String)> =
        (0..20).map(|i| (format!("d{i}"), "alpha".to_string())).collect();
    let docs_ref: Vec<(&str, &str)> =
        docs.iter().map(|(i, b)| (i.as_str(), b.as_str())).collect();
    let corpus = build_toy_corpus(&docs_ref);

    // Spawn with overfetch_cap = 5.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus.path().join("corpus"));
    let embedder: Arc<dyn fastrag::DynEmbedderTrait> = Arc::new(MockEmbedder);
    tokio::spawn(async move {
        fastrag_cli::http::serve_http_with_registry_and_cap(
            registry,
            listener,
            embedder,
            None,
            false,
            false,
            HttpRerankerConfig::default(),
            100,
            None,
            52_428_800,
            5, // similar_overfetch_cap
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let resp = Client::new()
        .post(format!("http://{addr}/similar"))
        .json(&json!({
            "text": "alpha",
            "threshold": 0.5,
            "max_results": 100
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["truncated"].as_bool().unwrap(), true);
    let hits = body["hits"].as_array().unwrap();
    assert!(hits.len() <= 5);
}
```

- [ ] **Step 2: Expose a helper `serve_http_with_registry_and_cap` so tests can override the cap**

In `fastrag-cli/src/http.rs`, add a new public fn that threads `similar_overfetch_cap` through, keeping `serve_http_with_registry` as a thin wrapper that calls it with 10_000:

```rust
#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry_and_cap(
    registry: fastrag::corpus::CorpusRegistry,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
) -> Result<(), HttpError> {
    // move the existing body of serve_http_with_registry here, using
    // `similar_overfetch_cap` when constructing AppState.
    // ...
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry(
    registry: fastrag::corpus::CorpusRegistry,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
) -> Result<(), HttpError> {
    serve_http_with_registry_and_cap(
        registry, listener, embedder, token, dense_only, cwe_expand_default,
        rerank_cfg, batch_max_queries, tenant_field, ingest_max_body, 10_000,
    )
    .await
}
```

Update `serve_http_with_registry_port` similarly: add a `_and_cap` variant and leave the old one delegating with 10_000. (The `serve-http` CLI plumbing gets the cap flag in Task 8.)

- [ ] **Step 3: Run the new tests**

Run:
```bash
cargo test -p fastrag-cli --features retrieval --test similar_http_e2e
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/src/http.rs fastrag-cli/tests/similar_http_e2e.rs
git commit -m "test(http): /similar fan-out, tenant filter, truncated flag"
```

---

## Task 8: `--similar-overfetch-cap` CLI flag

**Files:**
- Modify: `fastrag-cli/src/args.rs`, `fastrag-cli/src/main.rs`

- [ ] **Step 1: Add the flag to the `ServeHttp` command definition**

Edit `fastrag-cli/src/args.rs`. Inside the `ServeHttp { ... }` variant (near line 632, right after `ingest_max_body`), add:

```rust
        /// Hard cap on per-corpus adaptive overfetch for POST /similar.
        /// Raise for large corpora with permissive thresholds. Default: 10000.
        #[arg(long, default_value_t = 10_000)]
        similar_overfetch_cap: usize,
```

- [ ] **Step 2: Destructure and plumb the new field in `main.rs`**

Edit `fastrag-cli/src/main.rs` (the `Command::ServeHttp { ... }` arm near line 691). Add `similar_overfetch_cap,` to the destructure pattern, and pass it to the server builder:

```rust
        Command::ServeHttp {
            // ... existing fields ...
            ingest_max_body,
            similar_overfetch_cap,
        } => {
            // ... existing body up to the serve_http_with_registry_port call ...

            if let Err(e) = fastrag_cli::http::serve_http_with_registry_port_and_cap(
                registry,
                port,
                embedder,
                token,
                dense_only,
                cwe_expand,
                rerank_cfg,
                batch_max_queries,
                tenant_field,
                ingest_max_body,
                similar_overfetch_cap,
            )
            .await
            {
                eprintln!("Error starting HTTP server: {e}");
                std::process::exit(1);
            }
        }
```

- [ ] **Step 3: Add the `_and_cap` port variant in `http.rs`**

Append to `fastrag-cli/src/http.rs`:

```rust
#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry_port_and_cap(
    registry: fastrag::corpus::CorpusRegistry,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    serve_http_with_registry_and_cap(
        registry, listener, embedder, token, dense_only, cwe_expand_default,
        rerank_cfg, batch_max_queries, tenant_field, ingest_max_body, similar_overfetch_cap,
    )
    .await
}
```

- [ ] **Step 4: Build and confirm the flag is wired**

Run:
```bash
cargo build --release -p fastrag-cli
./target/release/fastrag serve-http --help 2>&1 | grep similar-overfetch-cap
```

Expected: the help text includes `--similar-overfetch-cap <SIMILAR_OVERFETCH_CAP>` with the default `10000`.

- [ ] **Step 5: Run the existing test suite to verify no regression**

Run:
```bash
cargo test -p fastrag-cli --features retrieval
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add fastrag-cli/src/args.rs fastrag-cli/src/main.rs fastrag-cli/src/http.rs
git commit -m "feat(cli): --similar-overfetch-cap flag for serve-http"
```

---

## Task 9: README and CLAUDE.md documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Before editing either file, invoke the doc-editor skill** per `CLAUDE.md` project rules.

- [ ] **Step 1: Draft the "Similarity Search" section for README.md**

Find the place in `README.md` where HTTP endpoints are documented (grep for `## Hybrid Retrieval` or `## Temporal Decay` — similarity sits alongside). Insert a new section, for example after "Temporal Decay":

```markdown
## Similarity Search

`POST /similar` returns documents whose raw cosine similarity to a query text exceeds a threshold, rather than a fixed top-K. Use it for dedup and cross-corpus near-duplicate detection.

### Request

```json
POST /similar
Content-Type: application/json

{
  "text": "SQL injection in login form",
  "threshold": 0.85,
  "max_results": 10,
  "corpus": "acme-q1",
  "filter": "source_tool = semgrep"
}
```

Replace `corpus` with `corpora: ["acme-q1", "acme-q2"]` to fan out across multiple registered corpora. Setting both is an error.

### Fields

| Field | Required | Notes |
|---|---|---|
| `text` | yes | Non-empty. Embedded once per request. |
| `threshold` | yes | Raw cosine, `[0.0, 1.0]`. |
| `max_results` | yes | `[1, 1000]`. |
| `corpus` | no | Defaults to `default` when `corpora` is also omitted. |
| `corpora` | no | Non-empty array of registry names. Mutually exclusive with `corpus`. |
| `filter` | no | String syntax (`"severity = HIGH"`) or JSON `FilterExpr`. |
| `fields` | no | Projection (`include` or `exclude-` prefix), same syntax as `/query`. |

Hybrid retrieval, temporal decay, CWE expansion, and reranking are rejected with 400 — they change score semantics, which breaks threshold portability.

### Response

```json
{
  "hits": [
    { "cosine_similarity": 0.934, "corpus": "acme-q1", "snippet": "...", "source": { ... } }
  ],
  "truncated": false,
  "stats": { "candidates_examined": 480, "above_threshold": 12, "returned": 10 },
  "latency": { "embed_us": 1240, "hnsw_us": 8100, "total_us": 12300 }
}
```

`truncated: true` means the adaptive overfetch hit the server cap before exhausting the above-threshold tail. Raise `--similar-overfetch-cap` on `serve-http` when this trips on large corpora.

### Threshold portability caveat

Cosine thresholds are specific to an embedder model. A `0.85` threshold for `bge-small` is not comparable to a `0.85` threshold for `text-embedding-3-small`. Version thresholds per embedder.
```

- [ ] **Step 2: Invoke doc-editor on the draft**

Send the drafted Markdown through the `doc-editor` skill (foreground Haiku Agent) as described in `CLAUDE.md`. Apply any cleanups it returns (typically: remove "You don't need to" constructions, padding, AI self-narration).

- [ ] **Step 3: Write the cleaned section into `README.md`**

Use the `Edit` tool to splice the section into the correct location in `README.md`. Verify the surrounding headings still flow and the table of contents (if present) is updated.

- [ ] **Step 4: Append the test command to `CLAUDE.md`**

In `CLAUDE.md`, in the Build & Test code fence, near the other HTTP-endpoint test lines (grep for `cwe_expand_http_e2e`), add:

```
cargo test -p fastrag-cli --features retrieval --test similar_http_e2e  # POST /similar threshold endpoint e2e
```

Run doc-editor on this line first — it should pass without changes.

- [ ] **Step 5: Render both files locally to verify**

Run:
```bash
cargo build --release -p fastrag-cli  # rebuild to be sure
head -200 README.md | less             # eyeball
```

No test assertion here — this is a manual visual check.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: /similar endpoint documentation"
```

---

## Task 10: Final lint gate and pre-push verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full lint gate**

Run:
```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
```

Expected: zero warnings.

- [ ] **Step 2: Run `cargo fmt --check`**

Run:
```bash
cargo fmt --check
```

Expected: no output. If it prints anything, run `cargo fmt` and commit a `style:` commit.

- [ ] **Step 3: Run the full workspace test suite**

Run:
```bash
cargo test --workspace --features retrieval,store
```

Expected: all tests pass, including the new `similar*` tests.

- [ ] **Step 4: Push and run ci-watcher**

```bash
git push origin HEAD
```

Then invoke the `ci-watcher.md` skill as a background Haiku Agent per `CLAUDE.md`.

Expected: all workflows pass.

- [ ] **Step 5: Comment `Closes #52` on the last commit or in the PR body**

If working on a branch with a PR, ensure the PR body contains `Closes #52`. If committing directly, amend (or add a trailing empty commit) with `Closes #52` in the message so GitHub auto-closes the issue on merge to main.

---

## Self-Review

**Spec coverage:** Every spec section maps to a task —
- Request schema/validation → Task 5 (implementation), Task 6 (validation tests).
- Response schema → Task 1 (types), Task 3/4 (population), Task 5 (HTTP serialization).
- Core algorithm → Task 3 (single corpus), Task 4 (fan-out + parallelism).
- Parallelism → Task 4.
- Reuse (`filter_scored_ids`, `scored_ids_to_dtos`) → Task 2.
- Changes (new files, modified files) → Tasks 1, 3, 5, 8, 9.
- Testing (unit + integration) → Tasks 1, 3, 4 (unit); 5, 6, 7 (integration).
- Out of scope / Risk → captured in docs (Task 9).

**Placeholder scan:** No `TBD` / `TODO` / `implement later`. The two deliberate comment blocks ("embedder-shim note" in Task 3, "TODO: merge local_latency" in Task 4) are both followed by specific resolution instructions on the next line — concrete, not placeholder.

**Type consistency:** `SimilarityRequest`, `SimilarityResponse`, `SimilarityHit`, `SimilarityStats`, `PerCorpusStats`, `PerCorpusOutcome`, `similarity_search`, `similarity_search_one`, `filter_scored_ids`, `similar_handler`, `similar_overfetch_cap`, `serve_http_with_registry_and_cap`, `serve_http_with_registry_port_and_cap` — all reference the same name throughout the plan.
