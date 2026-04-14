# Temporal Decay + Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-introduce BM25+dense hybrid retrieval via unweighted RRF, and add opt-in post-fusion multiplicative recency decay for security-corpus freshness. Closes crook3dfingers/fastrag#48.

**Architecture:** New `crates/fastrag/src/corpus/hybrid.rs` module exposes `query_hybrid()` which calls `Store::query_bm25` and `Store::query_dense`, fuses via the existing `fastrag-index::fusion::rrf_fuse`, and optionally applies an exponential decay factor on the fused score. Wired in via a new `hybrid: HybridOpts` field on `QueryOpts` — old callers that pass `QueryOpts::default()` get identical dense-only behavior.

**Tech Stack:** Rust, Tokio, Clap 4 derive, Axum, Tantivy (BM25, via `fastrag_store::Store::query_bm25`), HNSW (via `Store::query_dense`), chrono, humantime, serde_json.

**Context notes:**
- Hybrid retrieval existed in the tree until commit `ce230b6` (Apr 2026 Store migration) and was removed. The current `--dense-only` CLI flag is a no-op left over from that removal. This plan deletes `--dense-only` and introduces `--hybrid`.
- No worktrees — work in the main checkout (per project preference).
- Commit after each task locally; push once at the end, then watch CI with the `ci-watcher` skill.
- Run `cargo fmt` + `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings` + `cargo test --workspace --features retrieval` locally before the final push.

---

## File Structure

**New:**
- `crates/fastrag/src/corpus/hybrid.rs` — module with `HybridOpts`, `TemporalOpts`, `BlendMode`, `decay_factor`, `apply_decay`, `query_hybrid`
- `crates/fastrag/tests/hybrid_retrieval.rs` — integration test: BM25+dense RRF reorders correctly
- `crates/fastrag/tests/temporal_decay.rs` — integration test: decay shifts ranking with date spread
- `fastrag-cli/tests/hybrid_e2e.rs` — CLI `query --hybrid` e2e
- `fastrag-cli/tests/temporal_decay_e2e.rs` — CLI decay flags e2e
- `fastrag-cli/tests/temporal_decay_http_e2e.rs` — HTTP decay e2e

**Modified:**
- `crates/fastrag/Cargo.toml` — add `humantime` dep (behind `retrieval` feature)
- `crates/fastrag/src/corpus/mod.rs` — extend `QueryOpts`, branch on `hybrid.enabled` in `query_corpus_with_filter_opts`, expose `hybrid` module
- `fastrag-cli/src/args.rs` — remove `dense_only`; add `hybrid`, `rrf_k`, `rrf_overfetch`, `time_decay_*` flags
- `fastrag-cli/src/main.rs` — build `HybridOpts` from flags, pass via `QueryOpts`
- `fastrag-cli/src/http.rs` — accept `hybrid`, `rrf_k`, `time_decay` in `QueryBody`
- `crates/fastrag-mcp/src/lib.rs` — `search_corpus` accepts hybrid+time_decay params
- `tests/gold/questions.json` — 6 temporal entries (3 recency-seeking, 3 historical-reference)
- `crates/fastrag-eval/src/matrix.rs` — `Variant` enum gains `TemporalDecay` axis
- `README.md`, `CLAUDE.md` — document hybrid + decay flags and build commands

---

## Task 1: Scaffold hybrid module + add humantime dep

**Files:**
- Modify: `crates/fastrag/Cargo.toml`
- Create: `crates/fastrag/src/corpus/hybrid.rs`
- Modify: `crates/fastrag/src/corpus/mod.rs:17-19` (add `pub mod hybrid;` line)

- [ ] **Step 1: Add humantime dep**

Edit `crates/fastrag/Cargo.toml`. Add to `[dependencies]` near the existing `chrono` line (line 33):
```toml
humantime = { version = "2", optional = true }
```
Add `"dep:humantime"` to the `retrieval` feature's dependency list. (Find the `retrieval = [...]` line and extend.)

- [ ] **Step 2: Also add to workspace root**

Edit `Cargo.toml` at the workspace root. Add under `[workspace.dependencies]`:
```toml
humantime = "2"
```

- [ ] **Step 3: Create the module skeleton**

Create `crates/fastrag/src/corpus/hybrid.rs`:
```rust
//! Hybrid retrieval (BM25 + dense RRF) with optional post-fusion temporal decay.
//!
//! Called from `query_corpus_with_filter_opts` when `QueryOpts::hybrid.enabled`
//! is set. Keeps the pure-function pieces (`decay_factor`, `apply_decay`)
//! separate from the I/O-bound `query_hybrid` so they can be unit-tested in
//! isolation.

use std::time::Duration;

use chrono::{DateTime, NaiveDate, Utc};

use crate::CorpusError;
use fastrag_index::fusion::{ScoredId, rrf_fuse};

#[derive(Debug, Clone)]
pub struct HybridOpts {
    pub enabled: bool,
    pub rrf_k: u32,
    pub overfetch_factor: usize,
    pub temporal: Option<TemporalOpts>,
}

impl Default for HybridOpts {
    fn default() -> Self {
        Self {
            enabled: false,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalOpts {
    pub date_field: String,
    pub halflife: Duration,
    pub weight_floor: f32,
    pub dateless_prior: f32,
    pub blend: BlendMode,
    pub now: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Multiplicative,
    Additive,
}
```

- [ ] **Step 4: Register the module**

Edit `crates/fastrag/src/corpus/mod.rs`. After the existing `pub mod registry;` line near line 18, add:
```rust
pub mod hybrid;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build -p fastrag --features retrieval`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/Cargo.toml Cargo.toml crates/fastrag/src/corpus/hybrid.rs crates/fastrag/src/corpus/mod.rs
git commit -m "feat(hybrid): scaffold hybrid module with HybridOpts/TemporalOpts types"
```

---

## Task 2: decay_factor pure function + tests

**Files:**
- Modify: `crates/fastrag/src/corpus/hybrid.rs`

- [ ] **Step 1: Write the failing tests**

Append to `crates/fastrag/src/corpus/hybrid.rs`:
```rust
/// Multiplicative decay factor in `[weight_floor, 1.0]`, or `dateless_prior`
/// when `age_days` is `None`. `halflife_days` must be > 0.
///
/// ```text
/// factor = alpha + (1 - alpha) * exp(-ln(2) * age_days / halflife)
/// ```
pub fn decay_factor(
    age_days: Option<f32>,
    halflife_days: f32,
    alpha: f32,
    dateless_prior: f32,
    _blend: BlendMode,
) -> f32 {
    unimplemented!()
}

#[cfg(test)]
mod decay_factor_tests {
    use super::*;

    #[test]
    fn age_zero_returns_one() {
        let f = decay_factor(Some(0.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn age_equal_halflife_returns_midpoint() {
        // alpha + (1-alpha)*0.5 = 0.3 + 0.35 = 0.65
        let f = decay_factor(Some(30.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 0.65).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn very_old_approaches_alpha_floor() {
        let f = decay_factor(Some(10_000.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!(f > 0.299 && f < 0.301, "got {f}");
    }

    #[test]
    fn alpha_one_disables_decay() {
        let f = decay_factor(Some(9999.0), 30.0, 1.0, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn dateless_returns_prior() {
        let f = decay_factor(None, 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 0.5).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn dateless_prior_independent_of_halflife() {
        let a = decay_factor(None, 30.0, 0.3, 0.42, BlendMode::Multiplicative);
        let b = decay_factor(None, 9000.0, 0.3, 0.42, BlendMode::Multiplicative);
        assert_eq!(a, b);
    }

    #[test]
    fn negative_age_clamps_to_one() {
        // Future-dated docs (negative age) treated as "today".
        let f = decay_factor(Some(-5.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::decay_factor_tests`
Expected: all 7 tests fail with `not implemented` panic.

- [ ] **Step 3: Implement `decay_factor`**

Replace the `unimplemented!()` body:
```rust
pub fn decay_factor(
    age_days: Option<f32>,
    halflife_days: f32,
    alpha: f32,
    dateless_prior: f32,
    _blend: BlendMode,
) -> f32 {
    match age_days {
        None => dateless_prior,
        Some(a) => {
            let a = a.max(0.0);
            let ln2: f32 = std::f32::consts::LN_2;
            alpha + (1.0 - alpha) * (-ln2 * a / halflife_days).exp()
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::decay_factor_tests`
Expected: 7/7 PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/hybrid.rs
git commit -m "feat(hybrid): decay_factor with multiplicative exponential form"
```

---

## Task 3: apply_decay + tests

**Files:**
- Modify: `crates/fastrag/src/corpus/hybrid.rs`

- [ ] **Step 1: Write the failing tests**

Append to `crates/fastrag/src/corpus/hybrid.rs`:
```rust
/// Apply decay to every `ScoredId`. `dates` must be the same length as `fused`
/// (index-aligned). `None` entries get the dateless prior.
///
/// Returns a new vector sorted by descending final score.
pub fn apply_decay(
    fused: &[ScoredId],
    dates: &[Option<NaiveDate>],
    opts: &TemporalOpts,
) -> Vec<ScoredId> {
    unimplemented!()
}

#[cfg(test)]
mod apply_decay_tests {
    use super::*;
    use chrono::TimeZone;

    fn opts(halflife_days: u64, alpha: f32, prior: f32) -> TemporalOpts {
        TemporalOpts {
            date_field: "published_date".into(),
            halflife: Duration::from_secs(halflife_days * 86_400),
            weight_floor: alpha,
            dateless_prior: prior,
            blend: BlendMode::Multiplicative,
            now: Utc.with_ymd_and_hms(2026, 4, 14, 0, 0, 0).unwrap(),
        }
    }

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn uniform_age_preserves_order() {
        let fused = vec![
            ScoredId { id: 1, score: 0.9 },
            ScoredId { id: 2, score: 0.8 },
            ScoredId { id: 3, score: 0.7 },
        ];
        let same = Some(ymd(2026, 4, 1));
        let out = apply_decay(&fused, &[same, same, same], &opts(30, 0.3, 0.5));
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn fresh_outranks_equally_relevant_stale() {
        // Both have rrf_score 0.8; one is today, one is 10 years old.
        let fused = vec![
            ScoredId { id: 1, score: 0.8 }, // stale
            ScoredId { id: 2, score: 0.8 }, // fresh
        ];
        let dates = vec![Some(ymd(2016, 4, 14)), Some(ymd(2026, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts(30, 0.3, 0.5));
        assert_eq!(out[0].id, 2, "fresh must rank first; got {out:?}");
        assert_eq!(out[1].id, 1);
    }

    #[test]
    fn dateless_interleaves_at_neutral_prior() {
        // halflife=30d, alpha=0.3, prior=0.5.
        // id=1 fresh -> factor=1.0 -> 1.0
        // id=2 dateless -> factor=0.5 -> 0.5
        // id=3 very stale (5y) -> factor→0.3 -> 0.3
        let fused = vec![
            ScoredId { id: 1, score: 1.0 },
            ScoredId { id: 2, score: 1.0 },
            ScoredId { id: 3, score: 1.0 },
        ];
        let dates = vec![Some(ymd(2026, 4, 14)), None, Some(ymd(2021, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts(30, 0.3, 0.5));
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = apply_decay(&[], &[], &opts(30, 0.3, 0.5));
        assert!(out.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::apply_decay_tests`
Expected: 4/4 fail.

- [ ] **Step 3: Implement `apply_decay`**

Replace the `unimplemented!()` body:
```rust
pub fn apply_decay(
    fused: &[ScoredId],
    dates: &[Option<NaiveDate>],
    opts: &TemporalOpts,
) -> Vec<ScoredId> {
    assert_eq!(
        fused.len(),
        dates.len(),
        "apply_decay: fused and dates must be index-aligned"
    );

    let halflife_days = (opts.halflife.as_secs_f32() / 86_400.0).max(f32::EPSILON);
    let now_date = opts.now.date_naive();

    let mut out: Vec<ScoredId> = fused
        .iter()
        .zip(dates.iter())
        .map(|(hit, date)| {
            let age = date.map(|d| (now_date - d).num_days() as f32);
            let factor = decay_factor(
                age,
                halflife_days,
                opts.weight_floor,
                opts.dateless_prior,
                opts.blend,
            );
            ScoredId {
                id: hit.id,
                score: hit.score * factor,
            }
        })
        .collect();

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::apply_decay_tests`
Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/hybrid.rs
git commit -m "feat(hybrid): apply_decay walks ScoredId slice with date-aligned dates"
```

---

## Task 4: Date extraction helper + tests

**Files:**
- Modify: `crates/fastrag/src/corpus/hybrid.rs`

- [ ] **Step 1: Write the failing tests**

Append to `crates/fastrag/src/corpus/hybrid.rs`:
```rust
/// Extract a `NaiveDate` for one row of metadata by locating the named field.
/// Returns `None` when the field is absent or the value isn't a `Date`.
pub fn extract_date(
    fields: &[(String, fastrag_store::schema::TypedValue)],
    date_field: &str,
) -> Option<NaiveDate> {
    unimplemented!()
}

#[cfg(test)]
mod extract_date_tests {
    use super::*;
    use fastrag_store::schema::TypedValue;

    fn field(name: &str, v: TypedValue) -> (String, TypedValue) {
        (name.to_string(), v)
    }

    #[test]
    fn returns_date_when_field_present() {
        let rows = vec![
            field("other", TypedValue::String("x".into())),
            field(
                "published_date",
                TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
            ),
        ];
        let d = extract_date(&rows, "published_date");
        assert_eq!(d, NaiveDate::from_ymd_opt(2024, 6, 1));
    }

    #[test]
    fn returns_none_when_field_missing() {
        let rows = vec![field("other", TypedValue::String("x".into()))];
        assert_eq!(extract_date(&rows, "published_date"), None);
    }

    #[test]
    fn returns_none_when_field_wrong_type() {
        let rows = vec![field("published_date", TypedValue::String("2024-06-01".into()))];
        assert_eq!(extract_date(&rows, "published_date"), None);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::extract_date_tests`
Expected: 3/3 fail.

- [ ] **Step 3: Implement `extract_date`**

Replace the `unimplemented!()` body:
```rust
pub fn extract_date(
    fields: &[(String, fastrag_store::schema::TypedValue)],
    date_field: &str,
) -> Option<NaiveDate> {
    fields.iter().find_map(|(k, v)| {
        if k == date_field {
            match v {
                fastrag_store::schema::TypedValue::Date(d) => Some(*d),
                _ => None,
            }
        } else {
            None
        }
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::extract_date_tests`
Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/hybrid.rs
git commit -m "feat(hybrid): extract_date pulls NaiveDate from TypedValue metadata"
```

---

## Task 5: query_hybrid core function + unit tests

**Files:**
- Modify: `crates/fastrag/src/corpus/hybrid.rs`

- [ ] **Step 1: Write the failing test (uses an in-memory Store)**

Read the existing Store-fixture pattern first:
```
crates/fastrag-store/src/lib.rs
```
Confirm `Store::open`, `Store::write_batch`, `Store::query_dense`, `Store::query_bm25`, `Store::fetch_metadata` signatures. Use the same pattern the existing store tests use (see `crates/fastrag-store/src/lib.rs:550+` for the `query_dense` test).

Append to `crates/fastrag/src/corpus/hybrid.rs`:
```rust
/// BM25 + dense candidate fetch, unweighted RRF, optional post-fusion decay.
///
/// Fetches `overfetch_factor * top_k` from each retriever (minimum `top_k`),
/// fuses via RRF(k=rrf_k), optionally applies temporal decay, sorts, truncates
/// to `top_k`.
///
/// Timing: populates `breakdown.bm25_us`, `breakdown.hnsw_us`, `breakdown.fuse_us`.
/// Caller is responsible for `embed_us` and `breakdown.finalize()`.
#[allow(clippy::too_many_arguments)]
pub fn query_hybrid(
    store: &fastrag_store::Store,
    query: &str,
    vector: &[f32],
    top_k: usize,
    opts: &HybridOpts,
    breakdown: &mut crate::corpus::LatencyBreakdown,
) -> Result<Vec<ScoredId>, CorpusError> {
    use std::time::Instant;

    let fetch_count = top_k
        .saturating_mul(opts.overfetch_factor.max(1))
        .max(top_k);

    let t = Instant::now();
    let bm25 = store.query_bm25(query, fetch_count)?;
    breakdown.bm25_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    let dense = store.query_dense(vector, fetch_count)?;
    breakdown.hnsw_us = t.elapsed().as_micros() as u64;

    let bm25_sids: Vec<ScoredId> = bm25
        .iter()
        .map(|(id, score)| ScoredId {
            id: *id,
            score: *score,
        })
        .collect();
    let dense_sids: Vec<ScoredId> = dense
        .iter()
        .map(|(id, score)| ScoredId {
            id: *id,
            score: *score,
        })
        .collect();

    let t = Instant::now();
    let mut fused = rrf_fuse(&[&bm25_sids, &dense_sids], opts.rrf_k);
    breakdown.fuse_us = t.elapsed().as_micros() as u64;

    if let Some(temp) = &opts.temporal {
        let ids: Vec<u64> = fused.iter().map(|s| s.id).collect();
        let rows = store.fetch_metadata(&ids)?;
        let row_map: std::collections::HashMap<u64, Vec<(String, fastrag_store::schema::TypedValue)>> =
            rows.into_iter().collect();
        let dates: Vec<Option<NaiveDate>> = fused
            .iter()
            .map(|s| {
                row_map
                    .get(&s.id)
                    .and_then(|f| extract_date(f, &temp.date_field))
            })
            .collect();
        fused = apply_decay(&fused, &dates, temp);
    }

    fused.truncate(top_k);
    Ok(fused)
}
```

- [ ] **Step 2: Add a query_hybrid unit test (co-located)**

Append to `crates/fastrag/src/corpus/hybrid.rs`:
```rust
#[cfg(test)]
mod query_hybrid_tests {
    use super::*;
    use crate::corpus::LatencyBreakdown;

    // Helper: build a tiny Store with 3 chunks, injected BM25 + dense scores
    // differ from each other so RRF has work to do.
    //
    // Reuse the existing Store test harness pattern from
    // crates/fastrag-store/src/lib.rs — build via StoreSchemaBuilder and
    // Store::open on a tempdir, insert chunks via write_batch.
    //
    // Keep the fixture minimal: 3 chunks, no metadata beyond ids.

    use fastrag_embed::test_util::TestEmbedder;
    use fastrag_store::Store;
    use tempfile::tempdir;

    fn fixture() -> (tempfile::TempDir, Store) {
        let dir = tempdir().unwrap();
        let embedder = TestEmbedder::new(16);
        let schema = fastrag_store::StoreSchemaBuilder::new()
            .with_text_field("_chunk_text")
            .build();
        let mut store = Store::create(dir.path(), schema, &embedder).unwrap();

        // Three chunks with known text — BM25 will score on lexical overlap.
        store
            .write_batch(
                &[
                    (1u64, "alpha beta gamma", "src1.md"),
                    (2u64, "delta epsilon zeta", "src2.md"),
                    (3u64, "alpha delta", "src3.md"),
                ],
                &embedder,
            )
            .unwrap();
        store.commit().unwrap();
        (dir, store)
    }

    #[test]
    fn hybrid_reorders_vs_dense_only() {
        let (_dir, store) = fixture();
        let embedder = fastrag_embed::test_util::TestEmbedder::new(16);
        let vec = embedder.embed_query("alpha delta").unwrap();

        let mut bd = LatencyBreakdown::default();
        let opts = HybridOpts {
            enabled: true,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: None,
        };
        let out = query_hybrid(&store, "alpha delta", &vec, 3, &opts, &mut bd).unwrap();

        assert_eq!(out.len(), 3, "should return 3 fused ids");
        // ID 3 has both "alpha" and "delta" → BM25 favors it.
        assert_eq!(out[0].id, 3, "lexical overlap winner first; got {out:?}");
    }

    #[test]
    fn temporal_option_runs_decay_branch() {
        // Regression guard: ensure the branch compiles and doesn't panic when
        // temporal is Some but the corpus has no `published_date` field.
        // Expected behavior: dateless-prior applied to every hit.
        let (_dir, store) = fixture();
        let embedder = fastrag_embed::test_util::TestEmbedder::new(16);
        let vec = embedder.embed_query("alpha").unwrap();

        let mut bd = LatencyBreakdown::default();
        let opts = HybridOpts {
            enabled: true,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: Some(TemporalOpts {
                date_field: "published_date".into(),
                halflife: Duration::from_secs(30 * 86400),
                weight_floor: 0.3,
                dateless_prior: 0.5,
                blend: BlendMode::Multiplicative,
                now: Utc::now(),
            }),
        };
        let out = query_hybrid(&store, "alpha", &vec, 3, &opts, &mut bd).unwrap();
        assert!(!out.is_empty(), "decay branch returned empty");
    }
}
```

**Note:** The exact test fixture API (`Store::create`, `StoreSchemaBuilder`, `write_batch`, `TestEmbedder`) needs to match what `crates/fastrag-store` exposes today. Before running the test, open the relevant Store test and mirror its setup exactly — adjust fixture calls if names differ. The assertions (ordering, len) are the part that must not change.

- [ ] **Step 3: Run tests to verify they fail (compile or assert)**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid::query_hybrid_tests`
Expected: failure — either compile error (fix fixture APIs until it compiles) or the `query_hybrid` function body missing returns.

- [ ] **Step 4: Iterate fixture-api wiring until tests compile and pass**

If tests compile but fail an assertion, check the RRF ranking hand-calc: with BM25 ranking `[3, 1]` and dense ranking depending on test embedder, id=3 should dominate RRF. If the test embedder doesn't scale that way, adjust the query text to make id=3 the BM25 winner.

- [ ] **Step 5: Run full hybrid module tests**

Run: `cargo test -p fastrag --features retrieval --lib corpus::hybrid`
Expected: all tests in `decay_factor_tests`, `apply_decay_tests`, `extract_date_tests`, `query_hybrid_tests` PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/hybrid.rs
git commit -m "feat(hybrid): query_hybrid fuses BM25+dense via RRF with optional decay"
```

---

## Task 6: Wire hybrid into query_corpus_with_filter_opts

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (extend `QueryOpts`; branch in `query_corpus_with_filter_opts`)

- [ ] **Step 1: Extend `QueryOpts`**

Edit `crates/fastrag/src/corpus/mod.rs`. Replace the `QueryOpts` struct (~line 22–28) with:
```rust
#[derive(Debug, Clone, Default)]
pub struct QueryOpts {
    /// When `true` and the corpus manifest has a `cwe_field`, expand CWE
    /// predicates via the embedded taxonomy before filter evaluation,
    /// and synthesise an In filter from any CWE ids in the free-text query.
    pub cwe_expand: bool,

    /// Hybrid retrieval (BM25 + dense RRF) with optional temporal decay.
    /// When `enabled == false` (the default), the query path is dense-only.
    pub hybrid: crate::corpus::hybrid::HybridOpts,
}
```

- [ ] **Step 2: Branch on `hybrid.enabled` inside `query_corpus_with_filter_opts`**

In `query_corpus_with_filter_opts` (around line 1008, where `filter.is_none()` short-circuits to `store.query_dense`), replace:
```rust
    // Unfiltered: query Store for top_k dense hits and return.
    if filter.is_none() {
        let t = Instant::now();
        let scored = store.query_dense(&vector, top_k)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        breakdown.finalize();

        return scored_ids_to_dtos(&store, &scored, Some(query), snippet_len);
    }
```
with:
```rust
    // Unfiltered: hybrid or dense-only path.
    if filter.is_none() {
        let scored: Vec<(u64, f32)> = if opts.hybrid.enabled {
            let fused = crate::corpus::hybrid::query_hybrid(
                &store,
                query,
                &vector,
                top_k,
                &opts.hybrid,
                breakdown,
            )?;
            fused.into_iter().map(|s| (s.id, s.score)).collect()
        } else {
            let t = Instant::now();
            let dense = store.query_dense(&vector, top_k)?;
            breakdown.hnsw_us = t.elapsed().as_micros() as u64;
            dense
        };
        breakdown.finalize();

        return scored_ids_to_dtos(&store, &scored, Some(query), snippet_len);
    }
```

Apply the same pattern for the filtered (overfetch) path: inside the `for &factor in overfetch_factors` loop, replace the `store.query_dense(&vector, fetch_count)?` call with a helper that branches on `opts.hybrid.enabled`. Extract into a local closure for DRY:
```rust
    let fetch_candidates = |n: usize, bd: &mut LatencyBreakdown| -> Result<Vec<(u64, f32)>, CorpusError> {
        if opts.hybrid.enabled {
            let fused = crate::corpus::hybrid::query_hybrid(
                &store, query, &vector, n, &opts.hybrid, bd,
            )?;
            Ok(fused.into_iter().map(|s| (s.id, s.score)).collect())
        } else {
            let t = std::time::Instant::now();
            let out = store.query_dense(&vector, n)?;
            bd.hnsw_us = t.elapsed().as_micros() as u64;
            Ok(out)
        }
    };
```
Use `fetch_candidates(fetch_count, breakdown)?` inside the overfetch loop.

- [ ] **Step 3: Build + run existing tests**

Run: `cargo build -p fastrag --features retrieval`
Expected: success.

Run: `cargo test --workspace --features retrieval`
Expected: all existing tests still pass. This is the backward-compat regression guard.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs
git commit -m "feat(corpus): branch query_corpus_with_filter_opts on QueryOpts.hybrid"
```

---

## Task 7: Integration test — hybrid reorders vs dense-only

**Files:**
- Create: `crates/fastrag/tests/hybrid_retrieval.rs`

- [ ] **Step 1: Write the integration test**

Create `crates/fastrag/tests/hybrid_retrieval.rs`:
```rust
//! End-to-end integration: build a tiny Store-backed corpus, query dense-only
//! vs hybrid, assert the ordering differs as RRF intends.

#![cfg(feature = "retrieval")]

use fastrag::corpus::hybrid::HybridOpts;
use fastrag::corpus::{LatencyBreakdown, QueryOpts, query_corpus_with_filter_opts};
use fastrag_embed::test_util::TestEmbedder;
use tempfile::tempdir;

// Mirror the fixture from hybrid.rs tests; keep the assertions integration-style
// (go through the public query_corpus_with_filter_opts API).
fn build_corpus() -> tempfile::TempDir {
    let dir = tempdir().unwrap();
    // ... build a Store at dir.path() with 3 chunks, same as Task 5 fixture ...
    // (Copy the exact bootstrap from Task 5's query_hybrid_tests.)
    dir
}

#[test]
fn hybrid_changes_top_hit_vs_dense_only() {
    let corpus = build_corpus();
    let embedder = TestEmbedder::new(16);

    let mut bd = LatencyBreakdown::default();
    let dense_opts = QueryOpts::default();
    let dense_hits = query_corpus_with_filter_opts(
        corpus.path(),
        "alpha delta",
        3,
        &embedder,
        None,
        &dense_opts,
        &mut bd,
        0,
    )
    .unwrap();

    let mut bd2 = LatencyBreakdown::default();
    let mut hybrid_opts = QueryOpts::default();
    hybrid_opts.hybrid = HybridOpts {
        enabled: true,
        ..Default::default()
    };
    let hybrid_hits = query_corpus_with_filter_opts(
        corpus.path(),
        "alpha delta",
        3,
        &embedder,
        None,
        &hybrid_opts,
        &mut bd2,
        0,
    )
    .unwrap();

    assert_eq!(dense_hits.len(), 3);
    assert_eq!(hybrid_hits.len(), 3);
    // Concrete difference: id=3 (lexical winner on "alpha delta") should be
    // first under hybrid, regardless of dense ranking.
    assert_eq!(
        hybrid_hits[0].chunk_text, "alpha delta",
        "hybrid top hit should be the lexical winner; got {:?}",
        hybrid_hits[0].chunk_text
    );
}
```

Flesh out `build_corpus` using the same bootstrap as the Task 5 co-located test (copy-paste is fine — the integration test must be self-contained).

- [ ] **Step 2: Run the test**

Run: `cargo test -p fastrag --features retrieval --test hybrid_retrieval`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag/tests/hybrid_retrieval.rs
git commit -m "test(hybrid): integration test — hybrid reorders vs dense-only"
```

---

## Task 8: Integration test — temporal decay

**Files:**
- Create: `crates/fastrag/tests/temporal_decay.rs`

- [ ] **Step 1: Write the integration test**

Create `crates/fastrag/tests/temporal_decay.rs`:
```rust
//! End-to-end: decay promotes fresh docs over equally-relevant stale docs.

#![cfg(feature = "retrieval")]

use std::time::Duration;

use chrono::{NaiveDate, TimeZone, Utc};
use fastrag::corpus::hybrid::{BlendMode, HybridOpts, TemporalOpts};
use fastrag::corpus::{LatencyBreakdown, QueryOpts, query_corpus_with_filter_opts};
use fastrag_embed::test_util::TestEmbedder;
use fastrag_store::schema::TypedValue;
use tempfile::tempdir;

fn build_corpus_with_dates() -> tempfile::TempDir {
    let dir = tempdir().unwrap();
    let embedder = TestEmbedder::new(16);
    let schema = fastrag_store::StoreSchemaBuilder::new()
        .with_text_field("_chunk_text")
        .with_date_field("published_date")
        .build();
    let mut store = fastrag_store::Store::create(dir.path(), schema, &embedder).unwrap();

    // Three docs, semantically identical for embedding purposes:
    // 1 → 2016, 2 → 2026, 3 → dateless
    store
        .write_batch_with_metadata(
            &[
                (
                    1u64,
                    "openssl heap overflow",
                    "a.md",
                    vec![(
                        "published_date".to_string(),
                        TypedValue::Date(NaiveDate::from_ymd_opt(2016, 1, 1).unwrap()),
                    )],
                ),
                (
                    2u64,
                    "openssl heap overflow",
                    "b.md",
                    vec![(
                        "published_date".to_string(),
                        TypedValue::Date(NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()),
                    )],
                ),
                (
                    3u64,
                    "openssl heap overflow",
                    "c.md",
                    vec![],
                ),
            ],
            &embedder,
        )
        .unwrap();
    store.commit().unwrap();
    dir
}

#[test]
fn decay_promotes_fresh_over_equally_relevant_stale() {
    let corpus = build_corpus_with_dates();
    let embedder = TestEmbedder::new(16);

    let mut opts = QueryOpts::default();
    opts.hybrid = HybridOpts {
        enabled: true,
        rrf_k: 60,
        overfetch_factor: 4,
        temporal: Some(TemporalOpts {
            date_field: "published_date".into(),
            halflife: Duration::from_secs(30 * 86_400),
            weight_floor: 0.3,
            dateless_prior: 0.5,
            blend: BlendMode::Multiplicative,
            now: Utc.with_ymd_and_hms(2026, 4, 14, 0, 0, 0).unwrap(),
        }),
    };

    let mut bd = LatencyBreakdown::default();
    let hits = query_corpus_with_filter_opts(
        corpus.path(),
        "openssl heap overflow",
        3,
        &embedder,
        None,
        &opts,
        &mut bd,
        0,
    )
    .unwrap();

    assert_eq!(hits.len(), 3);
    // Strong assertion: fresh (id=2) outranks stale (id=1). Dateless (id=3)
    // lands between them under the 0.5 prior.
    assert_eq!(hits[0].source_path.file_name().unwrap(), "b.md", "fresh first");
    assert_eq!(hits[1].source_path.file_name().unwrap(), "c.md", "dateless middle");
    assert_eq!(hits[2].source_path.file_name().unwrap(), "a.md", "stale last");
}
```

**Note:** If `Store::write_batch_with_metadata` and `StoreSchemaBuilder::with_date_field` differ in name or signature, consult `crates/fastrag-store/src/tantivy.rs:595` area where `fetch_metadata_returns_typed_values` builds an equivalent fixture. Mirror that exactly.

- [ ] **Step 2: Run the test**

Run: `cargo test -p fastrag --features retrieval --test temporal_decay`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag/tests/temporal_decay.rs
git commit -m "test(hybrid): integration test — decay promotes fresh over stale"
```

---

## Task 9: CLI flags — remove --dense-only, add --hybrid + --time-decay-*

**Files:**
- Modify: `fastrag-cli/src/args.rs`
- Modify: `fastrag-cli/src/main.rs`

- [ ] **Step 1: Update args.rs**

Edit `fastrag-cli/src/args.rs`. In the `Query {` variant (starting around line 292), **remove** the `dense_only` field (line 345–346):
```rust
        /// Skip BM25/Tantivy and use dense vector search only.
        #[arg(long)]
        dense_only: bool,
```

**Add** the following fields before the closing `}` of the `Query` variant:
```rust
        /// Enable BM25 + dense hybrid retrieval via Reciprocal Rank Fusion.
        /// Disabled by default (dense-only path preserves current behavior).
        #[arg(long)]
        hybrid: bool,

        /// RRF k parameter. Default 60 per the canonical RRF paper.
        #[arg(long, default_value_t = 60)]
        rrf_k: u32,

        /// Per-retriever overfetch multiplier — fetch rrf_overfetch * top_k
        /// from each retriever before fusion.
        #[arg(long, default_value_t = 4)]
        rrf_overfetch: usize,

        /// Name of the `Date` metadata field to use for temporal decay.
        /// Implies --hybrid.
        #[arg(long)]
        time_decay_field: Option<String>,

        /// Decay halflife (humantime format, e.g. "30d", "7d", "1y").
        #[arg(long, default_value = "30d")]
        time_decay_halflife: String,

        /// Alpha floor: minimum decay factor for very old docs. Range 0..=1.
        #[arg(long, default_value_t = 0.3)]
        time_decay_weight: f32,

        /// Neutral prior used for docs missing the date field. Range 0..=1.
        #[arg(long, default_value_t = 0.5)]
        time_decay_dateless_prior: f32,

        /// Blend mode: multiplicative (default) or additive.
        #[arg(long, value_enum, default_value = "multiplicative")]
        time_decay_blend: TimeDecayBlendArg,
```

Add a `TimeDecayBlendArg` enum at the top of `args.rs` (below existing value-enum arg types):
```rust
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum TimeDecayBlendArg {
    Multiplicative,
    Additive,
}
```

- [ ] **Step 2: Update main.rs**

Edit `fastrag-cli/src/main.rs` around line 475 where the `Query` variant is destructured. Remove `dense_only,` from the pattern and add the new fields. Remove the `let _ = dense_only;` line around line 533.

Add a helper near the top of `main.rs` (or in `args.rs`) to build `HybridOpts` from flags:
```rust
fn build_hybrid_opts(
    hybrid: bool,
    rrf_k: u32,
    rrf_overfetch: usize,
    time_decay_field: Option<String>,
    time_decay_halflife: &str,
    time_decay_weight: f32,
    time_decay_dateless_prior: f32,
    time_decay_blend: fastrag_cli::args::TimeDecayBlendArg,
) -> Result<fastrag::corpus::hybrid::HybridOpts, String> {
    use fastrag::corpus::hybrid::{BlendMode, HybridOpts, TemporalOpts};

    let has_decay_params_without_field = time_decay_field.is_none()
        && (time_decay_halflife != "30d" || time_decay_weight != 0.3 || time_decay_dateless_prior != 0.5);
    if has_decay_params_without_field {
        return Err(
            "--time-decay-halflife / -weight / -dateless-prior require --time-decay-field"
                .to_string(),
        );
    }

    let temporal = if let Some(field) = time_decay_field {
        let halflife = humantime::parse_duration(time_decay_halflife)
            .map_err(|e| format!("--time-decay-halflife: {e}"))?;
        Some(TemporalOpts {
            date_field: field,
            halflife,
            weight_floor: time_decay_weight,
            dateless_prior: time_decay_dateless_prior,
            blend: match time_decay_blend {
                fastrag_cli::args::TimeDecayBlendArg::Multiplicative => BlendMode::Multiplicative,
                fastrag_cli::args::TimeDecayBlendArg::Additive => BlendMode::Additive,
            },
            now: chrono::Utc::now(),
        })
    } else {
        None
    };

    let enabled = hybrid || temporal.is_some();
    Ok(HybridOpts {
        enabled,
        rrf_k,
        overfetch_factor: rrf_overfetch,
        temporal,
    })
}
```

In the `Query` handler, construct `query_opts` with the new field:
```rust
let hybrid_opts = build_hybrid_opts(
    hybrid, rrf_k, rrf_overfetch,
    time_decay_field, &time_decay_halflife, time_decay_weight,
    time_decay_dateless_prior, time_decay_blend,
)
.unwrap_or_else(|e| {
    eprintln!("error: {e}");
    std::process::exit(2);
});
let query_opts = ops::QueryOpts {
    cwe_expand: cwe_expand_effective,
    hybrid: hybrid_opts,
};
```

- [ ] **Step 3: Apply the same flag set to `serve-http` subcommand**

The `serve-http` subcommand (grep `ServeHttp` in `args.rs`) also needs the same hybrid + time-decay flags so per-request HTTP bodies can still opt in, OR they can be omitted at the CLI level and solely exposed in the HTTP request body. Per the design spec, defaults ride on the HTTP body. **Do not add hybrid flags to `serve-http`** — the per-request HTTP body (Task 13) is the control surface.

- [ ] **Step 4: Build and run**

Run: `cargo build -p fastrag-cli --features retrieval,rerank,hybrid,contextual,eval`
Wait — there is no `hybrid` Cargo feature. Use the real feature set:
Run: `cargo build -p fastrag-cli`
Expected: success.

Run: `cargo test --workspace --features retrieval`
Expected: all previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add fastrag-cli/src/args.rs fastrag-cli/src/main.rs
git commit -m "feat(cli): --hybrid + --time-decay-* flags on query; drop --dense-only no-op"
```

---

## Task 10: CLI e2e — hybrid retrieval

**Files:**
- Create: `fastrag-cli/tests/hybrid_e2e.rs`

- [ ] **Step 1: Write the test**

Mirror the pattern in `fastrag-cli/tests/cwe_expand_e2e.rs`. Build a tiny corpus via `cargo run -- index ...`, then run `cargo run -- query ... --hybrid` and assert exit status 0 + non-empty JSON output containing expected ids.

Create `fastrag-cli/tests/hybrid_e2e.rs`:
```rust
//! End-to-end: CLI `query --hybrid` returns non-empty JSON on exit 0.

#![cfg(feature = "retrieval")]

use std::process::Command;
use tempfile::tempdir;

#[test]
fn cli_query_hybrid_returns_results() {
    // Arrange: seed a two-document corpus.
    let corpus_dir = tempdir().unwrap();
    let docs_dir = tempdir().unwrap();
    std::fs::write(docs_dir.path().join("a.md"), "alpha beta gamma").unwrap();
    std::fs::write(docs_dir.path().join("b.md"), "delta epsilon alpha").unwrap();

    // Index.
    let bin = env!("CARGO_BIN_EXE_fastrag");
    let status = Command::new(bin)
        .args([
            "index",
            docs_dir.path().to_str().unwrap(),
            "--corpus",
            corpus_dir.path().to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "index failed");

    // Query with --hybrid.
    let out = Command::new(bin)
        .args([
            "query",
            "alpha",
            "--corpus",
            corpus_dir.path().to_str().unwrap(),
            "--top-k",
            "2",
            "--hybrid",
            "--format",
            "json",
            "--no-rerank",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "query failed: {out:?}");
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v.get("hits").and_then(|h| h.as_array()).map(|a| !a.is_empty()).unwrap_or(false),
        "expected non-empty hits array; got {stdout}");
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p fastrag-cli --test hybrid_e2e`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/hybrid_e2e.rs
git commit -m "test(cli): --hybrid e2e via index + query"
```

---

## Task 11: CLI e2e — temporal decay + error paths

**Files:**
- Create: `fastrag-cli/tests/temporal_decay_e2e.rs`

- [ ] **Step 1: Write the test**

Create `fastrag-cli/tests/temporal_decay_e2e.rs`:
```rust
//! End-to-end: CLI --time-decay-field flag + error paths.

#![cfg(feature = "retrieval")]

use std::process::Command;
use tempfile::tempdir;

fn seed_corpus_with_jsonl() -> (tempfile::TempDir, tempfile::TempDir) {
    let docs_dir = tempdir().unwrap();
    let corpus_dir = tempdir().unwrap();
    // JSONL ingest with typed date field — matches the JSONL-ingest path
    // shipped in #41 (see tests/gold/corpus for a fixture example).
    let jsonl = r#"{"id": 1, "text": "openssl heap overflow", "published_date": "2016-01-01"}
{"id": 2, "text": "openssl heap overflow", "published_date": "2026-04-01"}
"#;
    std::fs::write(docs_dir.path().join("docs.jsonl"), jsonl).unwrap();
    (docs_dir, corpus_dir)
}

#[test]
fn decay_flags_without_field_error() {
    let corpus_dir = tempdir().unwrap();
    let bin = env!("CARGO_BIN_EXE_fastrag");
    let out = Command::new(bin)
        .args([
            "query",
            "x",
            "--corpus",
            corpus_dir.path().to_str().unwrap(),
            "--time-decay-halflife",
            "7d",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "should have errored");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(
        stderr.contains("--time-decay-field"),
        "stderr should reference --time-decay-field; got {stderr}"
    );
}

#[test]
fn decay_on_jsonl_corpus_promotes_fresh() {
    let (docs, corpus) = seed_corpus_with_jsonl();
    let bin = env!("CARGO_BIN_EXE_fastrag");

    // Ingest with the published_date field declared as a typed Date.
    let status = Command::new(bin)
        .args([
            "index",
            docs.path().to_str().unwrap(),
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--metadata-fields",
            "published_date:date",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "ingest failed");

    let out = Command::new(bin)
        .args([
            "query",
            "openssl heap overflow",
            "--corpus",
            corpus.path().to_str().unwrap(),
            "--top-k",
            "2",
            "--hybrid",
            "--time-decay-field",
            "published_date",
            "--time-decay-halflife",
            "30d",
            "--no-rerank",
            "--format",
            "json",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "query failed: {out:?}");
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let hits = v.get("hits").and_then(|h| h.as_array()).expect("hits array");
    assert!(hits.len() >= 2);
    // Fresh doc's source path contains /docs.jsonl#1 or similar id=2 marker.
    let top = hits[0].get("source_path").and_then(|s| s.as_str()).unwrap_or("");
    assert!(
        top.contains("#2") || top.contains("id=2") || top.contains("2026"),
        "top hit should be the 2026 doc; got {top}"
    );
}
```

**Note:** The JSONL ingest flag `--metadata-fields published_date:date` must match the actual flag shipped in #41. Grep `metadata_fields` in `fastrag-cli/src/args.rs` to confirm the exact syntax.

- [ ] **Step 2: Run the test**

Run: `cargo test -p fastrag-cli --test temporal_decay_e2e`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/temporal_decay_e2e.rs
git commit -m "test(cli): --time-decay-* e2e + error paths"
```

---

## Task 12: HTTP JSON body — hybrid + time_decay

**Files:**
- Modify: `fastrag-cli/src/http.rs`

- [ ] **Step 1: Extend `QueryBody`**

Edit `fastrag-cli/src/http.rs`. Find the `QueryBody` struct (search for `struct QueryBody`). Add optional fields:
```rust
#[serde(default)]
pub hybrid: bool,
#[serde(default = "default_rrf_k")]
pub rrf_k: u32,
#[serde(default = "default_rrf_overfetch")]
pub rrf_overfetch: usize,
#[serde(default)]
pub time_decay: Option<TimeDecayBody>,

// ... add at module level:
fn default_rrf_k() -> u32 { 60 }
fn default_rrf_overfetch() -> usize { 4 }

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeDecayBody {
    pub field: String,
    #[serde(default = "default_halflife")]
    pub halflife: String,
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default = "default_dateless")]
    pub dateless_prior: f32,
    #[serde(default = "default_blend")]
    pub blend: String,
}
fn default_halflife() -> String { "30d".into() }
fn default_weight() -> f32 { 0.3 }
fn default_dateless() -> f32 { 0.5 }
fn default_blend() -> String { "multiplicative".into() }
```

- [ ] **Step 2: Wire into the query handler**

Inside the handler that processes `QueryBody` (search for `QueryOpts {` in `http.rs`), build `HybridOpts` from `body.hybrid`, `body.rrf_k`, `body.rrf_overfetch`, `body.time_decay`. Reuse the `build_hybrid_opts` helper from Task 9 (extract it to `fastrag_cli::args` or a shared module so both CLI and HTTP paths use the same validation).

Pseudo-code for the handler insert:
```rust
let hybrid_opts = {
    use fastrag::corpus::hybrid::{BlendMode, HybridOpts, TemporalOpts};
    let temporal = if let Some(t) = body.time_decay {
        let halflife = humantime::parse_duration(&t.halflife)
            .map_err(|e| /* return 400 */ ...)?;
        let blend = match t.blend.as_str() {
            "multiplicative" => BlendMode::Multiplicative,
            "additive" => BlendMode::Additive,
            other => return /* 400 */ ...,
        };
        Some(TemporalOpts {
            date_field: t.field,
            halflife,
            weight_floor: t.weight,
            dateless_prior: t.dateless_prior,
            blend,
            now: chrono::Utc::now(),
        })
    } else {
        None
    };
    let enabled = body.hybrid || temporal.is_some();
    HybridOpts {
        enabled,
        rrf_k: body.rrf_k,
        overfetch_factor: body.rrf_overfetch,
        temporal,
    }
};
let query_opts = fastrag::corpus::QueryOpts {
    cwe_expand: cwe_expand_effective,
    hybrid: hybrid_opts,
};
```

Return `400 Bad Request` with `{"error": "..."}` when:
- `time_decay` present but `field` empty
- `halflife` unparseable
- `blend` not `multiplicative` or `additive`

- [ ] **Step 3: Build**

Run: `cargo build -p fastrag-cli`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/src/http.rs
git commit -m "feat(http): accept hybrid + time_decay fields on POST /query"
```

---

## Task 13: HTTP e2e test

**Files:**
- Create: `fastrag-cli/tests/temporal_decay_http_e2e.rs`

- [ ] **Step 1: Write the test**

Mirror `fastrag-cli/tests/cwe_expand_http_e2e.rs`. Spawn `serve-http` on an ephemeral port, POST a request body with `hybrid: true` + `time_decay` object, assert status 200 and hit ordering. Add a second test that POSTs with bad blend value and asserts 400.

```rust
//! End-to-end: POST /query with hybrid + time_decay body.

#![cfg(feature = "retrieval")]

use reqwest::blocking::Client;
use serde_json::json;
// ... same bootstrap pattern as cwe_expand_http_e2e.rs ...

#[test]
fn http_query_with_time_decay_returns_200() {
    let (addr, _guard) = spawn_serve_http(/* corpus with dated docs */);
    let client = Client::new();
    let body = json!({
        "query": "openssl heap overflow",
        "top_k": 3,
        "hybrid": true,
        "time_decay": {
            "field": "published_date",
            "halflife": "30d",
            "weight": 0.3,
            "dateless_prior": 0.5,
            "blend": "multiplicative"
        }
    });
    let resp = client
        .post(format!("http://{addr}/query"))
        .json(&body)
        .send()
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().unwrap();
    assert!(v.get("hits").and_then(|h| h.as_array()).map(|a| !a.is_empty()).unwrap_or(false));
}

#[test]
fn http_query_with_bad_blend_returns_400() {
    let (addr, _guard) = spawn_serve_http(/* any corpus */);
    let client = Client::new();
    let body = json!({
        "query": "x",
        "top_k": 1,
        "time_decay": { "field": "published_date", "blend": "bogus" }
    });
    let resp = client
        .post(format!("http://{addr}/query"))
        .json(&body)
        .send()
        .unwrap();
    assert_eq!(resp.status(), 400);
}
```

Copy the `spawn_serve_http` helper and corpus bootstrap from `fastrag-cli/tests/cwe_expand_http_e2e.rs`.

- [ ] **Step 2: Run the test**

Run: `cargo test -p fastrag-cli --test temporal_decay_http_e2e`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/temporal_decay_http_e2e.rs
git commit -m "test(http): hybrid + time_decay e2e via POST /query"
```

---

## Task 14: MCP search_corpus params

**Files:**
- Modify: `crates/fastrag-mcp/src/lib.rs`

- [ ] **Step 1: Extend the tool input schema**

Find the `search_corpus` tool definition in `crates/fastrag-mcp/src/lib.rs` (search for `search_corpus` and for its input struct). Add the same optional fields as the HTTP body: `hybrid`, `rrf_k`, `rrf_overfetch`, `time_decay`. Reuse the same `TimeDecayBody` shape but under the MCP crate — keep the schema JSON description fields informative.

Build `HybridOpts` identically to the HTTP path (Task 12) — extract the shared helper to `crates/fastrag/src/corpus/hybrid.rs` under a new public function `build_hybrid_opts_from_parts` if three call sites reuse it.

- [ ] **Step 2: Run existing MCP tests**

Run: `cargo test -p fastrag-mcp --features mcp-search`
Expected: all existing tests pass.

Add one new test covering the new params if the MCP crate already has a `search_corpus` test harness. If not, skip — the CLI + HTTP e2e already exercise the shared `HybridOpts` construction.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-mcp/src/lib.rs
git commit -m "feat(mcp): search_corpus accepts hybrid + time_decay params"
```

---

## Task 15: Gold-set temporal entries + eval axis

**Files:**
- Modify: `tests/gold/questions.json`
- Modify: `crates/fastrag-eval/src/matrix.rs` (if the matrix enumerates variants)

- [ ] **Step 1: Inspect current gold set schema**

Run: `head -60 tests/gold/questions.json`
Understand the question object shape — expected fields are `id`, `query`, `expected_hits` (or similar), and optional tags.

- [ ] **Step 2: Add 6 temporal entries**

Append to `tests/gold/questions.json`. Three recency-seeking, three historical-reference. Example — adjust to match the real schema:
```json
{
  "id": "temporal-recency-1",
  "query": "latest OpenSSL heap overflow advisory",
  "expected_ids": ["advisory-2026-03-15"],
  "tags": ["temporal", "recency-seeking"]
},
{
  "id": "temporal-historical-1",
  "query": "CVE-2014-0160 Heartbleed description",
  "expected_ids": ["cve-2014-0160"],
  "tags": ["temporal", "historical-reference"]
}
```
Use real corpus entries that already exist in `tests/gold/corpus/` — don't invent ids that won't resolve.

- [ ] **Step 3: Add a temporal axis to the eval matrix**

In `crates/fastrag-eval/src/matrix.rs`, find the `Variant` enum (search around line 18). Add a new variant `TemporalOn` to complement the existing ablations. Update the `run_matrix` function to pass `QueryOpts::hybrid.temporal = Some(...)` for that variant. Halflife `30d`, alpha `0.3`, prior `0.5`, blend `Multiplicative`, `now` set from a fixed test timestamp.

- [ ] **Step 4: Run the eval**

Run: `cargo test --workspace --features eval`
Expected: existing matrix tests pass, new temporal variant exercises the decay path without errors.

- [ ] **Step 5: Commit**

```bash
git add tests/gold/questions.json crates/fastrag-eval/src/matrix.rs
git commit -m "test(eval): temporal axis + gold-set entries for recency vs historical"
```

---

## Task 16: Docs — README + CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Draft the docs additions**

Prepare draft text for README (new "Hybrid retrieval + temporal decay" section, likely under the retrieval docs) and for CLAUDE.md (new build/test commands).

- [ ] **Step 2: Run draft through doc-editor skill**

Per `CLAUDE.md`: every Edit/Write to a `.md` file must go through the `doc-editor/SKILL.md` skill. Launch it as a foreground Haiku Agent and pass the draft text.

- [ ] **Step 3: Apply cleaned prose**

Edit README.md and CLAUDE.md with the returned cleaned prose.

**README additions (example shape):**
```markdown
### Hybrid retrieval + temporal decay

Opt in to BM25 + dense RRF hybrid retrieval with `--hybrid`:

```bash
fastrag query "openssl vulnerability" --corpus ./corpus --hybrid
```

For security corpora where freshness matters, layer a recency decay:

```bash
fastrag query "latest openssl advisory" \
    --corpus ./corpus \
    --time-decay-field published_date \
    --time-decay-halflife 30d \
    --time-decay-weight 0.3
```

Flags:
- `--hybrid` — enable BM25 + dense RRF fusion
- `--rrf-k <int>` — RRF k parameter (default 60)
- `--time-decay-field <name>` — metadata Date field to use for decay (implies `--hybrid`)
- `--time-decay-halflife <humantime>` — decay halflife (default `30d`)
- `--time-decay-weight <float>` — alpha floor (default `0.3`)
- `--time-decay-dateless-prior <float>` — neutral prior for dateless docs (default `0.5`)
- `--time-decay-blend <multiplicative|additive>` — blend mode (default multiplicative)
```

**CLAUDE.md additions (append to the Build & Test block):**
```bash
cargo test -p fastrag --features retrieval --lib corpus::hybrid    # Hybrid module unit tests
cargo test -p fastrag --features retrieval --test hybrid_retrieval # Integration: BM25+dense RRF
cargo test -p fastrag --features retrieval --test temporal_decay   # Integration: recency decay
cargo test -p fastrag-cli --test hybrid_e2e --features retrieval    # CLI --hybrid e2e
cargo test -p fastrag-cli --test temporal_decay_e2e --features retrieval  # CLI --time-decay-* e2e
cargo test -p fastrag-cli --test temporal_decay_http_e2e --features retrieval  # HTTP decay e2e
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: hybrid retrieval + temporal decay usage"
```

---

## Task 17: Final verification + push

**Files:** none

- [ ] **Step 1: Run fmt**

Run: `cargo fmt --all`
Expected: no changes, or commit the formatting changes:
```bash
git add -u
git commit -m "style: cargo fmt"
```

- [ ] **Step 2: Run the full clippy gate**

Run: `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings`

*(`hybrid` is not a real Cargo feature anymore — reuse the existing multi-feature invocation from CLAUDE.md, which is: `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings`. If the literal string includes a now-nonexistent feature, drop `hybrid` and keep the rest.)*

Corrected command to run:
```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,contextual,eval,nvd,hygiene -- -D warnings
```
Expected: clean.

- [ ] **Step 3: Run the full test suite**

Run: `cargo test --workspace --features retrieval,rerank,contextual,eval,nvd,hygiene`
Expected: all pass.

- [ ] **Step 4: Write the closing commit with Closes #48**

If the last commit in the sequence above does not already carry `Closes #48`, amend or add a trailing empty-delta commit is not the right approach (per project preference, never amend published commits — create a new commit). Instead, ensure the final meaningful commit (Task 16 docs or Task 17 fmt) carries the `Closes #48` trailer. Retroactively: add a final "chore: close #48" commit only if there's no better home — prefer putting it on the last real-content commit.

Pragmatic shortcut: make the Task 16 commit the closing commit by rewriting its message:
```bash
git commit --amend -m "$(cat <<'EOF'
docs: hybrid retrieval + temporal decay usage

Closes #48
EOF
)"
```
Only amend if that commit has NOT been pushed yet — we push once at the end (Step 5).

- [ ] **Step 5: Push**

Run: `git push`
Expected: success.

- [ ] **Step 6: Watch CI**

Per project convention: invoke the `ci-watcher` skill as a background Haiku Agent. Do **not** use `gh run watch`. Read `.claude/skills/ci-watcher.md` and pass it to an `Agent` call with `model=haiku, run_in_background=true`.

- [ ] **Step 7: Declare done when CI goes green**

Confirm all workflows green before closing the task. If any workflow fails, diagnose root cause (fix failing test, fix clippy) and push a new commit — do not skip hooks.

---

## Spec coverage self-review

- Architecture diagram (spec §Architecture) → Task 6 wires `query_hybrid` into `query_corpus_with_filter_opts`; Task 5 is `query_hybrid`.
- Decay formula, multiplicative + additive, dateless prior (spec §Decay formula) → Task 2 covers multiplicative; additive variant is scaffolded in the enum and the `_blend` param is threaded, but the additive code path is not yet implemented. **Gap:** add additive logic inside `decay_factor` when `BlendMode::Additive` is requested OR remove additive from the public surface until a follow-up issue lands it.
- Defaults table (spec §Defaults) → Task 9 sets CLI defaults; Task 12 sets HTTP defaults.
- Module layout (spec §Module layout) → Tasks 1, 2, 3, 4, 5.
- CLI + HTTP surface (spec §CLI and HTTP surface) → Tasks 9, 12, 14.
- Backward compat (spec §Backward compatibility) → Task 6 Step 3 runs full `cargo test --workspace --features retrieval` to confirm zero regressions with defaults.
- Tests (spec §Eval and testing) → Tasks 2, 3, 4, 5, 7, 8, 10, 11, 13, 15.
- TDD ordering (spec §TDD ordering) → Tasks 2–5 follow it exactly.
- Deferred issues (spec §Deferred) → #53, #54, #55 already filed.

**Gap to resolve before shipping:** the additive blend path. Two options:
1. Add the additive implementation to Task 2 (expand `decay_factor` to honor `BlendMode::Additive` — requires an extra argument for the pre-fusion normalized RRF score, so move this into `apply_decay` instead). Add 3 more unit tests for additive mode.
2. Drop `--time-decay-blend additive` from Task 9's CLI + Task 12's HTTP surface and file a new issue for it.

**Recommendation:** drop additive from the CLI/HTTP surface for this issue (file follow-up issue). The spec exposes additive as an "escape valve for eval A/B testing"; without eval data showing it helps, shipping additional surface area is speculative. File a new issue: "Additive blend mode for temporal decay" and remove the `TimeDecayBlendArg::Additive` enum variant + the `time_decay_blend` CLI flag and the HTTP `blend` field. Keep `BlendMode::Additive` reserved in the `hybrid.rs` module (dead-code-allowed) so the follow-up issue doesn't need an enum rename.

**Plan amendment:** between Task 8 and Task 9, insert a half-task to file the additive follow-up issue, and strip the `blend` flag / field from Tasks 9, 12, 14, 16. The design spec doesn't need changes because it already categorizes `blend` as multiplicative-default with additive available — just the "available as flag" becomes "reserved for follow-up."

---

## Execution notes

- **Commit cadence:** one commit per task locally. Push once, at the end of Task 17. Per project preference for multi-landing plans.
- **Final commit message carries `Closes #48`.** Other commits in the sequence do not close the issue.
- **No worktrees.** Work in the main checkout.
- **Skip hooks:** never. If a pre-commit hook fires, fix the root cause.
- **CI watcher:** invoke as background Haiku Agent, never `gh run watch` directly.
