# Temporal Decay + Hybrid Retrieval — Design

**Issue:** crook3dfingers/fastrag#48 (Phase 3 Step 8)
**Status:** draft
**Date:** 2026-04-14

## Problem

The current retrieval path is dense-only. A 2024 advisory for the same vulnerability ranks identically to a 2019 one, because the ranking has no recency signal. The corpus also lacks BM25+dense hybrid, even though BM25 is measurably stronger than dense alone for exact-match signals that dominate a security corpus (CVE-IDs, product names, version strings).

This issue lands both pieces together: a 2-way (BM25 + dense) hybrid retrieval path using Reciprocal Rank Fusion, and an opt-in post-fusion recency decay.

## Design decisions and why

The issue body proposed a 3-way RRF over `[BM25 rank, dense rank, recency rank]`. Research into 2025–2026 systems (Vespa `freshness`, Qdrant `exp_decay`, Elastic `function_score` with decay, OpenSearch, Azure AI Search, Snowflake Cortex, arXiv 2509.19376) showed that every production stack treats recency as an explicit decay function, not as a co-equal rank list. The rank-based approach collapses the time signal (a 5-year gap and a 5-day gap look identical once ranked), forces an arbitrary policy for dateless and tied-date documents, and makes a document's "recency boost" dependent on which other documents happened to be retrieved.

The ChatGPT second-opinion pass confirmed the multiplicative baseline but adjusted three defaults:

- `alpha` was `0.3` as a floor multiplier — kept for old docs, but dateless docs get a separate `0.5` neutral prior rather than the same floor (dateless should not be punished as if infinitely old).
- `halflife` was proposed at `14d` — moved to `30d` as the general opt-in default. `7–14d` belongs to an auto-aggressive mode that ships in the follow-up issue (crook3dfingers/fastrag#53).
- The additive blend from Grofsky 2509.19376 is exposed as an opt-in flag, not the default, since multiplicative combination avoids the "fresh but off-topic" failure mode that security searches are especially sensitive to.

## Architecture

```
query
  │
  ├──► BM25 (Tantivy)  ──► list of (id, score)  ──► rank list A
  │
  ├──► dense (HNSW)    ──► list of (id, score)  ──► rank list B
  │
  └──► optional recency decay factor on fused result

fuse(A, B) via unweighted RRF k=60  ──►  rrf_score per id
                                            │
                                            ▼
                           if time-decay opts present:
                               final = rrf_score · decay_factor(age, halflife, alpha, blend)
                           else:
                               final = rrf_score
                                            │
                                            ▼
                           sort desc, truncate to top_k
                                            │
                                            ▼
                           optional cross-encoder rerank stage (unchanged)
```

**Placement.** Decay applies post-fusion, pre-rerank. Applying decay inside the rerank stage (the issue's "simpler alternative") would only help when rerank is enabled, and would couple a retrieval-level concern to an optional reranker.

**Overfetch.** RRF needs each retriever to return more than `top_k` candidates for fusion to matter. Default factor is `4×`; the existing adaptive `4×/16×/32×` ladder used for filter survival is reused when a metadata filter is also active.

**Filter integration.** Filters run after fusion on fused candidates — identical semantics to today's dense-only filter path, just over fused results. No change to the filter expression language.

**Hybrid without temporal.** `--hybrid` is exposed on its own so users can opt into BM25+dense fusion without recency decay. `--time-decay-field` implies `--hybrid` (one less flag to remember; passing time-decay options without `--hybrid` is not a useful combination).

## Decay formula

```
age_days     = max(0, (query_time - doc_date) / 86400)
decay_factor = alpha + (1 - alpha) · exp(-ln(2) · age_days / halflife)
final_score  = rrf_score · decay_factor         # multiplicative (default)
```

Properties:
- `age = 0` → `decay_factor = 1.0`
- `age = halflife` → `decay_factor = (1 + alpha) / 2`
- `age → ∞` → `decay_factor → alpha` (floor; never zero)
- `alpha = 1.0` → `decay_factor ≡ 1.0` (escape valve: disables decay without removing flags)

**Dateless docs:** `decay_factor = dateless_prior` (default `0.5`), independent of halflife and alpha. Rationale: undated CVE/advisory records exist in practice (`dateReserved` / `datePublished` observed missing in real CVE JSON), they should sit between fresh and very-stale peers, not be punished as if infinitely old.

**Invalid / future dates:** parse failure is treated as dateless. Future-dated docs (negative age) clamp to age 0 (equivalent to "today"). Parse warnings happen at ingest time, not per-query.

**Additive blend (opt-in via `--time-decay-blend additive`):**
```
final = (1 - w) · normalized_rrf + w · exp(-ln(2) · age_days / halflife)
```
where `normalized_rrf` is min-max normalized within the candidate set and `w = 1 - alpha`, so the same `alpha` knob makes sense in both modes.

## Defaults

| Flag | Default | Rationale |
|---|---|---|
| `--hybrid` | off | opt-in; preserves current dense-only behavior |
| `--rrf-k` | `60` | RRF canonical default across Elastic, OpenSearch, Azure |
| `--rrf-overfetch` | `4` | enough for fusion to matter without doubling latency |
| `--time-decay-field` | unset | opt-in |
| `--time-decay-halflife` | `30d` | general freshness; narrower halflives belong to auto mode (#53) |
| `--time-decay-weight` (alpha) | `0.3` | floor; old docs keep 30% relevance |
| `--time-decay-dateless-prior` | `0.5` | neutral |
| `--time-decay-blend` | `multiplicative` | safer; `additive` available as escape |

## Module layout

**New:** `crates/fastrag/src/corpus/hybrid.rs`

```rust
pub fn query_hybrid(
    store: &fastrag_store::Store,
    query: &str,
    vector: &[f32],
    top_k: usize,
    filter: Option<&FilterExpr>,
    opts: &HybridOpts,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<(u64, f32)>, CorpusError>;

#[derive(Debug, Clone, Default)]
pub struct HybridOpts {
    pub enabled: bool,
    pub rrf_k: u32,
    pub overfetch_factor: usize,
    pub temporal: Option<TemporalOpts>,
}

#[derive(Debug, Clone)]
pub struct TemporalOpts {
    pub date_field: String,
    pub halflife: Duration,
    pub weight_floor: f32,
    pub dateless_prior: f32,
    pub blend: BlendMode,
    pub now: DateTime<Utc>,  // injected for test determinism
}

pub enum BlendMode { Multiplicative, Additive }
```

**Pure functions** (unit-test targets, no I/O):
- `fusion::rrf_fuse` — already in `fastrag-index`, reused as-is
- `hybrid::decay_factor(age_days, halflife_days, alpha, dateless_prior, blend) -> f32`
- `hybrid::apply_decay(rrf_scores, dates, opts) -> Vec<ScoredId>`

**Integration in `corpus/mod.rs::query_corpus_with_filter_opts`:**
The current `store.query_dense(...)` call becomes a branch:
- `opts.hybrid.enabled == false` → dense-only path (unchanged; regression-safe)
- `opts.hybrid.enabled == true` → `hybrid::query_hybrid(...)` returns the fused+decayed `(id, score)` list; existing metadata-filter + overfetch logic proceeds unchanged

`QueryOpts` grows a `hybrid: HybridOpts` field. Existing callers pick up `HybridOpts::default()` via the `Default` impl — no signature break.

**Store API additions:** none. `query_bm25` and `query_dense` are already on `fastrag_store::Store`. Date extraction reuses `fetch_metadata` + `TypedValue::Date`.

**Manifest:** no new required fields. The date field is a query-time parameter; per-type persisted date-field maps ship with crook3dfingers/fastrag#55.

**Feature flags:** hybrid + temporal decay sit behind the existing `retrieval` feature. No new workspace feature flag.

## CLI and HTTP surface

**CLI (`query` subcommand, all optional):**
```
--hybrid
--rrf-k <u32>                         # default 60
--rrf-overfetch <usize>               # default 4

--time-decay-field <name>             # enables decay; implies --hybrid
--time-decay-halflife <humantime>     # default "30d"
--time-decay-weight <f32>             # alpha; default 0.3
--time-decay-dateless-prior <f32>     # default 0.5
--time-decay-blend <multiplicative|additive>
```

Passing `--time-decay-*` without `--time-decay-field` is a hard error.

**HTTP (`POST /query` body, additive):**
```json
{
  "query": "...",
  "top_k": 10,
  "hybrid": true,
  "rrf_k": 60,
  "rrf_overfetch": 4,
  "time_decay": {
    "field": "published_date",
    "halflife": "30d",
    "weight": 0.3,
    "dateless_prior": 0.5,
    "blend": "multiplicative"
  }
}
```

Validation mirrors the CLI: `time_decay` object present forces `hybrid: true`; decay params with no `field` is rejected.

**MCP `search_corpus` tool:** identical `time_decay` + `hybrid` optional parameters with the same semantics.

**Response shape:** unchanged. `SearchHitDto.score` carries the final fused-and-decayed score. Debug breakdown (`rrf_score`, `decay_factor`) is not exposed in this issue — can be added behind a debug flag later if eval needs it.

## Backward compatibility

- Omitting every new flag → byte-identical behavior to today
- Corpora without `_chunk_text` (legacy HNSW-only, pre-Store) → `--hybrid` errors with a clear message pointing at reindex
- Unknown date field → query-time error, not a silent skip

## Eval and testing

**Unit tests** (`#[cfg(test)]` in-file):
- `hybrid::decay_factor`: age=0, age=halflife, age→∞, alpha=1.0, dateless branch, multiplicative vs additive on known inputs, negative-age clamp
- `hybrid::apply_decay`: ordering preserved for uniform-age candidates; changes with date spread; dateless candidates interleave at neutral prior
- `hybrid::query_hybrid`: empty BM25 falls back to dense; empty dense falls back to BM25; both empty returns empty; overfetch respected

**Integration tests** (`crates/fastrag/tests/`):
- `hybrid_retrieval.rs` — small Store fixture; 2-way RRF reorders vs pure dense and pure BM25 (concrete id order assertion)
- `temporal_decay.rs` — injected `published_date`s spanning 2 years; decay-off preserves baseline order; decay-on promotes fresh over equally-relevant stale; dateless interleaves between fresh and very-stale; `now` injected to prevent wall-clock flakiness

**CLI / HTTP e2e** (`fastrag-cli/tests/`):
- `hybrid_e2e.rs` — CLI `query --hybrid` returns valid JSON on exit 0
- `temporal_decay_e2e.rs` — CLI decay flags succeed on a corpus with the field; clear error on missing field; clear error on decay flags without `--time-decay-field`
- `temporal_decay_http_e2e.rs` — same assertions via `POST /query`

**Eval gold-set additions** (`tests/gold/questions.json`), tagged `temporal: true`:
- 3 recency-seeking entries — expected hits are the more recent advisories (decay helps)
- 3 historical-reference entries — query contains a specific CVE-ID, expected hits are the canonical older docs (the `alpha=0.3` floor preserves these; regression guard against decay overreach)

**Regression guards:**
- Existing `cargo test --workspace --features retrieval` passes unchanged with decay off
- Eval matrix gains a `temporal: [off, on]` axis; baseline comparison via the existing `--baseline docs/eval-baselines/current.json` flow
- Slack gate (existing eval harness feature): temporal=off matches current baseline within tolerance; temporal=on must not regress historical-reference queries

## TDD ordering

1. `decay_factor` pure function
2. `apply_decay` over a slice
3. `query_hybrid` with a Store fixture
4. CLI / HTTP wiring
5. Gold-set eval entries

## Deferred — follow-up issues

- crook3dfingers/fastrag#53 — query-conditional auto mode (mild vs aggressive halflife, recency-marker detection)
- crook3dfingers/fastrag#54 — weighted RRF for hybrid retrieval
- crook3dfingers/fastrag#55 — per-document-type date field semantics (manifest date-field map)

## References

- arXiv 2509.19376 — *Solving Freshness in RAG: A Simple Recency Prior* (cybersec evaluation; half-life prior)
- arXiv 2509.01306 — *Re3: Learning to Balance Relevance and Recency* (learned intent gate)
- arXiv 2601.22196 — *Linux Kernel Recency Matters, CVE Severity Doesn't*
- arXiv 2502.20245 — hybrid vs dense-only on technical corpora
- Elastic weighted RRF (2025), Qdrant decay functions (1.14+), Vespa freshness feature, Snowflake Cortex Search boost/decay (Apr 2025)
- Cormack, Clarke, Büttcher — original RRF paper
