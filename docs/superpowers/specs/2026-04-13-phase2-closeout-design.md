# Phase 2 Closeout — Design Spec

**Date:** 2026-04-13
**Issues:** #35, #36, #30
**Scope:** NVD security query set, CI eval-regression gate, eval-driven chunking tuning

## Summary

Three tightly coupled deliverables that close out Phase 2. The NVD query set (#35) enables real retrieval metrics on security baselines. The CI eval gate (#36) prevents quality regressions on every PR. The chunking sweep (#30) replaces the guessed defaults with data-driven ones.

Landing order: #35 → #36 → #30. After #30 lands, close epic #33 (Phase 2 complete).

## 1. NVD Security Query Set (#35)

### Goal

Hand-curated security query set bundled with the eval harness so NVD/CWE baselines report retrieval metrics (recall@10, MRR@10, nDCG@10) instead of index-footprint-only numbers.

### Design

**File:** `crates/fastrag-eval/src/datasets/security_queries.json`

**Schema:** Same as existing gold set — `{id, question, must_contain_cve_ids, must_contain_terms, notes}`.

**~30 queries across five categories:**

| Category | Count | Example |
|----------|-------|---------|
| Exact CVE lookup | ~8 | "What is CVE-2021-44228?" |
| Semantic CVE description | ~8 | "remote code execution in logging frameworks" |
| CWE/class queries | ~6 | "path traversal vulnerabilities in 2023" |
| Vendor/product faceting | ~4 | "Apache vulnerabilities rated CRITICAL" |
| Negative/edge cases | ~4 | Queries for Rejected/Disputed CVEs that must NOT appear |

Queries target CVEs present in the NVD 2023/2024 feeds that the eval harness already downloads.

### Wiring

`load_nvd()` in `crates/fastrag-eval/src/datasets/nvd.rs` uses the bundled query set when no external path is given. `load_nvd_corpus_with_queries()` already accepts external queries — this adds a default.

### Deliverables

- `security_queries.json` committed (not downloaded)
- NVD/CWE baselines refreshed in `docs/eval-baselines/` with retrieval metrics
- `docs/eval-baselines/README.md` updated to note NVD/CWE coverage

## 2. CI Eval-Regression Gate (#36)

### Goal

Prevent retrieval quality regressions on PRs with a fast, real-model eval gate backed by a cached pre-built corpus.

### Architecture

Two tiers: fast PR gate + full weekly eval.

#### PR Gate (`eval-gate` job in `ci.yml`)

**Trigger path filter:** PRs touching any of:
- `crates/fastrag-embed/`
- `crates/fastrag-eval/`
- `crates/fastrag/`
- `crates/fastrag-index/`
- `crates/fastrag-rerank/`
- `crates/fastrag-context/`
- `crates/fastrag-tantivy/`
- `fastrag-cli/`
- `tests/gold/`
- `docs/eval-baselines/`

**Steps:**
1. Restore pre-built corpus from `actions/cache`
   - Cache key: `eval-corpus-{hash(tests/gold/corpus/**, crates/fastrag-embed/**, crates/fastrag-index/**)}`
   - Covers both contextual and raw corpus variants
2. On cache hit: run query-only eval against 110-entry gold set (~1-2 min)
   - On cache miss: skip gate, emit GitHub warning annotation ("eval corpus cache miss — gate skipped, will run in next weekly")
3. Diff against `docs/eval-baselines/current.json`
   - Fail on >2% regression in hit@5 or MRR@10 for any variant

**Waiver mechanism:** `Eval-Regression-Justified: <reason>` in commit trailer. The gate reads the trailer, logs the reason, and exits 0.

**Target wall time:** ~1-2 min on cache hit (query + rerank, no indexing).

#### Weekly Workflow Changes

After running the full matrix eval, persist both corpus variants into `actions/cache` using the same cache key the PR gate restores. No other changes to the existing weekly logic.

### What the PR gate does NOT do

- Auto-refresh baselines (stays manual per existing policy in `docs/eval-baselines/README.md`)
- Run on pushes to main (weekly catches those)
- Download or run embedding models (corpus is pre-built)

## 3. Eval-Driven Chunking Tuning (#30)

### Goal

Replace the guessed chunking defaults (Basic, 1000 chars, 0 overlap) with a data-driven choice validated by the eval harness.

### Sweep Grid

| Dimension | Values |
|-----------|--------|
| Strategy | `basic`, `by-title`, `recursive` |
| Max characters | 500, 800, 1000, 1500 |
| Overlap | 0, 100, 200 |

36 combinations total. `semantic` excluded — requires an embedding call per chunk boundary, making it 10-50x slower to index. Watch-listed for a future one-off experiment.

### Sweep Script

**File:** `scripts/chunking-sweep.sh`

**Per combination:**
1. Rebuild the 50-doc fixture corpus with the given chunking params
2. Run gold set eval (all 4 matrix variants)
3. Append a row to `target/chunking-sweep/results.tsv`

**Output columns:** strategy, size, overlap, chunk_count, index_size_bytes, index_build_ms, hit@1, hit@5, hit@10, mrr@10, per-variant breakdowns.

**Prerequisites:** llama-server + GGUF models on PATH (same as nightly CI).

**Estimated wall time:** ~2-3 hours for 36 combos on the dev box.

### Winner Selection Criteria (priority order)

1. Best Primary hit@5 (quality is king)
2. Tiebreak: best MRR@10
3. Tiebreak: fewer chunks (smaller index, faster queries)

### Deliverables

- `docs/chunking-eval.md` — markdown table of full sweep results + analysis
- Updated defaults in `fastrag-cli/src/args.rs` and `crates/fastrag-core/src/chunking.rs`
- Refreshed `docs/eval-baselines/current.json` locked to the new defaults
- CHANGELOG note: "Rebuilding corpus recommended to benefit from new chunking defaults"

## Ordering and Dependencies

```
#35 NVD query set
  ↓
#36 CI eval gate (needs baselines refreshed by #35)
  ↓
#30 Chunking sweep (needs gate in place so new defaults don't regress)
```

One commit per issue with `Closes #N` in the message.

After #30 lands, close epic #33 (Phase 2 complete).

## Out of Scope

- Semantic chunking strategy (watch-listed)
- Auto-refreshing baselines
- Full NVD feed eval in CI (too slow — stays a local experiment)
- Phase 3 work
