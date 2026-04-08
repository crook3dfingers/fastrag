# fastrag Phase 2 — Roadmap Rewrite (2026-04-07)

## Why this document exists

The original Phase 2 roadmap sequenced multi-model candle embedders (#31b) before reranking (#39) and never included hybrid retrieval, Contextual Retrieval, or a type-safe embedder invariant. A fresh review of `docs/rag-research-2026-04.md` — cross-checked against 2026 MTEB leaderboards, fastembed-rs 5.x, bge-reranker-v2-m3 ONNX availability, and current Contextual Retrieval / late-chunking discussion — made clear that the sequencing was targeting the lowest-ROI lever first.

The #31b spec and plan have been deleted. Issue #31 will be closed as superseded, and this document is the new source of truth for Phase 2 ordering.

## Principles driving the rewrite

1. **Foundational correctness before new features.** The type-system embedder invariant (research doc §2 / shim lesson #1) must land before any multi-model work, because multi-model is the exact scenario where the invariant has to hold.
2. **Highest-ROI quality levers first.** Reranker and Contextual Retrieval each outweigh any embedder swap by an order of magnitude on retrieval quality. They come before the multi-model work they'd otherwise be blocked behind.
3. **Hybrid retrieval is not optional for security corpora.** BM25 + dense + exact CVE-ID short-circuit via Tantivy is a correctness requirement for fastrag's target users, not a performance enhancement.
4. **Back-compat with existing corpora.** Every step must either preserve the ability to read previously indexed corpora or provide an explicit migration path with clear errors. No silent format drift.
5. **TDD red-green-refactor on every change.** Research-backed priorities do not override the repo's testing discipline.

## Phase 2 — new ordering

Each step is a sub-project with its own spec, plan, and PR. Issue numbers are placeholders until the new umbrella issue and sub-issues are filed.

### Step 1 — Embedder invariant refactor (foundational)

**Goal.** Make the dim-4 silent-trap bug unrepresentable at the type level, without breaking the current single-model corpus or the CLI surface.

**Scope.**
- `Embedder` trait gains an associated const `DIM: usize` and two distinct input types, `QueryText` and `PassageText`, each with its own `embed` method. `embed()` in its untyped form is removed or narrowed.
- `Index<E: Embedder>` parameterizes the vector store by the embedder type so mixing an `Index<Nomic>` with a query against `Bge` is a compile error.
- On-disk manifest is extended with `{embedder_id, embedder_version, dim, prefix_scheme_hash}` and a canary vector. Index open re-embeds the canary and verifies cosine similarity against the stored vector; mismatch → hard error with a clear migration message.
- The CLI's dynamic dispatch (`Arc<dyn Embedder>`) is preserved via a type-erased wrapper layer that still enforces the static invariants at construction time. Users do not need to thread generics through their own code.
- All existing candle/BGE/OpenAI/Ollama/Mock embedders updated to the new trait shape. No functional change to their behavior.

**Out of scope.** New models, new backends, reranking, hybrid retrieval. Purely a refactor plus the canary.

**Why now.** Every subsequent step (new backend, new model, Contextual Retrieval, reranker that depends on embedder dim) benefits from a statically enforced embedder identity. Doing it later means re-plumbing the whole stack twice.

### Step 2 — fastembed-rs backend + flagship 2026 presets

**Goal.** Add `fastembed-rs` (ort + HF tokenizers, 3–5× faster than candle on CPU with 60–80% less memory per benchmarks) as a second backend under the new trait, and ship nomic-embed-text-v1.5 + snowflake-arctic-embed-m-v1.5 as flagship presets. Keep the existing candle/bge path as back-compat for previously indexed corpora.

**Scope.**
- New `fastrag-embed-fastembed` crate (or a feature flag inside `fastrag-embed`) wrapping `fastembed-rs`. Implements `Embedder` for the two flagship models with Matryoshka truncation to 512d as the default, 768d as an opt-in.
- Presets: `nomic-v1.5`, `arctic-m-v1.5`, and `embedding-gemma-300m` as a watch-list preset (`#[cfg(feature = "experimental-models")]`).
- Each preset declares its own prefix scheme via the typed trait from Step 1 (nomic uses `search_query: ` / `search_document: `; arctic uses a different pair; gemma uses yet another — the trait forces each to be explicit).
- Candle path with `BgeSmallEmbedder` stays as-is, re-exported under a `legacy-candle` feature for corpora already indexed with it.
- Eval matrix (nomic vs arctic vs legacy bge) on the security corpus + NFCorpus, committed under `docs/evals/`, decides the default.

**Out of scope.** Reranking, hybrid retrieval, E5 (not in the 2026 top tier — drop from the previous #31b preset list). Quantized weights stay default via fastembed-rs shipping quantized ONNX; fp32 is an opt-in.

**Why second.** This is where the ROI from better embedders actually shows up, and Step 1 gives us the type-safety guarantees to land a second backend without risking dim drift.

### Step 3 — Reranker (`bge-reranker-v2-m3` ONNX int8)

**Goal.** Plug a cross-encoder reranker into the query pipeline. This is the single highest-ROI quality change on security corpora per the research doc (+10–20 nDCG points, often the difference between "cites the right CVE" and "hallucinates a plausible neighbor").

**Scope.**
- New `fastrag-rerank` crate. Single reranker impl: `bge-reranker-v2-m3` via `ort` + `tokenizers` from HuggingFace, loaded from a community-published ONNX int8 variant (e.g. `onnx-community/bge-reranker-v2-m3-ONNX`).
- `Reranker` trait with `rerank(query: &QueryText, candidates: &[Passage]) -> Vec<Scored>`.
- Pipeline wire-up: query → retrieve top-50 → rerank → top-5 to caller. Top-k retrieval and top-k reranked are both configurable.
- Default batch size + threading tuned for CPU. Target: ~200–400ms for a top-50 rerank on a mid-range CPU.
- CLI and MCP both expose a `--rerank` / `--no-rerank` flag. Default on once ship-quality is confirmed by eval.
- Eval: rerank delta (hit@5 with rerank minus hit@5 without) committed to `docs/evals/`. Gate the default-on decision on a measurable win.

**Out of scope.** Alternative rerankers (jina, mxbai) — watch-listed for a follow-up.

**Why third.** Biggest quality lever after basic correctness. Does not depend on hybrid retrieval, but pairs well with it in Step 4.

### Step 4 — Hybrid retrieval via Tantivy

**Goal.** Ship BM25 + dense fusion with exact-match short-circuits. The research doc is emphatic that dense-only is the wrong default for CVE corpora, and this is foundational for fastrag's target users.

**Scope.**
- `fastrag-tantivy` crate (or feature flag). Schema with structured fields where they apply to the corpus (for security corpora: `cve_id`, `cwe`, `cvss_vector`, `publish_date`, `kev_flag`, `vendor`, `product`; for generic corpora: just full-text body + title + path).
- Query pipeline:
  1. Regex extract `CVE-\d{4}-\d{4,7}` and `CWE-\d+`. If present, do an exact Tantivy term lookup and prepend those hits.
  2. Tantivy BM25 full-text search, top-50.
  3. Dense vector search, top-50.
  4. Reciprocal Rank Fusion (k=60) over the two lists.
  5. Pass the fused top-50 into the reranker from Step 3.
- Payloads are stored once in Tantivy; the vector index holds only doc IDs and vectors.
- Ingest path writes to both the dense index and the Tantivy index atomically per chunk so a crash can't leave them divergent.
- CLI: `--hybrid` flag (default on post-ship), `--dense-only` for back-compat and ablation.

**Out of scope.** ColBERT, SPLADE, GraphRAG (per research doc skip list).

**Why fourth.** Reranker first ensures the downstream stage is already in place when hybrid lands, so the eval delta of hybrid is measured against a reranked baseline (the true production configuration).

### Step 5 — Contextual Retrieval

**Goal.** Adopt Anthropic's Contextual Retrieval (Sept 2024, still gold standard in 2026) as a pluggable ingest-time feature. Reported −49% retrieval failure, −67% combined with BM25 + reranker.

**Scope.**
- `Contextualizer` trait with impls `None` (default, no dependency), `LocalLlm(endpoint)`, `OpenAiCompat(endpoint)`.
- At ingest, for each chunk: call the contextualizer to generate a 50–100 token context prefix, store both `raw_text` and `contextualized_text` in chunk metadata. Embedding uses `contextualized_text`; display at retrieval uses `raw_text`; the prefix is never shown to the consuming LLM.
- Cache by chunk content hash (required — a 40k-chunk corpus on CPU inference is an overnight job, and partial failures must resume cleanly). Cache key includes `contextualizer_version` so prompt changes invalidate cleanly.
- Each chunk tracks `contextualized_at: Option<Timestamp>` and `contextualizer_version: Option<u32>` so re-contextualization is incremental.
- CLI: `--contextualizer <none|local|openai>` with appropriate endpoint flags.

**Out of scope.** Self-RAG, FLARE, CRAG (skip list). Late chunking is an alternative for near-8k-token chunks — watch-listed, not shipped.

**Why fifth.** Depends on Steps 1–4 being in place so the effect can be measured against a full production pipeline and the cache semantics don't have to work around in-flight architectural churn.

### Step 6 — Eval harness refresh + gold set

**Goal.** Rebuild the eval harness so #25's baselines are meaningful against the new stack, and formalize a hand-curated gold set as the primary CI gate.

**Scope.**
- Eval harness runs the real embedder + reranker + hybrid against a real (small) corpus on every PR that touches retrieval code. No bag-of-words stubs anywhere (shim lesson #2). Fast smoke eval runs in CI; slow real-model eval runs nightly.
- Gold set format: hand-curated JSON of `{question, must_contain_cve_ids: [...], must_contain_terms: [...]}` with 100 entries minimum. Checked into repo under `tests/gold/`. Versioned.
- CI gates on `hit@5` and `MRR@10` against the gold set. Regression → red build.
- Metrics emitted: retrieval hit@k, rerank delta, groundedness rate (if a generation step is wired), refusal rate, per-stage latency percentiles, cache hit rates.

### Step 7 — Corpus hygiene (interleavable)

**Goal.** Ingest-time filters that turn out to be correctness issues on security corpora, per the research doc. Can interleave with Step 4 or Step 5 as convenient.

**Scope.**
- Reject `vulnStatus: Rejected` / `Disputed` CVEs at ingest.
- Strip NVD boilerplate (`** REJECT `, `** DISPUTED **`, CPE 2.3 URIs, reference URLs, legal notices) from the embedded string; keep them in metadata.
- Cross-source dedup (NVD / GHSA / OSV / KEV overlap) by CVE-ID; merge descriptions, keep all provenance as metadata.
- Normalize vendor and product via CPE 2.3 before indexing → free Tantivy facets.
- Language detect at ingest; log non-English counts and expose a policy flag (skip / translate / flag).
- Temporal weighting: reranker-side or Tantivy boost on post-2020 + KEV-flagged CVEs.

**Out of scope.** Any of the above for non-security corpora. This is a security-corpus-specific ingest profile, selectable via a CLI flag.

## What carries forward from the deleted #31b plan

The #31b spec and plan are gone, but several pieces are still directly reusable and should be lifted into Step 1's spec when we write it:

- The `CandleHfEmbedder` rename from `BgeSmallEmbedder` — still correct; happens inside Step 1's refactor.
- The pre-existing BGE manifest-id bug (`BAAI/...` vs `fastrag/bge`) — still needs fixing in Step 1; same fix as Task 1 of the deleted plan.
- The `embed_query` / `embed_passage` trait split — becomes the `QueryText` / `PassageText` typed variant in Step 1. Semantically the same, structurally stronger.
- The `FASTRAG_OFFLINE=1` cache-miss error — rolls into Step 2 when fastembed-rs weight fetching lands.
- The backcompat alias pattern (`--embedder bge` → `candle bge-small`) — rolls into Step 1 for legacy corpora.

## What gets explicitly dropped from the deleted #31b plan

- **E5-small as a preset.** Not in the 2026 top tier; research doc implicitly supersedes it with nomic/arctic. Dropped.
- **bge-base as a preset.** Same. Keeping only `bge-small` under `legacy-candle` for back-compat with existing corpora, and nomic + arctic as the new flagships.
- **`--model bge-small | e5-small | bge-base` CLI surface.** Replaced by preset naming from Step 2.
- **Eval matrix comparing only candle-backed models.** Replaced by a broader matrix in Step 6 (candle-bge-small vs nomic vs arctic, all with and without reranker, across security + NFCorpus).

## Non-goals for all of Phase 2

Quoting the research doc's skip list so it's in-repo and can be cited in reviews:

- ColBERT v2 / PLAID
- SPLADE v3
- ColPali
- GraphRAG (CVEs are already a graph via CWE / CPE — Tantivy facets are strictly better)
- Semantic chunkers (`chunking-ai` et al.)
- Self-RAG, FLARE
- `rust-bert` (frozen on old torch bindings)

A follow-up PR opens these as "watch" issues so anyone who revisits in 6 months has context.

## Sequencing summary

```
Step 1: Embedder invariant refactor (foundational)
  ↓
Step 2: fastembed-rs backend + nomic/arctic presets
  ↓
Step 3: Reranker (bge-reranker-v2-m3 ONNX int8)
  ↓
Step 4: Hybrid retrieval via Tantivy + RRF
  ↓
Step 5: Contextual Retrieval (pluggable)
  ↓
Step 6: Eval harness refresh + gold set
  ↕
Step 7: Corpus hygiene (interleavable with 4–5)
```

Each step produces working, testable, mergeable software on its own. No step depends on a later step for correctness.

## Open questions for the next brainstorm (Step 1)

1. How much of the `Arc<dyn Embedder>` type-erasure layer can we preserve unchanged, and how much gets rewritten to carry the static dim/prefix invariants through construction time?
2. Does the canary live in the manifest, in a separate sidecar file, or in the vector index itself? The research doc is agnostic.
3. What is the migration story for corpora indexed with the current `BgeSmallEmbedder`? Best-effort upgrade or hard-error-with-clear-message?
4. Does `QueryText` / `PassageText` need owned `String` or can it borrow `&str`? The old trait used `&[&str]`; keeping that avoids allocation churn.

Answer these during the Step 1 brainstorm, not here.
