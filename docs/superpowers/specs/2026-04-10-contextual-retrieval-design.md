# Step 5 — Contextual Retrieval (2026-04-10)

**Status:** design approved, pending implementation plan
**Roadmap:** `docs/superpowers/roadmap-2026-04-phase2-rewrite.md` — Phase 2 Step 5
**Research:** `docs/rag-research-2026-04.md`
**Depends on:** Step 1 (embedder invariant), Step 2 (llama.cpp backend), Step 3 (reranker), Step 4 (hybrid retrieval via Tantivy)

## Summary

Add Anthropic's Contextual Retrieval (Sept 2024) as an optional ingest-time stage. For each chunk, a small instruct LLM generates a 50–100 token context prefix situating the chunk within its source document. The prefix is prepended to the chunk text for both dense embedding and BM25 indexing, while the raw chunk is preserved for display and CVE/CWE exact-match. Reported impact from the original paper: −49% retrieval failure alone, −67% combined with BM25 + reranker — additive on top of Steps 3 and 4.

Contextualization is opt-in per corpus (overnight cost on CPU), cached in a SQLite sidecar keyed by chunk content hash plus prompt and model versions, and resumable across runs.

## Goals

1. Ship `fastrag-context` crate with a `Contextualizer` trait, a `NoContextualizer` default, and a `LlamaCppContextualizer` impl that reuses Step 2's `LlamaServerHandle` pattern.
2. Extend `fastrag::ops::index_corpus` with a new pipeline stage between chunking and dual-write ingest. Stage is a no-op when the contextualizer is `None` (default).
3. Persist contextualization results in a SQLite sidecar at `<corpus>/contextualization.sqlite`. Key: `(blake3(raw_text), ctx_version, model_id, prompt_version)`. Value: raw text, doc title, context text, status (`ok`/`failed`), error, timestamp. The cache is self-contained — a `--retry-failed` pass can run against the SQLite file alone, without opening the Tantivy or HNSW indexes.
4. Bump manifest to `index_version: 2`. `HnswIndex::load` hard-errors on older corpora with a clear rebuild message.
5. Index both dense vectors and Tantivy BM25 over contextualized text. Store raw text as Tantivy's `display_text` (stored, not indexed). Run CVE/CWE regex over raw text only.
6. CLI gains `--contextualize`, `--context-model <preset>`, `--context-strict`, `--retry-failed` flags on `index`. `corpus-info` and `doctor` gain contextualizer reporting.
7. MCP surface is unchanged. Contextualization is an ingest/maintenance operation; it belongs in the CLI, not the agent-facing MCP tool set.
8. Default ingest behavior is unchanged. On successful ingest without contextualization, print a one-line hint pointing to the flag.

## Non-goals

- HTTP / OpenAI-compat / CLI-shellout Contextualizer backends. Trait is pluggable to accept them in a follow-up, but the first PR ships `None` + `LlamaCpp` only.
- Automatic migration of `index_version: 1` corpora. Hard error with a rebuild message.
- Eval harness or quality-delta measurement. That is Step 6's scope. This PR proves the mechanism works end-to-end, not that it improves hit@5 at scale.
- Optimized in-place HNSW update for `--retry-failed`. First PR does a full dense rebuild from SQLite cache rows when any retry succeeds. Optimization is a follow-up.
- Late chunking, Self-RAG, FLARE, CRAG (research-doc skip list).

## Constraints

- **Overnight max latency.** A 40k-chunk corpus on a mid-range CPU must complete within ~8 hours wall-clock with the chosen model. Size class is a guideline, not a hard gate.
- **Haiku-floor quality.** The chosen instruct model must produce context prefixes at least as useful as Claude Haiku on a qualitative read of 10 sample chunks. Verification happens during the research pass and is re-verified in the E2E test.
- **Data-file swappability.** Model selection is a GGUF path + preset tuple. Swapping models is a configuration change.
- **Resume correctness.** Partial-failure ingest must resume cleanly on re-run. No duplicate contextualization, no lost failures, no silent skips.
- **Back-compat.** `index_version: 1` corpora fail to open with a clear rebuild message.

## Architecture

New crate `crates/fastrag-context/` parallel to `fastrag-rerank` and `fastrag-tantivy`. Feature-gated behind `contextual` in the facade crate `fastrag`. Full lint gate extends to `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings`.

### Dependencies

- `fastrag-core` — chunk types
- `fastrag-embed` — reuses `LlamaServerHandle` via a new `LlamaServerPool` helper
- `rusqlite` — new workspace dep, WAL mode
- `blake3` — verify transitive presence during implementation; add as a direct workspace dep if missing
- `reqwest` — already present via `fastrag-embed`
- `serde`, `thiserror`, `tokio` — already present

### Module layout

```
crates/fastrag-context/
  Cargo.toml
  src/
    lib.rs          — trait re-exports, error type
    contextualizer.rs — Contextualizer trait, NoContextualizer
    llama.rs        — LlamaCppContextualizer
    cache.rs        — ContextCache (SQLite wrapper)
    prompt.rs       — PROMPT const, PROMPT_VERSION, template formatter
    stage.rs        — Stage::Contextualize helper consumed by fastrag::ops
  tests/
    cache_resume.rs
    stage_fallback.rs
```

### Contextualizer trait

```rust
#[async_trait]
pub trait Contextualizer: Send + Sync {
    async fn contextualize(
        &self,
        doc_title: &str,
        raw_chunk: &str,
    ) -> Result<String, ContextError>;

    fn model_id(&self) -> &str;
    fn prompt_version(&self) -> u32;
    fn ctx_version(&self) -> u32 { CTX_VERSION }
}
```

- `doc_title` is always passed; empty string is valid and the prompt template handles it.
- `model_id` and `prompt_version` feed the cache key — the trait owns its own identity so the cache layer stays oblivious to backends.
- `ctx_version` is a crate-level constant bumped when the SQLite schema or key shape changes. Independent from `prompt_version`, which tracks prompt text edits.

### LlamaServerPool

New helper in `fastrag-embed` owning up to two `LlamaServerHandle` instances: the embedder server and the optional completion server. Separate ports, separate GGUF paths, separate health checks. `fastrag doctor` iterates the pool. RAII drop tears down both.

`LlamaServerPool` does not replace `LlamaServerHandle`; it coordinates two of them.

### SQLite schema

```sql
CREATE TABLE IF NOT EXISTS context (
  chunk_hash      BLOB NOT NULL,
  ctx_version     INTEGER NOT NULL,
  model_id        TEXT NOT NULL,
  prompt_version  INTEGER NOT NULL,
  raw_text        TEXT NOT NULL,
  doc_title       TEXT NOT NULL,    -- empty string for untitled
  context_text    TEXT,
  status          TEXT NOT NULL CHECK(status IN ('ok','failed')),
  error           TEXT,
  created_at      INTEGER NOT NULL,
  PRIMARY KEY (chunk_hash, ctx_version, model_id, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_context_status ON context(status);
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

- `chunk_hash` is 32 bytes from `blake3(raw_text)`.
- `raw_text` and `doc_title` are stored so `--retry-failed` can recover the strings needed to re-call `contextualize(doc_title, raw_text)` without opening the Tantivy or HNSW indexes. This makes the cache a portable artifact — copy the SQLite file to another machine and run the repair pass there.
- `status='ok'` rows carry a non-null `context_text` and null `error`. `status='failed'` rows carry null `context_text` and a truncated (≤500 char) `error` string.
- `INSERT OR REPLACE` is used on `put` so a subsequent successful retry overwrites a prior failure cleanly.
- Rough storage cost: ~2 KB × rows for raw_text, negligible for everything else. A 40k-chunk corpus produces an ~80 MB SQLite file, small relative to the multi-GB GGUFs and HNSW indexes.

### Manifest additions

`manifest.json` v2:

```json
{
  "index_version": 2,
  "embedder": { ... },
  "contextualizer": {
    "model_id": "qwen-something-3b-q4-km",
    "prompt_version": 1,
    "prompt_hash": "blake3-of-PROMPT"
  }
}
```

`contextualizer` is omitted on corpora ingested without `--contextualize`. `HnswIndex::load`:

- `index_version: 1` → hard error with rebuild message
- `index_version: 2` without `contextualizer` field → loads fine, behaves like Step 4
- `index_version: 2` with `contextualizer` field → loads fine, query path is unchanged

### Tantivy schema changes

Adds one new field:

- `display_text` — stored, not indexed. Holds the raw chunk text for return to the consumer at query time.

The existing `body` field (indexed for BM25) holds contextualized text when contextualization is enabled, raw text otherwise.

CVE and CWE regex extraction runs over `raw_text`, not over contextualized text. A hallucinated `CVE-2024-99999` in a context prefix must not pollute exact-match lookup.

The repair path (`--retry-failed`) does not touch Tantivy at all — raw text and doc title live in the SQLite cache. This keeps the `fastrag-context` crate free of any Tantivy dependency.

## Data flow

### First-time ingest without contextualization (default)

```
fastrag index ./docs --corpus ./corpus
  ├─ spawn LlamaServerPool { embedder:8080 } — no completion server
  ├─ for each file: Parse → Chunk → Contextualize(NoOp) → DualWriteIngest
  ├─ persist manifest v2 without contextualizer block
  └─ print:
       "Indexed 12,480 chunks.
        Hint: re-run with --contextualize for better retrieval on technical
              queries (one-time per corpus, cached thereafter)."
```

### First-time ingest with `--contextualize`

```
fastrag index ./docs --corpus ./corpus --contextualize
  ├─ resolve GGUFs via ModelSource (embedder + completion)
  ├─ spawn LlamaServerPool { embedder:8080, completion:8081 }
  ├─ open ContextCache at ./corpus/contextualization.sqlite
  ├─ for each file:
  │    Parse → Chunk → Contextualize → DualWriteIngest
  │    Contextualize stage per chunk:
  │      cache.get(key):
  │        hit-ok    → chunk.ctx = Some(cached)
  │        hit-failed→ chunk.ctx = None (fallback)
  │        miss      → llama_ctx.contextualize(...)
  │                      ok      → cache.put(ok); chunk.ctx = Some(prefix+"\n\n"+raw)
  │                      err + strict → abort
  │                      err + !strict→ cache.mark_failed; chunk.ctx = None
  │    DualWriteIngest:
  │      text_for_index = chunk.ctx.unwrap_or(raw_text)
  │      dense.embed_passage(text_for_index)
  │      tantivy.add(body=text_for_index, display_text=raw_text,
  │                  cve_ids=regex(raw_text), cwe_ids=regex(raw_text))
  ├─ persist manifest v2 with contextualizer block
  ├─ drop LlamaServerPool
  └─ print:
       "Indexed 12,480 chunks.
        Contextualized: 12,091 ok / 389 fallback (3.1%)."
```

### Retry-failed pass

```
fastrag index --corpus ./corpus --retry-failed --contextualize
  ├─ open existing ContextCache (no docs re-parsed, no Tantivy/HNSW opened yet)
  ├─ spawn LlamaServerPool { completion:8081 }
  ├─ for row in cache.iter_failed():
  │    llama_ctx.contextualize(row.doc_title, row.raw_text)
  │    on ok: cache.put(ok, context_text)
  │    on err: leave failed, log
  ├─ if any retries succeeded:
  │    open Tantivy + HNSW
  │    rebuild dense HNSW from cache rows (O(corpus), not O(failed))
  │    rewrite Tantivy body fields for repaired chunks
  └─ print "Repaired 312 of 389 failed chunks (77 still failed)."
```

The rebuild is O(corpus) because HNSW does not support cheap in-place updates. This is acceptable for a manual repair path expected to run rarely and against <5% failure rates. The rebuild reads its inputs from the SQLite cache — every chunk has its `raw_text` and (if successful) its `context_text` stored there, so the rebuild does not need to re-parse source documents or pull text out of Tantivy.

### Query path

Unchanged from Step 4. Contextualizer is ingest-only. At query time the consumer receives `raw_text` via Tantivy's `display_text` field. Context prefixes are never shown to the downstream LLM.

## CLI surface

```
fastrag index <docs> --corpus <dir>
  [--contextualize]                 # opt in
  [--context-model <preset>]        # GGUF preset name
  [--context-strict]                # hard-fail on any chunk error
  [--retry-failed]                  # repair pass, no doc re-parsing

fastrag corpus-info --corpus <dir>
  # new fields when contextualized:
  #   contextualized: true
  #   contextualizer: { model_id, prompt_version, ok: N, failed: M }

fastrag doctor
  # new section:
  #   contextualizer: completion GGUF resolves, llama-server starts
```

`--contextualize=off` is accepted as an explicit opt-out. Omitting the flag is equivalent to `off`.

## MCP surface

No changes. Indexing is a long-running ingest/maintenance operation; an LLM agent does not reach for it mid-conversation. CLI is the correct surface.

## Error handling

### Error taxonomy

```rust
#[derive(thiserror::Error, Debug)]
pub enum ContextError {
    #[error("llama-server HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("llama-server returned non-200: status={status}, body={body}")]
    BadStatus { status: u16, body: String },
    #[error("llama-server request timed out after {0:?}")]
    Timeout(std::time::Duration),
    #[error("llama-server returned empty completion")]
    EmptyCompletion,
    #[error("cache error: {0}")]
    Cache(#[from] rusqlite::Error),
    #[error("prompt template error: {0}")]
    Template(String),
}
```

### Behaviors

- **Per-chunk error (non-strict).** Write `failed` row with truncated error, set `chunk.ctx = None`, continue. Chunk still indexed using raw text.
- **Per-chunk error (strict).** Abort the pipeline via `?` propagation. Already-written chunks remain in the corpus; the manifest is not finalized, so the corpus cannot be queried until a successful re-run.
- **Subprocess crash.** Next HTTP call returns a connect error. `LlamaServerHandle::check_alive()` reports dead; pool attempts one restart; in-flight chunk is retried once. If either fails, fall through to per-chunk error handling. No retry loops, no backoff.
- **Cache corruption.** `rusqlite` error at open time surfaces a clear error pointing to `rm corpus/contextualization.sqlite`. No automatic deletion.
- **Manifest version mismatch.** `HnswIndex::load` on `index_version: 1` returns a hard error with a rebuild message. No automatic migration.
- **Empty completion.** Treated as a per-chunk failure. A `failed` row is written and the chunk falls back to raw text.

## Testing

### Unit tests (`#[cfg(test)]`)

- `NoContextualizer::contextualize` returns input unchanged.
- `ContextCache` put/get round-trip with concrete bytes.
- `ContextCache::mark_failed` → `iter_failed` → subsequent `put(ok)` removes the row from `iter_failed`.
- `ContextCache` key independence: distinct `(ctx_version, model_id, prompt_version)` tuples coexist for the same `chunk_hash`.
- `ContextPrompt` template: formatted output for with-title and without-title cases. No `None` leakage, no `{title}` leakage.
- `LlamaCppContextualizer` against `wiremock`: 200, 500, empty completion, timeout — each maps to the correct `ContextError` variant.

### Integration tests (`crates/fastrag-context/tests/`)

- `cache_resume.rs` — open, put 100 rows, close, reopen, assert all retrievable.
- `stage_fallback.rs` — mock contextualizer failing every 3rd call. Assert 2/3 ok, 1/3 fallback to raw, cache reflects the split, no panic.

### E2E tests (`fastrag-cli/tests/`, ignored, gated on `FASTRAG_LLAMA_TEST=1`)

- `contextual_corpus_e2e.rs` — real llama-server + both GGUFs. Ingests 5 fixture docs, `corpus-info` reports `ok: 5, failed: 0`. Query that requires context (chunk references "the vulnerability" without naming it; doc title names the CVE) must return the right chunk only when contextualization is enabled.
- `contextual_retry_failed_e2e.rs` — inject 2 failures out of 5, run `--retry-failed` against a healthy server, assert all 5 become `ok` and the dense index was rebuilt.
- `contextual_strict_e2e.rs` — `--context-strict` against a failing server, assert non-zero exit and no manifest written.

### CI

- New nightly job `contextual-retrieval` on top of `llama-cpp-backend`: downloads completion GGUF, runs the E2E suite.
- Push CI runs only unit + integration tests. No GGUF download in push CI.

### Explicit non-tests

- No test asserting contextualization improves hit@5 numerically. That is Step 6's eval harness scope.
- No prompt mutation testing. Prompt version bumps are a release-note concern.
- No rubber-stamp tests. Every assertion must fail if the implementation is broken or no-op.

## Research pass — model selection

Before the first PR lands, run a short research pass following `feedback_research_recency.md`:

- Constraint 1: completes 40k chunks on mid-range CPU within ~8h wall-clock.
- Constraint 2: produces context prefixes at least as useful as Claude Haiku on a 10-chunk qualitative read.
- Strict recency filter: no model with primary release older than 12 months as of 2026-04-10. No stale-year fallbacks.
- Output: a specific GGUF preset name (e.g. `ModelSource::HuggingFace { repo, file }`) and the quantization choice.

The preset is committed to `fastrag-context` as a constant. The `--context-model` flag accepts the preset name so users can point at alternatives without a code change, but the default is the researched choice.

## Rollout

1. Land the crate and trait without CLI wiring. Unit tests green.
2. Land the cache, stage, and mock-driven integration tests. Clippy + fmt green.
3. Land `LlamaServerPool` + `LlamaCppContextualizer` + `fastrag doctor` wiring. Nightly CI job wired.
4. Land CLI and MCP surface. Update `CLAUDE.md` Build & Test section with the new feature flag and test commands.
5. Update `README.md` with a Contextual Retrieval section pointing at the `--contextualize` flag and its cost model.

Each step is a separate commit on `main` in the listed order. No worktrees (per `feedback_no_worktrees.md`). `cargo test`, `cargo clippy`, and `cargo fmt` gates run locally before every push.

## Open questions for the implementation plan

1. Does Step 2's `LlamaServerHandle` currently expose `check_alive()` and a restart method? If not, that is a Step 2 patch that must land before Step 5's subprocess-crash handling, not part of Step 5 itself.
2. Is there an existing `ModelSource` entry in `fastrag-embed` for chat-completion GGUFs, or does that enum need a new variant? Likely a new variant — embeddings and completions have different llama-server CLI flags (`--embedding` vs not).
3. What is the precise prompt text? The Anthropic blog post publishes one; lift it verbatim with attribution and bump `PROMPT_VERSION` only on our own edits.
4. How does `fastrag-cli`'s existing `index` command hand off to `ops::index_corpus`? The stage insertion point must be verified before the plan is written.

These are resolved in the writing-plans phase, not here.
