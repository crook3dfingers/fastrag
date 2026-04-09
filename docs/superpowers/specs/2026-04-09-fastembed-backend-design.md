# fastembed-rs Backend + Flagship Presets — Design

**Date:** 2026-04-09
**Status:** Approved, pending implementation plan
**Scope:** Phase 2 Step 2. Follows Step 1 (embedder invariant refactor, merge 7cc1890).
**Related:** `docs/superpowers/roadmap-2026-04-phase2-rewrite.md`, `docs/rag-research-2026-04.md`

## Goal

Replace candle/BGE-small as the default embedding backend with fastembed-rs (ort + tokenizers), and ship two flagship presets: `arctic-embed-m-v1.5` (new default) and `nomic-embed-v1.5` (long-context, asymmetric). Preserve the static `Embedder` trait invariants shipped in Step 1.

## Non-goals

- Reranker (Step 3).
- Hybrid retrieval / BM25 (Step 4).
- Eval harness refresh or gold set (Step 6).
- Removing candle support. It moves behind a feature flag; deletion is a future issue.
- GPU runtime testing in CI. Build-only.

## Background

Step 1 shipped a static `Embedder` trait with `const DIM / MODEL_ID / PREFIX_SCHEME`, a dyn-safe `DynEmbedderTrait`, `QueryText`/`PassageText` newtypes, `EmbedderIdentity`, canary verification on load, and manifest v3. The asymmetric-prefix invariant (`PrefixScheme` hashed into identity) has not yet been exercised by a real model — BGE-small is symmetric. Nomic's `search_query:` / `search_document:` scheme will be its first live test.

Per `docs/rag-research-2026-04.md`, arctic-embed-m-v1.5 and nomic-embed-v1.5 are the 2026 flagship small-model picks, both shipping cleanly via fastembed-rs. Candle's BGE-small is second-tier on MTEB retrieval, and candle is less actively maintained than ort.

## Architecture

### Crate layout

`crates/fastrag-embed/` gains a `fastembed` module gated by a new `fastembed` feature (on by default). Candle BGE moves behind a `legacy-candle` feature (off by default). The features are independent backends and can be enabled simultaneously.

### Preset implementations

Two concrete structs, each owning a `TextEmbedding` handle:

```rust
pub struct ArcticEmbedMV15 { inner: TextEmbedding }
pub struct NomicEmbedV15   { inner: TextEmbedding }
```

Each hand-implements the static `Embedder` trait with its own consts. Arctic:

```rust
impl Embedder for ArcticEmbedMV15 {
    const DIM: usize = 768;
    const MODEL_ID: &'static str = "fastrag/arctic-embed-m-v1.5";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::SYMMETRIC;
    // ...
}
```

Nomic:

```rust
const PREFIX_SCHEME: PrefixScheme = PrefixScheme {
    query:   "search_query: ",
    passage: "search_document: ",
};
```

`MODEL_ID` encodes the exact variant shipped (e.g., `fastrag/arctic-embed-m-v1.5` vs `fastrag/nomic-embed-v1.5-q` for the int8 build). The manifest identity pins the variant; swapping f32↔int8 triggers `IdentityMismatch` on load.

### Shared helper

A private module-level helper absorbs the fastembed call surface:

```rust
fn fastembed_batch(
    inner: &TextEmbedding,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, EmbedError>
```

It handles error mapping and normalization. Both presets delegate through it. When a third preset lands, this helper stays; only the struct+const boilerplate grows.

### Prefix application

For nomic, `embed_query` and `embed_passage` prepend `Self::PREFIX_SCHEME.query` / `.passage` before calling `fastembed_batch`. Arctic uses `PrefixScheme::SYMMETRIC` (both fields empty string) and skips the prepend. Call sites remain prefix-agnostic — the `QueryText`/`PassageText` newtypes from Step 1 carry the intent, and the embedder translates that intent into the right string.

### Batch API

Extend `DynEmbedderTrait`:

```rust
fn embed_passages_batch(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
    texts.iter().map(|t| self.embed_passage(t)).collect()
}
```

The default impl loops existing single-call methods so all current implementations compile unchanged. Fastembed presets override with the native batched call (`TextEmbedding::embed(batch, Some(batch_size))`). The indexing path switches to batched calls.

### Model variants (quantization)

Default to int8 quantized where fastembed-rs exposes it:

- **Nomic**: `EmbeddingModel::NomicEmbedTextV15Q` (int8). Roughly 4× smaller, ~2× faster on CPU, MTEB drop under 0.5 points.
- **Arctic**: whichever quantized variant fastembed-rs ships in the pinned version. Fall back to f32 if no Q variant exists at pin time.

The chosen variant is frozen into `MODEL_ID`, so indexes built with one variant cannot silently load under another.

### Model cache

Resolve via the `dirs` crate, honoring `XDG_CACHE_HOME`:

```
$XDG_CACHE_HOME/fastrag/models/fastembed/
```

Pass to fastembed via `InitOptions::with_cache_dir`. Keeps the fastembed cache separate from the existing candle cache and establishes the convention for any future backend.

### Build surface

Default build: fastembed-rs + ort, statically linked, CPU execution provider only. Matches fastrag's single-binary promise; no `libonnxruntime.so` required at runtime.

Opt-in features:
- `gpu-cuda` — flips ort to the CUDA execution provider. Requires system CUDA/cuDNN. Documented build-size impact.
- `gpu-coreml` — macOS CoreML execution provider.

ROCm is out of scope.

CI continues to build and test only the CPU default. GPU features get a build check with no runtime test.

## Error handling

Reuse existing `EmbedError` variants where possible. Add a new variant only if fastembed-rs surfaces a failure mode not already covered (e.g., `EmbedError::ModelDownload(String)` for first-use fetch failures). Identity and canary mismatches continue to surface as `IndexError::IdentityMismatch` / `IndexError::CanaryMismatch` — unchanged from Step 1.

## Testing

Unit:
- Each preset round-trips `EmbedderIdentity` (serialize → deserialize → equal).
- Dimensions match `const DIM`.
- `PrefixScheme` FNV hash is stable across builds.
- Nomic `embed_query("foo")` and `embed_passage("foo")` produce different vectors (prefix was applied).
- Arctic `embed_query("foo")` and `embed_passage("foo")` produce identical vectors (symmetric).

Integration:
- Build a small corpus with arctic, persist, reload with arctic → canary passes, cosine ≥ 0.999.
- Build with arctic, attempt reload with nomic → `IdentityMismatch`.
- Build with nomic-int8, attempt reload with nomic-f32 (if both available) → `IdentityMismatch`.
- Batched embed matches repeated single-call embed within f32 tolerance.

Eval smoke check:
- Run the existing eval harness against a fixture corpus with arctic vs legacy BGE-small.
- Log hit@5 and MRR@10 in the PR description.
- No hard gate (that lives in Step 6), but a regression against BGE-small blocks merge pending investigation.

## Risks and mitigations

1. **fastembed-rs downloads models on first use.** Non-deterministic in CI.
   *Mitigation:* Pre-populate the cache in the CI job before running tests, or mark the download-dependent tests `#[ignore]` and run them in a dedicated lane.

2. **ort static linking inflates the release binary.**
   *Mitigation:* Measure before/after in the PR. Document in release notes. If the size delta exceeds ~50 MB, revisit `load-dynamic` with a runtime loader.

3. **int8 quantization accuracy regression on our corpus.**
   *Mitigation:* Eval smoke check above. If arctic-int8 regresses against BGE-small, ship arctic-f32 as default and keep int8 behind an opt-in preset variant.

4. **Breaking change for users with candle-built indexes.**
   *Mitigation:* `legacy-candle` feature keeps candle BGE loadable with the existing manifest-v3 identity. README documents the rebuild path for users who want to migrate to arctic.

## Decision log

- **Concrete struct per preset, not generic-with-ZST.** Two presets don't earn the generic machinery. Revisit at the third preset.
- **Arctic as default, not nomic.** Better MTEB retrieval score in the relevant size class; symmetric (simpler). Nomic is the long-context option.
- **int8 default where available.** Speed and size win outweighs the <0.5pt MTEB cost.
- **Static-linked CPU default.** Preserves single-binary deployment story.
- **Keep candle behind `legacy-candle`, don't delete.** Smooth migration for existing v3 corpora.
