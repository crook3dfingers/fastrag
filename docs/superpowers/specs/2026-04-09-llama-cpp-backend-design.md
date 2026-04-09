# Spec: llama.cpp Embedder Backend (Phase 2 Step 2)

**Date:** 2026-04-09
**Replaces:** `2026-04-09-fastembed-backend-design.md` (abandoned — see handoff notes)
**Status:** Draft

## Goal

Ship a production embedder backend for fastrag that (a) preserves the Step 1 `Embedder` trait invariants and manifest v3 canary story, (b) tracks SOTA models via data-file swaps (GGUF) rather than code PRs, and (c) maximizes CPU inference speed. This step unifies the backend used by Phase 2 Steps 2 (embedder), 3 (reranker), and 5 (contextual retrieval LLM) onto a single native binary: `llama-server`.

## Non-goals

- GPU support. CPU-only, aggressively quantized, is the target.
- Python runtime. Zero Python dependency is maintained; `llama-server` is C++ single-binary, installed alongside `fastrag`.
- Dynamic model registry / runtime model selection. Each preset is a concrete Rust struct with burned-in identity (consistent with Step 1 invariants).

## Model selection

**Default preset:** `Qwen/Qwen3-Embedding-0.6B-GGUF` (official Qwen GGUF repo).

- License: Apache 2.0
- Parameters: 0.6B
- Dim: 1024
- Pooling: last-token
- Published benchmark: MTEB English v2 retrieval 61.83
- Provenance: Official Qwen HuggingFace repo (not a community re-quant)
- Quantization: `Q8_0` default (best quality), `Q4_K_M` as a smaller/faster variant

**Why Qwen3 over Arctic-embed-l-v2.0 or Harrier-OSS-v1:**

- Arctic v2.0 has a cleaner retrieval story on multilingual benchmarks but would force a second backend (ONNX Runtime / TEI), breaking the unification with Steps 3 and 5. English-only security corpus with controlled chunking does not exercise Arctic's multilingual or CLS-pooling advantages.
- Harrier-OSS-v1 has no published retrieval-only benchmark, weak GGUF provenance (community re-quant of a non-Microsoft intermediary), requires query-instruction prefixing, and depends on an undocumented `llama-server` flag (`--embd-normalize`). Rejected after adversarial review.
- Qwen3-Embedding-0.6B has an official GGUF, published English retrieval number, clean license, and fits the unified-binary architecture.

**Preset struct:**

```rust
pub struct Qwen3Embed600mQ8 { /* llama-server handle */ }

impl Embedder for Qwen3Embed600mQ8 {
    const DIM: usize = 1024;
    const MODEL_ID: &'static str = "Qwen/Qwen3-Embedding-0.6B-GGUF@Q8_0";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;
    // ...
}
```

`MODEL_ID` embeds the repo ID and quant tag so the manifest canary catches silent quant swaps.

## Backend architecture

**Subprocess model:** fastrag spawns `llama-server` as a child process on a configured local port and speaks HTTP to it.

- Why `llama-server` over subprocess-per-call to `llama-embedding`: startup cost amortized, batching possible, reused across Steps 3/5.
- Why HTTP over shared library / FFI: zero Rust↔C++ FFI surface, easier version pinning, matches the existing `http-embedders` code path in `fastrag-embed`.
- Transport: `reqwest` blocking client (already a dep behind `http-embedders` feature). The new backend lives behind a `llama-cpp` feature and reuses the same HTTP plumbing where possible.

**Lifecycle:**

1. `Qwen3Embed600mQ8::load(config)` spawns `llama-server --embedding --model <path> --port <port> --pooling last ...`.
2. Parent polls `/health` until ready (bounded timeout).
3. `embed_query` / `embed_passage` POST to `/embedding` (or `/v1/embeddings`).
4. On drop, parent sends SIGTERM and waits for exit.
5. Model path resolution: `$FASTRAG_MODEL_DIR/Qwen3-Embedding-0.6B-Q8_0.gguf`, with auto-download via `hf-hub` if missing (already a dep).

**Version pinning:** `llama-server` version is pinned via a documented minimum tag (e.g., `b5092` or later). Startup checks `llama-server --version` and errors if below the pinned minimum.

**Concurrency:** a single `llama-server` instance per preset handles all calls. Interior mutability not needed — HTTP client is `Send + Sync`.

## Invariant preservation

- Static `Embedder` trait (`DIM / MODEL_ID / PREFIX_SCHEME`) — unchanged from Step 1.
- Manifest v3 canary: each preset struct emits its `EmbedderIdentity` at index-build time; `HnswIndex::load` enforces identity+canary match. Quant tag in `MODEL_ID` means loading a Q8_0 index with a Q4_K_M backend is a hard error.
- `PrefixScheme::NONE` — Qwen3-Embedding-0.6B does not require per-passage/query prefixing for the retrieval use case (no instruction prompting).

## CLI / MCP wiring

- New `--backend qwen3-q8` (and `qwen3-q4`) flag on `index` / `query` / `serve-http` subcommands.
- Existing candle BGE path moves behind `legacy-candle` feature flag, non-default. No code is deleted — it becomes a fallback for constrained CI environments.
- MCP `search_corpus` tool unaffected at the interface level; backend choice is corpus-level metadata.

## CI concerns

- CI runners need `llama-server` binary. Plan: download prebuilt release from `ggml-org/llama.cpp` GitHub releases at the pinned tag, cache it.
- GGUF model download cached across runs (keyed by URL + size).
- New test: corpus round-trip (index → query → verify top-k) using Qwen3 backend, gated behind `--features llama-cpp`.
- Existing candle BGE tests continue to run under `--features legacy-candle`.

## Out of scope (deferred)

- Reranker integration (Step 3) — will reuse the same `llama-server` with `--reranking` flag, different model. Spec'd separately.
- Local LLM for contextual retrieval (Step 5) — same binary, different model. Spec'd separately.
- Q4_K_M preset — included as a struct but default presets used in benchmarks are Q8_0.

## Open questions

- Should `llama-server` be bundled in the fastrag release tarball or installed by the user? Lean: user-installed, version-pinned, with a `fastrag doctor` command that reports the detected version.
- Batch size tuning for Qwen3 at Q8_0 on typical pentest boxes (4-16 cores). Benchmark during implementation.
