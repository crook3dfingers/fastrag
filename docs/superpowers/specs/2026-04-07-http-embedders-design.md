# HTTP Embedder Backends — Design (#31a)

## Purpose

Add two HTTP-backed embedder implementations — OpenAI and Ollama — alongside the existing local `BgeSmallEmbedder`. Wire them through `fastrag index`, `query`, and `serve-http` so users can choose a backend at runtime. Auto-detect the embedder from the corpus manifest on read paths so users don't have to repeat themselves.

ONNX is out of scope here and lands as #31b. Cohere/Voyage are rejected as duplicative.

## Trait & error refactor

The `Embedder` trait is unchanged structurally; one signature evolves:

- `fn model_id(&self) -> &'static str` → `fn model_id(&self) -> String`. HTTP embedders need runtime ids like `openai:text-embedding-3-small`. `BgeSmallEmbedder` and `MockEmbedder` are updated.

`EmbedError` gains:

- `MissingEnv(&'static str)` — missing API key env var.
- `Http(String)` — reqwest transport errors.
- `Api { status: u16, message: String }` — non-2xx response (body truncated to 500 chars).
- `DimensionProbeFailed(String)` — Ollama probe failed at construction.
- `UnknownModel { backend: &'static str, model: String }` — OpenAI unknown model name.

These map cleanly through the existing `CorpusError::Embed(#[from] EmbedError)`.

## Module layout

```
crates/fastrag-embed/src/
  lib.rs           # trait, re-exports
  bge.rs           # existing BGE local
  http/
    mod.rs         # shared reqwest client construction, helpers
    openai.rs      # OpenAIEmbedder
    ollama.rs      # OllamaEmbedder
  error.rs
  test_utils.rs
```

Cargo features:

- `bge-local` — existing local BGE (kept).
- `http-embedders` — enables `http::openai` + `http::ollama`. Pulls `reqwest` (`blocking`, `json`, `rustls-tls`).
- `test-utils` — existing.

`fastrag-cli`'s `retrieval` feature enables both `bge-local` and `http-embedders`.

## OpenAIEmbedder

```rust
pub struct OpenAIEmbedder {
    model: String,
    api_key: String,
    base_url: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl OpenAIEmbedder {
    pub fn new(model: impl Into<String>) -> Result<Self, EmbedError>;
    pub fn with_base_url(self, url: impl Into<String>) -> Self;
}
```

- **Construction**: read `OPENAI_API_KEY` from env (error if absent). Look up `dim` in a static table for the supported models:
  - `text-embedding-3-small` → 1536
  - `text-embedding-3-large` → 3072
  Unknown model → `EmbedError::UnknownModel`. No silent probing.
- **Request**: blocking `POST {base_url}/embeddings` with `{"model": ..., "input": [texts]}`. OpenAI accepts batch input natively.
- **Response**: parse `data[].embedding`, asserting `len == texts.len()`.
- **`default_batch_size = 512`** (well below the 2048 token-count ceiling, comfortable for RSS).
- **Errors**: non-2xx → `Api { status, message }` with body excerpt; transport → `Http`.
- **Retry**: single retry on 5xx or connection reset, 500ms backoff. Anything more is a user concern.
- **`model_id()`** → `format!("openai:{}", self.model)`.
- **Blocking client** (not async). The corpus indexing path is sync; spinning up a runtime for one HTTP call is overkill.

## OllamaEmbedder

```rust
pub struct OllamaEmbedder {
    model: String,
    base_url: String,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl OllamaEmbedder {
    pub fn new(model: impl Into<String>) -> Result<Self, EmbedError>;
    pub fn with_base_url(self, url: impl Into<String>) -> Self;
}
```

- **Construction**: no API key. Honor `OLLAMA_HOST` env var if `base_url` not explicitly set.
- **Dimension probing on `new()`**: Ollama hosts arbitrary user-pulled models, so a static table is wrong. Issue one probe call (`POST /api/embeddings` with `{"model": ..., "prompt": "a"}`) and record `embedding.len()` as `dim`. Probe failure → `EmbedError::DimensionProbeFailed`.
- **Request**: `POST /api/embeddings` with `{"model": ..., "prompt": <text>}`. Ollama's endpoint is **single-prompt, not batched** — the embedder loops over inputs internally.
- **`default_batch_size = 1`** (overrides the trait default of 64) so callers don't expect batching that isn't happening. Document the per-request latency cost.
- **Errors**: same `Api`/`Http` mapping as OpenAI. 404 on the model id is a common case ("model not pulled") and surfaces as `Api { status: 404, ... }`.
- **`model_id()`** → `format!("ollama:{}", self.model)`.

## CLI surface

New flags on `index`, `query`, and `serve-http` (gated by `retrieval` feature):

```
--embedder <bge|openai|ollama>            [default: bge]
--openai-model <name>                     [default: text-embedding-3-small]
--openai-base-url <url>                   [default: https://api.openai.com/v1]
--ollama-model <name>                     [default: nomic-embed-text]
--ollama-url <url>                        [default: http://localhost:11434]
```

`--model-path` stays and applies only when `--embedder bge`. API keys come from env vars only — never as CLI flags.

### Auto-detect on read paths

For `query` and `serve-http`: if `--embedder` is omitted, parse `<backend>:<model>` out of the corpus manifest's `embedding_model_id` and reconstruct the matching embedder. Backend-specific flags (`--openai-base-url`, `--ollama-url`, etc.) still apply if passed. If the user explicitly passes a `--embedder` whose final `model_id()` differs from the manifest's, hard error and print both ids in the message.

### Mismatch protection on index

`index_path_with_metadata` already loads the existing manifest (via #28). When the manifest exists, compare its `embedding_model_id` to the new embedder's `model_id()`; on mismatch, return `CorpusError::EmbedderMismatch { existing, requested }`. The user's recovery is to delete the corpus directory or pick a different one. A `--reset` flag is out of scope here.

## Testing

### Unit tests

`crates/fastrag-embed/src/http/openai.rs` (`#[cfg(test)]`):

- Happy path: wiremock stubs `POST /embeddings` returning a fixed `data[].embedding` payload; assert vectors round-trip.
- 401: stub returns 401 + JSON error body; assert `EmbedError::Api { status: 401, .. }`.
- Length mismatch: response has fewer embeddings than inputs; assert error.
- Unknown model: `OpenAIEmbedder::new("text-embedding-9001")` → `UnknownModel`.
- `model_id()` returns `openai:text-embedding-3-small`.

`crates/fastrag-embed/src/http/ollama.rs` (`#[cfg(test)]`):

- Happy path: wiremock stubs `/api/embeddings` for both the probe call and a query call; assert dim and round-trip.
- Probe failure (connection refused): construction returns `DimensionProbeFailed`.
- 404 on missing model: stub returns 404; assert `Api { status: 404, .. }`.
- `model_id()` returns `ollama:nomic-embed-text`.

### Integration tests

`crates/fastrag-embed/tests/http_e2e.rs` — wiremock-backed end-to-end through the `Embedder` trait: a tiny `embed(&["a","b"])` round-trip per backend.

`fastrag-cli/tests/embedder_e2e.rs`:

- `--embedder openai --openai-base-url http://127.0.0.1:<wiremock>`: full `index` → `query` round-trip against the fake server.
- Auto-detect: `index --embedder openai`, then `query` (no flag), assert success and the manifest's `embedding_model_id` is `openai:text-embedding-3-small`.
- Mismatch: index with openai, query with `--embedder ollama` → non-zero exit, manifest model_id appears in stderr.

### Real-API smoke tests

`#[ignore]` tests gated on `FASTRAG_E2E_OPENAI=1` and the env-supplied `OPENAI_API_KEY`. Documented in README under "Testing against real APIs". Never run in CI.

### Existing tests

All existing `Embedder`-using tests (corpus, incremental, eval) must still pass against `MockEmbedder` and `BgeSmallEmbedder`.

## Dependencies added

- `wiremock` — dev-dep on `fastrag-embed` and `fastrag-cli`.
- `reqwest` (`blocking`, `json`, `rustls-tls`) — gated behind `http-embedders` on `fastrag-embed`. The workspace already has reqwest as a dev-dep; promote it to a regular workspace dep with optional features.

## Out of scope (deferred)

- ONNX backend (`#31b`).
- Cohere / Voyage (rejected).
- Async embedding, parallel HTTP batching beyond OpenAI's native batch.
- Token-aware chunking respecting per-model context limits.
- `--reset` flag for clobbering a corpus on embedder change.
- Disk cache of API responses.
- Cost telemetry.
