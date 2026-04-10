# Step 3 Design: Cross-Encoder Reranker

**Date:** 2026-04-09
**Phase:** 2, Step 3
**Status:** Spec approved, pending implementation plan

## Goal

Add a cross-encoder reranker as a second-stage retrieval step. After HNSW retrieval produces initial candidates, the reranker rescores each candidate against the query using a cross-encoder model, improving ranking quality.

Two backends ship in this step: ONNX (default) and llama-cpp. Both implement the existing `Reranker` trait. The dual-backend design enables empirical comparison of runtime characteristics (latency, memory, accuracy) before committing to one.

## Model Choices

| Backend | Model | Params | License | Format | Source |
|---------|-------|--------|---------|--------|--------|
| ONNX | gte-reranker-modernbert-base | 149M | Apache 2.0 | ONNX int8 | Alibaba-NLP/gte-reranker-modernbert-base |
| llama-cpp | bge-reranker-v2-m3 | 568M | MIT | GGUF Q8_0 | klnstpr/bge-reranker-v2-m3-Q8_0-GGUF |

**Why different models per backend:** Each backend uses the strongest model available in its format. gte-reranker-modernbert-base leads sub-1B reranking benchmarks (83% Hit@1, 149M params) and targets ONNX cleanly. bge-reranker-v2-m3 is the most mature GGUF reranker with established llama.cpp compatibility.

**Known risk (llama-cpp):** llama.cpp issue #16407 reports that community-converted GGUF rerankers can produce garbage scores when the `cls.output.weight` classifier tensor is missing. The `klnstpr` Q8_0 conversion is established and tested. Integration tests validate scores.

## Architecture

### Crate structure

Single `fastrag-rerank` crate with feature-gated backends:

```
crates/fastrag-rerank/
  Cargo.toml           # features: onnx (default), llama-cpp, test-utils
  src/
    lib.rs             # Reranker trait + RerankError (existing)
    onnx/
      mod.rs           # GteModernBertReranker
      model_source.rs  # 3-tier model resolution for ONNX files
    llama_cpp/
      mod.rs           # BgeRerankerV2M3Llama
      client.rs        # LlamaCppRerankClient (/v1/rerank HTTP)
```

### Reranker trait (existing, one addition)

The trait in `crates/fastrag-rerank/src/lib.rs` remains as-is:

```rust
pub trait Reranker: Send + Sync {
    fn model_id(&self) -> &'static str;
    fn rerank(&self, query: &str, hits: Vec<SearchHit>) -> Result<Vec<SearchHit>, RerankError>;
}
```

Add one `RerankError` variant:

```rust
#[error("HTTP: {0}")]
Http(String),
```

### ONNX backend: `GteModernBertReranker`

**Module:** `crates/fastrag-rerank/src/onnx/mod.rs`
**Feature gate:** `onnx`
**Dependencies:** `ort`, `tokenizers`

```rust
pub struct GteModernBertReranker {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
}
```

**Model loading:** 3-tier resolution matching the embedder pattern:
1. `$FASTRAG_MODEL_DIR/gte-reranker-modernbert-base/`
2. `~/.cache/fastrag/models/gte-reranker-modernbert-base/`
3. Download from HuggingFace on first use

**Session config:** `with_cpu_mem_arena(false)` (ORT's BFCArena never releases CPU memory; disabling is a precaution). `GraphOptimizationLevel::Level3`.

**Inference:** Tokenize each (query, passage) pair, batch into a single `session.run()` call, extract logits, apply sigmoid, assign scores to hits, sort descending.

### llama-cpp backend: `BgeRerankerV2M3Llama`

**Module:** `crates/fastrag-rerank/src/llama_cpp/mod.rs`
**Feature gate:** `llama-cpp`
**Dependencies:** `fastrag-embed` (for `LlamaServerHandle` reuse), `reqwest`, `serde_json`

```rust
pub struct BgeRerankerV2M3Llama {
    client: LlamaCppRerankClient,
    handle: LlamaServerHandle,
}
```

**Subprocess:** Spawns a second `llama-server` process with `--embedding --pooling rank` flags on a different port from the embedder server. Reuses `LlamaServerHandle` from `fastrag-embed::llama_cpp::handle` for spawn, health-check, and graceful shutdown.

**HTTP client:** `LlamaCppRerankClient` hits the `/v1/rerank` endpoint:

Request:
```json
{"model": "<path>", "query": "...", "documents": ["...", "..."], "top_n": 50}
```

Response:
```json
{"results": [{"index": 0, "relevance_score": 0.87}, ...]}
```

**Model resolution:** 3-tier via `ModelSource::HfHub { repo: "klnstpr/bge-reranker-v2-m3-Q8_0-GGUF", file: "bge-reranker-v2-m3-q8_0.gguf" }`.

## Pipeline Integration

The existing `query_corpus_reranked()` in `crates/fastrag/src/corpus/mod.rs:406` implements the two-stage pattern: fetch `top_k * over_fetch` from HNSW, pass to `&dyn Reranker`, truncate to `top_k`. No changes needed to this function.

```
query text
    ↓
embed query → Vec<f32>
    ↓
HNSW search → top_k × over_fetch candidates
    ↓
reranker.rerank(query, candidates) → rescored + reordered
    ↓
truncate to top_k
    ↓
return SearchHit[]
```

## CLI Surface

### `fastrag query`

New flags:
- `--rerank <backend>`: `onnx` (default), `llama-cpp`, `off`. Selects the reranker backend.
- `--no-rerank`: Shorthand for `--rerank off`.
- `--rerank-over-fetch <N>`: First-stage fan-out multiplier (default: 10). HNSW retrieves `top_k × N` candidates for the reranker.

Default behavior: reranking enabled with ONNX backend. Pass `--no-rerank` to disable.

### `fastrag serve-http`

Same `--rerank`, `--no-rerank`, `--rerank-over-fetch` flags. The `/query` endpoint gains optional query params: `rerank=onnx|llama-cpp|off`, `over_fetch=10`.

### `fastrag index`

No changes. Reranking is query-time only.

### MCP `search_corpus`

Add optional `rerank` param to `SearchCorpusParams`: `Option<String>` accepting `"onnx"`, `"llama-cpp"`, `"off"`. Default: `"onnx"`.

## Reranker Loader

New `fastrag-cli/src/rerank_loader.rs` (mirrors `embed_loader.rs`):

```rust
pub fn load_reranker(kind: RerankerKindArg) -> Result<Box<dyn Reranker>, RerankLoaderError>
```

Handles model download, ONNX session creation, or llama-server subprocess spawn depending on the selected backend.

`RerankerKindArg` enum: `Onnx`, `LlamaCpp`, added to `args.rs`.

## Feature Flags

| Feature | Crate | Pulls in |
|---------|-------|----------|
| `onnx` | fastrag-rerank | ort, tokenizers |
| `llama-cpp` | fastrag-rerank | fastrag-embed/llama-cpp, reqwest, serde_json |
| `rerank` | fastrag (facade) | fastrag-rerank/onnx (default backend) |
| `rerank-llama` | fastrag (facade) | fastrag-rerank/llama-cpp |

The `retrieval` feature in `fastrag-cli` enables `rerank` by default.

## Memory Budget (20 GB LXC)

| Component | RSS |
|-----------|-----|
| Qwen3 embedder (llama-server) | ~800 MB |
| ONNX reranker (gte-modernbert, int8) | ~200 MB |
| llama-cpp reranker (bge-v2-m3 Q8_0) | ~800 MB |
| fastrag process + index | ~200 MB |
| **ONNX path total** | **~1.2 GB** |
| **llama-cpp path total** | **~1.8 GB** |

Both paths fit within 20 GB.

## Testing

### Unit tests

- **ONNX:** Test with a real model in CI. Download gte-reranker-modernbert-base ONNX, run inference on 3-5 known pairs, assert scores in [0,1] and correct ordering.
- **llama-cpp:** Wiremock for `/v1/rerank` endpoint. Test request shape, response parsing, error handling. Same pattern as `LlamaCppClient` tests in `crates/fastrag-embed/src/llama_cpp/client.rs`.
- **Trait:** Existing `MockReranker` tests in `crates/fastrag-rerank/src/lib.rs` cover the trait contract.

### Integration tests

- `fastrag-cli/tests/rerank_onnx_e2e.rs`: Index small corpus, query with `--rerank=onnx`, verify results differ from `--no-rerank`. Gated behind `#[ignore]` + `FASTRAG_RERANK_TEST=1`.
- `fastrag-cli/tests/rerank_llama_cpp_e2e.rs`: Same with `--rerank=llama-cpp`. Gated behind `FASTRAG_LLAMA_TEST=1`.

### CI

- Extend `test` job: add `--features rerank` to clippy/test commands.
- New `rerank-onnx` job: download ONNX model, run ONNX e2e test.
- Extend `llama-cpp-backend` job: download bge-reranker GGUF alongside embedder GGUF, run reranker e2e test.

## Out of Scope

- Eval harness integration (Step 6). The `--rerank` flag on the eval subcommand will be wired for future use; quality validation happens in Step 6.
- Hybrid retrieval (Step 4). Reranking plugs into the existing dense retrieval path only.
- Default-on flip in production. The default is ONNX-on for query commands; quality validation happens in Step 6.

## Sources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
- [bge-reranker-v2-m3 Q8_0 GGUF](https://huggingface.co/klnstpr/bge-reranker-v2-m3-Q8_0-GGUF)
- [llama.cpp #16407 — reranker score bugs](https://github.com/ggml-org/llama.cpp/issues/16407)
- [Reranker Benchmark: Top 8 Models Compared](https://aimultiple.com/rerankers)
- [ORT CPU arena leak — tarmo-rag-shim lessons](/tmp/fastrag-ort-lessons.md)
