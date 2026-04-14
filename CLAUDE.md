# FastRAG — Project Conventions

## Build & Test

```bash
cargo test --workspace       # Run all tests
cargo test --workspace --features retrieval  # Retrieval stack tests
cargo clippy --workspace     # Lint
cargo clippy --workspace --all-targets --features retrieval -- -D warnings  # Retrieval lint gate
cargo test -p fastrag-embed --features llama-cpp       # llama-cpp backend unit tests
FASTRAG_LLAMA_TEST=1 cargo test -p fastrag-cli --test llama_cpp_corpus_e2e -- --ignored  # Requires llama-server + GGUF model
cargo test -p fastrag-rerank --features onnx           # ONNX reranker unit tests
cargo test -p fastrag-rerank --features llama-cpp      # llama-cpp reranker unit tests
cargo test --workspace --features hybrid                                          # Hybrid retrieval tests
cargo test -p fastrag-cli --test hybrid_e2e --features hybrid                     # Hybrid e2e integration test
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid -- -D warnings  # Full lint gate
FASTRAG_RERANK_TEST=1 cargo test -p fastrag-rerank --features onnx -- --ignored  # Requires ONNX model files
FASTRAG_RERANK_TEST=1 cargo test -p fastrag-cli --test rerank_onnx_e2e -- --ignored  # ONNX rerank e2e
FASTRAG_LLAMA_TEST=1 cargo test -p fastrag-cli --test rerank_llama_cpp_e2e -- --ignored  # llama-cpp rerank e2e
cargo test --workspace --features contextual                                # Contextual retrieval tests
cargo test -p fastrag-context                                               # Contextualizer crate unit tests
cargo test -p fastrag-context --features test-utils --test cache_resume     # Cache resume integration test
cargo test -p fastrag-context --features test-utils --test stage_fallback   # Stage fallback integration test
cargo test -p fastrag-context --features llama-cpp                          # LlamaCppContextualizer unit tests (wiremock)
FASTRAG_LLAMA_TEST=1 cargo test -p fastrag-cli --features contextual,contextual-llama --test contextual_corpus_e2e -- --ignored
FASTRAG_LLAMA_TEST=1 cargo test -p fastrag-cli --features contextual,contextual-llama --test contextual_retry_failed_e2e -- --ignored
FASTRAG_LLAMA_TEST=1 cargo test -p fastrag-cli --features contextual,contextual-llama --test contextual_strict_e2e -- --ignored
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings  # Full lint gate (with contextual)
cargo test --workspace --features eval                                        # Eval harness unit tests
cargo test -p fastrag-eval --features real-driver                             # Real-driver build (needed by CLI)
cargo test -p fastrag-eval --test matrix_stub                                 # Matrix orchestrator stub test
cargo test -p fastrag-eval --test gold_set_loader                             # Gold set loader validation branches
cargo test -p fastrag-eval --test union_match                                 # Union-of-top-k scorer
cargo test -p fastrag-eval --test baseline_diff                               # Baseline diff + slack gate
FASTRAG_LLAMA_TEST=1 FASTRAG_RERANK_TEST=1 cargo test -p fastrag-cli --features eval,contextual,contextual-llama,retrieval,rerank,hybrid --test eval_matrix_e2e -- --ignored
cargo test -p fastrag-cli --features eval --test eval_gold_set_rejects_invalid_e2e
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval -- -D warnings  # Full lint gate with eval
cargo test -p fastrag-nvd                                                     # NVD crate unit tests
cargo test -p fastrag-nvd --test nvd_end_to_end                               # NVD fixture integration test (5-CVE slice)
cargo test -p fastrag --features hygiene --lib                                # Hygiene filters unit tests
cargo test --workspace --features nvd,hygiene                                 # NVD parser + hygiene chain tests
FASTRAG_NVD_TEST=1 cargo test -p fastrag-cli --features nvd,hygiene,retrieval --test security_profile_e2e -- --ignored  # End-to-end ingest with --security-profile (requires fastrag binary on PATH)
cargo test -p fastrag-cwe                                                      # CWE taxonomy closure unit tests
cargo test -p fastrag-cwe --features compile-tool                              # compile-taxonomy binary tests
cargo test -p fastrag --test cwe_expansion --features retrieval                # CWE hierarchy expansion end-to-end
cargo test -p fastrag-cli --test cwe_expand_e2e --features "retrieval,store"   # CLI --cwe-expand e2e
cargo test -p fastrag-cli --test cwe_expand_http_e2e --features "retrieval,store"  # HTTP cwe_expand param e2e
cargo test -p fastrag --features retrieval --test hybrid_retrieval    # Hybrid BM25 + dense RRF integration test
cargo test -p fastrag --features retrieval --test temporal_decay      # Temporal decay integration test (Date-typed metadata)
cargo test -p fastrag-cli --features retrieval --test hybrid_e2e      # CLI --hybrid e2e (jsonl ingest + query)
cargo test -p fastrag-cli --features retrieval --test temporal_decay_e2e        # CLI --time-decay-* e2e + error paths
cargo test -p fastrag-cli --features retrieval --test temporal_decay_http_e2e   # HTTP GET /query decay params + 400 handling
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings  # Full lint gate with hygiene
cargo fmt --check            # Format check
cargo build --release        # Release build (binary at target/release/fastrag)
```

### PDF Feature Flags

The PDF parser has optional feature flags for advanced extraction:

```bash
cargo test -p fastrag-pdf --features images           # Image extraction tests
cargo test -p fastrag-pdf --features table-detect      # Table detection tests
cargo test -p fastrag-pdf --features footnotes         # Footnote extraction tests
cargo test -p fastrag-pdf --features column-detect     # Column detection tests
cargo test --workspace --features pdf-images,pdf-table-detect  # Full feature set
cargo bench -p fastrag-pdf --bench pdf_parsing --features images,table-detect  # Benchmarks
```

The OCR feature (`--features ocr`) requires the `tesseract` library and `pdfium-render` (statically linked). These dependencies must be available in the build environment.

## MCP Server

FastRAG includes an MCP (Model Context Protocol) server for AI assistant integration.

```bash
cargo run -- serve                    # Start MCP server (stdio transport)
cargo build --release                 # Build with MCP
cargo build -p fastrag-mcp --features mcp-search  # Enable corpus search tool
cargo build --release --no-default-features --features language-detection  # Build without MCP
```

### Retrieval CLI

The CLI includes semantic corpus retrieval commands behind the `retrieval` feature. The `fastrag-cli` binary enables this feature by default.

```bash
cargo run -- index ./documents --corpus ./corpus
cargo run -- index ./documents --corpus ./corpus --contextualize
cargo run -- index --corpus ./corpus --contextualize --retry-failed
cargo run -- query "invoice payment terms" --corpus ./corpus --top-k 5
cargo run -- corpus-info --corpus ./corpus  # Shows contextualized: true or false
cargo run -- serve-http --corpus ./corpus --port 8081
cargo run -- eval --gold-set tests/gold/questions.json \
                  --corpus ./corpus-ctx \
                  --corpus-no-contextual ./corpus-raw \
                  --config-matrix \
                  --baseline docs/eval-baselines/current.json \
                  --report target/eval/matrix.json
```

The retrieval path uses `fastrag-embed` for embeddings and `fastrag-index` for persistence (`manifest.json`, `index.bin`, `entries.bin`).

### MCP Tools

| Tool | Description |
|------|-------------|
| `parse_file` | Parse a single document file |
| `parse_directory` | Parse all files in a directory |
| `list_formats` | List supported file formats |
| `chunk_document` | Parse and chunk a file for RAG |
| `search_corpus` | Semantic search across a persisted corpus |

The MCP crate lives at `crates/fastrag-mcp/`.

## Architecture

- Workspace with crates in `crates/` and CLI in `fastrag-cli/`
- Every format parser implements the `Parser` trait from `fastrag-core`
- `crates/fastrag/` is the facade library with `ParserRegistry` for format detection and dispatch
- `crates/fastrag/src/ops.rs` is the shared operations layer used by both CLI and MCP
- `crates/fastrag/src/corpus.rs` contains the shared retrieval ops used by CLI, HTTP, and MCP search
- `crates/fastrag-embed/` provides embedding models and test doubles
- `crates/fastrag-index/` provides the persisted vector index
- `crates/fastrag-mcp/` is the MCP server crate (gated behind `mcp` feature flag)
- `fastrag-cli` enables retrieval by default and also exposes an HTTP query server
- Feature flags gate each parser crate (default: all enabled)
- Test fixtures live in `tests/fixtures/`

## Conventions

- Use `thiserror` for error types, not manual `impl Error`
- All parsers must be `Send + Sync`
- Element text should be trimmed, no trailing whitespace
- Table elements use markdown table format in their text field
- Unit tests go in the same file as the code (`#[cfg(test)]` module)
- Integration tests go in `tests/`

### Commit Messages

When a commit resolves a GitHub issue, include `Closes #N` in the commit message body. For multiple issues, use `Closes #N1, Closes #N2`. GitHub automatically closes the referenced issues when the commit lands on the default branch.

Example:
```
feat: add EPUB format support

Closes #6
```

## Adding a New Format Parser

1. Create `crates/fastrag-{format}/` with `Cargo.toml` and `src/lib.rs`
2. Implement `Parser` trait
3. Add to workspace `Cargo.toml` members
4. Add feature flag in `crates/fastrag/Cargo.toml`
5. Register in `ParserRegistry::default()` in `crates/fastrag/src/registry.rs`
6. Add test fixture in `tests/fixtures/`

## CLI and MCP Are Separate Surfaces

CLI and MCP serve different users with different workflows. **Do not build MCP tools as 1:1 mirrors of CLI commands** — that is an anti-pattern that produces awkward MCP tools and bloats both surfaces.

- **CLI** is for humans and scripts: batch operations, flags, exit codes, shell composition.
- **MCP** is for LLM agents: use-case-oriented tools matching how an LLM reasons about a task. Tools may be narrower, broader, or wholly different from their CLI counterparts.

Some operations belong only in CLI (ingest, maintenance, repair passes). Some belong only in MCP (query helpers tuned for LLM context). Some belong in both with different shapes. Decide per operation — don't default to mirroring.

Shared logic still goes in `crates/fastrag/src/ops.rs` so both surfaces can call it, but the surfaces themselves are designed independently.

When adding a new operation, ask:
1. Does a human running this in a shell need it? → CLI.
2. Does an LLM agent need this to answer a user question or drive a workflow? → MCP.
3. If both: design each surface for its own caller. Do not force a shared shape.

## Development Discipline

### TDD Red-Green-Refactor (Mandatory)

Every piece of code written in this repo must follow TDD red-green-refactor:

1. **Red** — Write a failing test first. Run `cargo test --workspace` to confirm it fails with the expected error.
2. **Green** — Write the minimal code to make it pass. Run `cargo test --workspace` again to confirm green.
3. **Refactor** — Clean up while keeping tests green.

This is not optional. Do not write production code before writing the test that exercises it.

- Unit tests go in `#[cfg(test)]` modules in the same file as the code.
- Integration tests go in `tests/`.

### No Rubber-Stamp Tests — Strictly Forbidden

A rubber-stamp test is one that passes trivially without actually verifying behaviour. Forbidden patterns:

- `assert!(true)` or `assert!(result.is_some())` when a concrete value can be asserted
- Tests where every call is mocked and the only assertion checks the mock was called (no real logic exercised)
- Tests written *after* the implementation that re-state what the code happens to return
- Catching panics and asserting success instead of the specific error type and message
- Tests whose assertion would still pass if the implementation were a no-op or returned a constant

Every test must have at least one assertion on a **concrete value** derived from real logic, and must fail if the implementation is broken or deleted.

## Skills (`.claude/skills/`)

Reusable prompts that keep main-context token usage low. To invoke a skill: read the skill file — no substitution needed — pass the skill file content directly as the `prompt` to a Haiku Agent call (`model=haiku`). Unless the table below specifies "foreground", use `run_in_background=true`.

| Skill file | Model | When to use |
|---|---|---|
| `ci-watcher.md` | Haiku, background | After every `git push` — **mandatory** |
| `doc-editor/SKILL.md` | Haiku, foreground | Before every `Edit` or `Write` to a `.md` file — **mandatory** |
