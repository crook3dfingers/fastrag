# Plan: llama.cpp Embedder Backend (Phase 2 Step 2)

**Spec:** `docs/superpowers/specs/2026-04-09-llama-cpp-backend-design.md`
**Worktree:** `.worktrees/fastembed-backend` (branch `feat/fastembed-backend`, will be renamed or reused)
**Date:** 2026-04-09

## Principles

- TDD red-green-refactor, mandatory. No production code before a failing test.
- Each task ends in a passing `cargo test --workspace` and a clean commit.
- No rubber-stamp tests. Every test asserts concrete values.

## Tasks

### Task 1 — Gate existing candle BGE behind `legacy-candle` feature

- Move `candle-core / candle-nn / candle-transformers / tokenizers / hf-hub` in `crates/fastrag-embed/Cargo.toml` behind `legacy-candle` feature.
- Gate the BGE module and its re-exports with `#[cfg(feature = "legacy-candle")]`.
- Update `fastrag-cli` to not depend on candle by default.
- Update all existing tests that use candle BGE to gate with `#[cfg(feature = "legacy-candle")]`.
- Green test: `cargo test -p fastrag-embed` passes with candle disabled. `cargo test -p fastrag-embed --features legacy-candle` still passes.
- Commit: `refactor(embed): gate candle BGE behind legacy-candle feature`.

### Task 2 — Add `llama-cpp` feature skeleton + reqwest client reuse

- Add `llama-cpp` feature in `fastrag-embed/Cargo.toml` that pulls in `reqwest` (blocking) — reuse the existing `http-embedders` client code where possible; refactor shared HTTP helpers into a private module.
- Add empty `src/llama_cpp/mod.rs` with `#[cfg(feature = "llama-cpp")]`.
- Green test: `cargo build -p fastrag-embed --features llama-cpp` succeeds.
- Commit: `feat(embed): scaffold llama-cpp backend feature`.

### Task 3 — `LlamaServerHandle` lifecycle manager (TDD)

- Red: write a test that spawns a fake `llama-server` (a small binary or shell script returning a `/health` 200 and an `/embedding` stub), waits for readiness, and shuts it down on drop.
- Green: implement `LlamaServerHandle` with:
  - `spawn(config: LlamaServerConfig) -> Result<Self>` — launches subprocess, polls `/health` until ready (bounded timeout, default 30s).
  - `Drop` — SIGTERM + bounded wait.
  - `base_url()` / `client()` accessors.
- Refactor: extract `LlamaServerConfig` (model path, port, pooling mode, extra args).
- Concrete assertions: process exits cleanly; `/health` was polled; drop terminates the child.
- Commit: `feat(embed): llama-server subprocess lifecycle manager`.

### Task 4 — HTTP embedding client against `/embedding` endpoint (TDD)

- Red: test that POSTs a batch of passages to a wiremock-stubbed `/embedding` endpoint and asserts the client parses `Vec<Vec<f32>>` correctly, with correct batch shape and dim.
- Green: `LlamaCppClient` with `embed(texts: &[&str]) -> Result<Vec<Vec<f32>>>`. Use `serde_json` to build the request, `reqwest::blocking` for transport.
- Error cases: HTTP 4xx/5xx, connection refused, dimension mismatch vs expected → distinct `EmbedError` variants.
- Commit: `feat(embed): llama.cpp embedding HTTP client`.

### Task 5 — `Qwen3Embed600mQ8` preset struct implementing `Embedder` (TDD)

- Red: test asserting `Qwen3Embed600mQ8::DIM == 1024`, `MODEL_ID == "Qwen/Qwen3-Embedding-0.6B-GGUF@Q8_0"`, `PREFIX_SCHEME == PrefixScheme::NONE`. Second test with a wiremock server stubs `/embedding` and verifies `embed_query` / `embed_passage` round-trip preserves dim and batch shape.
- Green: implement the struct. `load(config)` spawns `LlamaServerHandle`, holds the handle + client. `embed_query` and `embed_passage` both go through the client (no prefixing — `PrefixScheme::NONE`).
- Assert: dropping the struct shuts down the server.
- Commit: `feat(embed): Qwen3Embed600mQ8 preset`.

### Task 6 — Model path resolution + hf-hub auto-download

- Red: test that when `$FASTRAG_MODEL_DIR/Qwen3-Embedding-0.6B-Q8_0.gguf` exists, `resolve_model_path` returns it without network. Test that when missing, a configurable "downloader" trait is invoked (mock it — do not hit HF in tests).
- Green: `ModelSource` enum — `Local(PathBuf)` | `HfHub { repo: &'static str, file: &'static str }`. Resolution strategy: check `$FASTRAG_MODEL_DIR` → fall back to `~/.cache/fastrag/models` → invoke downloader.
- Commit: `feat(embed): model path resolution for GGUF presets`.

### Task 7 — llama-server version check on spawn

- Red: test that spawn fails with a clear error when the version string is below the pinned minimum tag. Use a fake `llama-server` script that prints an old version.
- Green: parse `llama-server --version` output, compare against `MIN_LLAMA_SERVER_TAG` const, error with actionable message if below.
- Commit: `feat(embed): enforce minimum llama-server version`.

### Task 8 — CLI wiring

- Add `--backend qwen3-q8` option to `fastrag-cli` `index`, `query`, `serve-http` subcommands.
- Default remains whatever it was pre-Task-1 (candle BGE via `legacy-candle` feature, or Qwen3 if legacy-candle is disabled — TBD during implementation, prefer Qwen3 as default once it works).
- Add `fastrag doctor` subcommand that checks for `llama-server` in `$PATH` and reports its version.
- Tests: CLI integration test (snapshot or assert exit code + stdout) for `fastrag doctor` in both "found" and "missing" cases (PATH manipulated).
- Commit: `feat(cli): wire llama-cpp backend and doctor subcommand`.

### Task 9 — Corpus round-trip integration test

- Integration test under `crates/fastrag-embed/tests/` or `fastrag-cli/tests/`:
  1. Skip if `llama-server` not in PATH (CI installs it in Task 10).
  2. Index a small fixture corpus with Qwen3 backend.
  3. Query with known-good query, assert top-1 result is the expected doc.
  4. Verify manifest v3 identity matches.
- Gate with `#[cfg(feature = "llama-cpp")]` and `#[ignore]`-by-default with an env-var opt-in for local, run in CI unconditionally.
- Commit: `test(embed): corpus round-trip with Qwen3 llama-cpp backend`.

### Task 10 — CI workflow: install llama-server, cache model

- Update `.github/workflows/ci.yml`:
  - New job `llama-cpp-backend` that installs `llama-server` from a pinned `ggml-org/llama.cpp` release, caches the binary and the GGUF model.
  - Runs `cargo test --workspace --features llama-cpp`.
- Commit: `ci: install llama.cpp and run qwen3 backend tests`.

### Task 11 — Docs

- Update `README.md`: new "Embedder backends" section explaining Qwen3 default, `legacy-candle` fallback, `fastrag doctor`, `$FASTRAG_MODEL_DIR`.
- Update `CLAUDE.md` with `--features llama-cpp` test commands.
- Delete stale spec+plan: `docs/superpowers/specs/2026-04-09-fastembed-backend-design.md` and `docs/superpowers/plans/2026-04-09-fastembed-backend.md`.
- Commit: `docs: llama.cpp backend + delete stale fastembed-rs spec/plan`.

## Definition of done

- `cargo test --workspace --features llama-cpp` green locally and in CI.
- `cargo test --workspace --features legacy-candle` still green (no regression on the legacy path).
- `cargo clippy --workspace --all-targets --features llama-cpp -- -D warnings` clean.
- Corpus round-trip test passes on CI with real `llama-server` and real Qwen3 GGUF.
- README + CLAUDE.md updated.
- Stale fastembed-rs spec/plan deleted.
