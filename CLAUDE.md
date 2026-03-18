# FastRAG — Project Conventions

## Build & Test

```bash
cargo test --workspace       # Run all tests
cargo clippy --workspace     # Lint
cargo fmt --check            # Format check
cargo build --release        # Release build (binary at target/release/fastrag)
```

### PDF Feature Flags

The PDF parser has optional feature flags for advanced extraction:

```bash
cargo test -p fastrag-pdf --features images           # Image extraction tests
cargo test -p fastrag-pdf --features table-detect      # Table detection tests
cargo test --workspace --features pdf-images,pdf-table-detect  # Full feature set
cargo bench -p fastrag-pdf --bench pdf_parsing --features images,table-detect  # Benchmarks
```

The OCR feature (`--features ocr`) requires the `tesseract` library and `pdfium-render` (statically linked). These dependencies must be available in the build environment.

## MCP Server

FastRAG includes an MCP (Model Context Protocol) server for AI assistant integration.

```bash
cargo run -- serve                    # Start MCP server (stdio transport)
cargo build --release                 # Build with MCP
cargo build --release --no-default-features --features language-detection  # Build without MCP
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `parse_file` | Parse a single document file |
| `parse_directory` | Parse all files in a directory |
| `list_formats` | List supported file formats |
| `chunk_document` | Parse and chunk a file for RAG |

The MCP crate lives at `crates/fastrag-mcp/`.

## Architecture

- Workspace with crates in `crates/` and CLI in `fastrag-cli/`
- Every format parser implements the `Parser` trait from `fastrag-core`
- `crates/fastrag/` is the facade library with `ParserRegistry` for format detection and dispatch
- `crates/fastrag/src/ops.rs` is the shared operations layer used by both CLI and MCP
- `crates/fastrag-mcp/` is the MCP server crate (gated behind `mcp` feature flag)
- Feature flags gate each parser crate (default: all enabled)
- Test fixtures live in `tests/fixtures/`

## Conventions

- Use `thiserror` for error types, not manual `impl Error`
- All parsers must be `Send + Sync`
- Element text should be trimmed, no trailing whitespace
- Table elements use markdown table format in their text field
- Unit tests go in the same file as the code (`#[cfg(test)]` module)
- Integration tests go in `tests/`

## Adding a New Format Parser

1. Create `crates/fastrag-{format}/` with `Cargo.toml` and `src/lib.rs`
2. Implement `Parser` trait
3. Add to workspace `Cargo.toml` members
4. Add feature flag in `crates/fastrag/Cargo.toml`
5. Register in `ParserRegistry::default()` in `crates/fastrag/src/registry.rs`
6. Add test fixture in `tests/fixtures/`

## CLI + MCP Parity

Every user-facing operation must be exposed in both CLI and MCP server, both calling shared ops in `crates/fastrag/src/ops.rs`. When adding a new operation:

1. Add the function to `crates/fastrag/src/ops.rs`
2. Add CLI subcommand/flag in `fastrag-cli/src/args.rs` and handler in `main.rs`
3. Add MCP tool in `crates/fastrag-mcp/src/lib.rs` with `#[tool]` attribute
4. Add tests for both CLI and MCP paths

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

Reusable prompts that keep main-context token usage low. To invoke a skill: read the skill file — no substitution needed — pass the skill file content directly as the `prompt` to a background Haiku Agent call (`model=haiku`, `run_in_background=true`). The task completion notification signals pass/fail.

| Skill file | Model | When to use |
|---|---|---|
| `ci-watcher.md` | Haiku, background | After every `git push` — **mandatory** |
| `doc-editor/SKILL.md` | (inherits) | Before every `Edit` or `Write` to a `.md` file — **mandatory** |
