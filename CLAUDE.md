# FastRAG — Project Conventions

## Build & Test

```bash
cargo test --workspace       # Run all tests
cargo clippy --workspace     # Lint
cargo fmt --check            # Format check
cargo build --release        # Release build (binary at target/release/fastrag)
```

## Architecture

- Workspace with crates in `crates/` and CLI in `fastrag-cli/`
- Every format parser implements the `Parser` trait from `fastrag-core`
- `crates/fastrag/` is the facade library with `ParserRegistry` for format detection and dispatch
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
