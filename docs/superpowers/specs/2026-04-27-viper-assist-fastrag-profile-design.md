# VIPER Assist FastRAG Profile — Design

Closes <https://github.com/Tarmo-Technologies/fastrag/issues/74>.
Parent roadmap: <https://github.com/crook3dfingers/VIPER_Dashboard/issues/1>.

## Goal

Make FastRAG consume the VIPER Assist corpus JSONL produced by
`crook3dfingers/VIPER_Dashboard`'s `build/build_viper_assist_corpus.py`, with
the asymmetric `nomic-ai/nomic-embed-text-v1.5` retrieval embedder, so the
VIPER Assist sidecar's `/search` and `/ask` return relevant cited content for
beginner-style pentest observations like:

> I found SMB, LDAP, Kerberos, and WinRM open on a Windows server. What
> should I look at first?

## Non-goals

- Building the corpus producer — that lives in `crook3dfingers/VIPER_Dashboard`
  (`build/build_viper_assist_corpus.py`).
- Changing FastRAG's embedder backend selection or runtime — only adding a
  catalog-level prefix entry for Nomic.
- Hybrid (BM25 + dense) tuning for the VIPER corpus — defaults are good
  enough for v1; a follow-up via `fastrag-eval` if needed.
- Adding VIPER to the `fastrag-eval` gold-set / matrix — the smoke shell
  test is enough for #74's acceptance.
- Wiring the local stack scripts in VIPER_Dashboard or the Playwright
  browser smoke — those are VIPER#6 and VIPER#4 respectively.

## What already exists (reused, not rebuilt)

- **JSONL ingest engine** — `fastrag::ingest::engine::index_jsonl` (Issue #41).
  Driven by `JsonlIngestConfig` in `crates/fastrag/src/ingest/jsonl.rs`.
- **Preset pattern** — `tarmo_finding_preset()` in
  `crates/fastrag/src/ingest/presets.rs` + `IngestPresetArg::TarmoFinding`
  in `fastrag-cli/src/args.rs` + dispatch in `fastrag-cli/src/main.rs`.
- **Embedder profile system** — `fastrag.toml`
  `[embedder.profiles.<name>]`; `Settings -> ResolvedEmbedderProfile`
  resolution in `fastrag-cli/src/config.rs`. Already supports
  `backend = "ollama" | "openai" | "llama-cpp"`.
- **Prefix scheme** — `PrefixScheme::new("search_query: ", "search_document: ")`
  in `crates/fastrag-embed/src/lib.rs`.
- **Catalog defaults** — `catalog_prefix_defaults` in
  `fastrag-cli/src/config.rs`; previously knew only mxbai. Extended for Nomic.

## Changes

### 1. `viper_assist_preset()`

`crates/fastrag/src/ingest/presets.rs` — new `JsonlIngestConfig` matching
the live producer's emitted fields:

| Block | Fields |
|---|---|
| `text_fields` | `text`, `title`, `section`, `summary` |
| `id_field` | `id` |
| `metadata_fields` (13) | `url`, `category`, `ports`, `tools`, `tags`, `cves`, `mitre_ids`, `risk_signals`, `requires_credentials`, `risk_level`, `source_file`, `content_hash`, `schema_version` |
| `metadata_types` | `requires_credentials → Bool`, `schema_version → Numeric` |
| `array_fields` (6) | `ports`, `tools`, `tags`, `cves`, `mitre_ids`, `risk_signals` |
| `cwe_field` | `None` (VIPER tracks CVE strings via `cves`, not numeric CWE) |

The fields the issue's example row mentions but the live producer doesn't
emit yet (`phase`, `engagement_types`, `platforms`) are intentionally
omitted — the dynamic schema is forward-compatible (Issue #41 design), so
adding them in the producer later does not require a fastrag change beyond
appending to `metadata_fields` and `array_fields`. The VIPER sidecar
already knows about these filters; today's queries with those filters
return no rows (correct, as no rows have the field) rather than erroring.

### 2. `IngestPresetArg::ViperAssist`

`fastrag-cli/src/args.rs` adds the variant; `fastrag-cli/src/main.rs`
dispatches it to `viper_assist_preset()`. No HTTP-side changes — the HTTP
ingest path already accepts raw `JsonlIngestConfig` params.

### 3. `catalog_prefix_defaults` extension

`fastrag-cli/src/config.rs` — match arm added for `nomic-ai/nomic-embed-text-v1.5`
and the bare Ollama tag `nomic-embed-text`, both mapping to:

```rust
PrefixConfig {
    query: "search_query: ".into(),
    passage: "search_document: ".into(),
}
```

The mxbai entry is preserved verbatim; `use_catalog_defaults = false` and
unknown models fall through to `PrefixConfig::default()`.

Five new unit tests in `fastrag-cli/src/config.rs::catalog_defaults_tests`
cover both Nomic aliases, the mxbai regression, the disabled path, and an
unknown-model fallthrough.

### 4. Operator recipe

`docs/profiles/viper-assist.md` — covers the `fastrag.toml` profile shape
(`backend = "llama-cpp"` with a Nomic GGUF path), `fastrag index` and
`fastrag query` commands using `--preset viper-assist`, the `serve-http`
mode the VIPER sidecar consumes (`127.0.0.1:8081`), the live filter table,
re-indexing trigger via the existing `viper-assist-corpus.sha256` file,
and a one-page Ollama appendix for operators who already run it.

### 5. Smoke test

`fastrag-cli/tests/viper_assist_smoke.rs` — ignored-by-default integration
test gated by `FASTRAG_LLAMA_TEST=1` (matches the existing
`llama_cpp_corpus_e2e.rs` gate). Indexes
`fastrag-cli/tests/fixtures/viper_assist/smoke_corpus.jsonl` (11 hand-curated
real VIPER rows: AD attack chain, AD playbook phases 1-3, web-app testing
phases, network-recon phases) and runs the four
`fastrag-cli/tests/fixtures/viper_assist/smoke_queries.json` prompts. Each
query asserts top-5 contains at least one allowlisted page id.

The fifth issue-#74 prompt (Jenkins/Artifactory) is omitted: the live VIPER
corpus producer does not yet emit CI/CD-platform pages, so there is nothing
to retrieve. Tracked in the corpus producer roadmap, not here.

`scripts/smoke_viper_assist.sh` is a shell wrapper that exports
`FASTRAG_LLAMA_TEST=1` + `VIPER_NOMIC_GGUF` (default
`/var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf`) and runs the
test. Suitable for nightly cron.

## Why llama-cpp (not Ollama)

The VIPER stack already runs Granite via `llama-server` (the chat model).
Choosing llama-cpp for embeddings keeps the operator on a single backend
family — same toolchain, same GPU bindings, same failure modes — and avoids
adding an Ollama daemon. Fastrag's `fastrag-embed` `llama-cpp` feature
loads the GGUF in-process, so there is no extra daemon, port, or HTTP hop
on the query path. The Nomic GGUF (~250 MB at Q5_K_M) coexists comfortably
with Granite-4.0-H-Tiny Q4_K_M (~5 GB) on the target RTX 2080 (8 GB VRAM).

The Ollama path is documented as an appendix; the smoke test does not cover
it, and operators picking it own their own validation.

## Verification

Standard fastrag gates per `CLAUDE.md` (no CI on push for this repo —
nightly only — so local verification is the gate):

| Gate | Result |
|---|---|
| `cargo test -p fastrag --features store --lib presets` | 14/14 passing (7 new viper + 7 existing tarmo) |
| `cargo test -p fastrag-cli --lib catalog` | 5/5 passing (4 new + mxbai regression) |
| `cargo test -p fastrag-cli` (full suite, no e2e) | all green |
| `cargo clippy -p fastrag --features store,retrieval --lib` | clean |
| `cargo clippy -p fastrag-cli --features retrieval --lib` | clean for changed files; pre-existing `if_same_then_else` (`config.rs:129`) and `derivable_impls` (`embed_profile.rs:19`) warnings remain — out of scope for this PR |
| `bash -n scripts/smoke_viper_assist.sh` | clean |
| `cargo test -p fastrag-cli --test viper_assist_smoke` (no env) | 1 ignored, 0 failed |
| End-to-end smoke (manual, requires GGUF) | tracked in `scripts/smoke_viper_assist.sh` |

End-to-end manual verification path:

```bash
# 1. Place the GGUF
curl -L -o /var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q5_K_M.gguf
# 2. Run the smoke
scripts/smoke_viper_assist.sh
# Expect: all 4 smoke queries pass (top-5 contains an allowlisted VIPER page).
```

VIPER sidecar wiring sanity (no fastrag code change needed; just
verification):

```bash
cd ~/github/VIPER_Dashboard && python3 build/build_viper_assist_corpus.py --write --validate
fastrag index ./artifacts/viper-assist-corpus.jsonl \
  --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml \
  --embedder-profile viper-assist \
  --preset viper-assist \
  --ingest-format jsonl
fastrag serve-http --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml --embedder-profile viper-assist --bind 127.0.0.1:8081 &
viper-assist &
curl -s http://127.0.0.1:8765/search \
  -d '{"query":"SMB LDAP Kerberos WinRM"}' \
  -H 'content-type: application/json' | jq
# Expect: ranked VIPER page hits with the AD playbook on top.
```

## Sequencing after this lands

1. **fastrag #74 (this PR)** — real retrieval works.
2. **VIPER_Dashboard #6** — local-stack docs/scripts wrap the now-working
   stack with `start_local_stack.sh`, `check_local_stack.sh`, `.env.example`.
3. **VIPER_Dashboard #4** — Playwright browser smoke test (Mode A mocked +
   Mode B real-stack) verifies the sidecar end-to-end.
