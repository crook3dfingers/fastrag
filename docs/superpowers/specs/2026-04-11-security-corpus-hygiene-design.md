# fastrag Phase 2 — Step 7: Security Corpus Hygiene

## Context

Phase 2 Steps 1–6 are all shipped and green on `main`:

- Step 1 — Embedder invariant refactor (manifest v4 with `EmbedderIdentity` + canary)
- Step 2 — llama.cpp embedder (Qwen3-Embedding-0.6B-Q8)
- Step 3 — Cross-encoder reranker (ONNX + llama-cpp)
- Step 4 — Hybrid retrieval (Tantivy BM25 + dense, RRF fusion)
- Step 5 — Contextual retrieval (Anthropic Sept 2024 prefix-prepend technique)
- Step 6 — Eval harness: 105-entry gold set, 50-doc corpus, 4-variant config matrix, weekly regression gate

**Why Step 7.** The research doc at `docs/rag-research-2026-04.md` §10 and the lessons doc both flag an ingest-time correctness bug specific to security corpora: NVD ships ~760 rejected CVEs per year, they dominate top-k on short queries, and they poisoned the earlier retrieval. Filtering them at ingest is the single biggest hygiene win documented. Four more hygiene tasks follow: boilerplate stripping, language filtering, temporal metadata for KEV/recency boosting, and vendor/product normalization.

**Scope decision (user-approved).** The full roadmap bullet list assumes structured NVD/GHSA/OSV/KEV input that fastrag does not currently parse. Fastrag today only parses documents (PDF, markdown, DOCX, etc.) through the `Parser` trait. CVE/CWE regex extraction happens at *query* time, not ingest. The user picked decomposition **Option A — "NVD parser + generic filters"**:

- **In Step 7:** a new `fastrag-nvd` parser crate that reads NVD 2.0 JSON feeds and emits `Document` objects with structured metadata, plus a generic hygiene filter chain enabled by `--security-profile`.
- **Deferred to Step 8 or later:** GHSA parser, OSV parser, KEV live-catalog join, cross-source dedup with GHSA prose merge, full CPE 2.3 structural normalization beyond vendor/product strings, Tantivy schema migration for a numeric year field.

This keeps the first PR focused and ships a working NVD-hygiene profile that measurably improves gold-set hit@5 via the Step 6 eval harness.

---

## Verified facts from the Phase 1 exploration

**Ingest pipeline (`crates/fastrag/src/corpus/mod.rs:196-420`).** The `index_path_with_metadata` function runs: `load_document` (line 299) → `chunk_document` (line 301) → optional contextualize stage (306–319, feature gated) → embed stage (321–349) → metadata merge + `IndexEntry` build (351–398) → `tantivy.add_entries(&entries)` (402–404). Hygiene filters plug in cleanly between chunk (301) and contextualize (306): chunks exist, file-level metadata is available, but the LLM and embedder have not been invoked yet, so filtering is cheap.

**Chunk struct (`crates/fastrag-core/src/chunking.rs:6-19`).** Fields: `elements`, `text`, `char_count`, `section`, `index`, `contextualized_text`. Per-chunk metadata is not supported. File-level metadata propagates uniformly to all chunks via `file_metadata.clone()` at line 394 of `corpus/mod.rs`. Hygiene decisions are **file-scoped** (reject the whole file) or **chunk-text-scoped** (strip boilerplate from text).

**Tantivy schema (`crates/fastrag-tantivy/src/schema.rs:26-69`).** Typed fields: `id`, `chunk_text` (BM25 indexed), `display_text` (stored raw), `source_path`, `section`, `cve_id`, `cwe`. A free-form `metadata_json` stored blob accepts arbitrary key-value pairs (serialized `BTreeMap<String, String>`) without schema migration. This is where `published_year`, `vuln_status`, `kev_flag`, `language`, `cpe_vendor`, `cpe_product`, and `cvss_severity` will live. Filtering by these fields uses the `query_corpus_with_filter` path (`corpus/mod.rs:498-548`).

**CVE/CWE regex location (`crates/fastrag-index/src/identifiers.rs:8-37`).** The regexes (`\bCVE-\d{4}-\d{4,7}\b`, `\bCWE-\d+\b`) run at query time, not ingest. Ingest-time extraction is not currently wired; metadata is expected via `.meta.json` sidecars or `--metadata k=v`. Step 7 does not change this — the NVD parser populates `cve_id` directly in structured metadata, bypassing the regex path.

**Parser registry (`crates/fastrag/src/registry.rs:11-52`).** Format parsers register in `ParserRegistry::default()`. Formats: text, CSV, markdown, HTML, PDF, XML, XLSX, DOCX, PPTX, EPUB, RTF, email. `fastrag-nvd` slots into this registry behind a `nvd` feature flag.

**Parser trait single-doc shape.** `Parser::parse(&self, path: &Path) -> Result<Document, Error>` returns one `Document` per file. An NVD feed is a multi-record file — 40k CVEs per yearly dump. Step 7 must emit one `Document` per CVE so each has its own metadata. The shape options are a new `MultiDocParser` trait alongside `Parser`, or an extended `Parser` with `parse_all(&self, path) -> Result<Vec<Document>>` defaulting to `vec![self.parse(path)?]`. Resolved in writing-plans.

**Eval-harness NVD loader (`crates/fastrag-eval/src/datasets/nvd.rs:14-19`).** Eval-only, not in the ingest path. Downloads NVD 2.0 feeds from NIST and deserializes `NvdFeed → vulnerabilities[] → cve { id, descriptions[] { lang, value } }`. Step 7 lifts the serde types into a shared `fastrag-nvd-schema` module so ingest and eval share one source of truth, avoiding parallel NVD schemas.

**Language detection already present.** `whatlang = { version = "0.16", optional = true }` in `crates/fastrag-core/Cargo.toml:13`, feature-gated behind `language-detection`. Step 7 reuses it.

**No numeric year/date field in Tantivy.** Step 7 stores `published_year` as a string in `metadata_json` and filters via `query_corpus_with_filter`. Numeric boost requires a schema migration, deferred.

---

## Design recommendation

### High-level architecture

Two parallel deliverables:

1. **`crates/fastrag-nvd/`** — new parser crate registered behind a `nvd` feature flag. Detects NVD 2.0 feeds by schema peek (`format` field == `"NVD_CVE"` and `vulnerabilities[]` array present) rather than `.json` extension. Emits one `Document` per CVE record with file-level metadata pre-populated: `cve_id`, `vuln_status`, `published_year`, `cvss_severity`, `cpe_vendor`, `cpe_product`, `description_lang`.

2. **`crates/fastrag/src/hygiene/`** — new module gated behind a `hygiene` feature flag (default off). Provides a `ChunkFilter` trait and a `SecurityProfile` chain composing:
   - `MetadataRejectFilter` — drops docs where `vuln_status` ∈ {`Rejected`, `Disputed`}
   - `BoilerplateStripper` — strips `** REJECT **`, `** DISPUTED **`, CPE 2.3 URIs, reference URL lists, legal notices from chunk text
   - `LanguageFilter` — runs `whatlang` on chunk text, drops or flags non-English chunks per a policy enum
   - `KevTemporalTagger` — metadata enricher that tags chunks with `kev_flag=true` when the CVE-ID is in a caller-supplied KEV ID set loaded from a JSON file path

   CLI surface: `fastrag index ... --security-profile [--security-lang en] [--security-kev-catalog path/to/kev.json] [--security-reject-statuses Rejected,Disputed]`.

### File layout

```
crates/fastrag-nvd/
  Cargo.toml            — serde, serde_json, thiserror, fastrag-core dep
  src/
    lib.rs              — re-exports, Parser impl shell
    schema.rs           — NVD 2.0 serde types (lifted from eval)
    parser.rs           — NvdFeedParser: schema detect + multi-doc emit
    metadata.rs         — CVE → BTreeMap<String, String> projection
  tests/
    fixture.rs          — real NVD feed slice, 5 CVEs

crates/fastrag/src/hygiene/
  mod.rs                — ChunkFilter trait, SecurityProfile chain builder
  reject.rs             — MetadataRejectFilter
  boilerplate.rs        — BoilerplateStripper (regex-driven)
  language.rs           — LanguageFilter (whatlang wrapper)
  kev.rs                — KevTemporalTagger (JSON loader + metadata enricher)

crates/fastrag-eval/src/datasets/nvd.rs
                        — refactor to depend on fastrag-nvd-schema types,
                          remove duplicate serde definitions

fastrag-cli/src/args.rs — new --security-profile + sub-flags on Index variant
fastrag-cli/src/main.rs — wire flag to HygieneChain::security_default()
```

No existing file exceeds ~500 lines; every new file is under 300 lines with a single responsibility.

### Metadata contract

Every CVE document carries this metadata map:

| Key | Source | Example | Purpose |
|---|---|---|---|
| `cve_id` | `cve.id` | `CVE-2024-12345` | exact-match lookup |
| `vuln_status` | `cve.vulnStatus` | `Analyzed` \| `Rejected` \| `Disputed` \| `Modified` | reject filter |
| `published_year` | `cve.published[0:4]` | `2024` | temporal filter (string) |
| `cvss_severity` | `metrics.cvssMetricV31[0].cvssData.baseSeverity` | `HIGH` | post-filter at query |
| `cpe_vendor` | first `cpeMatch.criteria` vendor | `apache` | facet |
| `cpe_product` | first `cpeMatch.criteria` product | `log4j` | facet |
| `description_lang` | `descriptions[0].lang` | `en` | language filter |
| `kev_flag` | JOIN on KEV catalog if supplied | `true` \| absent | temporal boost |
| `source` | constant | `nvd` | cross-source provenance |

Metadata lives in Tantivy's `metadata_json` blob, queryable via `query_corpus_with_filter`. Adding fields costs zero schema migration.

### Filter ordering

Inside `index_path_with_metadata`, between chunk (301) and contextualize (306):

```
chunks = chunk_document(&doc, chunking)
chunks, doc_metadata = hygiene.apply(chunks, doc_metadata)  // new step
if let Some(ctx) = contextualizer { run_contextualize_stage(&mut chunks, ctx) }
...
```

`HygieneChain::apply` runs filters in order: reject → strip → language → kev-tag. Reject runs first to avoid embedding docs about to be dropped. Boilerplate strip runs before language detect so the stripped text is what `whatlang` sees. KEV tag runs last because it only mutates metadata.

### Feature flags

- `nvd` on the facade `fastrag` crate — pulls in `fastrag-nvd`, registers the parser. Off by default.
- `hygiene` on the facade `fastrag` crate — enables the `hygiene` module and the `--security-profile` CLI branch. Off by default.
- Full lint gate extends to: `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings`

### Testing

**Unit tests** (`#[cfg(test)]`):

- `fastrag-nvd::schema` round-trip with a real NIST feed slice (5 CVEs).
- `fastrag-nvd::parser` multi-doc emission count + metadata correctness.
- `hygiene::reject` drops `Rejected` and `Disputed`, keeps `Analyzed`.
- `hygiene::boilerplate` strips `** REJECT **`, `** DISPUTED **`, CPE URIs (`cpe:2.3:a:...`), URL lists, leaves normal prose intact.
- `hygiene::language` flags a Spanish test string as `es`, English as `en`, configurable threshold.
- `hygiene::kev` tags CVE-2021-44228 when KEV catalog contains it, leaves CVE-2024-99999 untagged.

**Integration tests** (`crates/fastrag-nvd/tests/`, `crates/fastrag/tests/hygiene/`):

- `nvd_end_to_end.rs` — parse fixture NVD slice, produce 5 docs, verify each metadata field.
- `hygiene_chain.rs` — apply full `SecurityProfile` to a mix of CVE + boilerplate chunks, assert expected reject/keep counts.

**E2E test** (`fastrag-cli/tests/`, fast path):

- `security_profile_e2e.rs` — run `fastrag index fixtures/nvd_slice.json --security-profile --corpus tmp`, then query for a specific CVE, assert Rejected CVEs don't appear and metadata is queryable.

**Gold-set regression** (hooks into Step 6's eval harness):

- 5–10 gold-set entries in `tests/gold/questions.json` that exercise hygiene (e.g., "CVE-2023-XXXXX rejected status — should NOT appear", "What was the impact of Log4Shell" with `must_contain_cve_ids: [CVE-2021-44228]` and `must_contain_terms: [kev]`).
- Baseline refresh under the `docs/eval-baselines/README.md` flow after the profile lands. This is the end-to-end quality proof that Step 7 improves hit@5.

### Rollout — 9 landings, separate commits on `main`

1. **`fastrag-nvd` crate skeleton.** Cargo.toml, empty lib.rs, workspace registration. `cargo check` green. No tests.
2. **NVD 2.0 serde types.** `schema.rs`. Lift the existing eval types and extend them with the fields hygiene filters need. Move the canonical schema here; `fastrag-eval::datasets::nvd` becomes a thin re-export. Unit test: round-trip a 5-CVE NIST fixture slice. Clippy + fmt green.
3. **`fastrag-nvd::parser` single-record path.** Implement `Parser` trait for a single CVE (pretend the feed has one). Establish the metadata-projection logic. Unit test with a one-CVE fixture.
4. **Multi-doc emission.** New `MultiDocParser` trait in `fastrag-core` (or a `parse_all` method on `Parser`; pick one in writing-plans). Update `index_path_with_metadata` to call the multi-doc path when the parser opts in. Unit test with a 5-CVE fixture: 5 documents out. Verify CLI ingest still works for single-doc parsers (markdown, PDF).
5. **Hygiene module skeleton + `MetadataRejectFilter`.** New `crates/fastrag/src/hygiene/` with `ChunkFilter` trait, `HygieneChain` composer, and the reject filter. Unit tests: round-trip reject+keep.
6. **`BoilerplateStripper`.** Regex-driven. Unit tests against the four boilerplate patterns (REJECT, DISPUTED, CPE 2.3 URIs, URL lists) with both positive and negative cases.
7. **`LanguageFilter`.** `whatlang` wrapper. Unit tests for en/es/de detection and the "non-English → drop / flag" policy enum. Depends on `whatlang` feature being on.
8. **`KevTemporalTagger`.** Loads a KEV JSON (use the public CISA vulnerabilities.json schema; document the expected shape in a docstring). Tags chunks whose `cve_id` is in the set. Unit tests.
9. **CLI wiring + e2e + docs.** `--security-profile` flag, `SecurityProfile` default chain builder. E2E test runs the full ingest on a fixture. Update `CLAUDE.md` Build & Test section with the new feature flag + test commands. Add a "Security Corpus Hygiene" subsection to `README.md` after the Contextual Retrieval section. Gold-set entries + baseline refresh. Push + ci-watcher background.

Each landing is a separate commit on `main`. TDD red-green per `CLAUDE.md`. No worktrees. Local `cargo test + clippy + fmt` before every push. ci-watcher as background Haiku Agent after every push.

### Non-goals (explicit, to prevent scope creep)

- GHSA, OSV, KEV-as-primary-source parsers. (Step 8.)
- Cross-source dedup with GHSA prose merge. (Step 8.)
- Full CPE 2.3 structural normalization beyond vendor/product string extraction.
- Tantivy schema migration for a numeric `published_year` or `cvss_score` field. Step 7 stores everything in `metadata_json` strings.
- Live KEV catalog fetch. The user supplies a path to a JSON file; fastrag does not download.
- Automatic translation of non-English descriptions. `LanguageFilter` offers drop/flag; translate is out.
- Changing the query-time retrieval path. Reranker boost for post-2020+KEV is a follow-up after Step 7 proves the metadata is available.

### Open design questions to resolve in writing-plans

1. **Multi-doc emission shape.** New `MultiDocParser` trait alongside `Parser`, or extend `Parser` with a `parse_all(&self, path) -> Result<Vec<Document>>` method that defaults to `vec![self.parse(path)?]`? The first is cleaner but adds a second trait dispatch path. The second is more backwards-compatible but touches every parser trait impl.
2. **KEV file schema.** CISA's public JSON schema vs. a fastrag-specific `{cve_ids: []}` minimal format. Accept both — detect shape at load time.
3. **Reject-status default set.** Just `Rejected` + `Disputed`, or also `Awaiting Analysis` / `Undergoing Analysis`? Roadmap specifies the first two; stick with that and make it overridable via CLI.
4. **Strip patterns source of truth.** Hard-coded regex list in `boilerplate.rs`, or a pluggable config file? First PR uses a hard-coded list commented with NVD schema references. Revisit if churn becomes a problem.

---

## Critical files to touch (reference, not exhaustive)

- **Read before writing**: `crates/fastrag/src/corpus/mod.rs:196-420` (ingest pipeline), `crates/fastrag-core/src/chunking.rs:6-19` (Chunk struct), `crates/fastrag-core/src/parser.rs` (Parser trait), `crates/fastrag/src/registry.rs:11-52` (registry), `crates/fastrag-tantivy/src/schema.rs:26-69` (Tantivy schema), `crates/fastrag-eval/src/datasets/nvd.rs` (existing NVD serde to lift).
- **Create**: everything under `crates/fastrag-nvd/` and `crates/fastrag/src/hygiene/`.
- **Modify**: `crates/fastrag/src/corpus/mod.rs` (hygiene insertion point, multi-doc loop), `crates/fastrag/src/registry.rs` (register `fastrag-nvd`), `crates/fastrag/Cargo.toml` (feature flags), `fastrag-cli/src/args.rs` + `fastrag-cli/src/main.rs` (CLI flag), `crates/fastrag-eval/src/datasets/nvd.rs` (deduplicate serde types), `CLAUDE.md`, `README.md`, `docs/superpowers/roadmap-2026-04-phase2-rewrite.md` (mark Step 7 shipped).

## Verification

- `cargo test --workspace --features nvd,hygiene` — unit + integration green.
- `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings` — clean.
- `cargo run --release -p fastrag-cli --features nvd,hygiene,retrieval -- index tests/gold/corpus/nvd-fixture.json --corpus /tmp/nvd-corpus --security-profile` completes and prints a hygiene summary (`rejected: N, stripped: N, lang-dropped: N, kev-tagged: N`).
- Query for a known Rejected CVE in the fixture returns no hits.
- Query for a known KEV CVE returns hits with `kev_flag=true` in metadata.
- Gold-set eval matrix run shows hit@5 delta ≥ 0 vs. the pre-Step-7 baseline; ideally > 0 on hygiene-sensitive questions.
- Step 7 marked ✅ in the roadmap.
