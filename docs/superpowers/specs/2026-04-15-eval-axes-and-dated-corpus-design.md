# Eval Axes + Dated Gold Corpus + Markdown Frontmatter Metadata — Design

**Status:** proposed
**Date:** 2026-04-15
**Unblocks:** #53 (query-conditional halflife), #54 (weighted RRF), #55 (per-doctype date fields) — by producing the labeled data those features need, without implementing the features themselves.

## Context

Phase 2 shipped temporal decay and hybrid RRF with defaults chosen on engineering judgement rather than eval data. Three P3 enhancements (#53/#54/#55) were filed against those defaults but explicitly parked until labeled data exists. The gold set today has 120 questions across 50 markdown corpus docs with no axis labels, no attached dates, and no mixed doc types — neither the parked enhancements nor regressions in the shipped defaults are currently detectable.

This spec expands the eval harness's signal surface along three axes that matter for the parked work, without changing any retrieval behaviour.

## Goals

- Per-query axis labels (`style`, `temporal_intent`) on every gold-set entry.
- Per-bucket metrics in the eval report, per-bucket slack in the regression gate.
- Dated gold corpus: every `tests/gold/corpus/*.md` carries a real `published_date` and, where applicable, a `last_modified`.
- Markdown frontmatter flows into the index as typed metadata, reusing the JSONL type-inference layer.
- Baseline recapture under the new schema, committed in-tree.

## Non-goals

- Implementing query-conditional halflife (#53), weighted RRF (#54), or per-doctype date-field maps (#55). Those are separate specs that consume this spec's output.
- Expanding the corpus with new doc types (NVD JSON fixtures, advisory fixtures, blog fixtures). That is a follow-up that points the eval at a real external corpus.
- Changing chunking defaults, reranker defaults, or HNSW parameters.

## Architecture

Three stacked landings. Each is independently reviewable, independently revertible, and leaves the tree shippable.

### Landing 1 — Markdown frontmatter → typed metadata

**Scope:** `crates/fastrag-markdown`, `crates/fastrag/src/ingest`, `fastrag-cli/src/args.rs`.

- `MarkdownParser::parse` detects a leading `---\n...\n---\n` YAML block and parses it as a flat map. Keys become entries in `Metadata.extra: BTreeMap<String, String>` (existing field, currently unused by the markdown parser). Values are stringified from their YAML scalars. Malformed YAML returns a descriptive parse error with line context.
- `index_path_with_metadata` gains a `metadata_types: BTreeMap<String, TypedKind>` parameter. The resolution pipeline collects string-typed metadata from (1) CLI `base_metadata`, (2) sidecar `<path>.meta.json`, (3) `Document.metadata.extra` (parser-emitted frontmatter). Last wins on collision. Each named field is then promoted through the shared typing helpers in `fastrag::ingest::jsonl` using `metadata_types` as a hint. Fields not listed in the CLI flags pass through as `TypedValue::String`.
- CLI: the existing JSONL flags `--metadata-fields` and `--metadata-types` (both comma-separated, already on the `index` subcommand) lose their "JSONL:" restriction and apply to directory ingest as well. No new flag is added. `--metadata-fields` nominates which frontmatter keys to promote; `--metadata-types` supplies per-field kind overrides using the same `field=kind` syntax JSONL ingest accepts today.

**Contract:** a markdown file with `---\npublished_date: 2021-12-10\n---` plus `--metadata-fields published_date --metadata-types published_date=date` produces chunks carrying `user_fields` containing `("published_date", TypedValue::Date(2021-12-10))` — the same shape JSONL ingest produces today, consumable by the existing temporal-decay code in `crates/fastrag/src/corpus/hybrid.rs`.

### Landing 2 — Gold-set axes + per-bucket metrics + per-bucket gate

**Scope:** `crates/fastrag-eval/src/gold_set.rs`, `crates/fastrag-eval/src/report.rs`, `crates/fastrag-eval/src/baseline.rs`.

- `GoldEntry` gains a required `axes: Axes` field. `Axes { style: Style, temporal_intent: TemporalIntent }`. Both are non-optional; loading a file without `axes` on every entry returns `EvalError::MalformedDataset` with the offending entry id.
  - `Style`: `Identifier | Conceptual | Mixed`
  - `TemporalIntent`: `Historical | Neutral | RecencySeeking`
- `RunReport` gains a `buckets: BTreeMap<String, BTreeMap<String, BucketMetrics>>` computed from the existing `per_question` list at report-build time. Outer key is axis name (`"style"`, `"temporal_intent"`), inner is axis value (`"identifier"`, `"recency_seeking"`, …). `BucketMetrics { hit_at_1, hit_at_5, hit_at_10, mrr_at_10, n }`. Empty buckets are omitted.
- `BaselineFile.schema_version` bumps to `2`. Loading a v1 baseline against a v2 report fails fast with a descriptive error pointing at the recapture command.
- `BaselineDiff` gains per-bucket deltas. Baseline file gains an optional `per_bucket_slack: Option<f64>` (defaults to `slack`). A per-bucket regression exceeding its slack fails the gate with the same severity as an overall regression.

**Contract:** `fastrag eval --baseline docs/eval-baselines/current.json` exits non-zero if any bucket in any variant regresses beyond `per_bucket_slack`.

### Landing 3 — Backfill, new questions, baseline recapture

**Scope:** `tests/gold/corpus/*.md`, `tests/gold/questions.json`, `docs/eval-baselines/current.json`, `.github/workflows/weekly.yml`, `README.md`, `CLAUDE.md`.

- Add a YAML frontmatter block to each of the 50 corpus docs with `published_date:` (required, `YYYY-MM-DD`) and, for docs describing a CVE that was revised post-publication, `last_modified:`. Dates are sourced from the real CVE/NVD/KEV records each doc describes — public data, no synthesis.
- Backfill `axes` on all 120 existing questions via a one-pass heuristic (`scripts/axis-backfill.py`, committed alongside): CVE/CWE-ID regex → `identifier`; markers like `latest`, `newest`, `current`, recent-year → `recency_seeking`; explicit past-year (`2014`, `2017`) or `as of` phrasing → `historical`; everything else → `neutral`/`conceptual` per style. Hand-review the output and correct borderline cases before committing.
- Add 30 new questions balancing the thin buckets: ~10 `recency_seeking`, ~10 `conceptual`, ~10 `historical`. Target distribution after backfill + new: roughly 40 / 70 / 40 across temporal buckets, and 70 / 60 / 20 across style buckets. Final total 150.
- Update `.github/workflows/weekly.yml`: both `index` invocations (ctx + raw) gain `--metadata-fields published_date,last_modified --metadata-types published_date=date,last_modified=date`.
- Recapture `docs/eval-baselines/current.json` under schema v2. Capture happens locally with llama-server running, committed in the same commit as the new gold-set file so the gate stays green across the landing.
- README: add a "Metadata in markdown frontmatter" subsection under the CLI docs. CLAUDE.md: add the new build commands for the frontmatter tests.

## Data model

### `Axes` (fastrag-eval)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axes {
    pub style: Style,
    pub temporal_intent: TemporalIntent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style { Identifier, Conceptual, Mixed }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalIntent { Historical, Neutral, RecencySeeking }
```

### `BucketMetrics` (fastrag-eval)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketMetrics {
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub hit_at_10: f64,
    pub mrr_at_10: f64,
    pub n: usize,
}
```

### Report + baseline additions

`RunReport`:
```rust
#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
pub buckets: BTreeMap<String, BTreeMap<String, BucketMetrics>>,
```

`BaselineFile`:
```rust
pub schema_version: u32,              // 2
#[serde(default, skip_serializing_if = "Option::is_none")]
pub per_bucket_slack: Option<f64>,
```

## Testing

- **Landing 1:** markdown parser unit tests (frontmatter extract, malformed YAML error, trailing `---` handled, no frontmatter passes through). Integration test: index a markdown fixture with a date field and verify `user_fields` carries `TypedValue::Date` post-ingest. Precedence test: CLI base + sidecar + frontmatter on the same key, frontmatter wins.
- **Landing 2:** gold-set round-trip with axes, missing-axes rejection, bucket aggregation math against a hand-computed expected. Baseline v1 → v2 migration error. Per-bucket slack gate: under-threshold pass, over-threshold fail.
- **Landing 3:** weekly workflow YAML validates. `fastrag eval --baseline current.json` exits 0 against a freshly recaptured baseline. Spot-check five corpus docs have parseable frontmatter dates.

## Rollout

Each landing is its own commit (or tight sequence of commits) on `main`, pushed in order. Baselines stay green through Landing 1 and Landing 2 because neither changes the queried data. Landing 3 is the one commit where the baseline file itself changes shape and values — they change together.

## Open questions

- (resolved) Axis taxonomies: `style ∈ {identifier, conceptual, mixed}`, `temporal_intent ∈ {historical, neutral, recency_seeking}`.
- (resolved) Inline axes on gold entries vs sidecar bucket files: inline.
- (resolved) Per-bucket metrics baked into report vs computed at diff time: baked.
- (resolved) Date attachment shape: markdown frontmatter, unified with JSONL typing.
- (resolved) CLI surface: reuse existing `--metadata-fields` / `--metadata-types` rather than a new flag.

## References

- `docs/superpowers/specs/2026-04-14-temporal-decay-hybrid-retrieval-design.md` — the decay infrastructure this spec produces data for.
- `docs/superpowers/plans/2026-04-12-jsonl-ingest-record-storage.md` — the typed-metadata layer we reuse.
- Issues #53, #54, #55 — the parked features this spec unblocks.
