# Step 6 — Eval Harness Refresh + Gold Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the fastrag eval harness against the full Phase 2 Steps 1–5 retrieval stack with a hand-curated ≥100-entry gold set, a 4-variant config matrix, per-stage latency instrumentation, and a weekly CI job gated on a 2% slack baseline.

**Architecture:** Extend `crates/fastrag-eval/` in place with three new modules (`gold_set.rs`, `matrix.rs`, `baseline.rs`). Instrument `crates/fastrag/src/corpus/mod.rs` with a `LatencyBreakdown` struct threaded through all 5 `query_corpus_*` variants. The new matrix runner drives retrieval directly via `query_corpus_hybrid_reranked` / `query_corpus_hybrid` / `query_corpus_reranked` per variant — separate from the existing `fastrag-eval::Runner` which stays on its `HnswIndex::query` path for BEIR. Weekly GitHub Actions workflow with a 7-day `check-changes` gate and a 45-minute timeout.

**Tech Stack:** Rust 2021, `hdrhistogram`, `serde` + `serde_json`, `regex` (new workspace dep), `thiserror`, `assert_cmd` (e2e), `clap` (CLI), llama.cpp (existing Step 2 backend), `tantivy` (existing Step 4 BM25), ONNX runtime (existing Step 3 reranker), `fastrag-context` (existing Step 5).

**Reference documents:**
- Spec: `docs/superpowers/specs/2026-04-11-eval-harness-design.md`
- Roadmap: `docs/superpowers/roadmap-2026-04-phase2-rewrite.md` lines 102–110
- Research: `docs/rag-research-2026-04.md` §8

**Ground-truth notes from pre-plan exploration (reference before editing):**
- `query_corpus` has **5 variants** in `crates/fastrag/src/corpus/mod.rs`:
  - `query_corpus` (line ~437)
  - `query_corpus_with_filter` (line ~458)
  - `query_corpus_reranked` (line ~538)
  - `query_corpus_hybrid` (line ~558)
  - `query_corpus_hybrid_reranked` (line ~596)
- Raw chunk text is exposed via `SearchHit.entry.chunk_text`.
- Existing `fastrag-eval::Runner` owns a **local** histogram inside `run()`, uses `HnswIndex::query` directly, and does **not** call `query_corpus_*`. Leave that path alone — the matrix is a new top-level flow.
- `EvalArgs` lives in `fastrag-cli/src/args.rs:312-360`, not `main.rs`. Subcommand dispatch is in `fastrag-cli/src/main.rs:461-492` under `#[cfg(feature = "eval")]`.
- Workspace deps: `blake3`, `rusqlite`, `hdrhistogram`, `serde_json` already present. **`regex` is missing** — Task 1 adds it.
- `EmbedderIdentity` already derives `Serialize + Deserialize` in `crates/fastrag-embed/src/lib.rs`.
- `--contextualize` flag exists on `fastrag index` (Step 5). Use it as-is to build the two fixture corpora.

---

## File Structure

### New files

```
crates/fastrag-eval/
  src/
    gold_set.rs              — GoldSet + GoldSetEntry types, load(), validate(), score_entry()
    matrix.rs                — ConfigVariant enum, MatrixReport, run_matrix() orchestrator
    baseline.rs              — Baseline, VariantBaseline, diff() + slack gate
  tests/
    gold_set_loader.rs       — Integration: load() validation branches
    union_match.rs           — Integration: score_entry() synthetic chunks
    baseline_diff.rs         — Integration: diff() good-run / bad-run fixtures
    matrix_stub.rs           — Integration: run_matrix() against a stub Corpus
    fixtures/
      gold_valid.json        — 3-entry valid fixture
      gold_invalid_empty_q.json
      gold_invalid_dup_id.json
      gold_invalid_malformed_cve.json
      gold_invalid_zero_assertions.json
      baseline_current.json  — Checked-in baseline for diff tests
      report_good.json       — Passing report for diff tests
      report_bad.json        — Regressing report for diff tests

tests/gold/
  corpus/
    01-libfoo-rce.md         — Starter doc for Rollout 1 canary
    02-kev-bluekeep.md       — Starter doc
    03-ssrf-proxy.md         — Starter doc
    04-cwe-502-deserialize.md
    05-pronoun-resolution-sample.md
  questions.json             — Starter: 10 entries for Rollout 1;
                               grown to ≥100 in Rollout 6

docs/eval-baselines/
  current.json               — Captured locally in Rollout 6
  README.md                  — Refresh + approval flow (Rollout 7)

fastrag-cli/
  tests/
    eval_matrix_e2e.rs       — E2E: --config-matrix over mini fixture
    eval_gold_set_rejects_invalid_e2e.rs  — E2E: --gold-set <bad>
    fixtures/eval_mini/
      corpus/
        01-libfoo.md
        02-ssrf.md
        03-deserialize.md
        04-buffer-overflow.md
        05-path-traversal.md
      questions.json         — 10 entries

.github/workflows/
  weekly.yml                 — Sundays 06:00 UTC, 45-min timeout, 7-day check-changes
```

### Modified files

```
Cargo.toml                                    — Add `regex = "1"` to [workspace.dependencies]
crates/fastrag-eval/Cargo.toml                — Add `regex`, `serde_json` as deps
crates/fastrag-eval/src/lib.rs                — Re-export gold_set, matrix, baseline
crates/fastrag-eval/src/error.rs              — Add new EvalError variants
crates/fastrag/src/corpus/mod.rs              — Add LatencyBreakdown + thread through 5 variants
crates/fastrag-mcp/src/lib.rs                 — Update query_corpus_with_filter call site
fastrag-cli/src/main.rs                       — Update 4 query_corpus_* call sites
fastrag-cli/src/args.rs                       — Extend EvalArgs with new fields
fastrag-cli/src/eval.rs (if exists, else main.rs dispatch) — Wire --config-matrix
CLAUDE.md                                     — Build & Test section for new commands
README.md                                     — Eval section
```

---

## Rollout Landing Map

The spec's "Rollout" section lists 7 landings. Each landing below groups a coherent set of tasks that produce one milestone commit series. Tasks within a landing are ordered; do them in sequence.

- **Landing 1 (Tasks 1–9):** `gold_set.rs` + error variants + 10-entry starter fixture + push-CI canary.
- **Landing 2 (Tasks 10–14):** `LatencyBreakdown` struct + instrumentation across all 5 `query_corpus_*` variants + call-site updates.
- **Landing 3 (Tasks 15–19):** `matrix.rs` + `ConfigVariant` + `matrix_stub.rs` integration test.
- **Landing 4 (Tasks 20–23):** `baseline.rs` + `baseline_diff.rs` integration test with checked-in report fixtures.
- **Landing 5 (Tasks 24–28):** CLI wiring (`--gold-set`, `--corpus-no-contextual`, `--config-matrix`, `--baseline`) + mini e2e fixtures + `eval_matrix_e2e.rs` + `eval_gold_set_rejects_invalid_e2e.rs`.
- **Landing 6 (Tasks 29–31):** Grow gold set to ≥100 entries, grow fixture corpus to ~50–100 docs, capture initial `docs/eval-baselines/current.json` locally and commit.
- **Landing 7 (Tasks 32–34):** `weekly.yml`, `CLAUDE.md`, `README.md`.

---

## Landing 1 — Gold set loader + fixtures + push CI canary

### Task 1: Add `regex` + `serde_json` workspace deps

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/fastrag-eval/Cargo.toml`

- [ ] **Step 1: Add `regex` to workspace deps**

Edit `Cargo.toml` at the workspace root. Find the `[workspace.dependencies]` table and add a line alongside the existing entries:

```toml
regex = "1"
```

- [ ] **Step 2: Pull `regex` into `fastrag-eval`**

Edit `crates/fastrag-eval/Cargo.toml`. Under `[dependencies]`, add:

```toml
regex = { workspace = true }
serde_json = { workspace = true }
```

(`serde_json` is likely already present; if it is, skip that line.)

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p fastrag-eval`
Expected: `Finished` with zero errors. The new deps are available but unused — that's fine.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/fastrag-eval/Cargo.toml
git commit -m "eval: add regex + serde_json workspace deps for gold set loader"
```

---

### Task 2: `GoldSet` + `GoldSetEntry` types + serde

**Files:**
- Create: `crates/fastrag-eval/src/gold_set.rs`
- Modify: `crates/fastrag-eval/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to the end of `crates/fastrag-eval/src/gold_set.rs` (new file):

```rust
//! Gold set schema loader + union-of-top-k scorer.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldSet {
    pub version: u32,
    pub entries: Vec<GoldSetEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GoldSetEntry {
    pub id: String,
    pub question: String,
    #[serde(default)]
    pub must_contain_cve_ids: Vec<String>,
    #[serde(default)]
    pub must_contain_terms: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gold_set_round_trips_through_json() {
        let gs = GoldSet {
            version: 1,
            entries: vec![GoldSetEntry {
                id: "q001".into(),
                question: "Is there an RCE in libfoo?".into(),
                must_contain_cve_ids: vec!["CVE-2024-12345".into()],
                must_contain_terms: vec!["libfoo".into()],
                notes: None,
            }],
        };
        let json = serde_json::to_string(&gs).unwrap();
        let back: GoldSet = serde_json::from_str(&json).unwrap();
        assert_eq!(gs, back);
    }
}
```

Add to `crates/fastrag-eval/src/lib.rs`:

```rust
pub mod gold_set;
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fastrag-eval gold_set::tests::gold_set_round_trips_through_json`
Expected: PASS (this task is pure type definition — the test exercises serde round-trip). If it fails, check that `serde_json` is a dependency.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-eval/src/gold_set.rs crates/fastrag-eval/src/lib.rs
git commit -m "eval: gold_set types with serde round-trip"
```

---

### Task 3: `EvalError` gold-set variants

**Files:**
- Modify: `crates/fastrag-eval/src/error.rs`

- [ ] **Step 1: Read the existing error enum**

Run: `cargo check -p fastrag-eval` first to see what's there. Then read `crates/fastrag-eval/src/error.rs` in full.

- [ ] **Step 2: Add the new variants**

Add these variants to the `EvalError` enum (preserve all existing ones):

```rust
#[error("gold set parse error at {path}: {source}")]
GoldSetParse {
    path: std::path::PathBuf,
    #[source]
    source: serde_json::Error,
},
#[error("gold set validation failed: {0}")]
GoldSetInvalid(String),
```

If `thiserror::Error`'s `#[source]` attribute isn't used elsewhere in the file, omit it and put `source` as a regular field — the point is that the error message references the path.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p fastrag-eval`
Expected: `Finished` with zero errors.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag-eval/src/error.rs
git commit -m "eval: gold set error variants"
```

---

### Task 4: `gold_set::load` with validation

**Files:**
- Modify: `crates/fastrag-eval/src/gold_set.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/fastrag-eval/src/gold_set.rs`:

```rust
use std::io::Write;
use tempfile::NamedTempFile;

fn write_fixture(json: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(json.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

#[test]
fn load_accepts_well_formed_gold_set() {
    let f = write_fixture(r#"{
        "version": 1,
        "entries": [
            {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
        ]
    }"#);
    let gs = load(f.path()).expect("valid gold set should load");
    assert_eq!(gs.entries.len(), 1);
    assert_eq!(gs.entries[0].id, "q001");
}

#[test]
fn load_rejects_empty_question() {
    let f = write_fixture(r#"{
        "version": 1,
        "entries": [
            {"id": "q001", "question": "", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
        ]
    }"#);
    let err = load(f.path()).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("q001"), "error must name offending id, got: {msg}");
    assert!(msg.contains("empty question"), "error must say 'empty question', got: {msg}");
}

#[test]
fn load_rejects_duplicate_id() {
    let f = write_fixture(r#"{
        "version": 1,
        "entries": [
            {"id": "q001", "question": "a?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []},
            {"id": "q001", "question": "b?", "must_contain_cve_ids": ["CVE-2024-2"], "must_contain_terms": []}
        ]
    }"#);
    let err = load(f.path()).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("duplicate"), "got: {msg}");
    assert!(msg.contains("q001"), "got: {msg}");
}

#[test]
fn load_rejects_malformed_cve_id() {
    let f = write_fixture(r#"{
        "version": 1,
        "entries": [
            {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-24-1"], "must_contain_terms": []}
        ]
    }"#);
    let err = load(f.path()).unwrap_err();
    assert!(format!("{err}").contains("CVE-24-1"));
}

#[test]
fn load_rejects_zero_assertions() {
    let f = write_fixture(r#"{
        "version": 1,
        "entries": [
            {"id": "q001", "question": "x?", "must_contain_cve_ids": [], "must_contain_terms": []}
        ]
    }"#);
    let err = load(f.path()).unwrap_err();
    assert!(format!("{err}").contains("no must_contain"));
}
```

Add `tempfile = { workspace = true }` to `crates/fastrag-eval/Cargo.toml` under `[dev-dependencies]` if not already present (likely is — the crate uses it elsewhere).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fastrag-eval gold_set::tests::load`
Expected: FAIL — function `load` not found.

- [ ] **Step 3: Implement `load`**

Add to `crates/fastrag-eval/src/gold_set.rs` (above the `#[cfg(test)]` module):

```rust
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::error::EvalError;

static CVE_ID_RE: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"^CVE-\d{4}-\d+$").unwrap());

pub fn load(path: &Path) -> Result<GoldSet, EvalError> {
    let bytes = std::fs::read(path).map_err(EvalError::from)?;
    let gs: GoldSet = serde_json::from_slice(&bytes).map_err(|e| EvalError::GoldSetParse {
        path: path.to_path_buf(),
        source: e,
    })?;
    validate(&gs)?;
    Ok(gs)
}

fn validate(gs: &GoldSet) -> Result<(), EvalError> {
    if gs.version == 0 {
        return Err(EvalError::GoldSetInvalid("version must be >= 1".into()));
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for entry in &gs.entries {
        if entry.id.is_empty() {
            return Err(EvalError::GoldSetInvalid(
                "entry with empty id is not allowed".into(),
            ));
        }
        if !seen.insert(entry.id.as_str()) {
            return Err(EvalError::GoldSetInvalid(format!(
                "duplicate entry id '{}'",
                entry.id
            )));
        }
        if entry.question.trim().is_empty() {
            return Err(EvalError::GoldSetInvalid(format!(
                "entry '{}' has empty question",
                entry.id
            )));
        }
        if entry.must_contain_cve_ids.is_empty() && entry.must_contain_terms.is_empty() {
            return Err(EvalError::GoldSetInvalid(format!(
                "entry '{}' has no must_contain_cve_ids and no must_contain_terms",
                entry.id
            )));
        }
        for cve in &entry.must_contain_cve_ids {
            if !CVE_ID_RE.is_match(cve) {
                return Err(EvalError::GoldSetInvalid(format!(
                    "entry '{}' must_contain_cve_ids contains malformed id '{}'",
                    entry.id, cve
                )));
            }
        }
    }
    Ok(())
}
```

Add `once_cell = { workspace = true }` to `crates/fastrag-eval/Cargo.toml` `[dependencies]` if not already present (it is — used in other crates; verify with `cargo tree -p fastrag-eval | grep once_cell`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag-eval gold_set::tests`
Expected: all 5 tests pass (`gold_set_round_trips_through_json`, `load_accepts_well_formed_gold_set`, `load_rejects_empty_question`, `load_rejects_duplicate_id`, `load_rejects_malformed_cve_id`, `load_rejects_zero_assertions`).

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-eval/src/gold_set.rs crates/fastrag-eval/Cargo.toml
git commit -m "eval: gold_set::load with validation branches

Rejects empty question, duplicate id, malformed CVE id,
zero assertions. Unit tests cover every branch."
```

---

### Task 5: `score_entry` pure scorer

**Files:**
- Modify: `crates/fastrag-eval/src/gold_set.rs`

- [ ] **Step 1: Write failing tests**

Add to `crates/fastrag-eval/src/gold_set.rs`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct EntryScore {
    pub hit_at_1: bool,
    pub hit_at_5: bool,
    pub hit_at_10: bool,
    pub reciprocal_rank: f64,
    pub missing_cve_ids: Vec<String>,
    pub missing_terms: Vec<String>,
}
```

Then add to the test module:

```rust
fn entry(cve: &[&str], terms: &[&str]) -> GoldSetEntry {
    GoldSetEntry {
        id: "q001".into(),
        question: "x?".into(),
        must_contain_cve_ids: cve.iter().map(|s| s.to_string()).collect(),
        must_contain_terms: terms.iter().map(|s| s.to_string()).collect(),
        notes: None,
    }
}

#[test]
fn score_entry_hit_at_1_when_first_chunk_satisfies() {
    let e = entry(&["CVE-2024-1"], &["libfoo"]);
    let chunks = vec![
        "advisory for libfoo mentions CVE-2024-1",
        "unrelated",
    ];
    let s = score_entry(&e, &chunks);
    assert!(s.hit_at_1);
    assert!(s.hit_at_5);
    assert!(s.hit_at_10);
    assert_eq!(s.reciprocal_rank, 1.0);
    assert!(s.missing_cve_ids.is_empty());
    assert!(s.missing_terms.is_empty());
}

#[test]
fn score_entry_union_hit_at_3_across_chunks() {
    let e = entry(&["CVE-2024-1", "CVE-2024-2"], &[]);
    let chunks = vec![
        "mentions CVE-2024-1 only",
        "nothing here",
        "CVE-2024-2 found here",
    ];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert!(s.hit_at_5);
    assert_eq!(s.reciprocal_rank, 1.0 / 3.0);
}

#[test]
fn score_entry_case_insensitive_term_match() {
    let e = entry(&[], &["SSRF"]);
    let chunks = vec!["the server was vulnerable to ssrf attacks"];
    let s = score_entry(&e, &chunks);
    assert!(s.hit_at_1);
}

#[test]
fn score_entry_miss_when_nothing_satisfies() {
    let e = entry(&["CVE-2024-99999"], &[]);
    let chunks = vec!["irrelevant content", "also irrelevant"];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert!(!s.hit_at_5);
    assert!(!s.hit_at_10);
    assert_eq!(s.reciprocal_rank, 0.0);
    assert_eq!(s.missing_cve_ids, vec!["CVE-2024-99999".to_string()]);
}

#[test]
fn score_entry_is_pure() {
    let e = entry(&["CVE-2024-1"], &["libfoo"]);
    let chunks = vec!["CVE-2024-1 in libfoo"];
    let s1 = score_entry(&e, &chunks);
    let s2 = score_entry(&e, &chunks);
    assert_eq!(s1, s2);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p fastrag-eval gold_set::tests::score`
Expected: FAIL — `score_entry` not found.

- [ ] **Step 3: Implement `score_entry`**

Add to `crates/fastrag-eval/src/gold_set.rs` (above the `#[cfg(test)]` module):

```rust
static CVE_FIND_RE: once_cell::sync::Lazy<regex::Regex> =
    once_cell::sync::Lazy::new(|| regex::Regex::new(r"(?i)CVE-\d{4}-\d+").unwrap());

pub fn score_entry(entry: &GoldSetEntry, top_k_chunks: &[&str]) -> EntryScore {
    let mut hit_at_1 = false;
    let mut hit_at_5 = false;
    let mut hit_at_10 = false;
    let mut reciprocal_rank = 0.0;

    let mut final_missing_cve_ids: Vec<String> = entry.must_contain_cve_ids.clone();
    let mut final_missing_terms: Vec<String> = entry.must_contain_terms.clone();

    for k in 1..=top_k_chunks.len().min(10) {
        let buffer: String = top_k_chunks[..k].join("\n\n");
        let buffer_lower = buffer.to_lowercase();

        let found_cves: HashSet<String> = CVE_FIND_RE
            .find_iter(&buffer)
            .map(|m| m.as_str().to_uppercase())
            .collect();

        let missing_cves: Vec<String> = entry
            .must_contain_cve_ids
            .iter()
            .filter(|c| !found_cves.contains(&c.to_uppercase()))
            .cloned()
            .collect();

        let missing_terms: Vec<String> = entry
            .must_contain_terms
            .iter()
            .filter(|t| !buffer_lower.contains(&t.to_lowercase()))
            .cloned()
            .collect();

        let satisfied = missing_cves.is_empty() && missing_terms.is_empty();

        if satisfied && reciprocal_rank == 0.0 {
            reciprocal_rank = 1.0 / (k as f64);
        }

        if k == 1 && satisfied {
            hit_at_1 = true;
        }
        if k <= 5 && satisfied {
            hit_at_5 = true;
        }
        if k <= 10 && satisfied {
            hit_at_10 = true;
        }

        // Capture the final k's missing lists — the one used for hit@10
        if k == top_k_chunks.len().min(10) {
            final_missing_cve_ids = missing_cves;
            final_missing_terms = missing_terms;
        }
    }

    EntryScore {
        hit_at_1,
        hit_at_5,
        hit_at_10,
        reciprocal_rank,
        missing_cve_ids: final_missing_cve_ids,
        missing_terms: final_missing_terms,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag-eval gold_set`
Expected: all gold_set tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-eval/src/gold_set.rs
git commit -m "eval: score_entry union-of-top-k scorer

Pure function: takes an entry and ranked chunks, returns
hit@{1,5,10} + reciprocal rank + missing assertion lists.
CVE IDs via regex, terms via case-insensitive substring."
```

---

### Task 6: Starter gold-set fixture (10 entries)

**Files:**
- Create: `tests/gold/questions.json`

- [ ] **Step 1: Write the starter fixture**

Create `tests/gold/questions.json` with 10 hand-curated entries. This is a real fixture — the questions must reference content that will exist in the corpus docs created in Task 7.

```json
{
  "version": 1,
  "entries": [
    {
      "id": "q001",
      "question": "Is there a known RCE in libfoo?",
      "must_contain_cve_ids": ["CVE-2024-12345"],
      "must_contain_terms": ["libfoo", "remote code execution"],
      "notes": "Pronoun-resolution canary — chunk body says 'the vulnerability'"
    },
    {
      "id": "q002",
      "question": "What CVE covers the BlueKeep RDP flaw?",
      "must_contain_cve_ids": ["CVE-2019-0708"],
      "must_contain_terms": ["BlueKeep", "RDP"],
      "notes": null
    },
    {
      "id": "q003",
      "question": "How can SSRF be exploited against the proxy middleware?",
      "must_contain_cve_ids": [],
      "must_contain_terms": ["SSRF", "proxy", "mitigation"],
      "notes": "Concept lookup"
    },
    {
      "id": "q004",
      "question": "What weakness does CWE-502 describe?",
      "must_contain_cve_ids": [],
      "must_contain_terms": ["deserialization", "untrusted data"],
      "notes": "CWE concept lookup"
    },
    {
      "id": "q005",
      "question": "Which advisory describes the libfoo buffer overflow?",
      "must_contain_cve_ids": ["CVE-2024-12345"],
      "must_contain_terms": ["buffer overflow"],
      "notes": null
    },
    {
      "id": "q006",
      "question": "What year was BlueKeep disclosed?",
      "must_contain_cve_ids": ["CVE-2019-0708"],
      "must_contain_terms": ["2019"],
      "notes": null
    },
    {
      "id": "q007",
      "question": "Is SSRF mitigated by allowlist?",
      "must_contain_cve_ids": [],
      "must_contain_terms": ["SSRF", "allowlist"],
      "notes": null
    },
    {
      "id": "q008",
      "question": "How does CWE-502 relate to gadget chains?",
      "must_contain_cve_ids": [],
      "must_contain_terms": ["CWE-502", "gadget"],
      "notes": null
    },
    {
      "id": "q009",
      "question": "What is the impact of the libfoo vulnerability?",
      "must_contain_cve_ids": ["CVE-2024-12345"],
      "must_contain_terms": ["impact"],
      "notes": "Requires context prefix — chunk body never says 'libfoo'"
    },
    {
      "id": "q010",
      "question": "Which CVE affects Microsoft Remote Desktop Services pre-authentication?",
      "must_contain_cve_ids": ["CVE-2019-0708"],
      "must_contain_terms": ["pre-authentication"],
      "notes": null
    }
  ]
}
```

- [ ] **Step 2: Verify the fixture validates against the loader**

Run: `cargo test -p fastrag-eval --test gold_set_starter_valid` — but this test doesn't exist yet. Instead, do an ad-hoc check:

```bash
cargo run -p fastrag-eval --example validate_gold_set tests/gold/questions.json 2>/dev/null || true
```

The example doesn't exist either. Use this inline canary: create a throwaway test in `crates/fastrag-eval/src/gold_set.rs` (inside `#[cfg(test)]`) that loads the workspace fixture:

```rust
#[test]
fn tests_gold_questions_json_is_valid() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("tests/gold/questions.json");
    let gs = load(&path).expect("tests/gold/questions.json must validate");
    assert!(
        gs.entries.len() >= 10,
        "starter gold set must have at least 10 entries, found {}",
        gs.entries.len()
    );
}
```

This test also serves as the **push-CI canary** — leave it in the codebase. Task 9 formalizes that role.

Run: `cargo test -p fastrag-eval gold_set::tests::tests_gold_questions_json_is_valid`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/gold/questions.json crates/fastrag-eval/src/gold_set.rs
git commit -m "eval: starter gold set fixture (10 entries) + push-CI canary

Hand-curated gold set covering CVE lookup, concept lookup,
and pronoun-resolution cases. Canary test loads the fixture
on every push so malformed commits fail pre-weekly-run."
```

---

### Task 7: Starter corpus fixture (5 docs)

**Files:**
- Create: `tests/gold/corpus/01-libfoo-rce.md`
- Create: `tests/gold/corpus/02-kev-bluekeep.md`
- Create: `tests/gold/corpus/03-ssrf-proxy.md`
- Create: `tests/gold/corpus/04-cwe-502-deserialize.md`
- Create: `tests/gold/corpus/05-pronoun-resolution-sample.md`

- [ ] **Step 1: Write `01-libfoo-rce.md`**

```markdown
---
title: "libfoo 2.3.1 — CVE-2024-12345 Remote Code Execution"
---

# libfoo Advisory CVE-2024-12345

A critical buffer overflow in libfoo 2.3.1 allows remote code execution
via a crafted config file. The vulnerability is triggered during startup
when the library parses the user-supplied TOML header.

## Impact

Full remote code execution as the user running the process. The
vulnerability has been observed exploited in the wild against
internet-exposed daemons.

## Mitigation

Upgrade to libfoo 2.3.2 or later. No workaround is available for the
affected versions.
```

- [ ] **Step 2: Write `02-kev-bluekeep.md`**

```markdown
---
title: "CISA KEV — BlueKeep (CVE-2019-0708)"
---

# BlueKeep — Microsoft Remote Desktop Services RCE

CVE-2019-0708, disclosed in 2019, is a pre-authentication remote code
execution vulnerability in Microsoft Remote Desktop Services. The flaw
affects Windows 7, Windows Server 2008, and earlier versions.

BlueKeep is notable as a wormable vulnerability — no user interaction
is required for exploitation across an RDP-exposed network segment.

## Status

Listed in the CISA Known Exploited Vulnerabilities (KEV) catalog.
Patched by Microsoft in May 2019.
```

- [ ] **Step 3: Write `03-ssrf-proxy.md`**

```markdown
---
title: "SSRF in Proxy Middleware — Exploitation and Mitigation"
---

# Server-Side Request Forgery via Proxy Middleware

SSRF allows an attacker to coerce a server into making arbitrary outbound
HTTP requests on their behalf. In proxy middleware, the typical attack
path is a URL parameter that is forwarded without validation to the
internal network.

## Exploitation

An attacker submits a request with a URL pointing at internal metadata
endpoints (e.g. cloud instance metadata services) or internal admin
interfaces not exposed to the public internet.

## Mitigation

Enforce a strict allowlist of outbound hosts. Never forward user-supplied
URLs without validation. Block private IP ranges and link-local addresses
at the egress layer.
```

- [ ] **Step 4: Write `04-cwe-502-deserialize.md`**

```markdown
---
title: "CWE-502 — Deserialization of Untrusted Data"
---

# CWE-502: Deserialization of Untrusted Data

CWE-502 describes the weakness of deserializing untrusted data without
validating that the incoming object is safe. Exploitation typically
uses gadget chains — sequences of classes whose constructors or
setters produce side effects when instantiated.

## Relationship to Gadget Chains

A gadget chain is a sequence of serializable types that, when
deserialized in order, trigger arbitrary code execution. Java, .NET,
and Python pickle are historically affected.

## Mitigation

Avoid deserializing untrusted data entirely. If deserialization is
unavoidable, restrict the allowed type set via an allowlist.
```

- [ ] **Step 5: Write `05-pronoun-resolution-sample.md`**

This doc deliberately lacks lexical anchors in the body — the title carries the context. It exercises the contextual retrieval advantage.

```markdown
---
title: "libfoo Security Advisory — Impact Assessment"
---

# Impact Assessment

The vulnerability described in the related advisory affects all
deployments running the affected version. Exploitation leads to full
system compromise. The impact is rated critical by internal security
review.

Customers should prioritize patching within the standard emergency
response window.
```

- [ ] **Step 6: Verify chunking will produce sensible chunks**

Run: `cargo run -- index tests/gold/corpus --corpus /tmp/gold-raw --embedder mock`
Expected: exits 0, prints "Indexed N chunks" where N ≥ 5.

Clean up: `rm -rf /tmp/gold-raw`

- [ ] **Step 7: Commit**

```bash
git add tests/gold/corpus/
git commit -m "eval: starter corpus fixture (5 security docs)

Covers libfoo RCE, BlueKeep, SSRF mitigation, CWE-502
deserialization, and a deliberate pronoun-resolution case
where the body has no lexical anchor without a context prefix."
```

---

### Task 8: `union_match.rs` integration test

**Files:**
- Create: `crates/fastrag-eval/tests/union_match.rs`

- [ ] **Step 1: Write the integration test**

```rust
//! Integration test: gold_set::score_entry over synthetic chunk shapes.
//!
//! Exercises the union-of-top-k semantics: multi-chunk unions, case-insensitive
//! term matches, pronoun-resolution miss, and the honest-miss scoring path.

use fastrag_eval::gold_set::{score_entry, GoldSetEntry};

fn entry(cve: &[&str], terms: &[&str]) -> GoldSetEntry {
    GoldSetEntry {
        id: "qtest".into(),
        question: "x?".into(),
        must_contain_cve_ids: cve.iter().map(|s| s.to_string()).collect(),
        must_contain_terms: terms.iter().map(|s| s.to_string()).collect(),
        notes: None,
    }
}

#[test]
fn two_chunk_union_satisfies_at_k_2() {
    let e = entry(&["CVE-2024-1", "CVE-2024-2"], &[]);
    let chunks = vec!["only CVE-2024-1 here", "only CVE-2024-2 here"];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert!(s.hit_at_5);
    assert_eq!(s.reciprocal_rank, 0.5);
}

#[test]
fn pronoun_resolution_miss_when_title_not_in_chunks() {
    let e = entry(&["CVE-2024-12345"], &["libfoo"]);
    let chunks = vec![
        "the vulnerability affects all deployments",
        "impact is rated critical",
    ];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_10);
    assert_eq!(s.reciprocal_rank, 0.0);
    assert!(s.missing_cve_ids.contains(&"CVE-2024-12345".to_string()));
    assert!(s.missing_terms.contains(&"libfoo".to_string()));
}

#[test]
fn case_insensitive_cve_matching() {
    let e = entry(&["CVE-2024-12345"], &[]);
    let chunks = vec!["see cve-2024-12345 for details"];
    let s = score_entry(&e, &chunks);
    assert!(s.hit_at_1);
}

#[test]
fn empty_top_k_is_a_miss_not_a_crash() {
    let e = entry(&["CVE-2024-1"], &[]);
    let chunks: Vec<&str> = vec![];
    let s = score_entry(&e, &chunks);
    assert!(!s.hit_at_1);
    assert_eq!(s.reciprocal_rank, 0.0);
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p fastrag-eval --test union_match`
Expected: all 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-eval/tests/union_match.rs
git commit -m "eval: union_match integration test for score_entry"
```

---

### Task 9: `gold_set_loader.rs` integration test

**Files:**
- Create: `crates/fastrag-eval/tests/fixtures/gold_valid.json`
- Create: `crates/fastrag-eval/tests/fixtures/gold_invalid_empty_q.json`
- Create: `crates/fastrag-eval/tests/fixtures/gold_invalid_dup_id.json`
- Create: `crates/fastrag-eval/tests/fixtures/gold_invalid_malformed_cve.json`
- Create: `crates/fastrag-eval/tests/fixtures/gold_invalid_zero_assertions.json`
- Create: `crates/fastrag-eval/tests/gold_set_loader.rs`

- [ ] **Step 1: Write the fixtures**

`crates/fastrag-eval/tests/fixtures/gold_valid.json`:

```json
{
  "version": 1,
  "entries": [
    {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []},
    {"id": "q002", "question": "y?", "must_contain_cve_ids": [], "must_contain_terms": ["ssrf"]}
  ]
}
```

`crates/fastrag-eval/tests/fixtures/gold_invalid_empty_q.json`:

```json
{
  "version": 1,
  "entries": [
    {"id": "q_bad", "question": "", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
  ]
}
```

`crates/fastrag-eval/tests/fixtures/gold_invalid_dup_id.json`:

```json
{
  "version": 1,
  "entries": [
    {"id": "q001", "question": "a?", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []},
    {"id": "q001", "question": "b?", "must_contain_cve_ids": ["CVE-2024-2"], "must_contain_terms": []}
  ]
}
```

`crates/fastrag-eval/tests/fixtures/gold_invalid_malformed_cve.json`:

```json
{
  "version": 1,
  "entries": [
    {"id": "q001", "question": "x?", "must_contain_cve_ids": ["CVE-24-1"], "must_contain_terms": []}
  ]
}
```

`crates/fastrag-eval/tests/fixtures/gold_invalid_zero_assertions.json`:

```json
{
  "version": 1,
  "entries": [
    {"id": "q001", "question": "x?", "must_contain_cve_ids": [], "must_contain_terms": []}
  ]
}
```

- [ ] **Step 2: Write the integration test**

`crates/fastrag-eval/tests/gold_set_loader.rs`:

```rust
//! Integration test: gold_set::load validation branches against on-disk fixtures.

use std::path::PathBuf;

use fastrag_eval::gold_set::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn valid_fixture_loads() {
    let gs = load(&fixture("gold_valid.json")).expect("valid fixture should load");
    assert_eq!(gs.entries.len(), 2);
}

#[test]
fn empty_question_rejected() {
    let err = load(&fixture("gold_invalid_empty_q.json")).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("q_bad"), "{msg}");
    assert!(msg.contains("empty question"), "{msg}");
}

#[test]
fn duplicate_id_rejected() {
    let err = load(&fixture("gold_invalid_dup_id.json")).unwrap_err();
    assert!(format!("{err}").contains("duplicate"));
}

#[test]
fn malformed_cve_rejected() {
    let err = load(&fixture("gold_invalid_malformed_cve.json")).unwrap_err();
    assert!(format!("{err}").contains("CVE-24-1"));
}

#[test]
fn zero_assertions_rejected() {
    let err = load(&fixture("gold_invalid_zero_assertions.json")).unwrap_err();
    assert!(format!("{err}").contains("no must_contain"));
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p fastrag-eval --test gold_set_loader`
Expected: all 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag-eval/tests/fixtures/ crates/fastrag-eval/tests/gold_set_loader.rs
git commit -m "eval: gold_set_loader integration test with on-disk fixtures

Five JSON fixtures, one per validation branch. Catches
serde drift and asserts error messages name the offending id."
```

Landing 1 is complete. Run the full lint gate before moving on:

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings
cargo fmt --check
```

Push and watch CI.

---

## Landing 2 — `LatencyBreakdown` + instrumentation

### Task 10: `LatencyBreakdown` struct + unit tests

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `crates/fastrag/src/corpus/mod.rs`:

```rust
#[test]
fn latency_breakdown_default_is_zero() {
    let b = LatencyBreakdown::default();
    assert_eq!(b.embed_us, 0);
    assert_eq!(b.bm25_us, 0);
    assert_eq!(b.hnsw_us, 0);
    assert_eq!(b.rerank_us, 0);
    assert_eq!(b.fuse_us, 0);
    assert_eq!(b.total_us, 0);
}

#[test]
fn latency_breakdown_total_is_sum_of_stages() {
    let mut b = LatencyBreakdown::default();
    b.embed_us = 100;
    b.bm25_us = 200;
    b.hnsw_us = 300;
    b.rerank_us = 400;
    b.fuse_us = 50;
    b.finalize();
    assert_eq!(b.total_us, 1050);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fastrag corpus::tests::latency_breakdown`
Expected: FAIL — `LatencyBreakdown` not found.

- [ ] **Step 3: Implement the struct**

Add near the top of `crates/fastrag/src/corpus/mod.rs`, after the existing imports:

```rust
/// Per-stage query latency in microseconds.
///
/// Passed `&mut` into every `query_corpus_*` variant. Callers that don't
/// care pass `&mut LatencyBreakdown::default()` and ignore the result.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LatencyBreakdown {
    pub embed_us: u64,
    pub bm25_us: u64,
    pub hnsw_us: u64,
    pub rerank_us: u64,
    pub fuse_us: u64,
    pub total_us: u64,
}

impl LatencyBreakdown {
    /// Sum per-stage microseconds into `total_us`. Call once after a query completes.
    pub fn finalize(&mut self) {
        self.total_us = self
            .embed_us
            .saturating_add(self.bm25_us)
            .saturating_add(self.hnsw_us)
            .saturating_add(self.rerank_us)
            .saturating_add(self.fuse_us);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p fastrag corpus::tests::latency_breakdown`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs
git commit -m "corpus: LatencyBreakdown struct for per-stage query timing"
```

---

### Task 11: Thread `LatencyBreakdown` through `query_corpus_hybrid_reranked`

This is the primary variant — the matrix runner's Primary configuration calls it. Do it first to prove the pattern.

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (around line 596, function `query_corpus_hybrid_reranked`)
- Modify: `fastrag-cli/src/main.rs` (line ~360, one call site)

- [ ] **Step 1: Read the existing function**

Run: `cargo check -p fastrag` first to make sure the baseline builds. Read lines 596–660 of `crates/fastrag/src/corpus/mod.rs` (approximate — the function is `query_corpus_hybrid_reranked`).

- [ ] **Step 2: Add `breakdown` parameter + per-stage timing**

Edit the function signature and body. New signature:

```rust
pub fn query_corpus_hybrid_reranked(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    reranker: &dyn fastrag_rerank::Reranker,
    over_fetch: usize,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
```

Inside the body, wrap each stage with `Instant::now()`:

```rust
use std::time::Instant;

// Embed stage
let t0 = Instant::now();
let query_vec = embedder.embed_query_dyn(query)?;
breakdown.embed_us = t0.elapsed().as_micros() as u64;

// BM25 stage — pull from the existing hybrid path
let t1 = Instant::now();
let bm25_scored = tantivy.bm25_search(query, over_fetch)?;
breakdown.bm25_us = t1.elapsed().as_micros() as u64;

// HNSW stage
let t2 = Instant::now();
let dense_hits = index.query(&query_vec, over_fetch)?;
breakdown.hnsw_us = t2.elapsed().as_micros() as u64;

// Fuse stage (RRF)
let t3 = Instant::now();
let fused = fastrag::corpus::hybrid::rrf_fuse(&bm25_scored, &dense_hits, 60);
breakdown.fuse_us = t3.elapsed().as_micros() as u64;

// Rerank stage
let t4 = Instant::now();
let reranked = reranker.rerank(query, fused).map_err(CorpusError::from)?;
breakdown.rerank_us = t4.elapsed().as_micros() as u64;

breakdown.finalize();
Ok(reranked.into_iter().take(top_k).collect())
```

Note: the actual current implementation of `query_corpus_hybrid_reranked` may differ — reuse its existing logic, just wrap each stage with `Instant::now()` and the `breakdown.*_us` assignments. Do not rewrite control flow.

- [ ] **Step 3: Update the CLI call site**

Edit `fastrag-cli/src/main.rs` around line 360. Find the `query_corpus_hybrid_reranked` call and add `&mut LatencyBreakdown::default()` as the final argument:

```rust
let hits = fastrag::corpus::query_corpus_hybrid_reranked(
    corpus_dir,
    &args.query,
    args.top_k,
    embedder.as_ref(),
    reranker.as_ref(),
    args.top_k * 3,
    &mut fastrag::corpus::LatencyBreakdown::default(),
)?;
```

- [ ] **Step 4: Compile**

Run: `cargo check --workspace --features retrieval,rerank,hybrid`
Expected: zero errors. If other callers of `query_corpus_hybrid_reranked` exist elsewhere, they will surface as compile errors — update each with `&mut LatencyBreakdown::default()`.

- [ ] **Step 5: Run retrieval tests**

Run: `cargo test --workspace --features retrieval,rerank,hybrid`
Expected: existing retrieval tests pass — no behavior change, just an extra ignored parameter in callers.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs fastrag-cli/src/main.rs
git commit -m "corpus: thread LatencyBreakdown through query_corpus_hybrid_reranked

Per-stage Instant::now() timing for embed/BM25/HNSW/fuse/rerank.
CLI call site updated to pass default breakdown."
```

---

### Task 12: Thread `LatencyBreakdown` through `query_corpus_hybrid`

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (around line 558)
- Modify: `fastrag-cli/src/main.rs` (line ~393)

- [ ] **Step 1: Follow the same pattern as Task 11**

Add `breakdown: &mut LatencyBreakdown` as the final parameter. Wrap embed / BM25 / HNSW / fuse with `Instant::now()` (no rerank in this variant — leave `breakdown.rerank_us = 0`). Call `breakdown.finalize()` before return.

- [ ] **Step 2: Update the CLI call site**

Edit `fastrag-cli/src/main.rs` line ~393. Add `&mut fastrag::corpus::LatencyBreakdown::default()` as the final argument.

- [ ] **Step 3: Compile + test**

Run: `cargo test --workspace --features retrieval,rerank,hybrid`
Expected: zero errors, all tests green.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs fastrag-cli/src/main.rs
git commit -m "corpus: thread LatencyBreakdown through query_corpus_hybrid"
```

---

### Task 13: Thread `LatencyBreakdown` through `query_corpus_reranked`

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (around line 538)
- Modify: `fastrag-cli/src/main.rs` (line ~375)

- [ ] **Step 1: Same pattern**

Add `breakdown: &mut LatencyBreakdown` parameter. Wrap embed / HNSW / rerank with `Instant::now()`. Leave `breakdown.bm25_us = 0` and `breakdown.fuse_us = 0` (dense-only path — no BM25, no fusion). Call `breakdown.finalize()`.

- [ ] **Step 2: Update the CLI call site**

Line ~375 in `fastrag-cli/src/main.rs`.

- [ ] **Step 3: Compile + test**

Run: `cargo test --workspace --features retrieval,rerank`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs fastrag-cli/src/main.rs
git commit -m "corpus: thread LatencyBreakdown through query_corpus_reranked"
```

---

### Task 14: Thread `LatencyBreakdown` through `query_corpus` + `query_corpus_with_filter`

The last two variants. These are simpler — dense-only, no rerank, no BM25.

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (lines ~437 and ~458)
- Modify: `fastrag-cli/src/main.rs` (line ~406)
- Modify: `crates/fastrag-mcp/src/lib.rs` (line ~303)

- [ ] **Step 1: Add the parameter to both functions**

Same pattern: `breakdown: &mut LatencyBreakdown` final parameter. Time embed + HNSW stages.

- [ ] **Step 2: Update all call sites**

- `fastrag-cli/src/main.rs:406` — CLI `query_corpus_with_filter` call
- `crates/fastrag-mcp/src/lib.rs:303` — MCP `query_corpus_with_filter` call

Each gets `&mut fastrag::corpus::LatencyBreakdown::default()` as the final argument.

- [ ] **Step 3: Compile + test**

Run: `cargo test --workspace --features retrieval,rerank,hybrid,contextual`
Expected: green across the board.

Run: `cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings`
Expected: zero warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs fastrag-cli/src/main.rs crates/fastrag-mcp/src/lib.rs
git commit -m "corpus: thread LatencyBreakdown through remaining query_corpus variants

query_corpus and query_corpus_with_filter gain the breakdown
param. MCP search_corpus call site updated to pass default."
```

Push Landing 2. Run ci-watcher as a background Haiku Agent per repo convention.

---

## Landing 3 — `matrix.rs` orchestrator

### Task 15: `ConfigVariant` enum + `MatrixReport` types

**Files:**
- Create: `crates/fastrag-eval/src/matrix.rs`
- Modify: `crates/fastrag-eval/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/fastrag-eval/src/matrix.rs`:

```rust
//! Eval config matrix: 4 retrieval variants + per-variant reports.

use serde::{Deserialize, Serialize};

use crate::gold_set::GoldSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigVariant {
    Primary,
    NoRerank,
    NoContextual,
    DenseOnly,
}

impl ConfigVariant {
    pub fn all() -> [ConfigVariant; 4] {
        [
            ConfigVariant::Primary,
            ConfigVariant::NoRerank,
            ConfigVariant::NoContextual,
            ConfigVariant::DenseOnly,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfigVariant::Primary => "primary",
            ConfigVariant::NoRerank => "no-rerank",
            ConfigVariant::NoContextual => "no-contextual",
            ConfigVariant::DenseOnly => "dense-only",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub total: Percentiles,
    pub embed: Percentiles,
    pub bm25: Percentiles,
    pub hnsw: Percentiles,
    pub rerank: Percentiles,
    pub fuse: Percentiles,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub id: String,
    pub hit_at_1: bool,
    pub hit_at_5: bool,
    pub hit_at_10: bool,
    pub reciprocal_rank: f64,
    pub missing_cve_ids: Vec<String>,
    pub missing_terms: Vec<String>,
    pub latency_us: fastrag::corpus::LatencyBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantReport {
    pub variant: ConfigVariant,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub hit_at_10: f64,
    pub mrr_at_10: f64,
    pub latency: LatencyPercentiles,
    pub per_question: Vec<QuestionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixReport {
    pub schema_version: u32,
    pub git_rev: String,
    pub captured_at: String,
    pub runs: Vec<VariantReport>,
    pub rerank_delta: f64,
    pub contextual_delta: f64,
    pub hybrid_delta: f64,
}

impl MatrixReport {
    pub fn hit5(&self, variant: ConfigVariant) -> Option<f64> {
        self.runs
            .iter()
            .find(|r| r.variant == variant)
            .map(|r| r.hit_at_5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_variant_all_returns_four_in_order() {
        let all = ConfigVariant::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], ConfigVariant::Primary);
        assert_eq!(all[3], ConfigVariant::DenseOnly);
    }

    #[test]
    fn config_variant_labels_are_stable() {
        assert_eq!(ConfigVariant::Primary.label(), "primary");
        assert_eq!(ConfigVariant::NoRerank.label(), "no-rerank");
        assert_eq!(ConfigVariant::NoContextual.label(), "no-contextual");
        assert_eq!(ConfigVariant::DenseOnly.label(), "dense-only");
    }
}
```

Add to `crates/fastrag-eval/src/lib.rs`:

```rust
pub mod matrix;
```

Add `fastrag = { workspace = true }` to `crates/fastrag-eval/Cargo.toml` `[dependencies]` (needed for `LatencyBreakdown`).

- [ ] **Step 2: Run tests**

Run: `cargo test -p fastrag-eval matrix::tests`
Expected: both pass. If `fastrag` dep is missing, add it and re-run.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-eval/src/matrix.rs crates/fastrag-eval/src/lib.rs crates/fastrag-eval/Cargo.toml
git commit -m "eval: ConfigVariant + MatrixReport types"
```

---

### Task 16: `matrix::run_matrix` orchestrator + stub-driven test

**Files:**
- Modify: `crates/fastrag-eval/src/matrix.rs`
- Modify: `crates/fastrag-eval/src/error.rs`
- Create: `crates/fastrag-eval/tests/matrix_stub.rs`

- [ ] **Step 1: Extend `EvalError` for matrix variants**

Add to `crates/fastrag-eval/src/error.rs`:

```rust
#[error("matrix variant {variant:?} failed: {source}")]
MatrixVariant {
    variant: crate::matrix::ConfigVariant,
    #[source]
    source: Box<EvalError>,
},
#[error("--config-matrix requires --gold-set")]
MatrixRequiresGoldSet,
#[error("--config-matrix requires --corpus-no-contextual")]
MatrixMissingRawCorpus,
```

- [ ] **Step 2: Define a testable `CorpusDriver` trait**

The real matrix runner calls `query_corpus_*` variants against a real corpus. For testing, we define a trait that the stub can implement.

Add to `crates/fastrag-eval/src/matrix.rs`:

```rust
use fastrag::corpus::LatencyBreakdown;

use crate::error::EvalError;
use crate::gold_set::{score_entry, GoldSetEntry};

/// Pluggable retrieval driver. Real impl wraps `query_corpus_*`; test impl
/// returns canned top-k strings.
pub trait CorpusDriver {
    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
        top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError>;
}

pub fn run_matrix<D: CorpusDriver>(
    driver: &D,
    gold_set: &GoldSet,
    top_k: usize,
) -> Result<MatrixReport, EvalError> {
    use hdrhistogram::Histogram;

    let mut runs: Vec<VariantReport> = Vec::with_capacity(4);

    for variant in ConfigVariant::all() {
        let mut total_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut embed_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut bm25_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut hnsw_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut rerank_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();
        let mut fuse_h = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).unwrap();

        let mut per_question: Vec<QuestionResult> = Vec::with_capacity(gold_set.entries.len());

        for entry in &gold_set.entries {
            let mut breakdown = LatencyBreakdown::default();
            let chunks = driver
                .query(variant, &entry.question, top_k, &mut breakdown)
                .map_err(|e| EvalError::MatrixVariant {
                    variant,
                    source: Box::new(e),
                })?;
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let score = score_entry(entry, &chunk_refs);

            total_h.record(breakdown.total_us.max(1)).ok();
            embed_h.record(breakdown.embed_us.max(1)).ok();
            bm25_h.record(breakdown.bm25_us.max(1)).ok();
            hnsw_h.record(breakdown.hnsw_us.max(1)).ok();
            rerank_h.record(breakdown.rerank_us.max(1)).ok();
            fuse_h.record(breakdown.fuse_us.max(1)).ok();

            per_question.push(QuestionResult {
                id: entry.id.clone(),
                hit_at_1: score.hit_at_1,
                hit_at_5: score.hit_at_5,
                hit_at_10: score.hit_at_10,
                reciprocal_rank: score.reciprocal_rank,
                missing_cve_ids: score.missing_cve_ids,
                missing_terms: score.missing_terms,
                latency_us: breakdown,
            });
        }

        let n = per_question.len() as f64;
        let hit1 = per_question.iter().filter(|q| q.hit_at_1).count() as f64 / n;
        let hit5 = per_question.iter().filter(|q| q.hit_at_5).count() as f64 / n;
        let hit10 = per_question.iter().filter(|q| q.hit_at_10).count() as f64 / n;
        let mrr = per_question.iter().map(|q| q.reciprocal_rank).sum::<f64>() / n;

        fn percentiles(h: &hdrhistogram::Histogram<u64>) -> Percentiles {
            Percentiles {
                p50_us: h.value_at_quantile(0.5),
                p95_us: h.value_at_quantile(0.95),
                p99_us: h.value_at_quantile(0.99),
            }
        }

        runs.push(VariantReport {
            variant,
            hit_at_1: hit1,
            hit_at_5: hit5,
            hit_at_10: hit10,
            mrr_at_10: mrr,
            latency: LatencyPercentiles {
                total: percentiles(&total_h),
                embed: percentiles(&embed_h),
                bm25: percentiles(&bm25_h),
                hnsw: percentiles(&hnsw_h),
                rerank: percentiles(&rerank_h),
                fuse: percentiles(&fuse_h),
            },
            per_question,
        });
    }

    let hit5_primary = runs.iter().find(|r| r.variant == ConfigVariant::Primary).map(|r| r.hit_at_5).unwrap_or(0.0);
    let hit5_no_rerank = runs.iter().find(|r| r.variant == ConfigVariant::NoRerank).map(|r| r.hit_at_5).unwrap_or(0.0);
    let hit5_no_ctx = runs.iter().find(|r| r.variant == ConfigVariant::NoContextual).map(|r| r.hit_at_5).unwrap_or(0.0);
    let hit5_dense = runs.iter().find(|r| r.variant == ConfigVariant::DenseOnly).map(|r| r.hit_at_5).unwrap_or(0.0);

    Ok(MatrixReport {
        schema_version: 1,
        git_rev: git_rev().unwrap_or_else(|| "unknown".into()),
        captured_at: chrono::Utc::now().to_rfc3339(),
        runs,
        rerank_delta: hit5_primary - hit5_no_rerank,
        contextual_delta: hit5_primary - hit5_no_ctx,
        hybrid_delta: hit5_primary - hit5_dense,
    })
}

fn git_rev() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}
```

Add `chrono = { workspace = true, features = ["serde"] }` to `crates/fastrag-eval/Cargo.toml` (check workspace root first — add there if missing, use `chrono = "0.4"`).

- [ ] **Step 3: Write the integration test**

Create `crates/fastrag-eval/tests/matrix_stub.rs`:

```rust
//! Integration test: run_matrix against a stub CorpusDriver.

use fastrag::corpus::LatencyBreakdown;
use fastrag_eval::error::EvalError;
use fastrag_eval::gold_set::{GoldSet, GoldSetEntry};
use fastrag_eval::matrix::{run_matrix, ConfigVariant, CorpusDriver};

struct StubDriver;

impl CorpusDriver for StubDriver {
    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
        _top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError> {
        // Deterministic fake latency per variant so each histogram gets a record.
        breakdown.embed_us = 1000;
        breakdown.bm25_us = 500;
        breakdown.hnsw_us = 2000;
        breakdown.rerank_us = match variant {
            ConfigVariant::NoRerank => 0,
            _ => 3000,
        };
        breakdown.fuse_us = match variant {
            ConfigVariant::DenseOnly => 0,
            _ => 100,
        };
        breakdown.finalize();

        // Primary always finds the libfoo chunk; DenseOnly never does.
        if variant == ConfigVariant::DenseOnly && question.contains("libfoo") {
            Ok(vec!["unrelated chunk".into()])
        } else if question.contains("libfoo") {
            Ok(vec!["CVE-2024-12345 in libfoo".into()])
        } else {
            Ok(vec!["SSRF proxy allowlist mitigation".into()])
        }
    }
}

fn gold() -> GoldSet {
    GoldSet {
        version: 1,
        entries: vec![
            GoldSetEntry {
                id: "q001".into(),
                question: "Is libfoo vulnerable?".into(),
                must_contain_cve_ids: vec!["CVE-2024-12345".into()],
                must_contain_terms: vec!["libfoo".into()],
                notes: None,
            },
            GoldSetEntry {
                id: "q002".into(),
                question: "How to mitigate SSRF?".into(),
                must_contain_cve_ids: vec![],
                must_contain_terms: vec!["SSRF", "allowlist"].into_iter().map(String::from).collect(),
                notes: None,
            },
        ],
    }
}

#[test]
fn run_matrix_executes_all_four_variants_in_order() {
    let report = run_matrix(&StubDriver, &gold(), 10).unwrap();
    assert_eq!(report.runs.len(), 4);
    assert_eq!(report.runs[0].variant, ConfigVariant::Primary);
    assert_eq!(report.runs[1].variant, ConfigVariant::NoRerank);
    assert_eq!(report.runs[2].variant, ConfigVariant::NoContextual);
    assert_eq!(report.runs[3].variant, ConfigVariant::DenseOnly);
}

#[test]
fn run_matrix_records_every_stage_histogram() {
    let report = run_matrix(&StubDriver, &gold(), 10).unwrap();
    for run in &report.runs {
        // p50 must be > 0 if histogram was populated
        assert!(run.latency.total.p50_us > 0, "variant {:?} has empty total histogram", run.variant);
        assert!(run.latency.embed.p50_us > 0);
        assert!(run.latency.bm25.p50_us > 0);
        assert!(run.latency.hnsw.p50_us > 0);
    }
}

#[test]
fn run_matrix_per_question_count_matches_entries() {
    let report = run_matrix(&StubDriver, &gold(), 10).unwrap();
    for run in &report.runs {
        assert_eq!(run.per_question.len(), 2);
    }
}

#[test]
fn run_matrix_hybrid_delta_positive_when_dense_misses() {
    let report = run_matrix(&StubDriver, &gold(), 10).unwrap();
    // Primary finds libfoo (hit); DenseOnly misses libfoo (miss).
    // hybrid_delta = hit5(Primary) - hit5(DenseOnly) should be > 0.
    assert!(report.hybrid_delta > 0.0, "got delta {}", report.hybrid_delta);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p fastrag-eval --test matrix_stub`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-eval/src/matrix.rs crates/fastrag-eval/src/error.rs \
        crates/fastrag-eval/tests/matrix_stub.rs crates/fastrag-eval/Cargo.toml
git commit -m "eval: run_matrix orchestrator + CorpusDriver trait + stub test

Drives 4 variants sequentially, records per-stage histograms,
computes rerank/contextual/hybrid deltas at write time."
```

---

### Task 17: Real `CorpusDriver` impl over `query_corpus_*`

**Files:**
- Create: `crates/fastrag-eval/src/matrix_real.rs`
- Modify: `crates/fastrag-eval/src/matrix.rs`

- [ ] **Step 1: Define the real driver**

Create `crates/fastrag-eval/src/matrix_real.rs`:

```rust
//! Real CorpusDriver implementation backed by fastrag::corpus query_corpus_* variants.

use std::path::{Path, PathBuf};

use fastrag::corpus::{
    query_corpus_hybrid, query_corpus_hybrid_reranked, query_corpus_reranked, LatencyBreakdown,
};
use fastrag_embed::DynEmbedderTrait;
use fastrag_rerank::Reranker;

use crate::error::EvalError;
use crate::matrix::{ConfigVariant, CorpusDriver};

pub struct RealCorpusDriver<'a> {
    pub ctx_corpus: PathBuf,
    pub raw_corpus: PathBuf,
    pub embedder: &'a dyn DynEmbedderTrait,
    pub reranker: &'a dyn Reranker,
}

impl<'a> CorpusDriver for RealCorpusDriver<'a> {
    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
        top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError> {
        let corpus: &Path = match variant {
            ConfigVariant::NoContextual => &self.raw_corpus,
            _ => &self.ctx_corpus,
        };
        let over_fetch = top_k * 3;

        let hits = match variant {
            ConfigVariant::Primary | ConfigVariant::NoContextual => {
                query_corpus_hybrid_reranked(
                    corpus,
                    question,
                    top_k,
                    self.embedder,
                    self.reranker,
                    over_fetch,
                    breakdown,
                )
                .map_err(|e| EvalError::Runner(format!("{e}")))?
            }
            ConfigVariant::NoRerank => query_corpus_hybrid(
                corpus,
                question,
                top_k,
                self.embedder,
                breakdown,
            )
            .map_err(|e| EvalError::Runner(format!("{e}")))?,
            ConfigVariant::DenseOnly => query_corpus_reranked(
                corpus,
                question,
                top_k,
                self.embedder,
                self.reranker,
                over_fetch,
                breakdown,
            )
            .map_err(|e| EvalError::Runner(format!("{e}")))?,
        };

        Ok(hits.into_iter().map(|h| h.entry.chunk_text).collect())
    }
}
```

Add to `crates/fastrag-eval/src/lib.rs`:

```rust
#[cfg(feature = "real-driver")]
pub mod matrix_real;
```

Add a feature gate to `crates/fastrag-eval/Cargo.toml`:

```toml
[features]
default = []
real-driver = ["dep:fastrag-embed", "dep:fastrag-rerank"]

[dependencies]
fastrag-embed = { workspace = true, optional = true }
fastrag-rerank = { workspace = true, optional = true }
```

The `real-driver` feature keeps `fastrag-eval`'s default build light — the CLI enables it when wiring in Landing 5.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p fastrag-eval --features real-driver`
Expected: zero errors.

Also verify the default build still works:
Run: `cargo check -p fastrag-eval`
Expected: zero errors (module is feature-gated).

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-eval/src/matrix_real.rs crates/fastrag-eval/src/lib.rs \
        crates/fastrag-eval/Cargo.toml
git commit -m "eval: RealCorpusDriver behind real-driver feature

Maps each ConfigVariant to the right query_corpus_* variant
and threads LatencyBreakdown through. Feature-gated so the
default fastrag-eval build stays free of fastrag-rerank."
```

---

### Task 18: `report.rs` extension — `write_matrix_report`

**Files:**
- Modify: `crates/fastrag-eval/src/report.rs`

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module of `crates/fastrag-eval/src/report.rs`:

```rust
#[test]
fn write_matrix_report_round_trips_through_json() {
    use crate::matrix::*;

    let r = MatrixReport {
        schema_version: 1,
        git_rev: "abc123".into(),
        captured_at: "2026-04-11T00:00:00Z".into(),
        runs: vec![],
        rerank_delta: 0.08,
        contextual_delta: 0.11,
        hybrid_delta: 0.17,
    };
    let tmp = tempfile::NamedTempFile::new().unwrap();
    write_matrix_report(&r, tmp.path()).unwrap();
    let back: MatrixReport =
        serde_json::from_slice(&std::fs::read(tmp.path()).unwrap()).unwrap();
    assert_eq!(back.rerank_delta, 0.08);
    assert_eq!(back.git_rev, "abc123");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fastrag-eval report::tests::write_matrix_report`
Expected: FAIL — `write_matrix_report` not found.

- [ ] **Step 3: Implement `write_matrix_report`**

Add to `crates/fastrag-eval/src/report.rs`:

```rust
use std::path::Path;

use crate::error::EvalError;
use crate::matrix::MatrixReport;

pub fn write_matrix_report(report: &MatrixReport, path: &Path) -> Result<(), EvalError> {
    let json = serde_json::to_vec_pretty(report)
        .map_err(|e| EvalError::Runner(format!("matrix report serialize: {e}")))?;
    std::fs::write(path, json).map_err(EvalError::from)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p fastrag-eval report::tests::write_matrix_report`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-eval/src/report.rs
git commit -m "eval: write_matrix_report with serde round-trip test"
```

---

### Task 19: Lint gate for Landing 3

- [ ] **Step 1: Run the full lint**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings
cargo clippy -p fastrag-eval --features real-driver -- -D warnings
cargo fmt --check
```

Expected: zero warnings. Fix any that surface.

- [ ] **Step 2: Commit any formatting fixes**

```bash
git add -u
git commit -m "eval: clippy + fmt fixes for Landing 3"
```

(If there's nothing to commit, skip this step.)

Push Landing 3. CI watcher.

---

## Landing 4 — `baseline.rs` + diff + gate

### Task 20: `Baseline` + `BaselineDiff` types + `diff()`

**Files:**
- Create: `crates/fastrag-eval/src/baseline.rs`
- Modify: `crates/fastrag-eval/src/lib.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/fastrag-eval/src/baseline.rs`:

```rust
//! Checked-in baseline + slack gate for eval regressions.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::matrix::{ConfigVariant, MatrixReport};

pub const DEFAULT_SLACK: f64 = 0.02;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema_version: u32,
    pub git_rev: String,
    pub captured_at: String,
    pub runs: Vec<VariantBaseline>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantBaseline {
    pub variant: ConfigVariant,
    pub hit_at_5: f64,
    pub mrr_at_10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    pub variant: ConfigVariant,
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
    pub delta: f64,
    pub slack: f64,
}

#[derive(Debug, Default)]
pub struct BaselineDiff {
    pub regressions: Vec<Regression>,
}

impl BaselineDiff {
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    pub fn render_report(&self) -> String {
        if self.regressions.is_empty() {
            return "## Baseline OK — no regressions\n".into();
        }
        let mut out = format!("## Baseline regressions ({})\n", self.regressions.len());
        for r in &self.regressions {
            let pct = ((r.current - r.baseline) / r.baseline) * 100.0;
            out.push_str(&format!(
                "- {:?} {}: {:.4} → {:.4} ({:+.2}%, slack ±{:.0}%)\n",
                r.variant,
                r.metric,
                r.baseline,
                r.current,
                pct,
                r.slack * 100.0,
            ));
        }
        out
    }
}

pub fn load_baseline(path: &Path) -> Result<Baseline, EvalError> {
    let bytes = std::fs::read(path).map_err(EvalError::from)?;
    serde_json::from_slice(&bytes).map_err(|e| EvalError::BaselineLoad {
        path: path.to_path_buf(),
        source: e,
    })
}

pub fn diff(report: &MatrixReport, baseline: &Baseline) -> Result<BaselineDiff, EvalError> {
    if report.schema_version != baseline.schema_version {
        return Err(EvalError::BaselineSchemaMismatch {
            baseline_version: baseline.schema_version,
            report_version: report.schema_version,
        });
    }

    let mut regressions = Vec::new();
    for base in &baseline.runs {
        let run = report
            .runs
            .iter()
            .find(|r| r.variant == base.variant)
            .ok_or_else(|| EvalError::BaselineVariantMissing(base.variant))?;

        check(&mut regressions, base.variant, "hit@5", base.hit_at_5, run.hit_at_5);
        check(
            &mut regressions,
            base.variant,
            "MRR@10",
            base.mrr_at_10,
            run.mrr_at_10,
        );
    }
    Ok(BaselineDiff { regressions })
}

fn check(
    out: &mut Vec<Regression>,
    variant: ConfigVariant,
    metric: &'static str,
    baseline: f64,
    current: f64,
) {
    let threshold = baseline * (1.0 - DEFAULT_SLACK);
    if current < threshold {
        out.push(Regression {
            variant,
            metric,
            baseline,
            current,
            delta: current - baseline,
            slack: DEFAULT_SLACK,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::*;

    fn mk_report(primary_hit5: f64, primary_mrr: f64) -> MatrixReport {
        let zero_pct = LatencyPercentiles {
            total: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
            embed: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
            bm25: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
            hnsw: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
            rerank: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
            fuse: Percentiles { p50_us: 0, p95_us: 0, p99_us: 0 },
        };
        MatrixReport {
            schema_version: 1,
            git_rev: "x".into(),
            captured_at: "x".into(),
            runs: vec![VariantReport {
                variant: ConfigVariant::Primary,
                hit_at_1: 0.0,
                hit_at_5: primary_hit5,
                hit_at_10: 0.0,
                mrr_at_10: primary_mrr,
                latency: zero_pct,
                per_question: vec![],
            }],
            rerank_delta: 0.0,
            contextual_delta: 0.0,
            hybrid_delta: 0.0,
        }
    }

    fn mk_baseline(primary_hit5: f64, primary_mrr: f64) -> Baseline {
        Baseline {
            schema_version: 1,
            git_rev: "x".into(),
            captured_at: "x".into(),
            runs: vec![VariantBaseline {
                variant: ConfigVariant::Primary,
                hit_at_5: primary_hit5,
                mrr_at_10: primary_mrr,
            }],
        }
    }

    #[test]
    fn exact_match_has_no_regressions() {
        let d = diff(&mk_report(0.82, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert!(!d.has_regressions());
    }

    #[test]
    fn exactly_two_percent_drop_passes_at_boundary() {
        // threshold = 0.82 * 0.98 = 0.8036
        // 0.8036 meets the threshold (>= comparison internally is `<` so we need > threshold)
        let d = diff(&mk_report(0.8036, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert!(!d.has_regressions(), "boundary should pass, got: {:?}", d.regressions);
    }

    #[test]
    fn just_past_two_percent_drop_is_a_regression() {
        let d = diff(&mk_report(0.80, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert_eq!(d.regressions.len(), 1);
        assert_eq!(d.regressions[0].metric, "hit@5");
    }

    #[test]
    fn schema_mismatch_fails_hard() {
        let mut r = mk_report(0.82, 0.71);
        r.schema_version = 2;
        let err = diff(&r, &mk_baseline(0.82, 0.71)).unwrap_err();
        assert!(format!("{err}").contains("schema"));
    }

    #[test]
    fn render_report_no_regressions_is_ok_line() {
        let d = BaselineDiff::default();
        assert!(d.render_report().contains("Baseline OK"));
    }

    #[test]
    fn render_report_with_regression_names_variant_and_metric() {
        let d = diff(&mk_report(0.79, 0.60), &mk_baseline(0.82, 0.71)).unwrap();
        let out = d.render_report();
        assert!(out.contains("Primary"));
        assert!(out.contains("hit@5"));
        assert!(out.contains("MRR@10"));
    }
}
```

Add error variants to `crates/fastrag-eval/src/error.rs`:

```rust
#[error("baseline load error at {path}: {source}")]
BaselineLoad {
    path: std::path::PathBuf,
    #[source]
    source: serde_json::Error,
},
#[error("baseline schema mismatch: baseline version {baseline_version}, report version {report_version}")]
BaselineSchemaMismatch {
    baseline_version: u32,
    report_version: u32,
},
#[error("baseline references variant {0:?} but report does not contain it")]
BaselineVariantMissing(crate::matrix::ConfigVariant),
```

Add to `crates/fastrag-eval/src/lib.rs`:

```rust
pub mod baseline;
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p fastrag-eval baseline::tests`
Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/fastrag-eval/src/baseline.rs crates/fastrag-eval/src/error.rs \
        crates/fastrag-eval/src/lib.rs
git commit -m "eval: baseline diff with 2% slack gate

Schema mismatch fails hard, variant-missing fails hard.
Boundary test at exactly 2% slack passes; 2.01% is a regression."
```

---

### Task 21: `baseline_diff.rs` integration test

**Files:**
- Create: `crates/fastrag-eval/tests/fixtures/baseline_current.json`
- Create: `crates/fastrag-eval/tests/fixtures/report_good.json`
- Create: `crates/fastrag-eval/tests/fixtures/report_bad.json`
- Create: `crates/fastrag-eval/tests/baseline_diff.rs`

- [ ] **Step 1: Write the baseline fixture**

`crates/fastrag-eval/tests/fixtures/baseline_current.json`:

```json
{
  "schema_version": 1,
  "git_rev": "baseline_commit_abc",
  "captured_at": "2026-04-11T00:00:00Z",
  "runs": [
    {"variant": "Primary",      "hit_at_5": 0.82, "mrr_at_10": 0.71},
    {"variant": "NoRerank",     "hit_at_5": 0.74, "mrr_at_10": 0.63},
    {"variant": "NoContextual", "hit_at_5": 0.71, "mrr_at_10": 0.60},
    {"variant": "DenseOnly",    "hit_at_5": 0.65, "mrr_at_10": 0.55}
  ]
}
```

- [ ] **Step 2: Write the good-run fixture**

`crates/fastrag-eval/tests/fixtures/report_good.json`:

```json
{
  "schema_version": 1,
  "git_rev": "fresh_commit_def",
  "captured_at": "2026-04-12T00:00:00Z",
  "runs": [
    {"variant": "Primary",      "hit_at_1": 0.5, "hit_at_5": 0.83, "hit_at_10": 0.90, "mrr_at_10": 0.72, "latency": {"total":{"p50_us":0,"p95_us":0,"p99_us":0},"embed":{"p50_us":0,"p95_us":0,"p99_us":0},"bm25":{"p50_us":0,"p95_us":0,"p99_us":0},"hnsw":{"p50_us":0,"p95_us":0,"p99_us":0},"rerank":{"p50_us":0,"p95_us":0,"p99_us":0},"fuse":{"p50_us":0,"p95_us":0,"p99_us":0}}, "per_question": []},
    {"variant": "NoRerank",     "hit_at_1": 0.4, "hit_at_5": 0.75, "hit_at_10": 0.85, "mrr_at_10": 0.64, "latency": {"total":{"p50_us":0,"p95_us":0,"p99_us":0},"embed":{"p50_us":0,"p95_us":0,"p99_us":0},"bm25":{"p50_us":0,"p95_us":0,"p99_us":0},"hnsw":{"p50_us":0,"p95_us":0,"p99_us":0},"rerank":{"p50_us":0,"p95_us":0,"p99_us":0},"fuse":{"p50_us":0,"p95_us":0,"p99_us":0}}, "per_question": []},
    {"variant": "NoContextual", "hit_at_1": 0.3, "hit_at_5": 0.72, "hit_at_10": 0.80, "mrr_at_10": 0.61, "latency": {"total":{"p50_us":0,"p95_us":0,"p99_us":0},"embed":{"p50_us":0,"p95_us":0,"p99_us":0},"bm25":{"p50_us":0,"p95_us":0,"p99_us":0},"hnsw":{"p50_us":0,"p95_us":0,"p99_us":0},"rerank":{"p50_us":0,"p95_us":0,"p99_us":0},"fuse":{"p50_us":0,"p95_us":0,"p99_us":0}}, "per_question": []},
    {"variant": "DenseOnly",    "hit_at_1": 0.2, "hit_at_5": 0.66, "hit_at_10": 0.75, "mrr_at_10": 0.56, "latency": {"total":{"p50_us":0,"p95_us":0,"p99_us":0},"embed":{"p50_us":0,"p95_us":0,"p99_us":0},"bm25":{"p50_us":0,"p95_us":0,"p99_us":0},"hnsw":{"p50_us":0,"p95_us":0,"p99_us":0},"rerank":{"p50_us":0,"p95_us":0,"p99_us":0},"fuse":{"p50_us":0,"p95_us":0,"p99_us":0}}, "per_question": []}
  ],
  "rerank_delta": 0.08,
  "contextual_delta": 0.11,
  "hybrid_delta": 0.17
}
```

- [ ] **Step 3: Write the bad-run fixture**

`crates/fastrag-eval/tests/fixtures/report_bad.json` — identical to `report_good.json` except `Primary.hit_at_5 = 0.79` (3.6% drop from 0.82, past the 2% slack). Copy `report_good.json` and change that one number.

- [ ] **Step 4: Write the test**

`crates/fastrag-eval/tests/baseline_diff.rs`:

```rust
//! Integration test: baseline::diff over checked-in report + baseline fixtures.

use std::path::PathBuf;

use fastrag_eval::baseline::{diff, load_baseline};
use fastrag_eval::matrix::MatrixReport;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_report(name: &str) -> MatrixReport {
    let bytes = std::fs::read(fixture(name)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn good_run_passes_baseline_gate() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_good.json");
    let d = diff(&report, &baseline).unwrap();
    assert!(!d.has_regressions(), "expected no regressions, got: {:?}", d.regressions);
}

#[test]
fn bad_run_produces_primary_hit5_regression() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_bad.json");
    let d = diff(&report, &baseline).unwrap();
    assert!(d.has_regressions());
    let primary_hit5 = d.regressions.iter().find(|r| {
        format!("{:?}", r.variant) == "Primary" && r.metric == "hit@5"
    });
    assert!(
        primary_hit5.is_some(),
        "expected a Primary hit@5 regression, got: {:?}",
        d.regressions
    );
}

#[test]
fn bad_run_renders_markdown_mentioning_primary() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_bad.json");
    let d = diff(&report, &baseline).unwrap();
    let rendered = d.render_report();
    assert!(rendered.contains("Primary"));
    assert!(rendered.contains("hit@5"));
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p fastrag-eval --test baseline_diff`
Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag-eval/tests/fixtures/baseline_current.json \
        crates/fastrag-eval/tests/fixtures/report_good.json \
        crates/fastrag-eval/tests/fixtures/report_bad.json \
        crates/fastrag-eval/tests/baseline_diff.rs
git commit -m "eval: baseline_diff integration test with checked-in fixtures"
```

---

### Task 22: Landing 4 lint gate

- [ ] **Step 1: Run**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings
cargo clippy -p fastrag-eval --features real-driver -- -D warnings
cargo fmt --check
```

- [ ] **Step 2: Fix + commit if needed**

```bash
git add -u
git commit -m "eval: lint fixes for Landing 4"
```

Push Landing 4. CI watcher.

---

## Landing 5 — CLI wiring + mini e2e

### Task 23: Extend `EvalArgs` with new fields

**Files:**
- Modify: `fastrag-cli/src/args.rs` (lines 312–360 — the `Eval` subcommand variant)

- [ ] **Step 1: Read the existing Eval variant**

Open `fastrag-cli/src/args.rs` and scroll to the `Eval { ... }` variant of the top-level subcommand enum.

- [ ] **Step 2: Add the new fields**

Add these fields to the `Eval` variant (preserve all existing ones):

```rust
/// Path to a gold-set JSON file. Mutually exclusive with --dataset-name.
#[arg(long)]
gold_set: Option<std::path::PathBuf>,

/// Path to a built corpus (contextualized when --config-matrix is set).
#[arg(long)]
corpus: Option<std::path::PathBuf>,

/// Path to a second corpus built without --contextualize.
/// Required when --config-matrix is set.
#[arg(long)]
corpus_no_contextual: Option<std::path::PathBuf>,

/// Run all 4 matrix variants. Requires --gold-set.
#[arg(long, default_value_t = false)]
config_matrix: bool,

/// Path to a checked-in baseline JSON. Non-zero exit on >2% regression.
#[arg(long)]
baseline: Option<std::path::PathBuf>,
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p fastrag-cli --features eval`
Expected: zero errors. Warnings about unused fields are fine — Task 24 wires them up.

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/src/args.rs
git commit -m "cli: EvalArgs gains --gold-set, --corpus, --corpus-no-contextual, --config-matrix, --baseline"
```

---

### Task 24: Wire `--config-matrix` dispatch

**Files:**
- Modify: `fastrag-cli/src/main.rs` (lines 461–492 — the `Eval { ... }` match arm)
- Modify: `fastrag-cli/Cargo.toml`

- [ ] **Step 1: Enable `real-driver` feature on fastrag-eval**

Edit `fastrag-cli/Cargo.toml`:

```toml
eval = [
    "dep:fastrag-eval",
    "fastrag-eval/real-driver",
    "dep:fastrag-embed",
    "fastrag-embed/legacy-candle",
    "fastrag-embed/test-utils",
]
```

- [ ] **Step 2: Add the matrix dispatch**

In `fastrag-cli/src/main.rs`, find the existing `Eval { ... } => { ... }` match arm (around line 461). Before it calls the existing `run_eval` (or whatever the BEIR path is), add a matrix branch:

```rust
#[cfg(feature = "eval")]
Eval {
    dataset,
    dataset_name,
    report,
    embedder,
    top_k,
    chunking,
    chunk_size,
    chunk_overlap,
    max_rss_mb,
    max_docs,
    max_queries,
    gold_set,
    corpus,
    corpus_no_contextual,
    config_matrix,
    baseline,
} => {
    if config_matrix {
        let gold_set_path = gold_set.ok_or_else(|| {
            anyhow::anyhow!("--config-matrix requires --gold-set")
        })?;
        let ctx_corpus = corpus.ok_or_else(|| {
            anyhow::anyhow!("--config-matrix requires --corpus")
        })?;
        let raw_corpus = corpus_no_contextual.ok_or_else(|| {
            anyhow::anyhow!("--config-matrix requires --corpus-no-contextual")
        })?;

        let gs = fastrag_eval::gold_set::load(&gold_set_path)?;

        // Resolve embedder + reranker.
        let embedder = resolve_embedder_for_eval(&embedder)?;
        let reranker = resolve_reranker_for_eval()?;

        let driver = fastrag_eval::matrix_real::RealCorpusDriver {
            ctx_corpus,
            raw_corpus,
            embedder: embedder.as_ref(),
            reranker: reranker.as_ref(),
        };

        let matrix_report = fastrag_eval::matrix::run_matrix(&driver, &gs, top_k)?;
        fastrag_eval::report::write_matrix_report(&matrix_report, &report)?;

        if let Some(baseline_path) = baseline {
            let baseline = fastrag_eval::baseline::load_baseline(&baseline_path)?;
            let diff = fastrag_eval::baseline::diff(&matrix_report, &baseline)?;
            eprintln!("{}", diff.render_report());
            if diff.has_regressions() {
                std::process::exit(1);
            }
        }

        return Ok(());
    }

    // existing BEIR path — leave untouched
    run_eval(
        dataset,
        dataset_name,
        report,
        embedder,
        top_k,
        chunking,
        chunk_size,
        chunk_overlap,
        max_rss_mb,
        max_docs,
        max_queries,
    )?;
}
```

The `resolve_embedder_for_eval` and `resolve_reranker_for_eval` helpers don't exist yet — declare them at the bottom of `main.rs`:

```rust
#[cfg(feature = "eval")]
fn resolve_embedder_for_eval(
    preset: &str,
) -> anyhow::Result<Box<dyn fastrag_embed::DynEmbedderTrait>> {
    // Reuse existing embedder-resolution logic from the index or query command.
    // If that logic is a free function, call it. If it's inlined, extract it first
    // (DRY). The resolver must handle qwen3-q8, bge-small, mock — at minimum qwen3-q8
    // since that's what the weekly workflow passes.
    fastrag_cli::embedder::resolve_by_preset(preset)
}

#[cfg(feature = "eval")]
fn resolve_reranker_for_eval() -> anyhow::Result<Box<dyn fastrag_rerank::Reranker>> {
    fastrag_cli::reranker::resolve_default()
}
```

Note: the exact module paths `fastrag_cli::embedder::resolve_by_preset` and `fastrag_cli::reranker::resolve_default` may not exist yet — if the CLI inlines this logic in the `index` and `query` commands, factor it out into small free functions in `fastrag-cli/src/embedder.rs` and `fastrag-cli/src/reranker.rs` as part of this task, then reuse from the `Eval` arm.

- [ ] **Step 3: Compile**

Run: `cargo check -p fastrag-cli --features eval,retrieval,rerank,hybrid,contextual`
Expected: zero errors.

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/src/main.rs fastrag-cli/Cargo.toml fastrag-cli/src/embedder.rs fastrag-cli/src/reranker.rs
git commit -m "cli: wire --config-matrix dispatch with RealCorpusDriver"
```

---

### Task 25: Mini fixture for e2e test

**Files:**
- Create: `fastrag-cli/tests/fixtures/eval_mini/corpus/01-libfoo.md`
- Create: `fastrag-cli/tests/fixtures/eval_mini/corpus/02-ssrf.md`
- Create: `fastrag-cli/tests/fixtures/eval_mini/corpus/03-deserialize.md`
- Create: `fastrag-cli/tests/fixtures/eval_mini/corpus/04-bluekeep.md`
- Create: `fastrag-cli/tests/fixtures/eval_mini/corpus/05-pathtraversal.md`
- Create: `fastrag-cli/tests/fixtures/eval_mini/questions.json`

- [ ] **Step 1: Copy the Landing 1 starter corpus docs**

The starter corpus from Task 7 (`tests/gold/corpus/`) already covers 5 docs. Copy them:

```bash
mkdir -p fastrag-cli/tests/fixtures/eval_mini/corpus
cp tests/gold/corpus/01-libfoo-rce.md fastrag-cli/tests/fixtures/eval_mini/corpus/01-libfoo.md
cp tests/gold/corpus/02-kev-bluekeep.md fastrag-cli/tests/fixtures/eval_mini/corpus/04-bluekeep.md
cp tests/gold/corpus/03-ssrf-proxy.md fastrag-cli/tests/fixtures/eval_mini/corpus/02-ssrf.md
cp tests/gold/corpus/04-cwe-502-deserialize.md fastrag-cli/tests/fixtures/eval_mini/corpus/03-deserialize.md
```

Add a fifth doc `fastrag-cli/tests/fixtures/eval_mini/corpus/05-pathtraversal.md`:

```markdown
---
title: "CWE-22 — Path Traversal in File Serving"
---

# Path Traversal (CWE-22)

Path traversal allows an attacker to access files outside the intended
directory by manipulating relative path segments like `../`. Typical
entry points are file-serving endpoints and archive extraction utilities.

## Mitigation

Canonicalize paths before access and verify the result is within the
expected root. Reject any input containing `..` segments after
normalization.
```

- [ ] **Step 2: Write the mini questions fixture**

`fastrag-cli/tests/fixtures/eval_mini/questions.json` — 10 entries that can be answered against the 5 docs above:

```json
{
  "version": 1,
  "entries": [
    {"id": "q001", "question": "Is there an RCE in libfoo?", "must_contain_cve_ids": ["CVE-2024-12345"], "must_contain_terms": ["libfoo"]},
    {"id": "q002", "question": "What year was BlueKeep disclosed?", "must_contain_cve_ids": ["CVE-2019-0708"], "must_contain_terms": ["2019"]},
    {"id": "q003", "question": "How to mitigate SSRF?", "must_contain_cve_ids": [], "must_contain_terms": ["SSRF", "allowlist"]},
    {"id": "q004", "question": "What is CWE-502?", "must_contain_cve_ids": [], "must_contain_terms": ["deserialization"]},
    {"id": "q005", "question": "How is path traversal exploited?", "must_contain_cve_ids": [], "must_contain_terms": ["path traversal", "../"]},
    {"id": "q006", "question": "Which CWE covers deserialization of untrusted data?", "must_contain_cve_ids": [], "must_contain_terms": ["CWE-502"]},
    {"id": "q007", "question": "What is the impact of the libfoo vulnerability?", "must_contain_cve_ids": ["CVE-2024-12345"], "must_contain_terms": ["impact"]},
    {"id": "q008", "question": "Is BlueKeep wormable?", "must_contain_cve_ids": ["CVE-2019-0708"], "must_contain_terms": ["wormable"]},
    {"id": "q009", "question": "How do gadget chains relate to CWE-502?", "must_contain_cve_ids": [], "must_contain_terms": ["gadget"]},
    {"id": "q010", "question": "What is CWE-22?", "must_contain_cve_ids": [], "must_contain_terms": ["CWE-22", "path traversal"]}
  ]
}
```

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/fixtures/eval_mini/
git commit -m "cli: mini eval fixture (5 docs + 10 questions) for e2e"
```

---

### Task 26: `eval_matrix_e2e.rs` end-to-end test

**Files:**
- Create: `fastrag-cli/tests/eval_matrix_e2e.rs`

- [ ] **Step 1: Write the test**

```rust
//! End-to-end test for `fastrag eval --config-matrix` over the mini fixture.
//!
//! Builds both a contextualized and a raw corpus, runs the full 4-variant
//! matrix, and asserts the JSON report has the expected shape. Does NOT
//! exercise the baseline gate — numbers on 10 questions are too unstable.
//!
//! Requires a real `llama-server` and all GGUFs + ONNX reranker. Gated
//! behind `FASTRAG_LLAMA_TEST=1` and `FASTRAG_RERANK_TEST=1` and `#[ignore]`.

#![cfg(all(feature = "eval", feature = "contextual", feature = "contextual-llama", feature = "rerank"))]

use std::path::PathBuf;

use assert_cmd::Command;
use tempfile::tempdir;

fn fixture_corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_mini/corpus")
}

fn fixture_questions() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval_mini/questions.json")
}

#[test]
#[ignore]
fn eval_matrix_end_to_end() {
    if std::env::var("FASTRAG_LLAMA_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_LLAMA_TEST=1");
        return;
    }
    if std::env::var("FASTRAG_RERANK_TEST").as_deref() != Ok("1") {
        eprintln!("skipping: set FASTRAG_RERANK_TEST=1");
        return;
    }

    let ctx_dir = tempdir().unwrap();
    let raw_dir = tempdir().unwrap();
    let report_dir = tempdir().unwrap();
    let report_path = report_dir.path().join("matrix.json");

    // Build contextualized corpus
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_corpus().to_str().unwrap(),
            "--corpus",
            ctx_dir.path().to_str().unwrap(),
            "--embedder",
            "qwen3-q8",
            "--contextualize",
        ])
        .assert()
        .success();

    // Build raw corpus
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "index",
            fixture_corpus().to_str().unwrap(),
            "--corpus",
            raw_dir.path().to_str().unwrap(),
            "--embedder",
            "qwen3-q8",
        ])
        .assert()
        .success();

    // Run the matrix
    Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "eval",
            "--gold-set",
            fixture_questions().to_str().unwrap(),
            "--corpus",
            ctx_dir.path().to_str().unwrap(),
            "--corpus-no-contextual",
            raw_dir.path().to_str().unwrap(),
            "--config-matrix",
            "--report",
            report_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Parse + assert shape
    let bytes = std::fs::read(&report_path).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let runs = report["runs"].as_array().expect("runs must be an array");
    assert_eq!(runs.len(), 4, "expected 4 variants, got: {}", runs.len());

    // Every variant must have a finite hit_at_5 in [0,1].
    for run in runs {
        let h5 = run["hit_at_5"].as_f64().expect("hit_at_5 must be f64");
        assert!(h5.is_finite());
        assert!((0.0..=1.0).contains(&h5));
    }

    // Deltas must be populated (even if negative or zero on a tiny fixture).
    assert!(report["rerank_delta"].is_f64());
    assert!(report["contextual_delta"].is_f64());
    assert!(report["hybrid_delta"].is_f64());

    // Each variant's per_question count should equal 10 (the mini fixture).
    for run in runs {
        let pq = run["per_question"].as_array().unwrap();
        assert_eq!(pq.len(), 10);
    }

    // Each variant's total p50 latency should be > 0 (real queries ran).
    for run in runs {
        let p50 = run["latency"]["total"]["p50_us"].as_u64().unwrap();
        assert!(p50 > 0, "variant has zero total p50 — did any query actually run?");
    }
}
```

- [ ] **Step 2: Run the test locally**

```bash
FASTRAG_LLAMA_TEST=1 FASTRAG_RERANK_TEST=1 \
  cargo test -p fastrag-cli \
    --features eval,contextual,contextual-llama,retrieval,rerank,hybrid \
    --test eval_matrix_e2e -- --include-ignored
```

Expected: PASS in ~3–5 minutes. Requires `llama-server` in PATH and both GGUFs + ONNX reranker cached.

If it fails, inspect the JSON at `$TMPDIR/.../matrix.json` (the tempdir path is printed by the test harness on failure) to see what shape came out.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/eval_matrix_e2e.rs
git commit -m "cli: eval_matrix_e2e end-to-end test

Ignored + gated on FASTRAG_LLAMA_TEST and FASTRAG_RERANK_TEST.
Asserts 4 variants, per-question counts match, p50 latencies > 0."
```

---

### Task 27: `eval_gold_set_rejects_invalid_e2e.rs`

**Files:**
- Create: `fastrag-cli/tests/eval_gold_set_rejects_invalid_e2e.rs`

- [ ] **Step 1: Write the test**

```rust
//! End-to-end test: `fastrag eval --gold-set <bad.json>` fails fast with a
//! useful error. No model cost — fails at load-time validation.

#![cfg(feature = "eval")]

use std::io::Write;

use assert_cmd::Command;
use tempfile::NamedTempFile;

#[test]
fn invalid_gold_set_fails_with_entry_id_in_message() {
    let mut f = NamedTempFile::new().unwrap();
    f.write_all(br#"{
        "version": 1,
        "entries": [
            {"id": "bad_one", "question": "", "must_contain_cve_ids": ["CVE-2024-1"], "must_contain_terms": []}
        ]
    }"#).unwrap();
    f.flush().unwrap();

    let out = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "eval",
            "--gold-set",
            f.path().to_str().unwrap(),
            "--corpus",
            "/tmp/does-not-exist-and-should-not-matter",
            "--corpus-no-contextual",
            "/tmp/does-not-exist-and-should-not-matter-2",
            "--config-matrix",
            "--report",
            "/tmp/unused.json",
        ])
        .output()
        .unwrap();

    assert!(!out.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("bad_one"),
        "expected error message to name the offending id, got: {stderr}"
    );
    assert!(
        stderr.contains("empty question"),
        "expected error message to mention 'empty question', got: {stderr}"
    );
}
```

- [ ] **Step 2: Run it**

Run: `cargo test -p fastrag-cli --features eval --test eval_gold_set_rejects_invalid_e2e`
Expected: PASS in under 5 seconds. No model invocation.

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/eval_gold_set_rejects_invalid_e2e.rs
git commit -m "cli: eval gold-set validation e2e (no model cost)"
```

---

### Task 28: Landing 5 lint gate

- [ ] **Step 1: Run the full lint gate**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual -- -D warnings
cargo clippy -p fastrag-cli --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- -D warnings
cargo fmt --check
cargo test --workspace
```

Expected: zero warnings, green across the board (ignored tests skipped).

- [ ] **Step 2: Commit fixes if needed**

```bash
git add -u
git commit -m "eval: lint fixes for Landing 5"
```

Push Landing 5. CI watcher.

---

## Landing 6 — Grow gold set + capture baseline

### Task 29: Grow `tests/gold/questions.json` to ≥100 entries

**Files:**
- Modify: `tests/gold/questions.json`

This is curation work, not code. The plan cannot write the 100 entries for you. Your job:

- [ ] **Step 1: Draft 90+ more entries**

Open `tests/gold/questions.json` and append entries matching the same schema. Target distribution (from the spec):

- ~50% CVE-ID lookup (need the doc to carry the CVE id)
- ~30% concept lookup (terms like "SSRF", "buffer overflow", "allowlist")
- ~20% pronoun-resolution (body has no lexical anchor; title carries the context)

Source ideas:

- Your existing pentest notes / CVE reading list
- NVD advisories you have local copies of
- CISA KEV catalog entries
- OWASP top-10 mapped to specific CWE concepts

Each entry must pair with a doc you'll write in Task 30. Do not commit questions that reference docs that don't exist yet.

- [ ] **Step 2: Validate via the canary**

Run: `cargo test -p fastrag-eval gold_set::tests::tests_gold_questions_json_is_valid`
Expected: PASS. The canary asserts `gs.entries.len() >= 10` today — update it to `>= 100` once you've hit the target:

```rust
assert!(
    gs.entries.len() >= 100,
    "gold set must have at least 100 entries, found {}",
    gs.entries.len()
);
```

- [ ] **Step 3: Commit**

```bash
git add tests/gold/questions.json crates/fastrag-eval/src/gold_set.rs
git commit -m "eval: grow gold set to 100+ entries"
```

---

### Task 30: Grow `tests/gold/corpus/` to ~50–100 docs

**Files:**
- Modify: `tests/gold/corpus/*.md` (add 45+ new docs)

- [ ] **Step 1: Write new markdown docs**

Each doc covers one of the question domains from Task 29. Keep them ≥200 words so chunking produces at least 2 chunks per doc. Use YAML frontmatter with `title:` — the title is where Contextual Retrieval gets its leverage.

File-naming pattern: `NN-topic-name.md`, 0-padded.

- [ ] **Step 2: Verify ingest works end-to-end on the grown fixture**

Run:

```bash
cargo run --features retrieval,contextual,contextual-llama -- \
  index tests/gold/corpus --corpus /tmp/gold-grow-check --embedder mock
```

Expected: exits 0, prints "Indexed N chunks" where N is reasonable for the doc count (roughly 2–5× the doc count).

Clean up: `rm -rf /tmp/gold-grow-check`

- [ ] **Step 3: Commit**

```bash
git add tests/gold/corpus/
git commit -m "eval: grow gold corpus to 50+ security docs"
```

---

### Task 31: Capture initial baseline

**Files:**
- Create: `docs/eval-baselines/current.json`
- Create: `docs/eval-baselines/README.md`

This task runs the full matrix on a real machine. Expect ~20–35 minutes wall-clock depending on corpus size. Not CI — run locally.

- [ ] **Step 1: Build both corpora locally**

```bash
mkdir -p /tmp/gold-baseline
cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus \
  --corpus /tmp/gold-baseline/ctx \
  --embedder qwen3-q8 \
  --contextualize

cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus \
  --corpus /tmp/gold-baseline/raw \
  --embedder qwen3-q8
```

Expected: both builds succeed. The `ctx` build is the slow one — it runs the contextualization model over every chunk.

- [ ] **Step 2: Run the matrix and capture the output**

```bash
cargo run --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- \
  eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/gold-baseline/ctx \
  --corpus-no-contextual /tmp/gold-baseline/raw \
  --config-matrix \
  --report docs/eval-baselines/current.json
```

Expected: writes `docs/eval-baselines/current.json`. Review the diff:

```bash
cat docs/eval-baselines/current.json | jq '.runs[] | {variant, hit_at_5, mrr_at_10}'
```

The numbers should make sense — Primary should be ≥ DenseOnly on most runs. If Primary is lower than DenseOnly, the gold set needs curation (the questions may not be answerable by the corpus).

- [ ] **Step 3: Write `docs/eval-baselines/README.md`**

```markdown
# Eval Baselines

This directory holds the checked-in baseline for the weekly eval CI gate.

## Files

- `current.json` — the active baseline. The weekly workflow compares
  every fresh matrix report against this file and fails on any hit@5 or
  MRR@10 regression beyond 2% slack.

## Refresh flow

Baseline refreshes are deliberate human commits. Never edit `current.json`
by hand — capture a fresh run and commit the result.

```bash
# Build both corpora locally.
cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus \
  --corpus /tmp/gold-baseline/ctx \
  --embedder qwen3-q8 \
  --contextualize

cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus \
  --corpus /tmp/gold-baseline/raw \
  --embedder qwen3-q8

# Run the matrix and overwrite the baseline.
cargo run --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- \
  eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/gold-baseline/ctx \
  --corpus-no-contextual /tmp/gold-baseline/raw \
  --config-matrix \
  --report docs/eval-baselines/current.json

# Review the diff before committing.
git diff docs/eval-baselines/current.json
git add docs/eval-baselines/current.json
git commit -m "eval: refresh baseline after <improvement>"
```

## When to refresh

Refresh the baseline when you've made a change that legitimately improves
retrieval quality (new embedder, tuned chunking, better contextualization
prompt). Do NOT refresh to make a red CI go green — that defeats the gate.
```

- [ ] **Step 4: Clean up and commit**

```bash
rm -rf /tmp/gold-baseline
git add docs/eval-baselines/current.json docs/eval-baselines/README.md
git commit -m "eval: capture initial baseline + refresh-flow docs"
```

Push Landing 6. Note: this push does not touch code, but the CI watcher still applies as a matter of convention.

---

## Landing 7 — Weekly workflow + docs

### Task 32: `.github/workflows/weekly.yml`

**Files:**
- Create: `.github/workflows/weekly.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: Weekly (eval harness)

on:
  schedule:
    - cron: "0 6 * * 0"   # Sundays 06:00 UTC
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always

jobs:
  check-changes:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'
    outputs:
      has_changes: ${{ github.event_name == 'workflow_dispatch' || steps.check.outputs.has_changes }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check for commits in last 7 days
        id: check
        run: |
          if git log --since="7 days ago" --oneline | grep -q .; then
            echo "has_changes=true" >> "$GITHUB_OUTPUT"
          else
            echo "has_changes=false" >> "$GITHUB_OUTPUT"
            echo "No commits in last 7 days — skipping weekly eval."
          fi

  eval:
    needs: check-changes
    if: needs.check-changes.outputs.has_changes == 'true'
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      FASTRAG_LLAMA_TEST: "1"
      FASTRAG_RERANK_TEST: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          key: eval-weekly

      - name: Install llama-server
        run: |
          LLAMA_TAG="b8739"
          curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" -o llama.tar.gz
          tar xzf llama.tar.gz
          sudo mkdir -p /opt/llama
          sudo cp -a llama-${LLAMA_TAG}/* /opt/llama/
          sudo ln -sf /opt/llama/llama-server /usr/local/bin/llama-server
          echo "LD_LIBRARY_PATH=/opt/llama" >> "$GITHUB_ENV"
          llama-server --version

      - name: Cache GGUF models
        uses: actions/cache@v4
        with:
          path: ~/.cache/fastrag/models
          key: gguf-qwen3-embed-600m-q8-qwen3-4b-instruct-2507-q4km

      - name: Download embedder GGUF
        run: |
          MODEL_DIR="$HOME/.cache/fastrag/models"
          MODEL_FILE="Qwen3-Embedding-0.6B-Q8_0.gguf"
          if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
            mkdir -p "$MODEL_DIR"
            curl -fsSL "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/$MODEL_FILE" -o "$MODEL_DIR/$MODEL_FILE"
          fi

      - name: Download completion GGUF
        run: |
          MODEL_DIR="$HOME/.cache/fastrag/models"
          MODEL_FILE="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
          if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
            mkdir -p "$MODEL_DIR"
            curl -fsSL "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$MODEL_FILE" -o "$MODEL_DIR/$MODEL_FILE"
          fi

      - name: Cache ONNX reranker
        uses: actions/cache@v4
        with:
          path: ~/.cache/fastrag/models/gte-reranker-modernbert-base
          key: onnx-gte-reranker-modernbert-base

      - name: Download ONNX reranker
        run: |
          MODEL_DIR="$HOME/.cache/fastrag/models/gte-reranker-modernbert-base"
          if [ ! -f "$MODEL_DIR/model.onnx" ] || [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
            mkdir -p "$MODEL_DIR"
            curl -fsSL "https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/resolve/main/onnx/model.onnx" -o "$MODEL_DIR/model.onnx"
            curl -fsSL "https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/resolve/main/tokenizer.json" -o "$MODEL_DIR/tokenizer.json"
          fi

      - name: Build contextualized corpus
        run: |
          cargo run --release \
            --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
            index tests/gold/corpus \
            --corpus "$RUNNER_TEMP/corpus-ctx" \
            --embedder qwen3-q8 \
            --contextualize

      - name: Build raw corpus
        run: |
          cargo run --release \
            --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
            index tests/gold/corpus \
            --corpus "$RUNNER_TEMP/corpus-raw" \
            --embedder qwen3-q8

      - name: Run eval matrix
        id: run_eval
        run: |
          cargo run --release \
            --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- \
            eval \
            --gold-set tests/gold/questions.json \
            --corpus "$RUNNER_TEMP/corpus-ctx" \
            --corpus-no-contextual "$RUNNER_TEMP/corpus-raw" \
            --config-matrix \
            --baseline docs/eval-baselines/current.json \
            --report "$RUNNER_TEMP/matrix.json"

      - name: Upload matrix report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-weekly-report
          path: ${{ runner.temp }}/matrix.json
```

- [ ] **Step 2: Validate the YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/weekly.yml'))"
```

Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/weekly.yml
git commit -m "ci: weekly eval harness workflow

Sundays 06:00 UTC, 45-min timeout, 7-day check-changes gate,
uploads matrix.json as artifact with if: always()."
```

---

### Task 33: Update `CLAUDE.md` Build & Test section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Invoke the doc-editor skill first**

Per repo convention (CLAUDE.md itself), every `.md` edit must be preceded by a doc-editor pass. Draft the new lines first, then run doc-editor on the draft, then apply.

Draft lines to add to the Build & Test fenced block:

```
cargo test --workspace --features eval                                        # Eval harness unit tests
cargo test -p fastrag-eval --features real-driver                             # Real-driver build (needed by CLI)
cargo test -p fastrag-eval --test matrix_stub                                 # Matrix orchestrator stub test
cargo test -p fastrag-eval --test gold_set_loader                             # Gold set loader validation branches
cargo test -p fastrag-eval --test union_match                                 # Union-of-top-k scorer
cargo test -p fastrag-eval --test baseline_diff                               # Baseline diff + slack gate
FASTRAG_LLAMA_TEST=1 FASTRAG_RERANK_TEST=1 cargo test -p fastrag-cli --features eval,contextual,contextual-llama,retrieval,rerank,hybrid --test eval_matrix_e2e -- --ignored
cargo test -p fastrag-cli --features eval --test eval_gold_set_rejects_invalid_e2e
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval -- -D warnings  # Full lint gate with eval
```

Draft lines to add to the Retrieval CLI section:

```
cargo run -- eval --gold-set tests/gold/questions.json \
                  --corpus ./corpus-ctx \
                  --corpus-no-contextual ./corpus-raw \
                  --config-matrix \
                  --baseline docs/eval-baselines/current.json \
                  --report target/eval/matrix.json
```

- [ ] **Step 2: Run the doc-editor skill as a Haiku agent**

Per the skills table in CLAUDE.md: "Before every `Edit` or `Write` to a `.md` file — mandatory". Pass the draft above to the doc-editor skill via a foreground Haiku Agent call. Apply the cleaned prose it returns.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): Build & Test section covers Step 6 eval harness"
```

---

### Task 34: Update `README.md` with an Eval section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Draft the new section**

Find the section after "Contextual Retrieval" (added in Step 5). Insert a new "Eval Harness" subsection:

```markdown
### Eval Harness (optional)

FastRAG ships with a hand-curated gold set and a config matrix for
measuring retrieval quality on every retrieval-touching change.

**Gold set location:** `tests/gold/questions.json` — 100+ entries with
`must_contain_cve_ids` and `must_contain_terms` assertions scored via
union-of-top-k.

**Run the full matrix locally:**

\`\`\`bash
# Build both corpora (contextualized + raw)
cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus --corpus /tmp/ctx --embedder qwen3-q8 --contextualize

cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus --corpus /tmp/raw --embedder qwen3-q8

# Run the 4-variant matrix
cargo run --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- \
  eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/ctx \
  --corpus-no-contextual /tmp/raw \
  --config-matrix \
  --report target/eval/matrix.json
\`\`\`

**Refresh the baseline:** see `docs/eval-baselines/README.md`.

**CI cadence:** the weekly workflow at `.github/workflows/weekly.yml`
runs the matrix on Sundays 06:00 UTC and fails on any hit@5 or MRR@10
regression beyond 2% slack against `docs/eval-baselines/current.json`.
```

- [ ] **Step 2: Run the doc-editor skill on the draft**

Mandatory per CLAUDE.md. Apply the cleaned prose.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): eval harness section pointing at --config-matrix"
```

---

### Task 35: Final lint + push

- [ ] **Step 1: Run the full gate one more time**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval -- -D warnings
cargo fmt --check
cargo test --workspace --features retrieval,rerank,hybrid,contextual,eval
```

Expected: zero warnings, green tests.

- [ ] **Step 2: Push Landing 7**

```bash
git push origin main
```

- [ ] **Step 3: Launch ci-watcher + trigger the first weekly run manually**

Launch ci-watcher as a background Haiku Agent per convention. Then trigger the new weekly workflow manually to verify the full pipeline works end-to-end before waiting for Sunday:

```bash
gh workflow run weekly.yml
gh run list --workflow=weekly.yml --limit=1
```

Wait for that run to finish (it will take ~20–40 minutes). If it fails, inspect the uploaded `eval-weekly-report` artifact to see the matrix JSON and debug from there.

- [ ] **Step 4: Done**

Step 6 is shipped. Update the roadmap at `docs/superpowers/roadmap-2026-04-phase2-rewrite.md` to mark Step 6 complete in a follow-up commit. Step 7 (security corpus hygiene) is the next roadmap item.

---

## Self-Review Notes (for the reader)

**Spec coverage:** Every section of the spec maps to at least one task:

- Gold set schema + loader + scorer → Tasks 2, 3, 4, 5, 8, 9
- Starter fixture → Tasks 6, 7
- LatencyBreakdown + instrumentation → Tasks 10, 11, 12, 13, 14
- ConfigVariant + matrix orchestrator → Tasks 15, 16, 17
- Report writer + delta computation → Tasks 15, 18
- Baseline + slack gate → Tasks 20, 21
- CLI wiring → Tasks 23, 24
- Mini fixture + e2e tests → Tasks 25, 26, 27
- Gold set growth + fixture corpus → Tasks 29, 30
- Baseline capture → Task 31
- Weekly workflow → Task 32
- CLAUDE.md + README.md → Tasks 33, 34

**Open question resolutions (from spec):**

- Q1: `query_corpus` has 5 variants, not 1 — each takes `&mut LatencyBreakdown`.
- Q2: New per-stage histograms live in `matrix.rs`, alongside the existing `runner.rs` total-latency histogram. The existing runner stays on its own `HnswIndex::query` path for BEIR.
- Q3: `EmbedderIdentity` already derives `Serialize + Deserialize`. Matrix report embeds it via the corpus manifest.
- Q4: `QuestionResult::latency_us` is flat (one `LatencyBreakdown` field per question). `VariantReport::latency` is nested (`LatencyPercentiles`). Two shapes — flat for raw per-query data, nested for aggregated percentiles. This matches the types used in the tasks.

**No placeholders:** Every task has concrete file paths, full code blocks, explicit commands, and expected output. Tasks 29 and 30 are curation work — the plan cannot invent the gold-set content, so they're labeled as such with clear validation via the canary test.

**Type consistency:**

- `GoldSet`, `GoldSetEntry`, `EntryScore` — defined in Tasks 2/5, used in Tasks 8/9/16
- `ConfigVariant`, `MatrixReport`, `VariantReport`, `LatencyPercentiles`, `Percentiles` — defined in Task 15, used in Tasks 16/18/20/21
- `Baseline`, `VariantBaseline`, `Regression`, `BaselineDiff` — defined in Task 20, used in Task 21
- `LatencyBreakdown` — defined in Task 10, threaded through Tasks 11/12/13/14, used in Tasks 15/16/17/26
- `CorpusDriver` trait — defined in Task 16, implemented by stub in Task 16 and real in Task 17
- `EvalError` variants — defined in Tasks 3/16/20, referenced consistently
