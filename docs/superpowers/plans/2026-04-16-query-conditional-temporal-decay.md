# Query-Conditional Temporal Decay v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-query `TemporalPolicy` API, high-precision abstaining regex detector, and late-stage decay injection to replace the fixed-halflife `TemporalOn` variant (which regresses historical queries by 15.4pp hit@5).

**Architecture:** Three layered changes land as three commits:
1. New `corpus/temporal.rs` module with `TemporalPolicy` types, detector trait, `AbstainingRegexDetector`, `OracleDetector`, and `apply_temporal_policy` wrapper — self-contained, dead code.
2. Rewire: strip decay out of `query_hybrid`, thread `TemporalPolicy` through `QueryOpts` / HTTP / CLI, inject decay at the post-rerank stage in `query_corpus_reranked_opts`.
3. Retire `ConfigVariant::TemporalOn`, add `TemporalAuto` + `TemporalOracle`, add `route_regret` metric, recapture `docs/eval-baselines/current.json`, wire new gate assertions.

**Tech Stack:** Rust workspace, `serde`, `regex` (new direct dep on `fastrag` for the detector), `chrono` (existing), `thiserror` (existing). No new runtime deps beyond `regex`.

**Spec:** `docs/superpowers/specs/2026-04-16-query-conditional-temporal-decay-design.md`

---

## Orientation for the Implementer

**Key files you will touch (read first, before starting Task 1):**

- `crates/fastrag/src/corpus/mod.rs` — `QueryOpts` struct (line 31), `query_corpus_reranked_opts` orchestrator (line 1642). Late-injection lands here.
- `crates/fastrag/src/corpus/hybrid.rs` — `apply_decay` (line 129, pure function, **stays unchanged**), `query_hybrid` (line 607, loses its internal decay branch), `HybridOpts`/`TemporalOpts` (lines 18, 37).
- `crates/fastrag-eval/src/matrix.rs` — `ConfigVariant` enum (line 20), `VariantReport` / `MatrixReport` structs (lines 105, 122).
- `crates/fastrag-eval/src/matrix_real.rs` — `RealCorpusDriver::query` dispatch (line 77). The `TemporalOn` branch (line 114) gets replaced.
- `fastrag-cli/src/args.rs` — CLI flag definitions. The deprecated `--time-decay-*` family lives here (lines ~395-450).
- `fastrag-cli/src/serve.rs` (or wherever the HTTP `/query` handler lives — run `rg 'fn.*query.*handler|temporal' fastrag-cli/src/` to locate).
- `tests/gold/questions.json` — gold-set; `axes.temporal_intent` per entry drives `OracleDetector`.

**Build/test commands you will use repeatedly:**

```bash
# Full lint gate (run before every commit):
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings

# Core unit tests for this work:
cargo test -p fastrag --features hybrid,retrieval --lib corpus::temporal
cargo test -p fastrag --features retrieval --test hybrid_retrieval
cargo test -p fastrag --features retrieval --test temporal_decay
cargo test -p fastrag-eval --features real-driver
cargo test -p fastrag-cli --features retrieval --test temporal_decay_e2e
cargo test -p fastrag-cli --features retrieval --test temporal_decay_http_e2e

# Full format check:
cargo fmt --check
```

**Commit discipline:**
- One commit per task. Commit messages follow `feat(scope): description` / `test(scope): description` / `refactor(scope): description`.
- **Do not** batch multiple tasks into one commit — each task leaves the tree green and testable.
- Close #53 only in the final commit of Landing 3 (`Closes #53`).
- Run the full lint gate before every commit.

**TDD is mandatory:** every task starts with a failing test. If a task lists no test, write one before the implementation.

---

## Landing 1 — `corpus/temporal.rs` (dead code, self-contained)

This landing is pure new code. Nothing it defines is called from existing code paths. At the end of Landing 1, the tree compiles and all existing tests pass; the new module has its own unit tests.

### Task 1: Add `regex` dep and create empty `temporal.rs` module

**Files:**
- Modify: `crates/fastrag/Cargo.toml`
- Create: `crates/fastrag/src/corpus/temporal.rs`
- Modify: `crates/fastrag/src/corpus/mod.rs` (add `pub mod temporal;`)

- [ ] **Step 1: Add `regex` to `fastrag` deps**

Run `rg '^regex' crates/fastrag/Cargo.toml` to check if it's already present (likely is, transitively — check the `[dependencies]` block). If not, add:

```toml
# crates/fastrag/Cargo.toml, under [dependencies]
regex = "1"
```

- [ ] **Step 2: Create empty module file**

```rust
// crates/fastrag/src/corpus/temporal.rs
//! Per-query temporal decay policy: API types, detector trait, and
//! late-stage injection wrapper.
//!
//! See `docs/superpowers/specs/2026-04-16-query-conditional-temporal-decay-design.md`.
```

- [ ] **Step 3: Register module**

In `crates/fastrag/src/corpus/mod.rs`, add next to existing `pub mod hybrid;`:

```rust
pub mod temporal;
```

- [ ] **Step 4: Verify tree compiles**

Run: `cargo build -p fastrag --features retrieval,hybrid`
Expected: success, no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/Cargo.toml crates/fastrag/src/corpus/temporal.rs crates/fastrag/src/corpus/mod.rs
git commit -m "feat(temporal): scaffold corpus::temporal module"
```

---

### Task 2: `Strength` enum with `halflife()` / `weight_floor()`

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`

- [ ] **Step 1: Write the failing test**

Append to `temporal.rs`:

```rust
#[cfg(test)]
mod strength_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn light_constants() {
        assert_eq!(Strength::Light.halflife(), Duration::from_secs(365 * 86_400));
        assert!((Strength::Light.weight_floor() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn medium_constants() {
        assert_eq!(Strength::Medium.halflife(), Duration::from_secs(180 * 86_400));
        assert!((Strength::Medium.weight_floor() - 0.60).abs() < 1e-6);
    }

    #[test]
    fn strong_constants() {
        assert_eq!(Strength::Strong.halflife(), Duration::from_secs(60 * 86_400));
        assert!((Strength::Strong.weight_floor() - 0.45).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::strength_tests`
Expected: FAIL — `Strength` not defined.

- [ ] **Step 3: Implement `Strength`**

Add above the test module:

```rust
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    Light,
    Medium,
    Strong,
}

impl Strength {
    pub fn halflife(self) -> Duration {
        match self {
            Strength::Light => Duration::from_secs(365 * 86_400),
            Strength::Medium => Duration::from_secs(180 * 86_400),
            Strength::Strong => Duration::from_secs(60 * 86_400),
        }
    }

    pub fn weight_floor(self) -> f32 {
        match self {
            Strength::Light => 0.75,
            Strength::Medium => 0.60,
            Strength::Strong => 0.45,
        }
    }
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::strength_tests`
Expected: 3/3 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs
git commit -m "feat(temporal): add Strength enum with halflife and weight_floor constants"
```

---

### Task 3: `TemporalPolicy` enum + serde round-trip test

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`

- [ ] **Step 1: Write the failing test**

Append a new test module:

```rust
#[cfg(test)]
mod policy_serde_tests {
    use super::*;

    #[test]
    fn auto_serializes_as_string() {
        let v = TemporalPolicy::Auto;
        assert_eq!(serde_json::to_string(&v).unwrap(), "\"auto\"");
    }

    #[test]
    fn off_serializes_as_string() {
        let v = TemporalPolicy::Off;
        assert_eq!(serde_json::to_string(&v).unwrap(), "\"off\"");
    }

    #[test]
    fn favor_recent_serializes_as_tagged_object() {
        let v = TemporalPolicy::FavorRecent(Strength::Medium);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"favor_recent":"medium"}"#
        );
    }

    #[test]
    fn auto_is_default() {
        let v: TemporalPolicy = Default::default();
        assert!(matches!(v, TemporalPolicy::Auto));
    }

    #[test]
    fn deserialize_round_trip() {
        for p in [
            TemporalPolicy::Auto,
            TemporalPolicy::Off,
            TemporalPolicy::FavorRecent(Strength::Light),
            TemporalPolicy::FavorRecent(Strength::Medium),
            TemporalPolicy::FavorRecent(Strength::Strong),
        ] {
            let s = serde_json::to_string(&p).unwrap();
            let back: TemporalPolicy = serde_json::from_str(&s).unwrap();
            assert_eq!(p, back);
        }
    }
}
```

- [ ] **Step 2: Run test — expected FAIL**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::policy_serde_tests`
Expected: fails — `TemporalPolicy` not defined.

- [ ] **Step 3: Implement `TemporalPolicy`**

Add to `temporal.rs` above the test modules:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPolicy {
    #[default]
    Auto,
    Off,
    FavorRecent(Strength),
}
```

- [ ] **Step 4: Run test — expected PASS**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::policy_serde_tests`
Expected: 5/5 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs
git commit -m "feat(temporal): add TemporalPolicy enum with serde support"
```

---

### Task 4: `TemporalDetector` trait + `OracleDetector`

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`

**Context:** The gold-set has `axes.temporal_intent` per entry with values `"historical" | "neutral" | "recency_seeking"`. `OracleDetector` maps these to policies for eval-time upper-bound routing. It is constructed per-query in the eval driver, so it holds the intent on itself.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod oracle_tests {
    use super::*;

    #[test]
    fn recency_seeking_routes_to_medium_favor_recent() {
        let d = OracleDetector::new(Some(TemporalIntent::RecencySeeking));
        match d.detect("any query text") {
            TemporalPolicy::FavorRecent(Strength::Medium) => {}
            other => panic!("expected FavorRecent(Medium), got {other:?}"),
        }
    }

    #[test]
    fn neutral_routes_to_off() {
        let d = OracleDetector::new(Some(TemporalIntent::Neutral));
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }

    #[test]
    fn historical_routes_to_off() {
        let d = OracleDetector::new(Some(TemporalIntent::Historical));
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }

    #[test]
    fn missing_intent_routes_to_off() {
        let d = OracleDetector::new(None);
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }
}
```

- [ ] **Step 2: Run test — expected FAIL**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::oracle_tests`
Expected: fails — `TemporalDetector`, `OracleDetector`, `TemporalIntent` undefined.

- [ ] **Step 3: Implement trait + `TemporalIntent` + `OracleDetector`**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalIntent {
    Historical,
    Neutral,
    RecencySeeking,
}

pub trait TemporalDetector: Send + Sync {
    fn detect(&self, query: &str) -> TemporalPolicy;
}

pub struct OracleDetector {
    intent: Option<TemporalIntent>,
}

impl OracleDetector {
    pub fn new(intent: Option<TemporalIntent>) -> Self {
        Self { intent }
    }
}

impl TemporalDetector for OracleDetector {
    fn detect(&self, _query: &str) -> TemporalPolicy {
        match self.intent {
            Some(TemporalIntent::RecencySeeking) => {
                TemporalPolicy::FavorRecent(Strength::Medium)
            }
            _ => TemporalPolicy::Off,
        }
    }
}
```

- [ ] **Step 4: Run test — expected PASS**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::oracle_tests`
Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs
git commit -m "feat(temporal): add TemporalDetector trait and OracleDetector"
```

---

### Task 5: `AbstainingRegexDetector` — positive cases

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`

**Context:** The detector recognizes high-precision recency markers and returns `FavorRecent(Medium)` on a hit, `Off` otherwise. It never classifies historical intent — historical queries get `Off` (which is correct for v1 since we don't yet have anchored-past reference-time support).

Pattern families (see spec §Layer 2):

1. `\b(latest|newest|current(ly)?|newer)\b`
2. `\brecent(ly)?\b` followed within 5 tokens by one of `advisory|exploit|bypass|CVE|disclosure|vulnerability|patch|guidance`
3. `\bstill (exploited|in KEV|vulnerable|unpatched)\b`
4. `\bas of (today|now|this (week|month))\b`
5. `\b(this week|this month)\b`
6. `\b2026\b` within 5 tokens of one of `CVE|vulnerability|advisory|disclosure|exploit|PoC|mitigation|patch`

Everything else → `Off`.

- [ ] **Step 1: Write positive-case test**

```rust
#[cfg(test)]
mod regex_positive_tests {
    use super::*;

    fn fires(query: &str) -> bool {
        matches!(
            AbstainingRegexDetector::new().detect(query),
            TemporalPolicy::FavorRecent(Strength::Medium)
        )
    }

    #[test]
    fn latest_keyword() {
        assert!(fires("latest Log4j advisory"));
        assert!(fires("newest CVE in KEV"));
        assert!(fires("current mitigation for Shellshock"));
        assert!(fires("newer bypass for PrintNightmare"));
    }

    #[test]
    fn recent_plus_security_noun() {
        assert!(fires("recent advisory for PyYAML"));
        assert!(fires("recently disclosed CVE in libxml2"));
        assert!(fires("recent patch for sudoedit"));
    }

    #[test]
    fn still_exploited_family() {
        assert!(fires("is Heartbleed still exploited in 2026"));
        assert!(fires("CVE-2021-44228 still in KEV"));
        assert!(fires("still unpatched on Ubuntu LTS"));
    }

    #[test]
    fn as_of_now_variants() {
        assert!(fires("as of today, what is the fix"));
        assert!(fires("as of now any known exploits"));
        assert!(fires("as of this week is it patched"));
        assert!(fires("as of this month how bad is it"));
    }

    #[test]
    fn this_week_this_month() {
        assert!(fires("what dropped this week"));
        assert!(fires("advisories this month"));
    }

    #[test]
    fn current_year_plus_security_noun() {
        assert!(fires("2026 CVE for libsqlite"));
        assert!(fires("2026 advisory NVD"));
        assert!(fires("2026 mitigation guidance"));
    }
}
```

- [ ] **Step 2: Run — expected FAIL**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::regex_positive_tests`
Expected: fails — `AbstainingRegexDetector` undefined.

- [ ] **Step 3: Implement `AbstainingRegexDetector`**

Add to `temporal.rs`:

```rust
use std::sync::OnceLock;

use regex::Regex;

pub struct AbstainingRegexDetector;

impl AbstainingRegexDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AbstainingRegexDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn keyword_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\b(latest|newest|current|currently|newer)\b").unwrap()
    })
}

fn recent_plus_noun_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // `recent` / `recently`, then 0..=4 intermediate tokens, then a security noun.
        Regex::new(
            r"(?i)\brecent(?:ly)?\b(?:\W+\w+){0,4}\W+(advisory|exploit|bypass|cve|disclosure|vulnerabilit(?:y|ies)|patch|guidance)\b",
        )
        .unwrap()
    })
}

fn still_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\bstill\s+(exploited|in\s+kev|vulnerable|unpatched)\b").unwrap()
    })
}

fn as_of_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\bas\s+of\s+(today|now|this\s+(week|month))\b").unwrap()
    })
}

fn this_week_month_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)\bthis\s+(week|month)\b").unwrap())
}

fn year_2026_plus_noun_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        // 2026 either followed by or preceded by (within 5 tokens) a security noun.
        Regex::new(
            r"(?i)(?:\b2026\b(?:\W+\w+){0,4}\W+(cve|vulnerabilit(?:y|ies)|advisory|disclosure|exploit|poc|mitigation|patch)\b|\b(cve|vulnerabilit(?:y|ies)|advisory|disclosure|exploit|poc|mitigation|patch)\b(?:\W+\w+){0,4}\W+\b2026\b)",
        )
        .unwrap()
    })
}

impl TemporalDetector for AbstainingRegexDetector {
    fn detect(&self, query: &str) -> TemporalPolicy {
        let fires = keyword_re().is_match(query)
            || recent_plus_noun_re().is_match(query)
            || still_re().is_match(query)
            || as_of_re().is_match(query)
            || this_week_month_re().is_match(query)
            || year_2026_plus_noun_re().is_match(query);
        if fires {
            TemporalPolicy::FavorRecent(Strength::Medium)
        } else {
            TemporalPolicy::Off
        }
    }
}
```

- [ ] **Step 4: Run — expected PASS**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::regex_positive_tests`
Expected: all positive cases pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs
git commit -m "feat(temporal): abstaining regex detector — positive patterns"
```

---

### Task 6: `AbstainingRegexDetector` — negative cases (abstain contract)

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`

**Context:** These are the queries that must **not** fire the detector. Add them in a single test; fix the regex only when a case fails.

- [ ] **Step 1: Write negative-case test**

```rust
#[cfg(test)]
mod regex_negative_tests {
    use super::*;

    fn abstains(query: &str) -> bool {
        matches!(
            AbstainingRegexDetector::new().detect(query),
            TemporalPolicy::Off
        )
    }

    #[test]
    fn historical_queries_abstain() {
        assert!(abstains("describe CVE-2014-0160"));
        assert!(abstains("as of 2014 how did Shellshock work"));
        assert!(abstains("in 2021 what was Log4Shell"));
        assert!(abstains("back in 2017 the WannaCry worm"));
    }

    #[test]
    fn neutral_queries_abstain() {
        assert!(abstains("explain Kerberoasting"));
        assert!(abstains("what is NTLM relay"));
        assert!(abstains("how does ASLR work"));
    }

    #[test]
    fn bare_cve_identifier_abstains() {
        assert!(abstains("CVE-2026-0001"));
        assert!(abstains("CVE-2014-6271"));
    }

    #[test]
    fn bare_2026_abstains_without_noun() {
        assert!(abstains("port 2026 scan"));
        assert!(abstains("version 2026"));
        assert!(abstains("2026"));
    }

    #[test]
    fn false_friends_abstain() {
        // `current` in non-security sense
        assert!(abstains("current user guide"));
        assert!(abstains("current working directory"));
        // `latest` unrelated to freshness
        assert!(abstains("latest attempt to install X"));
        assert!(abstains("what is the latest episode about"));
        // bare `recent` without a noun
        assert!(abstains("recently I was thinking"));
        assert!(abstains("a recent commit"));
    }
}
```

- [ ] **Step 2: Run — most should already pass; fix any that don't**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::regex_negative_tests`

If `current user guide` or `current working directory` fires (likely, since `\bcurrent\b` matches): tighten `keyword_re` to require a security-ish context, OR accept this false-positive class and update the spec's R3 risk annotation. For v1, tighten to:

```rust
// Replace keyword_re() body:
Regex::new(
    r"(?i)\b(latest|newest|current(?:ly)?|newer)\b(?:\W+\w+){0,6}\W+(advisory|exploit|bypass|cve|disclosure|vulnerabilit(?:y|ies)|patch|guidance|kev|mitigation|poc)\b|\b(still|as\s+of)\b",
)
```

Actually — the simpler fix is to mirror `recent_plus_noun_re`: require a security noun within 6 tokens of the keyword. Replace `keyword_re` with:

```rust
fn keyword_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(latest|newest|current(?:ly)?|newer)\b(?:\W+\w+){0,6}\W+(advisory|exploit|bypass|cve|disclosure|vulnerabilit(?:y|ies)|patch|guidance|kev|mitigation|poc)\b",
        )
        .unwrap()
    })
}
```

Note: this also requires updating Task 5's positive-case test. `current mitigation for Shellshock`, `newer bypass for PrintNightmare`, `latest Log4j advisory`, `newest CVE in KEV` all still match (mitigation/bypass/advisory/CVE are listed nouns). If any positive case regresses, expand the noun list to cover it.

- [ ] **Step 3: Iterate until all tests green**

Run positive + negative together:
`cargo test -p fastrag --features hybrid --lib corpus::temporal::regex`
Expected: every `regex_positive_tests` and `regex_negative_tests` case passes.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs
git commit -m "feat(temporal): abstaining detector — tighten to require security-noun context"
```

---

### Task 7: `apply_temporal_policy` wrapper

**Files:**
- Modify: `crates/fastrag/src/corpus/temporal.rs`
- Modify: `crates/fastrag/src/corpus/hybrid.rs` (may need to expose `TemporalOpts::new_with` helper if constructors aren't public — check first)

**Context:** This is the function that the orchestrator (Task 9) will call after rerank. It takes the final rerank output, a policy, the query string, a detector, and per-doc dates, and returns re-scored results. For `Auto`, it delegates to the detector; for `Off`, it returns input unchanged; for `FavorRecent(s)`, it builds a `TemporalOpts` from `s`'s constants and calls the existing `apply_decay`.

The `ScoredHit` shape used by the rerank orchestrator is different from `ScoredId` (hybrid's internal type). Define a narrow helper that works with a slice of `(id, score)` pairs + `dates: &[Option<NaiveDate>]` and returns re-scored pairs. That keeps `apply_temporal_policy` independent of the orchestrator's hit DTO.

- [ ] **Step 1: Write failing test**

Append:

```rust
#[cfg(test)]
mod apply_policy_tests {
    use super::*;
    use chrono::{NaiveDate, TimeZone, Utc};

    fn input() -> Vec<(u64, f32)> {
        vec![(1, 0.9), (2, 0.5), (3, 0.3)]
    }

    fn dates() -> Vec<Option<NaiveDate>> {
        vec![
            Some(NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()),
            Some(NaiveDate::from_ymd_opt(2021, 1, 1).unwrap()),
            None,
        ]
    }

    #[test]
    fn off_policy_is_identity() {
        let det = AbstainingRegexDetector::new();
        let now = Utc.with_ymd_and_hms(2026, 4, 16, 0, 0, 0).unwrap();
        let out = apply_temporal_policy(
            &input(),
            &TemporalPolicy::Off,
            "anything",
            &det,
            &dates(),
            now,
        );
        assert_eq!(out, input());
    }

    #[test]
    fn auto_abstains_to_identity_for_neutral_query() {
        let det = AbstainingRegexDetector::new();
        let now = Utc.with_ymd_and_hms(2026, 4, 16, 0, 0, 0).unwrap();
        let out = apply_temporal_policy(
            &input(),
            &TemporalPolicy::Auto,
            "describe kerberoasting",
            &det,
            &dates(),
            now,
        );
        assert_eq!(out, input());
    }

    #[test]
    fn favor_recent_medium_decays_older_doc() {
        let det = AbstainingRegexDetector::new();
        let now = Utc.with_ymd_and_hms(2026, 4, 16, 0, 0, 0).unwrap();
        let out = apply_temporal_policy(
            &input(),
            &TemporalPolicy::FavorRecent(Strength::Medium),
            "anything",
            &det,
            &dates(),
            now,
        );
        // id=1 (fresh, Apr 2026) must keep its score essentially unchanged
        // id=2 (Jan 2021, ~5y old with 6mo halflife) must be pushed close to
        //      the weight_floor 0.60 multiplier.
        let s1 = out.iter().find(|(i, _)| *i == 1).unwrap().1;
        let s2 = out.iter().find(|(i, _)| *i == 2).unwrap().1;
        assert!(s1 > s2, "fresh doc must outrank old doc after decay");
        assert!(
            (s1 - 0.9).abs() < 0.05,
            "fresh doc retains near-original score; got {s1}"
        );
        assert!(
            s2 <= 0.5 * 0.60 + 1e-3,
            "old doc must clamp at or below weight_floor * orig; got {s2}"
        );
    }

    #[test]
    fn auto_fires_for_recency_query() {
        let det = AbstainingRegexDetector::new();
        let now = Utc.with_ymd_and_hms(2026, 4, 16, 0, 0, 0).unwrap();
        let out = apply_temporal_policy(
            &input(),
            &TemporalPolicy::Auto,
            "latest Log4j advisory",
            &det,
            &dates(),
            now,
        );
        // Detector fires → FavorRecent(Medium) → same decay as explicit.
        let s2 = out.iter().find(|(i, _)| *i == 2).unwrap().1;
        assert!(s2 < 0.5, "recency-intent query should decay old doc");
    }
}
```

- [ ] **Step 2: Run — expected FAIL**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::apply_policy_tests`
Expected: fails — `apply_temporal_policy` undefined.

- [ ] **Step 3: Implement**

Add to `temporal.rs`:

```rust
use chrono::{DateTime, NaiveDate, Utc};

use crate::corpus::hybrid::{BlendMode, ScoredId, TemporalOpts, apply_decay};

/// Resolve `TemporalPolicy` for one query and apply decay to `results`.
///
/// `results` is a slice of `(doc_id, score)` from the reranker. `dates`
/// must be index-aligned with `results`. `detector` is consulted only when
/// `policy == Auto`.
///
/// Returns a new `Vec<(u64, f32)>` sorted by descending score.
pub fn apply_temporal_policy(
    results: &[(u64, f32)],
    policy: &TemporalPolicy,
    query: &str,
    detector: &dyn TemporalDetector,
    dates: &[Option<NaiveDate>],
    now: DateTime<Utc>,
) -> Vec<(u64, f32)> {
    let effective = match policy {
        TemporalPolicy::Auto => detector.detect(query),
        other => *other,
    };
    let strength = match effective {
        TemporalPolicy::FavorRecent(s) => s,
        _ => return results.to_vec(), // Off or Auto-abstained
    };

    let opts = TemporalOpts {
        date_fields: vec![], // coalesce is upstream; not used here.
        halflife: strength.halflife(),
        weight_floor: strength.weight_floor(),
        dateless_prior: 1.0,
        blend: BlendMode::Multiplicative,
        now,
    };

    let fused: Vec<ScoredId> = results
        .iter()
        .map(|(id, score)| ScoredId {
            id: *id,
            score: *score,
        })
        .collect();

    let decayed = apply_decay(&fused, dates, &opts);
    decayed.into_iter().map(|s| (s.id, s.score)).collect()
}
```

You will likely need to add `pub` to `ScoredId` if it isn't already exported; check `crates/fastrag/src/corpus/hybrid.rs`. If `ScoredId`'s fields aren't all pub, construct via its existing constructor or expose one.

- [ ] **Step 4: Run — expected PASS**

Run: `cargo test -p fastrag --features hybrid --lib corpus::temporal::apply_policy_tests`
Expected: 4/4 pass.

- [ ] **Step 5: Run full lint gate**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
cargo fmt --check
```

Fix any issues. Re-run.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/temporal.rs crates/fastrag/src/corpus/hybrid.rs
git commit -m "feat(temporal): apply_temporal_policy wrapper for late-stage decay"
```

**Landing 1 end state:** `temporal.rs` is a complete, tested, dead-code module. Nothing in the existing retrieval path calls it yet. Tree compiles clean, all pre-existing tests pass.

---

## Landing 2 — Late-stage injection + wire-up

Remove pre-rerank decay from `query_hybrid`, thread `TemporalPolicy` through the public surfaces (QueryOpts, HTTP, CLI), and call `apply_temporal_policy` at the post-rerank stage.

### Task 8: Remove pre-rerank decay branch from `query_hybrid`

**Files:**
- Modify: `crates/fastrag/src/corpus/hybrid.rs` (lines 648-664)

**Context:** The spec says _the reranker always sees the un-decayed candidate set_. This means `query_hybrid` stops applying decay entirely. The pure function `apply_decay` stays, since `apply_temporal_policy` still calls it.

The one-line delete is small but it breaks a few existing tests that rely on the pre-rerank decay behavior. Those tests move to `temporal.rs` or get replaced by Task 15's late-injection contract.

- [ ] **Step 1: Find callers of the decay branch**

Run: `grep -n "opts.temporal" crates/fastrag/src/corpus/hybrid.rs`
Run: `grep -rn "HybridOpts" crates/fastrag/ fastrag-cli/ crates/fastrag-eval/`

Note every call site that currently populates `opts.temporal = Some(...)`. These will all shift to `TemporalPolicy` in subsequent tasks. For Landing 2, leave `HybridOpts.temporal` as a field (so the struct shape stays stable) but make `query_hybrid` ignore it.

- [ ] **Step 2: Delete the decay branch**

In `crates/fastrag/src/corpus/hybrid.rs::query_hybrid` (~lines 648-664), replace:

```rust
    if let Some(temp) = &opts.temporal {
        let ids: Vec<u64> = fused.iter().map(|s| s.id).collect();
        let rows = store.fetch_metadata(&ids)?;
        let row_map: std::collections::HashMap<
            u64,
            Vec<(String, fastrag_store::schema::TypedValue)>,
        > = rows.into_iter().collect();
        let dates: Vec<Option<NaiveDate>> = fused
            .iter()
            .map(|s| {
                row_map
                    .get(&s.id)
                    .and_then(|f| extract_date_coalesce(f, &temp.date_fields))
            })
            .collect();
        fused = apply_decay(&fused, &dates, temp);
    }

    fused.truncate(top_k);
    Ok(fused)
```

with:

```rust
    // Temporal decay moved to the post-rerank stage in
    // `corpus/mod.rs::query_corpus_reranked_opts`. `HybridOpts.temporal`
    // is retained for the deprecated direct-flags code path but is a
    // no-op here — the reranker must see undecayed candidates.
    let _ = &opts.temporal;
    fused.truncate(top_k);
    Ok(fused)
```

- [ ] **Step 3: Update `temporal_option_runs_decay_branch` test**

The pre-existing test at `hybrid.rs::query_hybrid_tests::temporal_option_runs_decay_branch` (around line 748) currently asserts decay happens. Change the assertion to verify decay is a no-op inside `query_hybrid` — scores must match the no-decay baseline:

```rust
#[test]
fn temporal_option_is_ignored_in_query_hybrid() {
    // ... keep existing setup ...
    let opts_with = HybridOpts {
        enabled: true,
        rrf_k: 60,
        overfetch_factor: 3,
        temporal: Some(TemporalOpts { /* arbitrary */ }),
    };
    let opts_without = HybridOpts {
        enabled: true,
        rrf_k: 60,
        overfetch_factor: 3,
        temporal: None,
    };
    let a = query_hybrid(/* with opts_with */).unwrap();
    let b = query_hybrid(/* with opts_without */).unwrap();
    // Same rank order, same scores.
    assert_eq!(a, b, "temporal opt must no longer affect query_hybrid");
}
```

(Preserve the existing test setup harness; only the assertion and test name change.)

- [ ] **Step 4: Run hybrid tests**

Run: `cargo test -p fastrag --features hybrid --lib corpus::hybrid`
Run: `cargo test -p fastrag --features retrieval --test hybrid_retrieval`
Run: `cargo test -p fastrag --features retrieval --test temporal_decay`

**Expected temporary breakage:** `temporal_decay` integration test likely fails because the end-to-end flow no longer decays. Do **not** fix it here — Task 15 adds back the test coverage via late injection, and Task 11 (CLI wire-up) re-exercises the end-to-end path.

Mark failing tests with `#[ignore = "relies on pre-rerank decay path — reintroduced by late injection in Task 15"]` temporarily. Record them in a list at the top of `tests/temporal_decay.rs` so Task 15 un-ignores them.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/hybrid.rs crates/fastrag/tests/temporal_decay.rs
git commit -m "refactor(hybrid): strip pre-rerank decay — moving to post-rerank stage"
```

---

### Task 9: Add `TemporalPolicy` to `QueryOpts` and wire into `query_corpus_reranked_opts`

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (struct `QueryOpts` ~line 31, function `query_corpus_reranked_opts` ~line 1642)

**Context:** The rerank orchestrator gets a `dates` lookup and calls `apply_temporal_policy` after reranking but before returning. The detector is held on `QueryOpts` too — callers set a default `AbstainingRegexDetector` or pass in their own.

For v1 the detector is constructed inline in the orchestrator when `TemporalPolicy::Auto` is used. No need for a trait-object field on `QueryOpts` yet — that is a future optimization (the regex detector is stateless).

- [ ] **Step 1: Write integration test for late injection**

Create `crates/fastrag/tests/temporal_late_injection.rs`:

```rust
//! Contract: the reranker sees un-decayed candidates; decay is applied
//! only to the final post-rerank scores.

#![cfg(all(feature = "retrieval", feature = "rerank", feature = "hybrid"))]

use chrono::{TimeZone, Utc};
use fastrag::corpus::temporal::{Strength, TemporalPolicy};
// ... import needed fastrag_embed/rerank test doubles ...

#[test]
fn old_canonical_doc_survives_to_rerank() {
    // Setup: synthetic 3-doc corpus where doc A is 5y old lexically strong,
    // doc B is fresh lexically weak, doc C is fresh lexically strong.
    // With pre-rerank decay, A would be decay-suppressed below the over-fetch
    // cutoff. With post-rerank decay, A reaches rerank, gets semantically
    // promoted, and appears in the final top-k even with FavorRecent(Medium).

    // (concrete setup: use the same synthetic-corpus harness as
    //  crates/fastrag/tests/temporal_decay.rs — copy its `build_store` helper)

    // TODO: engineer fills this in using the test harness already present
    // in `tests/temporal_decay.rs`. The minimum:
    //   - Index three docs with published_date metadata
    //   - Run query_corpus_reranked_opts with TemporalPolicy::FavorRecent(Medium)
    //   - Assert A is in top-3 (i.e., survived rerank over-fetch)
    //   - Run same query with TemporalPolicy::Off
    //   - Assert A is also in top-3 (baseline)
}
```

Since the test harness involves real embedder/reranker plumbing that's idiosyncratic to this repo, **before writing this test, read `crates/fastrag/tests/temporal_decay.rs` in full** to copy its fixture builders. Mirror that structure.

- [ ] **Step 2: Run — expected FAIL**

Run: `cargo test -p fastrag --features retrieval,rerank,hybrid --test temporal_late_injection`
Expected: fails to compile (symbols not wired) or fails at assertion.

- [ ] **Step 3: Add `temporal_policy` to `QueryOpts`**

In `crates/fastrag/src/corpus/mod.rs`:

```rust
#[derive(Debug, Clone, Default)]
pub struct QueryOpts {
    pub cwe_expand: bool,
    pub hybrid: crate::corpus::hybrid::HybridOpts,
    /// Per-query temporal policy. Default is `Auto`, which routes through
    /// the abstaining regex detector (returning `Off` when no recency
    /// signal is detected).
    pub temporal_policy: crate::corpus::temporal::TemporalPolicy,
    /// Ordered list of metadata field names to try for per-doc dates.
    /// Empty means no decay is attempted even when policy is non-Off.
    pub date_fields: Vec<String>,
}
```

- [ ] **Step 4: Wire late injection into `query_corpus_reranked_opts`**

Edit `query_corpus_reranked_opts` in `corpus/mod.rs` (~line 1642). After the `reranked.truncate(top_k)` line but before `breakdown.finalize()`, add:

```rust
    // ── Late-stage temporal decay ──────────────────────────────────────
    if !opts.date_fields.is_empty()
        && !matches!(opts.temporal_policy, crate::corpus::temporal::TemporalPolicy::Off)
    {
        use crate::corpus::temporal::{
            AbstainingRegexDetector, TemporalDetector, TemporalPolicy, apply_temporal_policy,
        };

        // Look up dates for surviving doc ids from the store metadata.
        // Reuse `hybrid::extract_date_coalesce` — make it `pub(crate)` if
        // it isn't already. Assumes dates live in the corpus DB; we read
        // them lazily for the top-k that survived rerank.
        let ids: Vec<u64> = reranked.iter().map(|rh| rh.id).collect();
        let dates = crate::corpus::hybrid::dates_for_ids(
            corpus_dir,
            &ids,
            &opts.date_fields,
        )
        .map_err(|e| CorpusError::Internal(format!("date lookup: {e}")))?;

        let pairs: Vec<(u64, f32)> = reranked
            .iter()
            .map(|rh| (rh.id, rh.score))
            .collect();

        let detector = AbstainingRegexDetector::new();
        let decayed = apply_temporal_policy(
            &pairs,
            &opts.temporal_policy,
            query,
            &detector,
            &dates,
            chrono::Utc::now(),
        );

        // Re-sort reranked by the decayed scores.
        let mut score_by_id: std::collections::HashMap<u64, f32> =
            decayed.into_iter().collect();
        for rh in reranked.iter_mut() {
            if let Some(s) = score_by_id.remove(&rh.id) {
                rh.score = s;
            }
        }
        reranked.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    }
```

You will need to add `dates_for_ids` to `hybrid.rs` — a small helper that opens the corpus store and runs the existing `extract_date_coalesce` logic against a specific id list. Hoist the existing in-function logic (currently inline in `query_hybrid`) into this helper. Signature:

```rust
#[cfg(feature = "store")]
pub fn dates_for_ids(
    corpus_dir: &std::path::Path,
    ids: &[u64],
    date_fields: &[String],
) -> Result<Vec<Option<chrono::NaiveDate>>, crate::corpus::CorpusError> { /* ... */ }
```

Open the store inside the helper (or accept a `&Store` parameter — whichever matches existing patterns; check the orchestrator's current store lifetime).

- [ ] **Step 5: Run integration test**

Run: `cargo test -p fastrag --features retrieval,rerank,hybrid --test temporal_late_injection`
Expected: PASS.

- [ ] **Step 6: Un-ignore temporal_decay tests**

The `#[ignore]` markers added in Task 8 should now pass under the new flow. Remove them one-by-one, running the test after each. Fix any that still fail by adjusting policy construction in the test (switch from direct `TemporalOpts` to `temporal_policy: TemporalPolicy::FavorRecent(Strength::Medium) + date_fields`).

Run: `cargo test -p fastrag --features retrieval --test temporal_decay`
Expected: all un-ignored now pass.

- [ ] **Step 7: Lint gate**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval -- -D warnings
cargo fmt --check
```

- [ ] **Step 8: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs crates/fastrag/src/corpus/hybrid.rs crates/fastrag/tests/temporal_late_injection.rs crates/fastrag/tests/temporal_decay.rs
git commit -m "feat(corpus): late-stage decay injection post-rerank"
```

---

### Task 10: HTTP `/query` body accepts `temporal_policy`

**Files:**
- Modify: the HTTP query handler (locate with `rg 'fn.*query' fastrag-cli/src/` or `rg 'POST.*query' fastrag-cli/src/`)
- Test: add `crates/fastrag-cli/tests/temporal_policy_http_e2e.rs`

- [ ] **Step 1: Locate the HTTP request struct**

Run: `rg 'temporal|time_decay' fastrag-cli/src/`
Identify the serde struct that represents the request body. It likely has fields like `query`, `top_k`, `time_decay_halflife`, etc.

- [ ] **Step 2: Write failing HTTP integration test**

```rust
// crates/fastrag-cli/tests/temporal_policy_http_e2e.rs
#![cfg(feature = "retrieval")]

//! POST /query with temporal_policy in the body.
//!
//! Uses the same corpus fixture as temporal_decay_http_e2e.

// Minimal shape: copy setup from tests/temporal_decay_http_e2e.rs,
// swap the body to:
//   { "query": "latest Log4j advisory",
//     "top_k": 5,
//     "date_fields": ["published_date"],
//     "temporal_policy": {"favor_recent": "medium"} }
// Assert response returns HTTP 200, returns the 2026 doc above the 2021 doc.
```

Run: `cargo test -p fastrag-cli --features retrieval --test temporal_policy_http_e2e`
Expected: FAIL — field not yet accepted.

- [ ] **Step 3: Add `temporal_policy` and `date_fields` to the request struct**

```rust
#[derive(Debug, Deserialize)]
struct QueryBody {
    // ... existing fields ...

    /// Optional per-query temporal policy. Default `auto` (abstaining detector).
    #[serde(default)]
    temporal_policy: fastrag::corpus::temporal::TemporalPolicy,

    /// Ordered list of date metadata fields. Empty disables decay.
    #[serde(default)]
    date_fields: Vec<String>,
}
```

Pass both into the `QueryOpts` constructed inside the handler.

- [ ] **Step 4: Run — expected PASS**

Run: `cargo test -p fastrag-cli --features retrieval --test temporal_policy_http_e2e`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fastrag-cli/ crates/fastrag-cli/
git commit -m "feat(http): accept temporal_policy in POST /query body"
```

---

### Task 11: CLI `--temporal-policy` flag + deprecation warnings for old flags

**Files:**
- Modify: `fastrag-cli/src/args.rs` (locate the `query` subcommand args struct)
- Modify: wherever args → `QueryOpts` wiring happens (same file or neighbor)
- Test: `crates/fastrag-cli/tests/temporal_policy_e2e.rs`

- [ ] **Step 1: Write failing CLI integration test**

```rust
// crates/fastrag-cli/tests/temporal_policy_e2e.rs
#![cfg(feature = "retrieval")]

use std::process::Command;

#[test]
fn query_with_temporal_policy_favor_recent_medium() {
    // Use the prebuilt `target/release/fastrag` binary or `cargo run`.
    // Use the same fixture corpus that temporal_decay_e2e.rs uses.
    // Assert: `fastrag query "latest Log4j advisory" --corpus <dir> \
    //            --temporal-policy favor-recent-medium \
    //            --time-decay-field published_date --top-k 3`
    // returns 2026 doc first.
}

#[test]
fn deprecated_halflife_flag_emits_stderr_warning() {
    // Run with `--time-decay-halflife 90d` but no --temporal-policy.
    // Assert exit code 0 AND stderr contains "deprecated".
}

#[test]
fn temporal_policy_auto_abstains_on_historical_query() {
    // Same corpus. Query: "describe CVE-2014-0160".
    // With --temporal-policy auto (default) + --time-decay-field set,
    // top hit matches no-decay behavior (the 2014 doc wins).
}
```

Copy fixture setup from `crates/fastrag-cli/tests/temporal_decay_e2e.rs`.

Run: `cargo test -p fastrag-cli --features retrieval --test temporal_policy_e2e`
Expected: FAIL — flag undefined.

- [ ] **Step 2: Add `--temporal-policy` to the `query` subcommand**

In `fastrag-cli/src/args.rs`, under the `Query` subcommand struct, add:

```rust
/// Per-query temporal policy: auto (default), off, favor-recent-light,
/// favor-recent-medium, favor-recent-strong.
///
/// `auto` routes through the abstaining recency detector — neutral and
/// historical queries receive no decay. Explicit `favor-recent-*` values
/// override the detector.
#[arg(long, value_enum, default_value = "auto")]
temporal_policy: TemporalPolicyCli,
```

And next to existing enums:

```rust
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TemporalPolicyCli {
    Auto,
    Off,
    FavorRecentLight,
    FavorRecentMedium,
    FavorRecentStrong,
}

impl From<TemporalPolicyCli> for fastrag::corpus::temporal::TemporalPolicy {
    fn from(c: TemporalPolicyCli) -> Self {
        use fastrag::corpus::temporal::{Strength, TemporalPolicy};
        match c {
            TemporalPolicyCli::Auto => TemporalPolicy::Auto,
            TemporalPolicyCli::Off => TemporalPolicy::Off,
            TemporalPolicyCli::FavorRecentLight => {
                TemporalPolicy::FavorRecent(Strength::Light)
            }
            TemporalPolicyCli::FavorRecentMedium => {
                TemporalPolicy::FavorRecent(Strength::Medium)
            }
            TemporalPolicyCli::FavorRecentStrong => {
                TemporalPolicy::FavorRecent(Strength::Strong)
            }
        }
    }
}
```

Wire it into whatever builds `QueryOpts`:

```rust
let opts = QueryOpts {
    temporal_policy: args.temporal_policy.into(),
    date_fields: args.time_decay_field.clone(),
    hybrid: existing_hybrid_opts(&args),
    cwe_expand: args.cwe_expand,
};
```

- [ ] **Step 3: Add deprecation warning for legacy flags**

In the args → opts wiring function, after building `opts`:

```rust
// Deprecation warning: legacy --time-decay-* flags are still parsed
// but take effect only when --temporal-policy is non-auto. Emit a
// single-line stderr warning when any are set with auto policy.
let legacy_flags_set = args.time_decay_halflife.is_some()
    || args.time_decay_weight.is_some()
    || args.time_decay_blend.is_some()
    || args.time_decay_dateless_prior.is_some();
if legacy_flags_set
    && matches!(args.temporal_policy, TemporalPolicyCli::Auto)
{
    eprintln!(
        "warning: --time-decay-halflife/--time-decay-weight/\
         --time-decay-blend/--time-decay-dateless-prior are deprecated \
         and ignored when --temporal-policy=auto (default). Use \
         --temporal-policy=favor-recent-{{light|medium|strong}} instead."
    );
}
```

Check the current types of those fields — they may be `Option<String>` or `Option<Duration>`; adjust `.is_some()` as needed.

- [ ] **Step 4: Run — expected PASS**

Run: `cargo test -p fastrag-cli --features retrieval --test temporal_policy_e2e`
Expected: PASS, including the stderr-warning test.

Also re-run the legacy CLI e2e to make sure backwards compat is preserved where policy is non-auto:
Run: `cargo test -p fastrag-cli --features retrieval --test temporal_decay_e2e`
Expected: PASS (these tests may need updating if they previously relied on legacy flags alone to activate decay — switch them to the new flag).

- [ ] **Step 5: Apply same flag to `serve-http` subcommand**

If the HTTP server takes default decay flags at boot (check `fastrag-cli/src/args.rs` for the `ServeHttp` subcommand), add the same `--temporal-policy` flag there to set the default when a request omits `temporal_policy`.

- [ ] **Step 6: Lint + commit**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval -- -D warnings
cargo fmt --check

git add fastrag-cli/ crates/fastrag-cli/
git commit -m "feat(cli): --temporal-policy flag with legacy-flag deprecation warning"
```

**Landing 2 end state:** Rust API, HTTP, and CLI all speak `TemporalPolicy`. Legacy flags emit warnings but still function. Integration tests cover the abstain-on-historical contract and the late-injection survival contract.

---

## Landing 3 — Eval matrix swap + baseline recapture

Replace `ConfigVariant::TemporalOn` with `TemporalAuto`; add `TemporalOracle` (opt-in); add `route_regret`; recapture baseline; update gate assertions.

### Task 12: Retire `TemporalOn`, add `TemporalAuto`

**Files:**
- Modify: `crates/fastrag-eval/src/matrix.rs` (ConfigVariant enum, all/from_label/label methods, matrix test)
- Modify: `crates/fastrag-eval/src/matrix_real.rs` (TemporalOn branch → TemporalAuto)

- [ ] **Step 1: Update ConfigVariant enum tests**

In `crates/fastrag-eval/src/matrix.rs::tests`:

```rust
#[test]
fn all_variants_ordered() {
    let all = ConfigVariant::all();
    assert_eq!(all.len(), 5);
    assert_eq!(all[0], ConfigVariant::Primary);
    assert_eq!(all[1], ConfigVariant::NoRerank);
    assert_eq!(all[2], ConfigVariant::NoContextual);
    assert_eq!(all[3], ConfigVariant::DenseOnly);
    assert_eq!(all[4], ConfigVariant::TemporalAuto);
}

#[test]
fn labels_stable() {
    assert_eq!(ConfigVariant::Primary.label(), "primary");
    assert_eq!(ConfigVariant::NoRerank.label(), "no_rerank");
    assert_eq!(ConfigVariant::NoContextual.label(), "no_contextual");
    assert_eq!(ConfigVariant::DenseOnly.label(), "dense_only");
    assert_eq!(ConfigVariant::TemporalAuto.label(), "temporal_auto");
}

#[test]
fn temporal_on_label_still_parses_for_baseline_back_compat() {
    // Old baselines use "temporal_on" — reader should parse it as a
    // no-longer-canonical variant. Strategy: map it to TemporalAuto so
    // baseline diffs don't fatally break; flag on stderr via the loader.
    // Alternatively: reject it with a clear error. Decision: reject,
    // require baseline recapture. See Task 15.
    assert_eq!(ConfigVariant::from_label("temporal_on"), None);
    assert_eq!(
        ConfigVariant::from_label("temporal_auto"),
        Some(ConfigVariant::TemporalAuto)
    );
}
```

Run: `cargo test -p fastrag-eval --features real-driver matrix::tests`
Expected: FAIL — `TemporalAuto` doesn't exist.

- [ ] **Step 2: Rename variant in matrix.rs**

In `crates/fastrag-eval/src/matrix.rs`:

```rust
pub enum ConfigVariant {
    Primary,
    NoRerank,
    NoContextual,
    DenseOnly,
    TemporalAuto,
}

impl ConfigVariant {
    pub fn all() -> [ConfigVariant; 5] {
        [
            ConfigVariant::Primary,
            ConfigVariant::NoRerank,
            ConfigVariant::NoContextual,
            ConfigVariant::DenseOnly,
            ConfigVariant::TemporalAuto,
        ]
    }

    pub fn from_label(s: &str) -> Option<ConfigVariant> {
        match s {
            "primary" => Some(ConfigVariant::Primary),
            "no_rerank" => Some(ConfigVariant::NoRerank),
            "no_contextual" => Some(ConfigVariant::NoContextual),
            "dense_only" => Some(ConfigVariant::DenseOnly),
            "temporal_auto" => Some(ConfigVariant::TemporalAuto),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfigVariant::Primary => "primary",
            ConfigVariant::NoRerank => "no_rerank",
            ConfigVariant::NoContextual => "no_contextual",
            ConfigVariant::DenseOnly => "dense_only",
            ConfigVariant::TemporalAuto => "temporal_auto",
        }
    }
}
```

- [ ] **Step 3: Update `matrix_real.rs` dispatch**

In `crates/fastrag-eval/src/matrix_real.rs`, replace the `ConfigVariant::TemporalOn` match arm (around line 114). The new arm runs plain hybrid (no pre-rerank decay) and then applies `apply_temporal_policy` post-rerank:

```rust
ConfigVariant::TemporalAuto => {
    // Plain hybrid — pre-rerank decay is gone.
    let opts = HybridOpts {
        enabled: true,
        rrf_k: 60,
        overfetch_factor: 3,
        temporal: None,
    };
    let fused = query_hybrid(store, question, query_vector, over_fetch, &opts, breakdown)
        .map_err(|e| EvalError::Runner(format!("hybrid search: {e}")))?;
    fused.into_iter().map(|s| (s.id, s.score)).collect()
}
```

And at the post-rerank stage (after the existing `ordered_texts` assembly), for `TemporalAuto` we still need to apply `apply_temporal_policy` against the reranked `(id, score)` pairs. Since this driver's rerank path currently drops ids after sort, refactor the rerank block to keep `(id, score)` pairs, apply policy, then map to texts:

```rust
// Inside the `needs_rerank` branch, replace the final sort + truncate block:
let mut pairs_with_text: Vec<(u64, String, f32)> = /* merged cached + miss outputs */;
pairs_with_text.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

// Apply TemporalAuto late injection:
if matches!(variant, ConfigVariant::TemporalAuto) {
    use fastrag::corpus::temporal::{AbstainingRegexDetector, TemporalPolicy, apply_temporal_policy};
    let pairs: Vec<(u64, f32)> = pairs_with_text.iter().map(|(id, _, s)| (*id, *s)).collect();
    let ids: Vec<u64> = pairs.iter().map(|(id, _)| *id).collect();
    let dates = fastrag::corpus::hybrid::dates_for_ids(
        /* ctx_corpus_path */ &self.ctx_corpus_path,
        &ids,
        &["published_date".to_string()],
    ).unwrap_or_else(|_| vec![None; ids.len()]);

    let det = AbstainingRegexDetector::new();
    let decayed = apply_temporal_policy(
        &pairs,
        &TemporalPolicy::Auto,
        question,
        &det,
        &dates,
        Utc::now(),
    );
    let score_by_id: std::collections::HashMap<u64, f32> =
        decayed.into_iter().collect();
    for triple in pairs_with_text.iter_mut() {
        if let Some(&s) = score_by_id.get(&triple.0) {
            triple.2 = s;
        }
    }
    pairs_with_text.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
}

let ordered_texts: Vec<String> =
    pairs_with_text.into_iter().map(|(_, t, _)| t).take(top_k).collect();
```

(The `RealCorpusDriver` currently doesn't expose a path; you will need to add a `ctx_corpus_path: PathBuf` field at construction time so the dates lookup can open the corpus dir. Check what `RealCorpusDriver::load` already has — it may already hold `ctx_corpus` path implicitly via `Store::open`; if so, pass the path explicitly.)

- [ ] **Step 4: Run matrix tests**

```bash
cargo test -p fastrag-eval --features real-driver matrix
cargo test -p fastrag-eval --test matrix_stub
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-eval/
git commit -m "refactor(eval): retire TemporalOn, introduce TemporalAuto late-injection variant"
```

---

### Task 13: Add `TemporalOracle` + opt-in `--variants` flag

**Files:**
- Modify: `crates/fastrag-eval/src/matrix.rs` (add `TemporalOracle` variant — NOT in `all()`)
- Modify: `crates/fastrag-eval/src/matrix_real.rs` (add oracle dispatch)
- Modify: `fastrag-cli/src/args.rs` (Eval subcommand: `--variants` flag)
- Modify: gold-set plumbing to pass `temporal_intent` into the driver

- [ ] **Step 1: Extend `ConfigVariant` with `TemporalOracle`**

In `crates/fastrag-eval/src/matrix.rs`:

```rust
pub enum ConfigVariant {
    Primary,
    NoRerank,
    NoContextual,
    DenseOnly,
    TemporalAuto,
    /// Eval-only: routes via gold-set `axes.temporal_intent` instead of
    /// the regex detector. Serves as upper bound on regex quality.
    TemporalOracle,
}
```

Keep `ConfigVariant::all()` at 5 variants — `TemporalOracle` is not in the default set. Update `from_label` / `label` to include `"temporal_oracle"`.

Add a constructor:

```rust
impl ConfigVariant {
    pub fn canonical_plus_oracle() -> [ConfigVariant; 6] {
        [
            ConfigVariant::Primary,
            ConfigVariant::NoRerank,
            ConfigVariant::NoContextual,
            ConfigVariant::DenseOnly,
            ConfigVariant::TemporalAuto,
            ConfigVariant::TemporalOracle,
        ]
    }
}
```

- [ ] **Step 2: Test the new label round-trip**

```rust
#[test]
fn oracle_label_round_trips() {
    assert_eq!(
        ConfigVariant::from_label("temporal_oracle"),
        Some(ConfigVariant::TemporalOracle)
    );
    assert_eq!(ConfigVariant::TemporalOracle.label(), "temporal_oracle");
}

#[test]
fn oracle_not_in_default_all() {
    for v in ConfigVariant::all() {
        assert_ne!(v, ConfigVariant::TemporalOracle);
    }
}
```

Run: `cargo test -p fastrag-eval --features real-driver matrix`
Expected: PASS after enum update.

- [ ] **Step 3: Thread `temporal_intent` into driver queries**

The driver's `query` signature takes `question: &str`. Oracle needs the per-entry intent. Two options:

(a) Add a new method `fn query_with_intent(..., intent: Option<TemporalIntent>) -> Result<...>` with a default impl that ignores intent and calls `query`.
(b) Pre-resolve the intent in the driver by holding a `HashMap<String, TemporalIntent>` keyed on question text.

**Pick (a) — cleaner.** In `CorpusDriver`:

```rust
fn query(
    &self,
    variant: ConfigVariant,
    question: &str,
    query_vector: &[f32],
    top_k: usize,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<String>, EvalError> {
    // Default: ignore intent.
    self.query_with_intent(variant, question, query_vector, top_k, None, breakdown)
}

fn query_with_intent(
    &self,
    variant: ConfigVariant,
    question: &str,
    query_vector: &[f32],
    top_k: usize,
    intent: Option<fastrag::corpus::temporal::TemporalIntent>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<String>, EvalError>;
```

Or simpler: change the `run_matrix` loop to resolve the per-entry intent and pass it via a new parameter on `query`. Do not keep both APIs — clap a single `query(..., intent: Option<TemporalIntent>)` and update both `RealCorpusDriver` and the stub driver.

- [ ] **Step 4: Implement `TemporalOracle` branch in `matrix_real.rs`**

Same as `TemporalAuto` except use `OracleDetector::new(intent)` as the detector, and pass `TemporalPolicy::Auto` so the oracle's `detect()` is invoked. OR just resolve the policy up-front:

```rust
ConfigVariant::TemporalOracle => {
    // ... identical retrieval as TemporalAuto ...
}
```

Then in the post-rerank late-injection block:

```rust
let policy = match variant {
    ConfigVariant::TemporalAuto => TemporalPolicy::Auto,
    ConfigVariant::TemporalOracle => {
        // Oracle resolves policy directly from the intent.
        let det = OracleDetector::new(intent);
        det.detect(question)
    }
    _ => return /* no decay */,
};
// call apply_temporal_policy with the regex detector — when policy is
// already non-Auto, the detector is not consulted.
```

- [ ] **Step 5: Update `run_matrix` to pass per-entry `intent` into `query`**

Around line 244 (`for (qi, (entry, vector)) in gold_set.entries.iter().zip(&vectors).enumerate()`), compute:

```rust
let intent: Option<fastrag::corpus::temporal::TemporalIntent> =
    entry.axes.as_ref().and_then(|a| a.temporal_intent).map(|ti| {
        // Map from fastrag-eval's axes type to the core crate's type.
        // They should have matching discriminants.
        match ti {
            crate::axes::TemporalIntent::Historical => fastrag::corpus::temporal::TemporalIntent::Historical,
            crate::axes::TemporalIntent::Neutral => fastrag::corpus::temporal::TemporalIntent::Neutral,
            crate::axes::TemporalIntent::RecencySeeking => fastrag::corpus::temporal::TemporalIntent::RecencySeeking,
        }
    });
let result = driver.query(variant, &entry.question, vector, top_k, intent, breakdown)?;
```

Check `crates/fastrag-eval/src/axes.rs` (or wherever the axes types live) to match the mapping.

- [ ] **Step 6: Add `--variants` CLI flag**

In `fastrag-cli/src/args.rs`, under the `eval` subcommand:

```rust
/// Comma-separated list of variant labels to run. Defaults to the
/// canonical 5-variant set. Pass `temporal_oracle` to opt into oracle
/// diagnostics.
///
/// Examples:
///   --variants primary,temporal_auto
///   --variants primary,temporal_auto,temporal_oracle
#[arg(long, value_delimiter = ',')]
variants: Option<Vec<String>>,
```

Wire it into `run_matrix`:

```rust
let variants: Option<Vec<ConfigVariant>> = args.variants.as_ref().map(|list| {
    list.iter()
        .filter_map(|s| ConfigVariant::from_label(s.as_str()))
        .collect()
});
run_matrix(&driver, &gold_set, top_k, variants.as_deref())
```

- [ ] **Step 7: Test — run a canonical matrix + an opt-in oracle matrix on stub driver**

Run: `cargo test -p fastrag-eval --test matrix_stub`
Expected: PASS. Add a stub-driver test that runs `TemporalOracle` + `TemporalAuto` and asserts both produce results.

- [ ] **Step 8: Commit**

```bash
git add crates/fastrag-eval/ fastrag-cli/
git commit -m "feat(eval): TemporalOracle variant + --variants opt-in flag"
```

---

### Task 14: `route_regret` metric

**Files:**
- Modify: `crates/fastrag-eval/src/matrix.rs` (MatrixReport struct, summary build)
- Modify: test assertions

- [ ] **Step 1: Write failing test**

In `matrix.rs::tests` or a sibling test module:

```rust
#[test]
fn route_regret_populated_when_both_variants_present() {
    let mut report = MatrixReport { /* ... mock with TemporalAuto and TemporalOracle ... */ };
    // set auto.buckets.temporal_intent.recency_seeking.mrr_at_10 = 0.5
    // set oracle.buckets.temporal_intent.recency_seeking.mrr_at_10 = 0.7

    report.populate_summary();

    assert_eq!(
        report.summary.route_regret,
        Some(0.7_f64 - 0.5_f64)
    );
}

#[test]
fn route_regret_absent_when_oracle_missing() {
    let mut report = MatrixReport { /* ... mock with only TemporalAuto ... */ };
    report.populate_summary();
    assert_eq!(report.summary.route_regret, None);
}
```

Run: `cargo test -p fastrag-eval matrix::tests::route_regret`
Expected: FAIL.

- [ ] **Step 2: Extend `MatrixReport` with a `summary` field**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MatrixSummary {
    /// oracle(recency_seeking).mrr@10 − auto(recency_seeking).mrr@10,
    /// computed at build time if both variants ran. None otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_regret: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixReport {
    // ... existing fields ...
    #[serde(default)]
    pub summary: MatrixSummary,
}
```

Add `populate_summary`:

```rust
impl MatrixReport {
    pub fn populate_summary(&mut self) {
        let lookup = |v: ConfigVariant| -> Option<f64> {
            self.runs
                .iter()
                .find(|r| r.variant == v)
                .and_then(|r| r.buckets.get("temporal_intent"))
                .and_then(|b| b.get("recency_seeking"))
                .map(|m| m.mrr_at_10)
        };
        let auto = lookup(ConfigVariant::TemporalAuto);
        let oracle = lookup(ConfigVariant::TemporalOracle);
        self.summary.route_regret = match (auto, oracle) {
            (Some(a), Some(o)) => Some(o - a),
            _ => None,
        };
    }
}
```

Call `populate_summary()` once at the end of `run_matrix` before returning.

- [ ] **Step 3: Run — expected PASS**

Run: `cargo test -p fastrag-eval matrix::tests::route_regret`
Expected: 2/2 pass.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag-eval/
git commit -m "feat(eval): add route_regret metric (oracle mrr@10 − auto mrr@10)"
```

---

### Task 15: Baseline recapture + gate assertions

**Files:**
- Modify: `docs/eval-baselines/current.json` (regenerated, not hand-edited)
- Modify: the baseline-diff gate module (locate with `rg 'per_bucket_slack\|baseline' crates/fastrag-eval/src/`)

**Context:** The new baseline must contain the 5 canonical variants (Primary/NoRerank/NoContextual/DenseOnly/**TemporalAuto**). The old baseline with `temporal_on` is no longer parseable after Task 12 — so baseline recapture is strictly required to keep CI green.

- [ ] **Step 1: Re-capture the baseline**

**Preflight:** confirm `/tmp/eval-ctx` and `/tmp/eval-raw` still have the expanded corpus from the 2026-04-16 01:00 capture. If they've been purged, rebuild per `docs/eval-baselines/README.md`.

Run:

```bash
target/release/fastrag eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/eval-ctx \
  --corpus-no-contextual /tmp/eval-raw \
  --config-matrix \
  --report docs/eval-baselines/current.json
```

(Use `target/release/fastrag` not `cargo run` — binary ambiguity bites here.)

This writes a new baseline using the 5 canonical variants. The run takes ~2h against the 185-question gold set. **Kick this off in a screen/tmux and move on to Step 2 (gate-code edits).**

- [ ] **Step 2: Add new per-bucket gate assertions**

Locate the existing gate check (try `rg 'per_bucket_slack' crates/fastrag-eval/src/`). Add four new assertions into the `--baseline` enforcement path. Pseudocode — adapt to the existing gate style:

```rust
// New-baseline gates (in addition to existing per-bucket slack):
//
// 1. temporal_auto historical must not regress Primary by >2pp hit@5.
// 2. temporal_auto neutral must not regress Primary by >1pp hit@5.
// 3. temporal_auto recency_seeking mrr@10 must not regress Primary's.
// 4. temporal_oracle recency_seeking mrr@10 ≥ temporal_auto's (if both present).

fn enforce_temporal_gates(report: &MatrixReport) -> Result<(), GateViolation> {
    let get = |v: ConfigVariant, axis: &str, bucket: &str, metric: &str| -> Option<f64> {
        report.runs.iter().find(|r| r.variant == v)
            .and_then(|r| r.buckets.get(axis))
            .and_then(|b| b.get(bucket))
            .map(|m| match metric {
                "hit_at_5" => m.hit_at_5,
                "mrr_at_10" => m.mrr_at_10,
                _ => unreachable!(),
            })
    };

    let p_hist = get(ConfigVariant::Primary, "temporal_intent", "historical", "hit_at_5");
    let a_hist = get(ConfigVariant::TemporalAuto, "temporal_intent", "historical", "hit_at_5");
    if let (Some(p), Some(a)) = (p_hist, a_hist) {
        if a < p - 0.02 {
            return Err(GateViolation::new(format!(
                "TemporalAuto historical hit@5 regressed >2pp: primary={p:.4} auto={a:.4}"
            )));
        }
    }

    // ... similar for neutral (1pp), recency_seeking (mrr@10), oracle ≥ auto
    Ok(())
}
```

Call `enforce_temporal_gates` alongside the existing gate check when `--baseline` is set.

- [ ] **Step 3: Unit-test the gate**

Add a test with mocked `MatrixReport` values that flip each gate:

```rust
#[test]
fn historical_regression_above_2pp_fails_gate() {
    let report = mock_report(/* primary historical = 1.0, auto historical = 0.97 */);
    let err = enforce_temporal_gates(&report).unwrap_err();
    assert!(err.to_string().contains("historical"));
}

#[test]
fn historical_regression_within_2pp_passes() {
    let report = mock_report(/* primary = 1.0, auto = 0.985 */);
    assert!(enforce_temporal_gates(&report).is_ok());
}

// ... mirror for neutral (1pp), recency_seeking mrr@10, oracle ≥ auto.
```

Run: `cargo test -p fastrag-eval`
Expected: all pass.

- [ ] **Step 4: Validate the new baseline when the recapture finishes**

Once the eval run from Step 1 completes:

```bash
jq '.schema_version' docs/eval-baselines/current.json
# => 2

jq '[.runs[] | {variant, hit_at_5, mrr_at_10}]' docs/eval-baselines/current.json
# => 5 entries, variants: primary, no_rerank, no_contextual, dense_only, temporal_auto

jq '.runs[] | select(.variant == "temporal_auto") | .buckets.temporal_intent' \
   docs/eval-baselines/current.json
# => non-null object with recency_seeking, neutral, historical keys

# Smoke-check gates:
target/release/fastrag eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/eval-ctx \
  --corpus-no-contextual /tmp/eval-raw \
  --config-matrix \
  --baseline docs/eval-baselines/current.json \
  --report /tmp/selfcheck.json
# => exit 0, "baseline OK" message
```

- [ ] **Step 5: Verify `route_regret` with an opt-in oracle run**

```bash
target/release/fastrag eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/eval-ctx \
  --corpus-no-contextual /tmp/eval-raw \
  --config-matrix \
  --variants primary,no_rerank,no_contextual,dense_only,temporal_auto,temporal_oracle \
  --report /tmp/matrix-with-oracle.json

jq '.summary.route_regret' /tmp/matrix-with-oracle.json
# => a positive number; log it. If >0.1, file a learned-classifier follow-up
#    note on #65.
```

This is verification only — don't commit `/tmp/matrix-with-oracle.json`.

- [ ] **Step 6: Commit the new baseline + gates (Closes #53)**

```bash
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
cargo fmt --check
cargo test --workspace --features retrieval,rerank,hybrid,contextual,eval

git add docs/eval-baselines/current.json crates/fastrag-eval/ crates/fastrag-cli/
git commit -m "$(cat <<'EOF'
feat(temporal): query-conditional decay v1 (abstaining detector + late injection)

Retires the fixed-halflife TemporalOn variant (regressed historical by 15.4pp
hit@5) and replaces it with TemporalAuto: abstaining regex detector gates
FavorRecent(Medium) decay, applied to final post-rerank scores. Adds
TemporalOracle (opt-in) for upper-bound diagnostics and route_regret metric.

- Per-query TemporalPolicy API (Auto | Off | FavorRecent(Light|Medium|Strong))
- AbstainingRegexDetector: high-precision recency patterns, no CVE-year heuristic
- Late-stage injection: reranker always sees undecayed candidates
- CLI: --temporal-policy, deprecation warnings on legacy --time-decay-* flags
- HTTP: temporal_policy field on POST /query body
- Eval: TemporalAuto in canonical matrix, TemporalOracle via --variants
- Gate: 2pp/1pp/non-regression bounds on historical/neutral/recency buckets

Anchored-past (#59), multi-period (#60), volatility (#61), parameter sweep
(#62), timestamp hierarchy (#63), dateless telemetry (#64), learned classifier
(#65) tracked as follow-ups.

Closes #53
EOF
)"
```

- [ ] **Step 7: Push and run ci-watcher**

```bash
git push origin HEAD
```

Then invoke the ci-watcher skill per repo convention (background Haiku Agent — do not use raw `gh run watch`).

---

## Self-Review Checklist (run before handing off)

- [ ] **Spec coverage:** each section of `2026-04-16-query-conditional-temporal-decay-design.md` maps to at least one task. Verified:
  - §Layer 1 TemporalPolicy API → Tasks 2-3
  - §Layer 2 Abstaining detector → Tasks 4-6
  - §Layer 3 Late-stage injection → Tasks 7-9
  - §Wire-up Rust/HTTP/CLI → Tasks 9, 10, 11
  - §Matrix eval changes → Tasks 12-14
  - §Verification unit → Tasks 2-7 (all with tests)
  - §Verification integration → Task 9 (late-injection contract test), Task 11 (abstain-on-historical CLI test)
  - §Eval gate → Task 15
  - §Rollout three commits → Landings 1/2/3 mapped to commit sets (not literal single commits — each landing is a set, which is a superset of the spec's 3-commit rollout; spec explicitly says "each commit keeps the tree shippable" — same principle holds for each sub-task commit)
- [ ] **No placeholders:** inspected — every step either has complete code or instructs the engineer to read a specific file (e.g., `tests/temporal_decay.rs`) for a harness to copy. One known weak point: Task 9's integration test leaves the corpus-setup to the engineer (via "copy the harness from tests/temporal_decay.rs"). This is acceptable because the harness is already documented by the passing tests there.
- [ ] **Type consistency:** `TemporalPolicy::FavorRecent(Strength)` is a tuple variant (Task 3) and is used that way everywhere (Tasks 7, 11, 12, 13). `TemporalIntent` discriminants (`Historical|Neutral|RecencySeeking`) match `fastrag-eval/src/axes.rs` conventions (verified by mapping in Task 13). `apply_temporal_policy` signature `(results: &[(u64, f32)], policy: &TemporalPolicy, query: &str, detector: &dyn TemporalDetector, dates: &[Option<NaiveDate>], now: DateTime<Utc>)` is consistent across its use sites (Tasks 7, 9, 12).

---

## Execution Choice

**Plan complete and saved to `docs/superpowers/plans/2026-04-16-query-conditional-temporal-decay.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

User's memory `feedback_auto_approve_design_plan.md` says skip review-gate pauses → proceeding to subagent-driven execution.
