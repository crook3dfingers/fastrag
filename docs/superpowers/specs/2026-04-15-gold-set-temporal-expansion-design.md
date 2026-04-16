---
name: Gold-set temporal expansion + baseline capture — Design
description: Expand recency_seeking (14→~40) and historical (29→~40) buckets in tests/gold/questions.json, add 8-12 paired 2025/2026 corpus docs, capture v2 baseline. Unblocks #53 and #54 for empirical tuning.
type: design
---

# Gold-set temporal expansion + baseline capture — Design

**Status:** proposed
**Date:** 2026-04-15
**Unblocks:** #53 (query-conditional halflife), #54 (weighted RRF) — by producing the bucket signal their tuning requires. Neither feature is implemented in this spec.

## Context

The 2026-04-15 eval-axes spec landed `Axes` on `GoldSetEntry`, per-bucket `BucketMetrics` in `VariantReport`, and per-bucket slack in the regression gate. Schema v2 is live. What did not land: enough questions in each temporal bucket to actually tune decay and RRF weights.

Current `tests/gold/questions.json` distribution across `temporal_intent`:

- `neutral` — 107
- `historical` — 29
- `recency_seeking` — 14

The recency bucket is too thin to separate a mild-halflife variant from an aggressive-halflife variant statistically. The corpus itself also lacks any docs dated 2025 or later — "latest" queries have no newer candidate to prefer, so decay has no way to help.

A one-off 01:00 cron (`/tmp/fastrag-eval-1am.sh`, scheduled for 2026-04-16 01:00) is already in place to capture `docs/eval-baselines/current.json`. If we land the expansion first, the capture reflects the expanded set in one shot.

## Goals

- ~25 new recency_seeking questions (bucket target ≥40).
- ~10 new historical questions (bucket target ≥40).
- 8-12 new `tests/gold/corpus/*.md` pairs (old + new on the same topic) drawn from real 2025/2026 CVE/KEV/advisory records. YAML frontmatter with `published_date` and optional `last_modified`.
- `/tmp/eval-ctx` and `/tmp/eval-raw` rebuilt against the expanded corpus before the 01:00 cron.
- `docs/eval-baselines/current.json` committed under schema v2 with populated per-bucket numbers.

## Non-goals

- Implementing #53 (query-conditional halflife) or #54 (weighted RRF). Separate specs that consume this one's output.
- Style-axis rebalancing. The 73/46/31 identifier/conceptual/mixed distribution stays as-is. Tuning signal for #53/#54 is temporal, not stylistic.
- Adding new doc types beyond markdown. The pair strategy works within the existing format.
- Changing the gold-set schema, the axes enum, or the report/baseline shape — all already at v2.
- Any retrieval behaviour changes.
- Synthesized CVE IDs or fake advisories. All new docs cite real public records.

## Architecture

Four checkpoints in one landing. Each can be committed standalone; the tree stays shippable at each boundary.

### Checkpoint A — Corpus docs (8-12 new files)

**Scope:** `tests/gold/corpus/51-*.md` through `tests/gold/corpus/62-*.md` (8-12 files, stopping number depends on how many real pairings hold up).

- Each doc cites a real 2025 or 2026 CVE/KEV/advisory record.
- Docs are structured as **pairs on the same topic**: an old (pre-2024) doc already in the corpus or added here, plus a new (2025+) doc describing a follow-up advisory, revised analysis, KEV addition, or newer related CVE.
- YAML frontmatter matches the existing 50 docs: `published_date: YYYY-MM-DD` required, `last_modified: YYYY-MM-DD` optional.
- Example pair patterns (final selection during implementation):
  - A 2021 CVE writeup already in the corpus + a 2025 retrospective or newly-disclosed variant.
  - A 2024 CVE + a 2026 KEV addition.
  - A 2022 original advisory + a 2025 update advisory.
- No synthesized records. If no real pairing exists for a topic we'd like to cover, drop the topic.

### Checkpoint B — New questions (35 new entries in `tests/gold/questions.json`)

**Scope:** 25 `recency_seeking` + 10 `historical` entries, IDs `q151` through `q185`.

- Inline `axes` on every new entry.
- **Recency-seeking quality bar** — all four must hold:
  - Query contains an explicit recency marker: `latest`, `newest`, `current`, `this year`, `2026`, `recently`, `as of today`.
  - Corpus has at least one pre-2024 doc AND one 2025+ doc on the topic.
  - `must_contain_cve_ids` / `must_contain_terms` match **only** the newer doc. Any overlap with the older doc is a disqualifier.
  - A question that can't satisfy all of the above is dropped, not retagged.
- **Historical quality bar** — both must hold:
  - Query contains a historical marker: `as of YYYY`, explicit past year, or targets a specifically-dated event where the old description is the correct answer.
  - Aggressive decay should actively hurt this question — that's the signal #53 needs.
- Target distribution on the new 25 recency questions: roughly balanced across `identifier` / `conceptual` style to avoid skewing the style axis.

### Checkpoint C — Corpora rebuild

**Scope:** `/tmp/eval-ctx`, `/tmp/eval-raw` — rebuilt from `tests/gold/corpus/` via the commands in `docs/eval-baselines/README.md`.

- Raw (no-contextualize) rebuild runs first — no LLM dependency, fast.
- Contextualize rebuild runs second — requires llama-server reachable, may take longer.
- `manifest.json` in both dirs must reflect the expanded chunk count and match each other (same chunk_id set across dense and contextualized variants).
- No commit (corpora live in `/tmp`).

### Checkpoint D — Baseline capture

**Scope:** `docs/eval-baselines/current.json`.

- If the 01:00 cron fires against the expanded data, it writes the baseline and self-removes from crontab. Commit is a manual post-fire step.
- If the cron was disabled at the 00:30 go/no-go, the same command runs manually post-deadline.
- Post-capture sanity:
  - `jq '.schema_version' current.json` returns `2`.
  - `jq '.runs[].buckets.temporal_intent' current.json` shows populated `recency_seeking` and `historical` keys across every variant.
  - `jq '.runs[0].buckets.temporal_intent.recency_seeking.n' current.json` ≥ 35.
  - `jq '.runs[0].buckets.temporal_intent.historical.n' current.json` ≥ 35.
- Commit message: `eval: capture v2 baseline against expanded gold set`.

## Data model

No changes. Schema v2 covers everything:

- `GoldSetEntry.axes` — already required.
- `VariantReport.buckets: BTreeMap<String, BTreeMap<String, BucketMetrics>>` — already populated.
- `BaselineFile.schema_version: 2` — already set.
- `BaselineFile.per_bucket_slack: Option<f64>` — already supported.

## Verification

**During authoring (pre-commit):**

- `cargo test -p fastrag-eval --test gold_set_loader` — parser accepts the expanded file.
- `jq '.entries | length' tests/gold/questions.json` returns between 180 and 185 (target 185, R5 allows flex down).
- `jq '[.entries[].axes.temporal_intent] | group_by(.) | map({intent: .[0], count: length})'` confirms `recency_seeking ≥ 35` (target 39) and `historical ≥ 35` (target 39).
- `ls tests/gold/corpus/*.md | wc -l` returns 58-62.
- `grep -c "^published_date:" tests/gold/corpus/*.md` returns 1 for every new file.
- For each new question: `grep -c` over each `must_contain_terms` entry against `tests/gold/corpus/` confirms at least one match (the question is answerable) and, for recency_seeking, no match in the paired older doc (the question actually stresses decay).

**Post-corpus-rebuild (before cron):**

- `jq .num_chunks /tmp/eval-ctx/manifest.json` equals `jq .num_chunks /tmp/eval-raw/manifest.json`.
- Dry-run smoke query: `cargo run --release --features retrieval -- query "latest Log4j guidance" --corpus /tmp/eval-raw --top-k 5` returns at least one hit.

**Post-baseline-capture:**

- Schema v2 confirmed.
- Per-bucket entries present for every variant.
- `hit_at_5` non-zero for `Primary` and `DenseOnly` — sanity that end-to-end scoring works.

No new code, no new tests. Existing eval test suite covers the machinery.

## Rollout

The 01:00 cron is the happy path. The 00:30 local go/no-go is the safety valve.

1. Commit Checkpoint A (corpus docs) when authored.
2. Commit Checkpoint B (questions) when authored.
3. Rebuild `/tmp/eval-ctx` and `/tmp/eval-raw` (Checkpoint C) — no commit.
4. At **00:30 local**, assess: are A, B, and C complete? Is a smoke query returning sensible hits against the expanded corpora?
   - **Yes:** leave the cron alone. At 01:00 it captures against expanded data and writes `current.json` into the checkout. Commit the result manually post-fire.
   - **No:** disable cron with `crontab -l | grep -v fastrag-eval-1am | crontab -`. Finish A/B/C at natural pace. Re-add cron or invoke `fastrag eval` manually for the capture.
5. Post-capture commit: `eval: capture v2 baseline against expanded gold set`. Verify per-bucket numbers pass the sanity checks before pushing.

No partial data commits. If only A lands tonight, B and C come together later — never half-baked recency_seeking questions with placeholder corpus docs.

## Risks

**R1. Cron deadline miss.** Likely within the ~2h window.
Mitigation: the 00:30 go/no-go cleanly disables cron. No data loss, just a rescheduled capture.

**R2. Low-quality recency questions.** Ambiguous `must_contain_*` that matches both old and new docs poisons tuning signal.
Mitigation: the § Verification `grep -c` pair check is mandatory for every recency question. Drop anything that fails rather than shipping it.

**R3. Corpus rebuild fails or contextualize takes too long.** llama-server hiccups, rate limits, or wall-clock overrun.
Mitigation: rebuild raw corpus first (fast path, no LLM). If contextualize fails, capture a raw-only baseline; ctx baseline follows later. Acceptable interim state.

**R4. New corpus docs pollute retrieval for existing questions.** The 2025/2026 docs show up as false positives for pre-existing neutral queries.
Mitigation: post-capture, run an ad-hoc eval against the old 150-question subset only (filter by ID) and compare `hit_at_5`. A >5pp regression is a signal to revert specific corpus docs and re-cron.

**R5. No real records fit some planned pair.** Some topics just don't have a 2025+ follow-up in public data.
Mitigation: drop the pair. Better 8 high-quality pairs than 12 where two were fudged. Targets (25/10 new questions) flex down proportionally.

**R6. Unintended style-axis skew.** New recency questions clustered on `conceptual` tilts the style distribution.
Mitigation: track identifier/conceptual split during authoring on the 25 recency entries — aim roughly 50/50. Pre-commit, filter `tests/gold/questions.json` to IDs `q151…q185` with `jq` and verify the style distribution on that slice.

## References

- `docs/superpowers/specs/2026-04-15-eval-axes-and-dated-corpus-design.md` — the axes/schema work this spec builds on.
- `docs/superpowers/specs/2026-04-14-temporal-decay-hybrid-retrieval-design.md` — the retrieval features whose defaults this spec enables tuning for.
- `docs/eval-baselines/README.md` — canonical capture command, cron script path.
- Issues #53, #54 — the parked enhancements this spec unblocks.
- `/tmp/fastrag-eval-1am.sh` — the pre-scheduled 2026-04-16 01:00 capture.
