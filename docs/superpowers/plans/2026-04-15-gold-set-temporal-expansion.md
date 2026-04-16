# Gold-set temporal expansion + baseline capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 8-12 paired 2025/2026 CVE/KEV/advisory corpus docs + 25 recency_seeking + 10 historical gold-set questions, rebuild `/tmp/eval-ctx` and `/tmp/eval-raw`, and capture `docs/eval-baselines/current.json` via the pre-scheduled 01:00 cron — unblocking #53 and #54 for empirical tuning.

**Architecture:** Four checkpoints in one landing. Data-only: no code changes, no schema changes (v2 already live). Checkpoint A adds corpus docs; B adds questions; C rebuilds indices; D captures the baseline. A hard 00:30 local go/no-go decides whether tonight's cron fires against expanded data or gets disabled and rescheduled.

**Tech Stack:** Markdown + YAML frontmatter (data files), `jq`/`grep`/`ls` for validation, existing `fastrag` CLI for indexing and eval, existing `/tmp/fastrag-eval-1am.sh` cron for capture.

**Timing constraint:** Cron fires **2026-04-16 01:00 local**. Checkpoints A+B+C must be complete by **00:30 local** or cron gets disabled (see Task 12).

**Spec:** `docs/superpowers/specs/2026-04-15-gold-set-temporal-expansion-design.md`

---

## Checkpoint A — Corpus docs

### Task 1: Draft the pair target list

**Files:**
- Create: `/tmp/pair-targets.md` (scratch file, not committed)

**Context:** We need 8-12 new `tests/gold/corpus/*.md` docs (slots `51-` through `62-`). Each new doc is the "new" side of a pair; the "old" side is an already-existing doc in the corpus. Select topics where a real 2025 or 2026 public record exists (KEV addition, CVE disclosure, update advisory, retrospective). No synthesis.

- [ ] **Step 1: Survey existing pre-2024 corpus docs for pair candidates**

Run:
```bash
cd /home/ubuntu/github/fastrag
grep -l "^published_date: 20[12][0-9]" tests/gold/corpus/*.md | while read f; do
  year=$(grep "^published_date:" "$f" | awk '{print substr($2,1,4)}')
  title=$(grep "^title:" "$f" | head -1)
  echo "$year  $f  $title"
done | sort | head -40
```
Expected: listing of the 50 docs with year + filename + title. Use this to pick ~10 topics where real 2025/2026 follow-up data exists.

- [ ] **Step 2: Pick 10 real pair targets**

Candidate source patterns (use at least one real public source per entry):
- **2025 CISA KEV additions** for an older CVE (pair: old CVE writeup in corpus + new KEV addition doc).
- **2025 disclosed CVEs** with strong public write-ups (Tenable, Rapid7, GitHub security advisories, vendor PSIRT).
- **2026 advisories** revising a pre-2022 CVE (e.g., vendor re-analysis, new exploitation primitives discovered).
- **2025/2026 follow-up CVEs** related to a known family (e.g., another Spring RCE, another Log4j-style JNDI issue, another OpenSSL memory bug).

Write `/tmp/pair-targets.md` with 10 lines, one per pair:
```
51 | <topic> | old=<existing doc file> | new=<CVE/KEV/advisory ID + URL> | published_date=YYYY-MM-DD
```

Drop any row where you cannot cite a public URL for the new-side record.

- [ ] **Step 3: Sanity check the list before authoring**

Run:
```bash
wc -l /tmp/pair-targets.md
```
Expected: 8-12 lines. If fewer than 8, widen your source search (vendor PSIRTs, OSS-Security mailing list, ZDI upcoming disclosures). If you still can't hit 8, ship fewer — § Non-goals permit flexing down, and R5 covers it.

### Task 2: Author the 10 new corpus docs

**Files:**
- Create: `tests/gold/corpus/51-<slug>.md` through `tests/gold/corpus/~60-<slug>.md`

**Template — every new doc follows this shape:**

```markdown
---
title: "<short descriptive title>"
published_date: YYYY-MM-DD
last_modified: YYYY-MM-DD   # optional, omit if not applicable
---

# <Title matching frontmatter>

<Opening paragraph: CVE ID (if any), what it is, who's affected, public source citation.>

## Technical Details

<What the vulnerability/issue is, attack prerequisites, impact.>

## References

- <URL to CVE record, NVD entry, KEV entry, or vendor advisory>
- <Secondary source if available>
```

Slugs use lowercase-with-dashes and match conventions in existing filenames (e.g., `51-kev-2025-citrixbleed-ii.md`).

- [ ] **Step 1: Author all 10 new docs following the template**

Work through `/tmp/pair-targets.md` row by row. For each row, create the file, fill in the template, paste real content from the cited public source. Do not synthesize CVE IDs, dates, or vendor details.

- [ ] **Step 2: Validate frontmatter on every new doc**

Run:
```bash
cd /home/ubuntu/github/fastrag
for f in tests/gold/corpus/5[1-9]-*.md tests/gold/corpus/6[0-2]-*.md; do
  [[ -f "$f" ]] || continue
  grep -q "^published_date: 20" "$f" || { echo "FAIL: missing published_date in $f"; exit 1; }
  grep -q "^title:" "$f" || { echo "FAIL: missing title in $f"; exit 1; }
done
echo "OK"
```
Expected: `OK`. Any `FAIL` means fix that doc.

- [ ] **Step 3: Confirm no synthesized CVE IDs**

Run:
```bash
cd /home/ubuntu/github/fastrag
grep -rhoE "CVE-[0-9]{4}-[0-9]{4,}" tests/gold/corpus/5[1-9]-*.md tests/gold/corpus/6[0-2]-*.md 2>/dev/null | sort -u
```
Expected: list of CVE IDs. Eyeball: every ID should be a real public record you cited a URL for. If any ID wasn't in `/tmp/pair-targets.md`, cross-check it against a public source or remove it.

### Task 3: Commit Checkpoint A

- [ ] **Step 1: Run pre-commit hooks manually to surface early failures**

Run:
```bash
cd /home/ubuntu/github/fastrag
git add tests/gold/corpus/5[1-9]-*.md tests/gold/corpus/6[0-2]-*.md 2>/dev/null
git status | head -20
```
Expected: the new `.md` files listed under "Changes to be committed".

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
eval: add 2025/2026 corpus pairs for decay tuning

Adds N new corpus docs (slots 51-6N) covering real 2025/2026 CVE/KEV/
advisory records paired against pre-2024 existing docs. Each new doc
follows the existing frontmatter convention (published_date required).
Unblocks recency_seeking question authoring in Checkpoint B.

Refs docs/superpowers/specs/2026-04-15-gold-set-temporal-expansion-design.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```
Update `N` in the commit message to the actual count. Hooks may run; fix any failures with new commits (never `--no-verify`).

---

## Checkpoint B — New gold-set questions

### Task 4: Author 25 new recency_seeking questions

**Files:**
- Modify: `tests/gold/questions.json` (append into the existing `entries` array)

**Context:** IDs continue the existing `q-new-rec-` prefix pattern. Current last ID is `q-new-rec-010`; new IDs are `q-new-rec-011` through `q-new-rec-035`.

**Template entry:**
```json
{
  "id": "q-new-rec-011",
  "question": "What is the latest <topic> advisory?",
  "must_contain_cve_ids": ["<CVE-YYYY-NNNN from the NEW doc only>"],
  "must_contain_terms": ["<term unique to the NEW doc>"],
  "notes": "new-2026: recency, targets <new doc filename>",
  "axes": { "style": "conceptual", "temporal_intent": "recency_seeking" }
}
```

**Quality bar (all four must hold per question):**
1. `question` contains an explicit recency marker: `latest`, `newest`, `current`, `this year`, `2026`, `recently`, `as of today`.
2. Corpus has a pre-2024 doc AND a 2025+ doc on the topic (verified by `grep -l` across `tests/gold/corpus/*.md`).
3. Every `must_contain_cve_ids` and `must_contain_terms` entry appears in the NEW doc and does NOT appear in the paired OLD doc. This is the anti-ambiguity check.
4. `axes.temporal_intent` is `recency_seeking`.

- [ ] **Step 1: Author 25 entries targeting the 10 new docs**

Distribute ~2-3 questions per new doc. Mix style across identifier and conceptual roughly 50/50 (see R6):
- `identifier` style when the query is an ID lookup (`the latest CVE-2025-...`).
- `conceptual` style when the query is an open-ended ask (`what is the current guidance on ...`).
- `mixed` sparingly.

Insert all 25 entries into the `entries` array immediately before the closing `]`. Preserve the two-space indent and trailing-comma style used by existing entries.

- [ ] **Step 2: Validate each new question's anti-ambiguity**

Run (for each new question's must_contain_terms and must_contain_cve_ids):
```bash
cd /home/ubuntu/github/fastrag
jq -r '.entries[] | select(.id | test("^q-new-rec-0(1[1-9]|2[0-9]|3[0-5])$")) | "\(.id)\t\(.must_contain_terms + .must_contain_cve_ids | join(","))"' tests/gold/questions.json > /tmp/rec-q-terms.tsv
```
Then for each line, run:
```bash
while IFS=$'\t' read -r qid terms; do
  IFS=',' read -ra termlist <<< "$terms"
  for term in "${termlist[@]}"; do
    new_match=$(grep -lF "$term" tests/gold/corpus/5[1-9]-*.md tests/gold/corpus/6[0-2]-*.md 2>/dev/null | wc -l)
    old_match=$(grep -lF "$term" tests/gold/corpus/0[0-9]-*.md tests/gold/corpus/[1-4][0-9]-*.md tests/gold/corpus/50-*.md 2>/dev/null | wc -l)
    [[ "$new_match" -ge 1 && "$old_match" -eq 0 ]] || echo "AMBIGUOUS: $qid term=$term new_match=$new_match old_match=$old_match"
  done
done < /tmp/rec-q-terms.tsv
```
Expected: no `AMBIGUOUS:` lines. Any ambiguous term must be replaced with one unique to the new doc, or the question dropped.

### Task 5: Author 10 new historical questions

**Files:**
- Modify: `tests/gold/questions.json` (append after the recency block)

**Context:** IDs `q-new-hist-011` through `q-new-hist-020` (continues existing `q-new-hist-` series).

**Template entry:**
```json
{
  "id": "q-new-hist-011",
  "question": "As of <past year>, what was <topic>?",
  "must_contain_cve_ids": ["<CVE from the OLD doc>"],
  "must_contain_terms": ["<term unique to the OLD doc>"],
  "notes": "new-2026: historical, targets <old doc filename>",
  "axes": { "style": "mixed", "temporal_intent": "historical" }
}
```

**Quality bar:**
1. `question` contains a historical marker: `as of YYYY`, explicit past year (e.g., `2019`), or targets a specifically-dated event.
2. `must_contain_*` matches the OLD doc (pre-2020 preferred to give strong decay contrast).
3. `axes.temporal_intent` is `historical`.

- [ ] **Step 1: Author 10 entries**

Use existing pre-2020 corpus docs (CWE-2006 entries, 2014/2017/2019 CVE docs) as targets. Each entry frames a query whose correct answer is the OLD doc, phrased so that aggressive temporal decay would actively hurt it.

- [ ] **Step 2: Validate JSON still parses**

Run:
```bash
cd /home/ubuntu/github/fastrag
jq -e '.entries | length' tests/gold/questions.json
```
Expected: `185` (150 existing + 25 recency + 10 historical). If the number is wrong or `jq` errors on parsing, fix the JSON syntax.

### Task 6: Validate gold-set expansion

- [ ] **Step 1: Run the gold-set loader test**

Run:
```bash
cd /home/ubuntu/github/fastrag
cargo test -p fastrag-eval --test gold_set_loader
```
Expected: `test result: ok`. Any failure means malformed JSON or missing `axes` fields.

- [ ] **Step 2: Check bucket distribution**

Run:
```bash
jq '[.entries[].axes.temporal_intent] | group_by(.) | map({intent: .[0], count: length})' tests/gold/questions.json
```
Expected (target):
```json
[
  {"intent":"historical","count":39},
  {"intent":"neutral","count":107},
  {"intent":"recency_seeking","count":39}
]
```
Minimum acceptable (per R5): `historical ≥ 35`, `recency_seeking ≥ 35`. If below 35, author more until threshold met or abandon the landing for tonight.

- [ ] **Step 3: Check style distribution on new recency entries (R6)**

Run:
```bash
jq '.entries | map(select(.id | test("^q-new-rec-0(1[1-9]|2[0-9]|3[0-5])$"))) | [.[].axes.style] | group_by(.) | map({style: .[0], count: length})' tests/gold/questions.json
```
Expected: roughly balanced identifier + conceptual on the 25 new entries (e.g., 10-15 of each). Adjust question wordings if heavily skewed.

### Task 7: Commit Checkpoint B

- [ ] **Step 1: Stage and commit**

```bash
cd /home/ubuntu/github/fastrag
git add tests/gold/questions.json
git commit -m "$(cat <<'EOF'
eval: add recency/historical questions targeting new corpus pairs

Appends 25 recency_seeking (q-new-rec-011..035) and 10 historical
(q-new-hist-011..020) entries. Every recency question's must_contain_*
matches only the 2025/2026 NEW doc of its pair, enforcing the
anti-ambiguity rule the tuning signal requires. Target distribution:
recency_seeking=39, historical=39 (min acceptable 35 per spec R5).

Refs docs/superpowers/specs/2026-04-15-gold-set-temporal-expansion-design.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Checkpoint C — Rebuild eval corpora

### Task 8: Rebuild /tmp/eval-raw (no contextualize, fast)

- [ ] **Step 1: Remove stale corpus**

Run:
```bash
rm -rf /tmp/eval-raw
```

- [ ] **Step 2: Rebuild raw corpus against expanded `tests/gold/corpus/`**

Run:
```bash
cd /home/ubuntu/github/fastrag
cargo run --release --features retrieval,rerank,hybrid -- \
  index tests/gold/corpus \
  --corpus /tmp/eval-raw \
  --embedder qwen3-q8 \
  --metadata-fields published_date,last_modified \
  --metadata-types published_date=date,last_modified=date
```
Expected: completes without errors, prints chunk count reflecting the expanded corpus (roughly `chunks_50_original + chunks_per_new_doc * num_new_docs`).

- [ ] **Step 3: Confirm manifest reflects expanded count**

Run:
```bash
jq '{num_chunks, num_documents}' /tmp/eval-raw/manifest.json
```
Expected: `num_documents` ≥ 58. If less, Step 2 failed silently.

### Task 9: Rebuild /tmp/eval-ctx (with contextualize)

**Context:** Contextualize requires llama-server reachable. If it hiccups, see R3 — raw-only baseline is an acceptable fallback.

- [ ] **Step 1: Remove stale corpus**

Run:
```bash
rm -rf /tmp/eval-ctx
```

- [ ] **Step 2: Rebuild with contextualize**

Run:
```bash
cd /home/ubuntu/github/fastrag
cargo run --release --features retrieval,rerank,hybrid,contextual,contextual-llama -- \
  index tests/gold/corpus \
  --corpus /tmp/eval-ctx \
  --embedder qwen3-q8 \
  --metadata-fields published_date,last_modified \
  --metadata-types published_date=date,last_modified=date \
  --contextualize
```
Expected: completes without errors. May take minutes depending on llama-server throughput. Monitor for timeouts.

- [ ] **Step 3: Confirm manifest reflects expanded count + contextualization**

Run:
```bash
jq '{num_chunks, num_documents, contextualized}' /tmp/eval-ctx/manifest.json
```
Expected: `num_documents` matches `/tmp/eval-raw/manifest.json`; `contextualized` is `true`.

### Task 10: Cross-check manifests and smoke test

- [ ] **Step 1: Confirm manifests agree on chunk count**

Run:
```bash
raw_chunks=$(jq '.num_chunks' /tmp/eval-raw/manifest.json)
ctx_chunks=$(jq '.num_chunks' /tmp/eval-ctx/manifest.json)
[[ "$raw_chunks" == "$ctx_chunks" ]] && echo "OK: $raw_chunks chunks" || echo "MISMATCH: raw=$raw_chunks ctx=$ctx_chunks"
```
Expected: `OK: N chunks`. Mismatch means one index is stale — rebuild the offender.

- [ ] **Step 2: Run a smoke query against the raw corpus**

Run:
```bash
cd /home/ubuntu/github/fastrag
cargo run --release --features retrieval -- \
  query "latest Log4j guidance" \
  --corpus /tmp/eval-raw \
  --top-k 5
```
Expected: at least one result. Pick any recency-ish query you know the corpus should answer; adjust the string if `Log4j` isn't covered by your pair list.

---

## Checkpoint D — Baseline capture

### Task 11: 00:30 local go/no-go gate

- [ ] **Step 1: Assess readiness**

At **00:30 local** (or as soon as Checkpoint C finishes — whichever is earlier), check:
- Task 3 committed? (`git log --oneline -5` shows the corpus commit)
- Task 7 committed? (shows the questions commit)
- Tasks 8 + 9 succeeded? (`jq '.num_documents' /tmp/eval-ctx/manifest.json` ≥ 58)
- Task 10 smoke query returned hits?

- [ ] **Step 2: If NOT ready — disable the cron**

Run:
```bash
crontab -l | grep -v fastrag-eval-1am | crontab -
crontab -l | grep fastrag-eval
```
Expected: second command returns nothing. Cron disabled cleanly.

Then: resume Checkpoints A/B/C at natural pace. When ready, skip to Task 13 and run the capture command manually.

- [ ] **Step 3: If ready — leave the cron alone**

Cron fires at 01:00 and runs the capture against the now-expanded data. Script writes `docs/eval-baselines/current.json` and self-removes from crontab.

### Task 12: Await capture (or run manually)

- [ ] **Step 1: (If cron fires) Wait for it and check the log**

At 01:10 local (or later), run:
```bash
tail -40 /tmp/fastrag-eval-1am.log
ls -la docs/eval-baselines/current.json
crontab -l | grep fastrag-eval
```
Expected: log shows `exit: 0`; `current.json` exists; no crontab entry remains.

- [ ] **Step 2: (If running manually) Invoke capture directly**

Run:
```bash
cd /home/ubuntu/github/fastrag
FASTRAG_LLAMA_TEST=1 FASTRAG_RERANK_TEST=1 \
  target/release/fastrag eval \
    --gold-set tests/gold/questions.json \
    --corpus /tmp/eval-ctx \
    --corpus-no-contextual /tmp/eval-raw \
    --config-matrix \
    --report docs/eval-baselines/current.json
```
Expected: completes with exit 0; `current.json` written. If `target/release/fastrag` is stale, rebuild first with `cargo build --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama`.

### Task 13: Validate baseline

- [ ] **Step 1: Schema + buckets populated**

Run:
```bash
jq '.schema_version' docs/eval-baselines/current.json
```
Expected: `2`.

Run:
```bash
jq '[.runs[].buckets.temporal_intent | keys] | flatten | unique' docs/eval-baselines/current.json
```
Expected: `["historical","neutral","recency_seeking"]`.

- [ ] **Step 2: Bucket counts match expectations**

Run:
```bash
jq '.runs[0].buckets.temporal_intent | {historical: .historical.n, neutral: .neutral.n, recency_seeking: .recency_seeking.n}' docs/eval-baselines/current.json
```
Expected: `recency_seeking.n ≥ 35`, `historical.n ≥ 35`, `neutral.n == 107`.

- [ ] **Step 3: Hit rates non-zero (end-to-end sanity)**

Run:
```bash
jq '.runs[] | {variant, hit_at_5, mrr_at_10}' docs/eval-baselines/current.json
```
Expected: `hit_at_5 > 0` for `Primary` and `DenseOnly` variants. If zero, the capture didn't hydrate chunks — likely a driver regression. Investigate before committing.

- [ ] **Step 4: Regression sanity on original 150 questions (R4)**

Run:
```bash
jq '.runs[0].per_question | map(select(.id | startswith("q-new-") | not)) | [.[] | .hit_at_5] | add / length' docs/eval-baselines/current.json
```
Expected: some reasonable hit@5 average (0.4-0.8 range). Compare against your memory of pre-expansion numbers if available; a drop >0.05 suggests new corpus docs are polluting retrieval for existing questions — see R4 mitigation.

### Task 14: Commit Checkpoint D

- [ ] **Step 1: Stage and commit the baseline**

```bash
cd /home/ubuntu/github/fastrag
git add docs/eval-baselines/current.json
git commit -m "$(cat <<'EOF'
eval: capture v2 baseline against expanded gold set

Runs the full config matrix over the expanded 185-question gold set
(recency_seeking=39, historical=39, neutral=107) with the 2025/2026
paired corpus. Baseline file carries schema_version=2 and populated
per-axis buckets. Unblocks #53 + #54 implementation against real
bucket-level eval signal.

Refs docs/superpowers/specs/2026-04-15-gold-set-temporal-expansion-design.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Local gates before push (per CLAUDE.md global)**

Run:
```bash
cd /home/ubuntu/github/fastrag
cargo test --workspace --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
cargo fmt --check
```
Expected: all three pass. Any failure is fixed in new commits before push.

- [ ] **Step 3: Push and run ci-watcher**

Run:
```bash
git push
```
Then invoke the ci-watcher skill as a background Haiku agent per CLAUDE.md.

---

## Rollback

- **Bad baseline** (e.g., zero `hit_at_5`, Task 13 Step 3 fails): do NOT commit. Delete `docs/eval-baselines/current.json`, investigate (rebuild `target/release/fastrag`, re-check corpora, re-run `eval` manually). Recommit once numbers sanity-check.
- **Bad new corpus docs** (R4: regression on old 150): identify the offending doc via per-question diff, `git revert` the corpus commit for that doc, rebuild, recapture. Leave questions in place; they're independent.
- **Bad new questions** (Task 6 Step 2 fails minimum threshold): either author more to hit ≥35, or revert the Checkpoint B commit and skip this landing for tonight.

## Open assumptions

- `/tmp/fastrag-eval-1am.sh` assumes `target/release/fastrag` is built. If the binary is stale vs the current tree, rebuild before the cron fires: `cargo build --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama`.
- llama-server is running and reachable on the default port. If not, Task 9 fails; raw-only baseline is an acceptable fallback (see R3).
