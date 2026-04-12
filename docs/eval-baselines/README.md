# Eval Baselines

Checked-in baselines for the weekly retrieval eval gate.

## Files

- `current.json` — active baseline for the weekly workflow's regression gate. Compared against every fresh matrix report; any `hit@5` or `MRR@10` drop beyond 2% slack fails the job. Absent until the first manual capture.

## Initial capture

The baseline is captured on a workstation with a real embedder + reranker + llama-server available.

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

# Run the matrix and write the baseline.
cargo run --release --features eval,retrieval,rerank,hybrid,contextual,contextual-llama -- \
  eval \
  --gold-set tests/gold/questions.json \
  --corpus /tmp/gold-baseline/ctx \
  --corpus-no-contextual /tmp/gold-baseline/raw \
  --config-matrix \
  --report docs/eval-baselines/current.json
```

Review the captured numbers:

```bash
jq '.runs[] | {variant, hit_at_5, mrr_at_10}' docs/eval-baselines/current.json
```

`Primary` should score at or above `DenseOnly` on `hit@5` for most questions. If not, the gold set or corpus needs curation — the questions may not be answerable by the docs.

## Refresh flow

After Phase 2 Step 7 (security corpus hygiene) landed on 2026-04-11, the gold set grew from 105 to 110 entries (5 new `hygiene-*` questions: Log4Shell, HTTP/2 Rapid Reset, Spring4Shell, Apache vendor facet, and KEV tagging). Refresh the baseline by running the capture command on a workstation with llama-server before the first post-Step-7 commit to `docs/eval-baselines/current.json`.

Refreshes are deliberate human commits. Re-run the capture command above when a change legitimately improves retrieval quality — new embedder, tuned chunking, improved contextualization prompt. Review the diff and commit:

```bash
git diff docs/eval-baselines/current.json
git add docs/eval-baselines/current.json
git commit -m "eval: refresh baseline after <change>"
```

Never refresh to make a red CI go green — that defeats the gate.

## Weekly workflow

Until `current.json` is committed, the weekly workflow runs `eval --config-matrix` without `--baseline` and uploads the report as an artifact. Once the baseline is committed, the workflow passes `--baseline docs/eval-baselines/current.json` and fails on regression.
