# Chunking Strategy Evaluation

## Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `--chunk-strategy` | `recursive` | Preserves sentence and paragraph boundaries better than fixed-size splits, reducing mid-sentence cuts that degrade embedding quality |
| `--chunk-size` | `1000` | Fits within Qwen3-Embedding-0.6B's 8k context window while keeping chunks short enough for precise retrieval |
| `--chunk-overlap` | `200` | 20% overlap retains cross-boundary context without redundant storage; standard recommendation in RAG literature |

## Why no sweep results

A systematic sweep (all combinations of strategy × size × overlap) requires a corpus
large enough to produce low-variance measurements. The gold set (`tests/gold/corpus/`)
has 50 documents — too few to reliably distinguish a 1–2% hit@5 difference caused by
chunking from one caused by sampling noise.

A second factor: contextual retrieval generates a 50–100 token context prefix for
every chunk at ingest. The LLM prefix substantially reduces sensitivity to chunking
boundaries, compressing the performance gap between strategies.

For a meaningful sweep, use a corpus of at least 500 documents with at least 500
questions and run it with GPU-accelerated llama-server. The `scripts/chunking-sweep.sh`
script is ready for this.

## Re-evaluating defaults

Run the sweep against a larger corpus and compare hit@5 across combos:

```bash
bash scripts/chunking-sweep.sh
sort -t$'\t' -k7 -rn target/chunking-sweep/results.tsv | head -10
```

If a different combo wins by more than 2% hit@5, update the defaults in
`fastrag-cli/src/args.rs` (`default_value` for `--chunk-strategy` and
`default_value_t` for `--chunk-overlap`).

## Sweep grid

| Axis | Values |
|------|--------|
| Strategy | `basic`, `by-title`, `recursive` |
| `--chunk-size` | 500, 800, 1000, 1500 |
| `--chunk-overlap` | 0, 100, 200 |

`semantic` chunking is excluded — it requires an embedder pass during chunking,
which multiplies wall-clock by another order of magnitude on CPU.

## References

- Anthropic Contextual Retrieval (Sept 2024): chunking sensitivity reduced when context prefixes are added
- `docs/rag-research-2026-04.md`: §3 chunking, recommendation for recursive + overlap on technical prose
- `scripts/chunking-sweep.sh`: sweep implementation (36 combos: 3 strategies × 4 sizes × 3 overlaps)
