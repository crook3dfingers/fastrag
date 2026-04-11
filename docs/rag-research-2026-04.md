# RAG Research Notes for fastrag (2026-04)

Distilled from a 2025–2026 best-practices review done while critiquing the Python shim (`tarmo-llm-rag`) that fastrag is slated to replace. Target reader: whoever is implementing `fastrag --features retrieval`. Everything here is framed around the constraints that actually matter for fastrag — single binary, Rust, CPU inference, security corpora (NVD / KEV / GHSA / OSV / exploit writeups), single-engineer maintenance.

All recommendations are annotated **Adopt / Watch / Skip**. Dates and benchmark numbers come from training cutoff May 2025 plus inference for the gap through 2026-04 — verify before committing to a dependency.

## TL;DR — if you only read one section

1. **Ship hybrid retrieval from commit 1.** Dense-only is wrong for CVE corpora. Exact CVE-ID + CWE lookup must short-circuit dense. Lesson #4 from the shim's `fastrag-lessons-learned.md` is non-negotiable.
2. **Default embedder: nomic-embed-text-v1.5 or Snowflake arctic-embed-m-v1.5**, Matryoshka-truncated to 512d. Retire `all-MiniLM-L6-v2`. +8–12 MTEB points on technical text.
3. **Ship a reranker on day one**: `bge-reranker-v2-m3` exported to ONNX int8. Single highest-ROI quality lever. `ort` crate works fine for ONNX in Rust.
4. **Adopt Contextual Retrieval** (Anthropic, Sept 2024) as the ingest-time default. Prepend 50–80 token LLM-generated context per chunk before embedding. −49% retrieval failure, −67% with reranking. Cache by chunk hash.
5. **Enforce the embedder invariant in the type system.** Lesson #1 (the 4-dim silent-trap bug) must be impossible to write in Rust — not merely tested for. See §2.
6. **Structured field index next to the vector store.** CVE-ID, CWE, CVSS, KEV flag, publish date, vendor, product. Use `tantivy` (Rust native, Lucene-grade BM25, facets, ranges). Route `CVE-\d{4}-\d{4,7}` and `CWE-\d+` as exact lookups; everything else through hybrid.
7. **Grounded generation:** structured JSON output with `{claim, source_id, quote}` tuples, post-hoc verify quotes appear verbatim in retrieved chunks. This is the 2025 attribution pattern.
8. **Skip:** ColBERT, SPLADE, GraphRAG, semantic chunkers, Self-RAG, FLARE. All cost ops complexity disproportionate to gain on a structured corpus like NVD.

## 1. Embeddings

`all-MiniLM-L6-v2` (2021, 384d) is no longer defensible for technical corpora in 2026. Loses 5–15 nDCG@10 points on technical retrieval vs. modern small models. Do not default to it.

Candidates, all open (Apache/MIT), all CPU-friendly:

| Model | Dim | Notes | Verdict |
|---|---|---|---|
| `nomic-embed-text-v1.5` (Feb 2024) | 768 | Matryoshka (truncate 256/512), 8k context | **Adopt** — top pick |
| `Snowflake/snowflake-arctic-embed-m-v1.5` (mid-2024) | 768 | Matryoshka, beats BGE on MTEB retrieval | **Adopt** — top pick alt |
| `BAAI/bge-base-en-v1.5` | 768 | Solid floor, well-supported | Adopt if nomic/arctic blocked |
| `BAAI/bge-small-en-v1.5` | 384 | Drop-in MiniLM replacement, +8–12 MTEB | Adopt if 384d is a hard budget |
| `Alibaba-NLP/gte-base-en-v1.5` | 768 | Competitive but surpassed | Watch |
| `jinaai/jina-embeddings-v3` (late 2024) | 1024 | Task-specific LoRA adapters, 8k | Watch — some variants CC-BY-NC, check license |
| `dunzhang/stella_en_400M_v5` | 1024 | Quality leader in small class, heavier | Watch |

**Security-domain finetunes:** SecureBERT and CySecBERT exist but are encoder-only classifiers, not sentence embedders. No mainstream CVE-finetuned sentence embedder worth trusting yet. Finetuning nomic or arctic on (CVE-description, CVE-ID) contrastive pairs from your own corpus is a weekend project and will beat any generic model. **Watch / DIY.**

**Rust inference:** `fastembed-rs` (qdrant) is the most complete CPU path for modern embedders — ONNX under the hood, ships quantized weights, supports nomic and BGE families. Alternative: roll your own with `ort` + `tokenizers` (HF) if fastembed-rs lags on a specific model. Avoid `rust-bert` — it's frozen on older torch bindings.

**Query/passage asymmetry:** nomic, arctic, and jina v3 all require a task prefix (`search_query: ` vs. `search_document: ` or similar). Embed-time bugs in this prefix are a silent quality killer — enforce it in the type system (see §2).

## 2. The embedder invariant — make lesson #1 impossible in Rust

The shim's dimension-4 bug (`chromadb.get_or_create_collection` silently returning a 4-dim dummy embedder when `embedding_function` was omitted) corrupted 40k chunks and only surfaced at query time. In Rust you can make this unrepresentable.

Guidance for fastrag's API design:

- **The vector store handle must be parameterized by the embedder.** `Index<E: Embedder>` where `E::DIM` is an associated const. Mixing an index built with `NomicV15` with a query using `MiniLm` is a compile error, not a runtime log line.
- **Query and passage prefixes are distinct types.** `QueryText(String)` and `PassageText(String)`, each with its own `embed` method on the embedder trait. Impossible to embed a passage with the query prefix or vice versa.
- **Collection metadata persisted alongside the index** records `{embedder_id, embedder_version, dim, prefix_scheme_hash}`. On open, the index refuses to load if the embedder the binary knows about does not match — no silent dimension drift across upgrades.
- **A canary embedding is written at index creation** and re-verified on open. The canary's vector is stored; on open, re-embedding the canary and comparing (with a tight cosine threshold) catches model-file drift, tokenizer changes, and quantization mismatches that metadata alone will not.

This is the single most important lesson for fastrag to internalize. The Python shim caught the bug in hours; a Rust crate with the right types cannot have the bug at all.

## 3. Hybrid retrieval — BM25 + dense + exact, fused with RRF

BM25 + dense with Reciprocal Rank Fusion is still the 2025 baseline and nothing has cleanly displaced it for small teams. Do not be tempted by exotic replacements:

- **ColBERT v2 / PLAID** — better recall but 10–100× index size and painful on CPU. **Skip.**
- **SPLADE v3** (2024) — learned sparse, competitive with dense, but needs a GPU for query encoding at acceptable latency. **Skip** for CPU-first fastrag.
- **ColPali** — for visual document layouts (PDFs with tables). Not relevant to NVD JSON. **Skip.**

**Recommended architecture for fastrag retrieval:**

```
query
  ├── regex extract CVE-\d{4}-\d{4,7}, CWE-\d+  →  tantivy exact term lookup
  ├── tantivy BM25 (full-text)
  └── dense vector search (fastembed-rs + hnsw-rs or usearch)
        │
        └── fuse via RRF (k=60)
              │
              └── rerank top-50 with bge-reranker-v2-m3 (ONNX int8)
                    │
                    └── top-5 to caller
```

- **Tantivy** is the right structured-index choice for Rust: Lucene-grade BM25, facets, range queries, single-embedded library, no server. **Adopt.**
- **HNSW:** `hnsw_rs` and `usearch` (Rust bindings) are both production-grade. `usearch` has better recall/latency on small corpora (<1M chunks) and supports int8 quantization of stored vectors natively. **Adopt usearch** unless you hit a deal-breaker.
- **Store payloads out-of-band** (sled, redb, or just Tantivy itself). Keep the vector index holding only IDs and vectors.

## 4. Reranking — cheapest big quality win

The shim does not rerank. Adding a reranker is the single highest-ROI change you can make on a security corpus.

- **`BAAI/bge-reranker-v2-m3`** (2024) — 568M params, multilingual, best quality/cost trade-off. Export to ONNX, quantize int8, run via `ort` crate. Expect ~200–400ms for a top-50 rerank on CPU. **Adopt.**
- **`jinaai/jina-reranker-v2-base-multilingual`** — smaller (278M), faster, slight quality drop. Good if your latency budget is tight. **Watch.**
- **`mixedbread-ai/mxbai-rerank-base-v1`** — MIT, competitive. **Watch.**

**Pipeline:** hybrid returns top-50, reranker scores all 50, top-5 go to the LLM. Expect +10–20 nDCG points on a security-QA eval — this alone is often the difference between "cites the right CVE" and "hallucinates a plausible neighbor."

Quantization matters: int8 ONNX cuts memory and CPU by ~3–4× with single-digit quality loss on reranking tasks. Ship quantized by default, keep fp32 as a config opt-in for quality-sensitive workloads.

## 5. Contextual Retrieval (Anthropic, Sept 2024) — adopt, it's proven

**What:** Before embedding a chunk, have an LLM generate a 50–100 token context situating it in its parent document. Prepend to chunk text. Embed the augmented string. Store the original (un-augmented) chunk for display at retrieval time.

**Why it matters:** Reported −49% retrieval failure rate vs. naive RAG, −67% when combined with BM25 + dense + reranker. It is the single most-cited RAG upgrade of 2024–2025 and has held up in practice through 2025. Cost: one LLM pass over the corpus at ingest time. With a local model it is effectively free (just slow).

**For fastrag specifically:**

- Make contextualization pluggable. An `Contextualizer` trait with impls `None`, `LocalLlm(endpoint)`, `OpenAiCompat(endpoint)`. Default is `None` so the library is usable without an LLM dependency.
- **Cache by chunk content hash.** Idempotency is mandatory — a 40k-chunk corpus on CPU inference is an overnight job, partial failures must resume cleanly.
- Store both `raw_text` and `contextualized_text` in chunk metadata. Embedding uses `contextualized_text`, display at retrieval time uses `raw_text`. Never show the contextualization prefix to the final LLM.
- Track a per-chunk `contextualized_at` timestamp and a `contextualizer_version`. Re-contextualization on model or prompt change must be incremental.

**CRAG** (Corrective RAG, 2024) — lightweight retrieval-quality classifier that triggers a web search fallback when retrieval confidence is low. **Watch** — potentially useful for fastrag-as-library, but orthogonal to core retrieval.

**Skip:** FLARE (active retrieval mid-generation, adds latency for niche gains), Self-RAG (requires a finetuned model). Both are research-grade, not production patterns.

## 6. Chunking for security corpora — go structured, not semantic

For NVD / KEV / GHSA / OSV, "semantic chunking" is a waste. The records are already structured. Chunk per-field:

- **CVE record → one chunk** per CVE: `{cve_id} {description}` as the embedding target. Separate indexed fields (in Tantivy): CVSS vector, CWE, references, affected CPEs, publication date, KEV flag, vendor, product.
- **Exploit writeups / long-form KEV notes** → 512-token fixed chunks with 64-token overlap, and **prepend the CVE-ID as a header** (`CVE-ID: CVE-2024-3094\n\n`). This cheap structural hint dramatically helps dense retrieval associate narrative chunks with their CVE.
- **Markdown / HTML writeups** → split on headers first, fall back to fixed-size only inside long sections. `text-splitter` crate is the right Rust tool.

**Skip** semantic chunkers (`chunking-ai`, various langchain experiments). They add latency, non-determinism, and no measured win on structured security corpora.

## 7. Grounded generation and citation verification

Instructing the model to cite sources is table stakes. Verifying it did so is the 2025 standard:

- **Structured output.** Ask the model for JSON like `[{"claim": "...", "source_id": "S3", "quote": "..."}]`. Every claim carries the source it came from and the exact quoted substring.
- **Post-hoc quote verification.** After generation, for each `{source_id, quote}` tuple, confirm the quote appears verbatim in the retrieved chunk with ID `source_id`. Fuzzy match tolerance ~5 edits for whitespace/punctuation drift. Reject or downgrade generations where the quote check fails.
- **Refusal tracking as a signal.** When retrieval returns junk, a well-prompted model says "the provided evidence does not contain information about X" instead of hallucinating. That is the correct failure mode — surface it as a metric (`refusal_rate_by_tool`), not an error. A spike in refusal rate means retrieval regressed, not that the model broke.
- **Prompt-injection defense.** Chunks come from attacker-controlled vendor CVE text. Wrap each chunk in a delimited envelope (`<EVIDENCE id="S1" source="...">...</EVIDENCE>`) and instruct the model to treat envelope contents as untrusted data. Sanitize: strip control characters, truncate to a fixed max length (~500 chars), reject chunks containing the envelope delimiter itself.

**Hallucination gate:** the stable evaluation metric is **delta** — cold (no retrieval) vs. grounded (with retrieval), same question, same model. A stable gate is "does adding evidence change the answer from hallucination to evidence-cited." Do NOT gate on absolute correctness; modern MoE models produce non-deterministic cold hallucinations and "wrong answer" is not a stable target. This is a direct lesson from Layer 1 validation.

## 8. Evaluation

- **RAGAS** is still dominant in 2025. Nothing has cleanly displaced it.
- **DeepEval** (confident-ai) is the rising alternative — pytest-native, good DX. Python-only, so for fastrag's own eval harness you'll want to re-implement the core metrics in Rust or expose a CLI that DeepEval can call.
- **TruLens** and **ARES** exist but add operational burden without proportionate gain for single-engineer projects. **Skip.**

**Security-specific eval sets** exist but are QA benchmarks, not RAG-native: CyberMetric, SecQA, CTIBench (2024). Useful as source material.

**The most valuable eval asset is a hand-curated gold set.** 100 questions with known correct CVE IDs and must-contain keywords will beat any generic benchmark for a security-tooling use case. Check this fixture into the repo, version it, gate CI on hit@5 and MRR@10 against it. fastrag should ship a canonical gold-set format and a loader so downstream consumers can bring their own.

**Critical eval rule** (lesson #2 from the shim): eval harnesses that stub embeddings with bag-of-words cannot catch ingest/runtime mismatches. **The eval must run the real embedder against a real (small) corpus.** If you want a fast deterministic smoke test, that is fine, but clearly label it as a smoke and run a slow real-embedder eval in nightly CI. The shim shipped a bag-of-words-only eval and it did not catch the dimension-4 bug — do not repeat that mistake.

## 9. Observability

Production RAG metrics teams actually track in 2025:

- **Retrieval hit@k** against the gold set (offline, on every corpus or embedder change).
- **Rerank delta** — hit@k with rerank minus hit@k without. Is the reranker earning its CPU budget.
- **Groundedness rate** — fraction of generated claims whose quote-verification passes.
- **Refusal rate** — per tool, per query type. Spikes mean retrieval regressed.
- **Latency percentiles per stage** — embed, retrieve, rerank, generate. Know where the CPU budget goes.
- **Cache hit rate** — contextualization cache, embedding cache, rerank cache.

**Langfuse** (self-hosted, MIT, Postgres-backed) is the best OSS observability option in the 2025 landscape. Trace-level drill-down, reasonable DX. **Adopt** if fastrag ships an HTTP service mode. For the library path, emit OpenTelemetry spans and let the consumer wire it up — do not take a hard Langfuse dependency.

**Arize Phoenix** is notebook-first, less prod-ready. **Watch.**

## 10. Corpus hygiene — not optional, a correctness issue

Lessons from shipping the shim, each one a concrete pre-processing step fastrag should bake into its ingest pipeline:

1. **Reject `vulnStatus: Rejected` and `Disputed` CVEs.** NVD ships ~760 rejected CVEs per year with short boilerplate (`** REJECT `, `Rejected reason: ...`). They dominate top-k for any short query and poison semantic retrieval. Filter at ingest. This alone fixed a significant fraction of the shim's early retrieval failures.
2. **Strip NVD boilerplate from embedding text.** `** DISPUTED **`, CPE 2.3 URIs, reference URLs, legal notices — keep in metadata, exclude from the embedded string.
3. **Cross-source deduplication.** NVD, GHSA, OSV, and KEV overlap heavily. Dedup by CVE-ID, merge descriptions (GHSA prose is consistently better than NVD's), keep all source provenance as metadata. A single canonical chunk per CVE beats four noisy ones.
4. **Temporal weighting.** Boost KEV and post-2020 CVEs in the reranker score or as a Tantivy boost. Old CVEs dominate by volume but rarely match modern intent.
5. **Normalize vendor and product via CPE 2.3** before indexing — free facets in Tantivy, cheap disambiguation at query time.
6. **Language filtering.** NVD has a fraction of non-English descriptions. Silently falling back to "first description" got the shim non-English text in a few chunks. Detect language at ingest, log the count, and choose explicit behavior (skip, translate, or flag).

## 11. The "adopt one thing, skip one thing" summary

**Adopt first, above everything else:** `bge-reranker-v2-m3` (ONNX int8) + Contextual Retrieval. Two changes, a week of work, will likely double retrieval quality on a security corpus. This is the highest-ROI path. Everything else in this document is a follow-on.

**Skip regardless of hype:** ColBERT, SPLADE, agentic / self-correcting RAG, semantic chunkers, GraphRAG. Especially GraphRAG — CVEs are already a graph (CWE → CVE → CPE → affected product), and you do not need Microsoft's LLM-extracted one. Building a real graph from NVD structure and querying it with Tantivy facets is strictly better and orders of magnitude cheaper.

## 12. Caveats

- Training cutoff on this research is May 2025. Specific model names, benchmark numbers, and "best in class" verdicts should be verified against current releases before committing to a dependency. In particular, check for: any arctic-embed v2, jina-embeddings v4, bge-reranker v3, new fastembed-rs model support, and whether ONNX int8 reranker weights have been published for any of the above.
- Anthropic Contextual Retrieval is from Sept 19 2024 — confirm the technique has not been superseded by a newer Anthropic post before treating it as the default.
- Everything in this doc assumes a CPU-first, single-binary, single-engineer constraint. Teams with GPUs and ops headcount can and should make different choices (SPLADE, ColBERT, larger rerankers, live agentic retrieval).
