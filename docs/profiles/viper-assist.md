# VIPER Assist profile

Indexing and query recipe for the VIPER Dashboard's offline retrieval layer
(`crook3dfingers/VIPER_Dashboard` repo, `viper_assist/` sidecar). FastRAG owns
retrieval; the sidecar consumes it via the typed Python client.

## Models

| Role | Model | Notes |
|---|---|---|
| Retrieval embedder | `nomic-ai/nomic-embed-text-v1.5` (GGUF) | 512-dimensional Matryoshka embeddings; asymmetric prefixes (`search_query: ` / `search_document: `) — applied automatically by the catalog defaults |
| Chat (separate process) | `granite-4.0-h-tiny-Q4_K_M.gguf` via `llama-server` on `127.0.0.1:8080/v1` | Owned by VIPER Assist sidecar, not FastRAG |

The retrieval embedder runs in-process inside FastRAG via the `fastrag-embed`
`llama-cpp` feature. There is no separate embedder daemon.

## One-time setup

```bash
# 1. Place the embedder GGUF
mkdir -p /var/lib/fastrag/models
curl -L -o /var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q5_K_M.gguf

# 2. Build/index dir for the corpus bundle
mkdir -p /var/lib/fastrag/bundles/viper-assist

# 3. Generate the VIPER corpus JSONL (in the VIPER repo)
cd ~/github/VIPER_Dashboard
python3 build/build_viper_assist_corpus.py --write --validate
```

Install `llama-server` (from `llama.cpp`) or set `LLAMA_SERVER_PATH`; see
`docs/embedder-profile-migration.md` for backend-resolution details.

## `fastrag.toml`

```toml
[embedder]
default_profile = "viper-assist"

[embedder.profiles.viper-assist]
backend = "llama-cpp"
model = "/var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf"
use_catalog_defaults = true
```

`use_catalog_defaults = true` is what wires the Nomic asymmetric prefixes
(`search_query: ` / `search_document: `). Override per-profile only if you
need a different embedder; mixing prefixes will silently degrade recall.

## Index

```bash
fastrag index ~/github/VIPER_Dashboard/artifacts/viper-assist-corpus.jsonl \
  --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml \
  --embedder-profile viper-assist \
  --preset viper-assist \
  --ingest-format jsonl
```

The `viper-assist` preset declares the section-level chunk schema:

| Field | Type | Notes |
|---|---|---|
| `id` | string (id) | section-anchored, e.g. `cve-cwe:remote-code-execution-rce` |
| `text`, `title`, `section`, `summary` | text | concatenated for embedding |
| `url`, `category`, `risk_level`, `source_file`, `content_hash` | string | filterable |
| `requires_credentials` | bool | filterable |
| `schema_version` | numeric | filterable |
| `ports`, `tools`, `tags`, `cves`, `mitre_ids`, `risk_signals` | array | filterable via `IN`/`CONTAINS` |

Re-index trigger: when `artifacts/viper-assist-corpus.sha256` changes, rerun
the same `fastrag index` command. Indexing is idempotent — content-addressed
IDs cause unchanged chunks to skip.

## Query

```bash
fastrag query "I found SMB LDAP Kerberos and WinRM open" \
  --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml \
  --embedder-profile viper-assist \
  --top-k 8
```

With a metadata filter:

```bash
fastrag query "Where do I start with this Windows server?" \
  --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml \
  --embedder-profile viper-assist \
  --filter "category = playbook AND risk_level = safe" \
  --top-k 5
```

Filter examples for every key the VIPER sidecar (`viper_assist/config.py`)
sends today:

| Filter | Example |
|---|---|
| `category` | `category = playbook` |
| `risk_level` | `risk_level = intrusive` |
| `requires_credentials` | `requires_credentials = false` |
| `ports` | `ports IN [445, 88]` |
| `tools` | `tools IN ["impacket"]` |
| `tags` | `tags IN ["smb"]` |
| `mitre_ids` | `mitre_ids IN ["T1190"]` |
| `cves` | `cves IN ["CVE-2021-44228"]` |

Sidecar filters that are forwarded but currently match no corpus rows
(producer doesn't emit these fields yet): `engagement_types`, `phase`,
`platforms`. Adding them is corpus-producer work in `VIPER_Dashboard`, not
fastrag.

## HTTP serve

The sidecar consumes FastRAG over HTTP. Start one server bound to the port
the sidecar expects (default `127.0.0.1:8081`, see
`viper_assist/config.py`):

```bash
fastrag serve-http \
  --corpus /var/lib/fastrag/bundles/viper-assist \
  --config ./fastrag.toml \
  --embedder-profile viper-assist \
  --bind 127.0.0.1:8081
```

```bash
# sanity probe — same probes the VIPER sidecar uses
curl -fsS http://127.0.0.1:8081/ready
curl -fsS http://127.0.0.1:8081/corpora
curl -fsS 'http://127.0.0.1:8081/query?q=SMB%20LDAP%20Kerberos%20WinRM&corpus=viper-assist&top_k=5'
```

## Smoke test

`fastrag-cli/tests/viper_assist_smoke.rs` runs the issue-#74 prompts against
a curated mini-corpus (`tests/fixtures/viper_assist/`). Run it with:

```bash
FASTRAG_LLAMA_TEST=1 \
  scripts/smoke_viper_assist.sh
```

Same gate as the existing llama-cpp e2e tests; requires the GGUF in the path
above (or override `VIPER_NOMIC_GGUF`).

## Ollama appendix (untested path)

If you already run Ollama for other reasons, a one-liner profile works:

```toml
[embedder.profiles.viper-assist]
backend = "ollama"
model = "nomic-embed-text"
base_url = "http://127.0.0.1:11434"
use_catalog_defaults = true
```

Ollama serves the v1.5 architecture under the bare `nomic-embed-text` tag
today; `use_catalog_defaults` recognises both names. The smoke test does not
cover this path — operators using Ollama validate locally.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `error: missing default embedder profile` | `[embedder] default_profile = ` not set in `fastrag.toml` |
| All queries return empty | embedder GGUF path wrong, or index not built |
| Top results irrelevant for short queries | profile is missing `use_catalog_defaults = true`, prefixes not applied |
| `MissingField { field: "id" }` during ingest | wrong `--preset` (or `--id-field`) — corpus row uses `id`, not `_id` |
| Numeric-port filter returns nothing | confirm `ports: [445]` is array-of-int in source JSON, not strings |
