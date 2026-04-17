# fastrag for VAMS

How to wire [fastrag](../README.md) into VAMS as a retrieval + reference-lookup sidecar, including airgap packaging. This is a greenfield integration — VAMS has no prior RAG service to migrate from.

## TL;DR

- fastrag is a standalone HTTP service VAMS calls for **finding dedup** (`POST /similar`) and **CWE/KEV reference lookups** (`GET /cwe/{id}`, `GET /cwe/relation`).
- fastrag is **not** an LLM proxy. VAMS vetting keeps calling llama-server directly via `TARMO_VETTING_TIER*` — unchanged.
- Two install paths: (A) add fastrag as a service in VAMS's `docker-compose.yml` (default), or (B) run the standalone [airgap DVD image](./airgap-install.md) on a separate host and point VAMS at it via `FASTRAG_URL`.
- Smoke test: `curl -fsS $FASTRAG_URL/ready` — expect `{"ready": true, ...}`.

## What fastrag is (and isn't) for VAMS

fastrag exposes a `vams-lookup-v1` bundle of pre-built reference corpora (`cwe`, `kev`) plus a separately mounted VAMS-owned corpus of past findings (`vams-findings`). VAMS uses it for three things:

| Use case | Endpoint(s) | Status |
|---|---|---|
| Finding dedup at ingest | `POST /similar` (+ MinHash verifier) | Primary |
| CWE / KEV reference lookups | `GET /cwe/{id}`, `GET /cwe/relation` | Primary |
| Cross-engagement pattern matching | `POST /similar` with `corpora: [...]` + `X-Fastrag-Tenant` | Future (VAMS is single-tenant today) |

What fastrag does **not** do:

- **No LLM inference.** Vetting LLM calls stay on `TARMO_VETTING_TIER1_ENDPOINT` / `TARMO_VETTING_TIER2_ENDPOINT` (see `vams/core/vetting_service.py`). fastrag's internal embedder and reranker subprocesses bind to the container's loopback only — they are not VAMS-accessible LLM endpoints.
- **No session state or secrets.** Bundles contain public CVE/CWE/KEV data; the findings corpus must never contain credentials, session tokens, or customer secrets. Scrub before `POST /similar`.
- **No bundle signature verification** (yet). VAMS verifies bundles via `tarmo_vuln_core.signing.ReportSigner` before calling `POST /admin/reload`. Rust-side verification is deferred to fastrag issue #66.

## Architecture

```
┌────────────────────────────── compose network ──────────────────────────────┐
│                                                                             │
│   vams-api ──► fastrag (port 8080)                                          │
│        │         │                                                          │
│        │         ├─ internal llama-server: embedder (Qwen3-Embed-0.6B Q8)   │
│        │         ├─ internal llama-server: reranker (BGE-rerank-v2-m3 Q8)   │
│        │         └─ /var/lib/fastrag/bundles/<name>/  (mounted volume)      │
│        │                                                                    │
│        └──► llama-server (8081/8082) ◄── vetting LLM, unchanged             │
│                                                                             │
│   vams-frontend ──► vams-api                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

Single container, single port exposed. All ML subprocesses live inside the fastrag container and never bind a public port.

## Install — Path A: bundled into VAMS docker-compose (default)

### 1. Extend `docker-compose.yml`

Add a `fastrag` service block alongside the existing `vams-api` / `vams-frontend` stanzas in `vams/docker-compose.yml` (the existing env-var style at lines 23–35 is the template to match):

```yaml
  fastrag:
    image: fastrag:${FASTRAG_TAG:-latest}
    ports:
      - "${FASTRAG_PORT:-8080}:8080"
    volumes:
      - ${FASTRAG_BUNDLES_DIR:-./bundles}:/var/lib/fastrag/bundles:ro
    environment:
      BUNDLE_NAME: "${FASTRAG_BUNDLE_NAME:-vams-lookup-v1}"
      FASTRAG_TOKEN: "${FASTRAG_TOKEN:-}"
      FASTRAG_ADMIN_TOKEN: "${FASTRAG_ADMIN_TOKEN:-}"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8080/ready"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 120s
    restart: unless-stopped
```

Wire the dependency and URL into `vams-api`:

```yaml
  vams-api:
    # ... existing keys unchanged ...
    environment:
      # ... existing env unchanged ...
      FASTRAG_URL: "${FASTRAG_URL:-http://fastrag:8080}"
      FASTRAG_TOKEN: "${FASTRAG_TOKEN:-}"
      FASTRAG_ADMIN_TOKEN: "${FASTRAG_ADMIN_TOKEN:-}"
    depends_on:
      fastrag:
        condition: service_healthy
```

`start_period: 120s` is deliberate: first boot loads two GGUFs (~1 GB combined) before `/ready` flips to 200.

### 2. Extend the airgap save-bundle

VAMS's airgap flow (`vams/docs/installation.md` lines 37–89) packs every required image into a single `docker save` tarball. Add `fastrag:<tag>` to the staging list:

```bash
docker save \
    vams-vams-api:latest \
    vams-vams-frontend:latest \
    fastrag:<tag> \
    # ... existing scanner images ...
    -o vams-images.tar
```

Ship the fastrag bundle directory (CWE + KEV corpora) on the same media. On the target:

```bash
docker load -i vams-images.tar
sudo mkdir -p /srv/vams/bundles
sudo cp -r /path/on/media/fastrag-<date> /srv/vams/bundles/
# In .env:
#   FASTRAG_BUNDLES_DIR=/srv/vams/bundles
#   FASTRAG_BUNDLE_NAME=fastrag-<date>
#   FASTRAG_TOKEN=$(openssl rand -hex 32)
#   FASTRAG_ADMIN_TOKEN=$(openssl rand -hex 32)
docker compose up -d
```

## Install — Path B: standalone DVD ISO on a separate host

Use when fastrag runs on its own VM (e.g. a dedicated reference host shared across VAMS / scribe / storm). Follow [`docs/airgap-install.md`](./airgap-install.md) verbatim for DVD mount + `docker load` + token generation. Then in VAMS's `.env`:

```bash
FASTRAG_URL=http://<fastrag-host>:8080
FASTRAG_TOKEN=<the token printed during install>
FASTRAG_ADMIN_TOKEN=<admin token — only needed if VAMS will reload bundles>
```

Bundle management is done out-of-band on the fastrag host. VAMS consumes only the HTTP surface.

## Python client

The typed client at `clients/python/` wraps every endpoint VAMS needs.

### Install

```bash
# VAMS uses uv:
uv pip install -e /path/to/fastrag/clients/python
# Or, in VAMS's pyproject.toml under [tool.uv.sources]:
#   fastrag-client = { path = "/opt/fastrag/clients/python", editable = true }
```

Requires Python ≥ 3.11. Runtime deps: `httpx`, `pydantic>=2`.

### Construct

```python
from fastrag_client import FastRAGClient

client = FastRAGClient(
    base_url=os.environ["FASTRAG_URL"],
    token=os.environ.get("FASTRAG_TOKEN"),
    admin_token=os.environ.get("FASTRAG_ADMIN_TOKEN"),  # only for reload_bundle()
    timeout=30.0,
)
```

An `AsyncFastRAGClient` with the same surface is available for async call sites.

## Use case 1 — Finding dedup at ingest

**Goal:** collapse near-duplicate findings across scanners (Semgrep / Bandit / Trivy / ZAP / …) before they land in SQLite.

**Call site to hook:** `vams/vams/core/ingest_service.py:27–54`, inside `ingest_file()`. The existing flow already fingerprint-filters against the DB (line 87) and runs the Rust dedup engine (`tarmo_vuln_core.dedup.deduplicate_findings`). The fastrag hook sits *after* those and catches semantic near-dups the fingerprint/dedup passes miss (same bug described in different words across tools).

**Corpus bootstrap (one-time).** Export existing findings to JSONL, then ingest:

```bash
fastrag index vams-findings.jsonl \
    --corpus /var/lib/fastrag/corpora/vams-findings \
    --text-fields title,description,location \
    --id-field finding_id \
    --metadata-fields source_tool,severity,cwe_id,detected_at \
    --metadata-types severity=enum,cwe_id=int,detected_at=date
```

See the complete recipe in [`README.md` lines 448–514](../README.md). Index `vams-findings` to a path **outside** the bundle directory so bundle reloads leave it untouched.

**Per-finding call (new):**

```python
from fastrag_client import FastRAGClient, SimilarHit

def is_semantic_duplicate(client: FastRAGClient, finding) -> list[SimilarHit]:
    text = f"{finding.title} — {finding.description} — {finding.location}"
    return client.similar(
        text=text,
        threshold=0.85,                                         # cosine floor
        max_results=5,
        corpus="vams-findings",
        filter=f"source_tool != {finding.source_tool}",         # same tool → fingerprint already caught it
        verify={"method": "minhash", "threshold": 0.7},         # lexical confirmation
    )

for hit in is_semantic_duplicate(client, finding):
    if hit.score >= 0.95:
        merge_into_existing(hit.metadata["finding_id"], finding)
    elif hit.score >= 0.92:
        queue_for_review(finding, candidate=hit.metadata["finding_id"])
    # else: novel — insert
```

**Threshold bands:**

| Cosine | MinHash Jaccard | Action |
|---|---|---|
| ≥ 0.95 | ≥ 0.7 | Hard merge |
| 0.92 – 0.95 | ≥ 0.7 | Human review |
| 0.85 – 0.92 | — | Keep, tag as `related` |
| < 0.85 | — | Novel — insert |

Tune on a labelled sample before promoting these thresholds. Cosine thresholds drift with embedder version; re-tune if the bundle's embedder changes.

**After insert:** append the new finding to the corpus via `client.ingest([{...}], id_field="finding_id", text_fields=["title", "description", "location"], corpus="vams-findings")` so future dedup catches it.

## Use case 2 — CWE / KEV reference lookups

**Goal:** enrich vetting context with authoritative vuln references without external network calls.

**Call site to hook:** `vams/vams/core/context_builder/__init__.py:40–49`, inside `build_vetting_context()`. The function already accepts `cwe_id`; extend the result to carry CWE hierarchy and KEV status pulled from fastrag.

```python
# CWE record (parents + children pre-populated at bundle build time)
cwe = client.get_cwe(89)                 # accepts 89 or "CWE-89"
parents = cwe.metadata.get("parents", [])
children = cwe.metadata.get("children", [])

# Deep hierarchy traversal
relation = client.cwe_relation(89, direction="both", max_depth=3)
# relation.ancestors = [943, 707, 74, 20]; relation.descendants = [564, ...]

# KEV membership for a CVE (queried via the kev corpus)
hits = client.similar(text="CVE-2021-44228", corpus="kev", max_results=1, threshold=0.0)
kev_listed = any(h.metadata.get("cve_id") == "CVE-2021-44228" for h in hits)
```

Both CWE endpoints are structured lookups — they reject `q`, `top_k`, and `filter` query params. Use `GET /query` for free-text search instead. The `vams-lookup-v1` bundle has no `/cve/{id}` endpoint; CVE ↔ KEV membership is resolved via the `kev` corpus.

**What VAMS should attach to the vetting prompt:**

- CWE name, extended description, direct parents (for taxonomy grounding), and first-level children (so the LLM can judge whether to reclassify).
- KEV presence (from the `kev` corpus) when the finding cites a CVE; drives prioritisation only, not a confidence signal.

## Use case 3 — Cross-engagement pattern matching (future)

VAMS is single-tenant today. When multi-tenant lands, fastrag's fan-out pattern is:

```python
hits = client.similar(
    text="Prototype pollution via unsanitised Object.assign merge",
    threshold=0.88,
    max_results=20,
    corpora=["acme-q1", "acme-q2", "globex-q1"],   # fan out
)
# Per-tenant scoping: pass tenant_id=<id> to the client constructor to
# AND an X-Fastrag-Tenant header into the request; the server uses it
# to restrict which records are visible.
```

Leave `FastRAGClient(tenant_id=...)` at its default (None) until VAMS exposes a tenant model.

## Bundle operations

### Initial bundle

The `vams-lookup-v1` bundle ships two fixed-name corpora — `cwe` and `kev` — alongside a `cwe-taxonomy.json` closure for hierarchy traversal. A third corpus, `vams-findings`, holds VAMS's finding history and is **not** part of the bundle; it lives on a separate mount and is registered with a second `--corpus` flag, so bundle reloads leave it untouched. The docker entrypoint passes both to `fastrag serve-http`:

```bash
fastrag serve-http \
    --bundle-path /var/lib/fastrag/bundles/vams-lookup-v1 \
    --corpus vams-findings=/var/lib/fastrag/corpora/vams-findings \
    --port 8080
```

The bundle's `cwe` and `kev` corpora are auto-registered from `bundle/corpora/` and drive `/ready`, `/cwe/{id}`, and `/cwe/relation`. The findings corpus is reached via `POST /similar` with `corpus: "vams-findings"`. Reload swaps the bundle's reference data without touching the findings corpus.

Mount the bundle directory read-only at `/var/lib/fastrag/bundles/` and set `BUNDLE_NAME` to the directory name (`vams-lookup-v1`). Mount the findings corpus separately at `/var/lib/fastrag/corpora/vams-findings` (read-write, since `POST /ingest` appends to it).

### Hot reload on new NVD/KEV snapshot

1. Copy the new bundle directory alongside the current one under `/var/lib/fastrag/bundles/`.
2. Verify the bundle's signature using `tarmo_vuln_core.signing.ReportSigner` before issuing the reload.
3. Trigger reload:
   ```python
   result = client.reload_bundle("fastrag-20260501")  # directory name, not absolute path
   # → ReloadResult(reloaded=True, bundle_id="fastrag-20260501", previous_bundle_id="fastrag-20260416")
   ```
4. Rollback by re-issuing `reload_bundle(<prior_name>)`.

Guarantees:

- **Atomic.** In-flight queries complete against the prior bundle; the swap is a single `ArcSwap::store`.
- **Serialised.** Concurrent reloads return HTTP 409; a single reload mutex guards the handler.
- **Path-escape safe.** `bundle_path` is resolved relative to `--bundles-dir`; any `..` or absolute-path attempt returns 400 `path_escape`.
- **Memory.** Peak RSS during swap is ~2× the resident bundle size. Budget 4 GiB RAM headroom.

Full operator workflow: [`docs/airgap-install.md` § Updating bundles](./airgap-install.md#updating-bundles).

### `/ready` reason codes

On 503, `ReadyStatus.reasons` contains one or more of:

| Code | Meaning |
|---|---|
| `bundle_not_loaded` | Startup still in progress, or initial bundle invalid |
| `corpus_{name}_missing` | A required corpus (derived from the bundle's `manifest.corpora`) is absent from the loaded bundle |
| `embedder_unreachable` | Internal `llama-server` embedder subprocess not responding |
| `reranker_unreachable` | Internal `llama-server` reranker subprocess not responding |

## Environment variable reference

| Variable | Scope | Default | Required | Purpose |
|---|---|---|---|---|
| `BUNDLE_NAME` | fastrag container | — | yes | Directory under `/var/lib/fastrag/bundles/` to load at startup |
| `BUNDLES_DIR` | fastrag container | `/var/lib/fastrag/bundles` | no | Override the bundles root |
| `FASTRAG_TOKEN` | fastrag + VAMS | — | recommended | Read token for `/query`, `/similar`, `/cwe`, `/metrics` |
| `FASTRAG_ADMIN_TOKEN` | fastrag + VAMS | — | reload only | Admin token for `POST /admin/reload`; must differ from the read token |
| `FASTRAG_URL` | VAMS | `http://fastrag:8080` | yes | Where VAMS reaches fastrag |
| `FASTRAG_PORT` | compose host | `8080` | no | Host-side port mapping |
| `FASTRAG_BUNDLES_DIR` | compose host | `./bundles` | no | Host directory mounted into the container |
| `FASTRAG_BUNDLE_NAME` | compose host | `vams-lookup-v1` | yes | Bundle directory to load |
| `PORT` | fastrag container | `8080` | no | Listen port inside the container |

## Verification checklist

Run these from a shell with access to the compose network (e.g. `docker compose exec vams-api sh`) or from outside with `FASTRAG_URL` pointing at the exposed port.

```bash
# 1. Liveness + readiness.
curl -fsS "$FASTRAG_URL/health"
curl -fsS "$FASTRAG_URL/ready"
# expect: {"ready": true, ...}

# 2. CWE lookup + hierarchy.
curl -fsS -H "x-fastrag-token: $FASTRAG_TOKEN" \
    "$FASTRAG_URL/cwe/89"
curl -fsS -H "x-fastrag-token: $FASTRAG_TOKEN" \
    "$FASTRAG_URL/cwe/relation?cwe_id=89&direction=both"
# expect: ancestors includes 943, 707, 74, 20

# 3. Python dedup smoke.
python - <<'PY'
import os
from fastrag_client import FastRAGClient
c = FastRAGClient(os.environ["FASTRAG_URL"], token=os.environ.get("FASTRAG_TOKEN"))
print(c.ready())
print(c.similar("SQL injection in /api/v1/login", threshold=0.5, corpus="vams-findings", max_results=3))
PY
```

Exit criteria: all three commands return 200 / non-empty results, and `ready()` returns `ReadyStatus(ok=True, ...)`.

## Out of scope

These belong elsewhere, not in fastrag:

- **LLM inference / chat completion.** Keep calling llama-server directly via `TARMO_VETTING_TIER{1,2}_*`.
- **Grounded response auditing** (citation density, hallucination gates). If VAMS wants this, it belongs in a thin shim between `vams.core.vetting_service._call_llm` and llama-server. tarmo-llm-rag did this for scribe; the pattern is portable but not a fastrag concern.
- **Storing secrets, session tokens, or raw credentials** in any corpus. Scrub before `POST /similar` or `POST /ingest`.
- **Cross-tenant queries without explicit `X-Fastrag-Tenant`.** Until VAMS has a tenant model, don't opt into multi-corpus fan-out.

## References

- Design spec: [`docs/superpowers/specs/2026-04-16-fastrag-for-vams-design.md`](./superpowers/specs/2026-04-16-fastrag-for-vams-design.md)
- Airgap operator guide: [`docs/airgap-install.md`](./airgap-install.md)
- HTTP surface + metrics + auth: [`README.md` § Deployment](../README.md#deployment)
- Dedup pipeline recipe: [`README.md` § Similarity Search](../README.md#similarity-search)
- Python client source: `clients/python/src/fastrag_client/`
- VAMS ingest call site: `vams/vams/core/ingest_service.py` (`ingest_file`)
- VAMS vetting-context call site: `vams/vams/core/context_builder/__init__.py` (`build_vetting_context`)
