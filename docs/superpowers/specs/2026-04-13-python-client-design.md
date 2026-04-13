# Python Client Library — #45

## Goal

Typed Python client for fastrag's HTTP API so that VAMS, pentest-scribe, and pentest-storm can query, ingest, and manage corpora without manual HTTP construction.

## Architecture

Two concrete client classes — `FastRAGClient` (sync, backed by `httpx.Client`) and `AsyncFastRAGClient` (async, backed by `httpx.AsyncClient`). Both share the same method signatures and use shared helpers for URL/param construction. A filter builder (`F.field.op(value)`) produces the server's string filter syntax.

Separate repo: `~/github/fastrag-client-python/`. Package name: `fastrag-client`. Import as `fastrag_client`.

## Package Structure

```
fastrag-client-python/
├── pyproject.toml              # hatchling, Python 3.11+
├── src/
│   └── fastrag_client/
│       ├── __init__.py         # re-exports FastRAGClient, AsyncFastRAGClient, F
│       ├── client.py           # FastRAGClient (sync)
│       ├── async_client.py     # AsyncFastRAGClient
│       ├── models.py           # Pydantic v2 response models
│       ├── filters.py          # F.field.op(value) filter builder
│       └── errors.py           # Exception hierarchy
└── tests/
    ├── conftest.py             # shared fixtures (respx mock transport)
    ├── test_client.py          # sync client tests
    ├── test_async_client.py    # async client tests
    ├── test_filters.py         # filter builder tests
    └── test_models.py          # Pydantic model parsing tests
```

## Dependencies

**Runtime:** `httpx`, `pydantic>=2.0`

**Dev:** `pytest`, `pytest-asyncio`, `respx` (httpx mocking), `ruff`

## Client Construction

```python
client = FastRAGClient(
    base_url="http://localhost:8081",
    token="secret",           # optional — sets X-Fastrag-Token header
    tenant_id="acme",         # optional — sets X-Fastrag-Tenant header
    timeout=30.0,             # optional — request timeout in seconds, default 30
)
```

Both `FastRAGClient` and `AsyncFastRAGClient` accept the same constructor arguments.

When `token` is provided, every request includes `X-Fastrag-Token: <token>`. When `tenant_id` is provided, every request includes `X-Fastrag-Tenant: <tenant_id>`.

## Endpoints Wrapped

| Server Endpoint | Client Method | Returns |
|-----------------|---------------|---------|
| `GET /query` | `client.query(q, ...)` | `list[SearchHit]` |
| `POST /batch-query` | `client.batch_query(queries)` | `list[BatchResult]` |
| `POST /ingest` | `client.ingest(records, ...)` | `IngestResult` |
| `DELETE /ingest/{id}` | `client.delete(id, ...)` | `DeleteResult` |
| `GET /stats` | `client.stats(...)` | `dict[str, Any]` |
| `GET /corpora` | `client.corpora()` | `list[CorpusInfo]` |
| `GET /health` | `client.health()` | `bool` |

`GET /metrics` is not wrapped — Prometheus scrapes it directly.

## Method Signatures

### `query`

```python
def query(
    self,
    q: str,
    *,
    top_k: int = 5,
    corpus: str = "default",
    filter: FilterExpr | None = None,
    snippet_len: int = 150,
    fields: list[str] | None = None,
    rerank: str | None = None,
    over_fetch: int | None = None,
) -> list[SearchHit]:
```

`fields` accepts a list of strings. Include mode: `["score", "snippet"]`. Exclude mode: `["-chunk_text", "-source"]`. The client joins them with commas for the `fields` query param.

### `batch_query`

```python
def batch_query(
    self,
    queries: list[dict[str, Any]],
) -> list[BatchResult]:
```

Each dict in `queries` can contain: `q` (required), `top_k`, `corpus`, `filter` (a `FilterExpr` or string), `snippet_len`, `fields`. The client serializes `FilterExpr` to string before sending.

### `ingest`

```python
def ingest(
    self,
    records: list[dict[str, Any]],
    *,
    id_field: str,
    text_fields: list[str],
    metadata_fields: list[str] | None = None,
    metadata_types: dict[str, str] | None = None,
    array_fields: list[str] | None = None,
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    corpus: str = "default",
) -> IngestResult:
```

Serializes `records` to NDJSON body. Sends `Content-Type: application/x-ndjson`. `metadata_types` is a dict like `{"cvss": "numeric", "published": "date"}` — the client formats it as `cvss=numeric,published=date` for the query param.

### `delete`

```python
def delete(self, id: str, *, corpus: str = "default") -> DeleteResult:
```

### `stats`

```python
def stats(self, *, corpus: str = "default") -> dict[str, Any]:
```

Returns raw dict — the stats response shape is dynamic and evolving.

### `corpora`

```python
def corpora(self) -> list[CorpusInfo]:
```

### `health`

```python
def health(self) -> bool:
```

Returns `True` if the server responds with 200, `False` on connection error. Does not raise exceptions.

## Response Models

```python
class SearchHit(BaseModel):
    score: float
    chunk_text: str = ""
    snippet: str | None = None
    source: dict[str, Any] | None = None
    source_path: str = ""
    chunk_index: int = 0
    section: str | None = None
    pages: list[int] = []
    element_kinds: list[str] = []
    language: str | None = None
    metadata: dict[str, Any] = {}

class BatchResult(BaseModel):
    index: int
    hits: list[SearchHit] | None = None
    error: str | None = None

class IngestResult(BaseModel):
    corpus: str
    records_new: int
    records_updated: int
    records_unchanged: int
    chunks_added: int

class DeleteResult(BaseModel):
    corpus: str
    id: str
    deleted: bool

class CorpusInfo(BaseModel):
    name: str
    path: str
    status: str
```

All models use `model_config = ConfigDict(extra="allow")` so that new server fields don't break the client.

## Filter Builder

```python
from fastrag_client.filters import F

F.severity == "HIGH"                          # "severity = HIGH"
F.cvss >= 7.0                                 # "cvss >= 7.0"
F.cvss > 5.0                                  # "cvss > 5.0"
F.severity.in_(["HIGH", "CRITICAL"])          # "severity IN (HIGH, CRITICAL)"
(F.severity == "HIGH") & (F.cvss >= 7.0)      # "severity = HIGH, cvss >= 7.0"
```

### Implementation

`F` is a module-level `FieldFactory` instance. Attribute access (`F.severity`) returns a `FieldExpr(name="severity")`.

`FieldExpr` implements `__eq__`, `__ge__`, `__gt__`, `__le__`, `__lt__`, and `in_()`. Each returns a `FilterExpr`.

`FilterExpr` implements `__and__` to combine expressions. `__str__` produces the server's string syntax.

No `__or__` — the server's filter syntax only supports comma-separated AND. Add OR support when the server gains it.

## Error Handling

```python
class FastRAGError(Exception):
    """Base exception. Carries status_code and body when available."""
    status_code: int | None
    body: str | None

class AuthenticationError(FastRAGError):    # 401
class NotFoundError(FastRAGError):          # 404
class ValidationError(FastRAGError):        # 400
class PayloadTooLargeError(FastRAGError):   # 413
class ServerError(FastRAGError):            # 500, 503
class ConnectionError(FastRAGError):        # connection refused, timeout, DNS
```

The client maps HTTP status codes to exceptions. No raw `httpx.HTTPStatusError` leaks to callers. `ConnectionError` wraps `httpx.ConnectError`, `httpx.TimeoutException`, etc.

## Testing

Unit tests only — no integration tests requiring a running fastrag server. Use `respx` to mock httpx transports.

### `test_client.py`

One test per method (7 methods × happy path), plus error mapping tests:
- `test_query_returns_search_hits` — mock 200 with JSON fixture, verify `list[SearchHit]`
- `test_query_with_filter` — verify filter string appears in query params
- `test_query_with_fields` — verify fields param joined correctly
- `test_batch_query_returns_batch_results`
- `test_ingest_sends_ndjson` — verify Content-Type and body format
- `test_delete_returns_delete_result`
- `test_stats_returns_dict`
- `test_corpora_returns_list`
- `test_health_returns_true`
- `test_health_returns_false_on_connection_error`
- `test_auth_error_raises` — mock 401, verify `AuthenticationError`
- `test_not_found_raises` — mock 404, verify `NotFoundError`
- `test_validation_error_raises` — mock 400, verify `ValidationError`
- `test_token_header_sent` — verify `X-Fastrag-Token` header present
- `test_tenant_header_sent` — verify `X-Fastrag-Tenant` header present

### `test_async_client.py`

Same tests as sync, using `pytest-asyncio`.

### `test_filters.py`

- `test_eq` — `F.severity == "HIGH"` → `"severity = HIGH"`
- `test_ge` — `F.cvss >= 7.0` → `"cvss >= 7.0"`
- `test_gt` — `F.cvss > 5.0` → `"cvss > 5.0"`
- `test_le` — `F.cvss <= 3.0` → `"cvss <= 3.0"`
- `test_lt` — `F.cvss < 3.0` → `"cvss < 3.0"`
- `test_in` — `F.severity.in_(["HIGH", "CRITICAL"])` → `"severity IN (HIGH, CRITICAL)"`
- `test_and` — `(F.severity == "HIGH") & (F.cvss >= 7.0)` → `"severity = HIGH, cvss >= 7.0"`
- `test_str_renders` — verify `str()` on all expr types

### `test_models.py`

- Parse raw JSON fixtures into each Pydantic model
- Verify extra fields don't break parsing (`extra="allow"`)
- Verify optional fields default correctly when absent

## Out of Scope

- WebSocket streaming
- Retry/backoff (callers can wrap with tenacity)
- Connection pooling configuration (httpx defaults are fine)
- PyPI publishing automation (manual for now)
- CLI wrapper
