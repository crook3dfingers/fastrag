# Python Client Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `fastrag-client`, a typed Python client for fastrag's HTTP API with sync/async support, filter builder, and Pydantic v2 response models.

**Architecture:** Two concrete client classes (`FastRAGClient`, `AsyncFastRAGClient`) backed by `httpx`, sharing helpers for URL/param construction. A filter builder (`F.field.op(value)`) produces server-compatible filter strings. Pydantic v2 models parse responses.

**Tech Stack:** Python 3.11+, httpx, pydantic v2, hatchling (build), pytest + respx (test)

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata, dependencies, build config |
| `src/fastrag_client/__init__.py` | Public API re-exports |
| `src/fastrag_client/errors.py` | Exception hierarchy |
| `src/fastrag_client/models.py` | Pydantic v2 response models |
| `src/fastrag_client/filters.py` | Filter builder (`F.field.op(value)`) |
| `src/fastrag_client/client.py` | `FastRAGClient` (sync) |
| `src/fastrag_client/async_client.py` | `AsyncFastRAGClient` |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_filters.py` | Filter builder tests |
| `tests/test_models.py` | Pydantic model tests |
| `tests/test_client.py` | Sync client tests |
| `tests/test_async_client.py` | Async client tests |

**Working directory:** `~/github/fastrag-client-python/` (created in Task 1)

---

### Task 1: Project scaffolding

**Files:**
- Create: `~/github/fastrag-client-python/pyproject.toml`
- Create: `~/github/fastrag-client-python/src/fastrag_client/__init__.py`
- Create: `~/github/fastrag-client-python/tests/__init__.py`

- [ ] **Step 1: Create repo directory and initialize git**

```bash
mkdir -p ~/github/fastrag-client-python
cd ~/github/fastrag-client-python
git init
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fastrag-client"
version = "0.1.0"
description = "Typed Python client for fastrag's HTTP API"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "respx>=0.22",
    "ruff>=0.4",
]

[tool.hatch.build.targets.wheel]
packages = ["src/fastrag_client"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

- [ ] **Step 3: Create package directory structure**

```bash
mkdir -p src/fastrag_client tests
touch src/fastrag_client/__init__.py
touch tests/__init__.py
```

Write `src/fastrag_client/__init__.py` (placeholder — filled in Task 7):

```python
"""Typed Python client for fastrag's HTTP API."""
```

- [ ] **Step 4: Install in dev mode**

```bash
cd ~/github/fastrag-client-python
pip install -e ".[dev]"
```

- [ ] **Step 5: Verify pytest runs**

```bash
cd ~/github/fastrag-client-python
pytest tests/ -v
```

Expected: `no tests ran` (0 collected), exit 0 or 5 (no tests).

- [ ] **Step 6: Commit**

```bash
cd ~/github/fastrag-client-python
git add .
git commit -m "chore: project scaffolding with hatchling + dev deps"
```

---

### Task 2: Error hierarchy

**Files:**
- Create: `src/fastrag_client/errors.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_errors.py`:

```python
from fastrag_client.errors import (
    AuthenticationError,
    ConnectionError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)


def test_base_error_carries_status_and_body():
    err = FastRAGError("fail", status_code=500, body='{"error":"boom"}')
    assert str(err) == "fail"
    assert err.status_code == 500
    assert err.body == '{"error":"boom"}'


def test_base_error_defaults_none():
    err = FastRAGError("fail")
    assert err.status_code is None
    assert err.body is None


def test_subclasses_inherit_from_base():
    assert issubclass(AuthenticationError, FastRAGError)
    assert issubclass(NotFoundError, FastRAGError)
    assert issubclass(ValidationError, FastRAGError)
    assert issubclass(PayloadTooLargeError, FastRAGError)
    assert issubclass(ServerError, FastRAGError)
    assert issubclass(ConnectionError, FastRAGError)


def test_subclass_carries_status():
    err = AuthenticationError("denied", status_code=401, body="unauthorized")
    assert err.status_code == 401
    assert isinstance(err, FastRAGError)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_errors.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `errors.py`**

Create `src/fastrag_client/errors.py`:

```python
"""Exception hierarchy for fastrag client."""


class FastRAGError(Exception):
    """Base exception for all fastrag client errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthenticationError(FastRAGError):
    """401 — missing or invalid token."""


class NotFoundError(FastRAGError):
    """404 — unknown corpus or record."""


class ValidationError(FastRAGError):
    """400 — bad filter, mixed field selectors, malformed JSON."""


class PayloadTooLargeError(FastRAGError):
    """413 — ingest body exceeds server limit."""


class ServerError(FastRAGError):
    """500/503 — server-side failure."""


class ConnectionError(FastRAGError):
    """Connection refused, timeout, DNS failure."""
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_errors.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/errors.py tests/test_errors.py
git commit -m "feat: add exception hierarchy"
```

---

### Task 3: Response models

**Files:**
- Create: `src/fastrag_client/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
from typing import Any

from fastrag_client.models import (
    BatchResult,
    CorpusInfo,
    DeleteResult,
    IngestResult,
    SearchHit,
)


def test_search_hit_from_full_json():
    raw: dict[str, Any] = {
        "score": 0.87,
        "chunk_text": "SQL injection vulnerability",
        "snippet": "<b>SQL</b> injection",
        "source": {"id": "cve-1", "body": "SQL injection..."},
        "source_path": "cve-1",
        "chunk_index": 0,
        "section": "intro",
        "pages": [1, 2],
        "element_kinds": ["text"],
        "language": "en",
        "metadata": {"severity": "HIGH", "cvss": 9.8},
    }
    hit = SearchHit.model_validate(raw)
    assert hit.score == 0.87
    assert hit.snippet == "<b>SQL</b> injection"
    assert hit.source["id"] == "cve-1"
    assert hit.metadata["cvss"] == 9.8


def test_search_hit_minimal_json():
    raw: dict[str, Any] = {"score": 0.5}
    hit = SearchHit.model_validate(raw)
    assert hit.score == 0.5
    assert hit.chunk_text == ""
    assert hit.snippet is None
    assert hit.source is None
    assert hit.metadata == {}


def test_search_hit_extra_fields_allowed():
    raw: dict[str, Any] = {"score": 0.5, "new_field": "value"}
    hit = SearchHit.model_validate(raw)
    assert hit.score == 0.5


def test_batch_result_with_hits():
    raw: dict[str, Any] = {
        "index": 0,
        "hits": [{"score": 0.9, "chunk_text": "test"}],
    }
    result = BatchResult.model_validate(raw)
    assert result.index == 0
    assert len(result.hits) == 1
    assert result.hits[0].score == 0.9
    assert result.error is None


def test_batch_result_with_error():
    raw: dict[str, Any] = {"index": 1, "error": "corpus not found"}
    result = BatchResult.model_validate(raw)
    assert result.hits is None
    assert result.error == "corpus not found"


def test_ingest_result():
    raw: dict[str, Any] = {
        "corpus": "default",
        "records_new": 2,
        "records_updated": 0,
        "records_unchanged": 0,
        "chunks_added": 6,
    }
    result = IngestResult.model_validate(raw)
    assert result.corpus == "default"
    assert result.records_new == 2
    assert result.chunks_added == 6


def test_delete_result():
    raw: dict[str, Any] = {"corpus": "default", "id": "cve-1", "deleted": True}
    result = DeleteResult.model_validate(raw)
    assert result.deleted is True


def test_corpus_info():
    raw: dict[str, Any] = {"name": "default", "path": "/data/corpus", "status": "ready"}
    info = CorpusInfo.model_validate(raw)
    assert info.name == "default"
    assert info.status == "ready"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_models.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `models.py`**

Create `src/fastrag_client/models.py`:

```python
"""Pydantic v2 response models for fastrag API."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    model_config = ConfigDict(extra="allow")

    index: int
    hits: list[SearchHit] | None = None
    error: str | None = None


class IngestResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    corpus: str
    records_new: int
    records_updated: int
    records_unchanged: int
    chunks_added: int


class DeleteResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    corpus: str
    id: str
    deleted: bool


class CorpusInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    path: str
    status: str
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_models.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/models.py tests/test_models.py
git commit -m "feat: add Pydantic v2 response models"
```

---

### Task 4: Filter builder

**Files:**
- Create: `src/fastrag_client/filters.py`
- Create: `tests/test_filters.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_filters.py`:

```python
from fastrag_client.filters import F


def test_eq_string():
    expr = F.severity == "HIGH"
    assert str(expr) == "severity = HIGH"


def test_eq_numeric():
    expr = F.cvss == 9.8
    assert str(expr) == "cvss = 9.8"


def test_ge():
    expr = F.cvss >= 7.0
    assert str(expr) == "cvss >= 7.0"


def test_gt():
    expr = F.cvss > 5.0
    assert str(expr) == "cvss > 5.0"


def test_le():
    expr = F.cvss <= 3.0
    assert str(expr) == "cvss <= 3.0"


def test_lt():
    expr = F.cvss < 3.0
    assert str(expr) == "cvss < 3.0"


def test_in():
    expr = F.severity.in_(["HIGH", "CRITICAL"])
    assert str(expr) == "severity IN (HIGH, CRITICAL)"


def test_and_two():
    expr = (F.severity == "HIGH") & (F.cvss >= 7.0)
    assert str(expr) == "severity = HIGH, cvss >= 7.0"


def test_and_three():
    expr = (F.severity == "HIGH") & (F.cvss >= 7.0) & (F.source_tool == "semgrep")
    assert str(expr) == "severity = HIGH, cvss >= 7.0, source_tool = semgrep"


def test_field_access_returns_field_expr():
    field = F.severity
    assert hasattr(field, "in_")
    assert hasattr(field, "__eq__")


def test_in_single_value():
    expr = F.severity.in_(["HIGH"])
    assert str(expr) == "severity IN (HIGH)"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_filters.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `filters.py`**

Create `src/fastrag_client/filters.py`:

```python
"""Filter builder for fastrag query API.

Usage:
    from fastrag_client.filters import F

    expr = (F.severity == "HIGH") & (F.cvss >= 7.0)
    str(expr)  # "severity = HIGH, cvss >= 7.0"
"""

from __future__ import annotations

from typing import Any


class FilterExpr:
    """A single filter condition or a conjunction of conditions."""

    def __init__(self, parts: list[str]) -> None:
        self._parts = parts

    def __and__(self, other: FilterExpr) -> FilterExpr:
        return FilterExpr(self._parts + other._parts)

    def __str__(self) -> str:
        return ", ".join(self._parts)

    def __repr__(self) -> str:
        return f"FilterExpr({self._parts!r})"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        # Drop trailing zeros but keep at least one decimal
        s = f"{value:g}"
        return s
    return str(value)


class FieldExpr:
    """Proxy for a field name. Comparison operators produce FilterExpr."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __eq__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} = {_format_value(other)}"])

    def __ge__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} >= {_format_value(other)}"])

    def __gt__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} > {_format_value(other)}"])

    def __le__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} <= {_format_value(other)}"])

    def __lt__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} < {_format_value(other)}"])

    def in_(self, values: list[Any]) -> FilterExpr:
        formatted = ", ".join(_format_value(v) for v in values)
        return FilterExpr([f"{self._name} IN ({formatted})"])


class FieldFactory:
    """Factory that creates FieldExpr via attribute access.

    Usage: F.severity returns FieldExpr("severity")
    """

    def __getattr__(self, name: str) -> FieldExpr:
        return FieldExpr(name)


F = FieldFactory()
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_filters.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/filters.py tests/test_filters.py
git commit -m "feat: add filter builder (F.field.op(value))"
```

---

### Task 5: Sync client

**Files:**
- Create: `src/fastrag_client/client.py`
- Create: `tests/conftest.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/conftest.py`:

```python
"""Shared test fixtures."""

import json

import pytest

SEARCH_HIT_FIXTURE: dict = {
    "score": 0.87,
    "chunk_text": "SQL injection vulnerability allows remote code execution",
    "snippet": "<b>SQL</b> <b>injection</b> vulnerability",
    "source": {"id": "cve-1", "body": "SQL injection..."},
    "source_path": "cve-1",
    "chunk_index": 0,
    "metadata": {"severity": "HIGH"},
}

INGEST_FIXTURE: dict = {
    "corpus": "default",
    "records_new": 2,
    "records_updated": 0,
    "records_unchanged": 0,
    "chunks_added": 6,
}

DELETE_FIXTURE: dict = {
    "corpus": "default",
    "id": "cve-1",
    "deleted": True,
}

CORPORA_FIXTURE: dict = {
    "corpora": [
        {"name": "default", "path": "/data/corpus", "status": "ready"},
    ],
}

BATCH_FIXTURE: dict = {
    "results": [
        {"index": 0, "hits": [SEARCH_HIT_FIXTURE]},
        {"index": 1, "error": "corpus not found"},
    ],
}


@pytest.fixture
def base_url() -> str:
    return "http://testserver:8081"
```

Create `tests/test_client.py`:

```python
"""Tests for sync FastRAGClient."""

import json

import httpx
import pytest
import respx

from fastrag_client.client import FastRAGClient
from fastrag_client.errors import (
    AuthenticationError,
    NotFoundError,
    ValidationError,
)
from fastrag_client.filters import F
from fastrag_client.models import (
    BatchResult,
    CorpusInfo,
    DeleteResult,
    IngestResult,
    SearchHit,
)

from .conftest import (
    BATCH_FIXTURE,
    CORPORA_FIXTURE,
    DELETE_FIXTURE,
    INGEST_FIXTURE,
    SEARCH_HIT_FIXTURE,
)


@respx.mock
def test_query_returns_search_hits(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE])
    )
    client = FastRAGClient(base_url=base_url)
    hits = client.query("SQL injection")
    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].score == 0.87
    assert hits[0].snippet == "<b>SQL</b> <b>injection</b> vulnerability"


@respx.mock
def test_query_with_filter(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE])
    )
    client = FastRAGClient(base_url=base_url)
    client.query("SQL injection", filter=F.severity == "HIGH")
    request = route.calls[0].request
    assert "filter=severity+%3D+HIGH" in str(request.url) or "filter=severity" in str(request.url)


@respx.mock
def test_query_with_fields(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE])
    )
    client = FastRAGClient(base_url=base_url)
    client.query("SQL injection", fields=["score", "snippet"])
    request = route.calls[0].request
    assert "fields=" in str(request.url)


@respx.mock
def test_batch_query_returns_batch_results(base_url: str):
    respx.post(f"{base_url}/batch-query").mock(
        return_value=httpx.Response(200, json=BATCH_FIXTURE)
    )
    client = FastRAGClient(base_url=base_url)
    results = client.batch_query([{"q": "SQL injection", "top_k": 3}])
    assert len(results) == 2
    assert isinstance(results[0], BatchResult)
    assert results[0].hits is not None
    assert len(results[0].hits) == 1
    assert results[1].error == "corpus not found"


@respx.mock
def test_ingest_sends_ndjson(base_url: str):
    route = respx.post(f"{base_url}/ingest").mock(
        return_value=httpx.Response(200, json=INGEST_FIXTURE)
    )
    client = FastRAGClient(base_url=base_url)
    records = [{"id": "v1", "body": "test"}, {"id": "v2", "body": "test2"}]
    result = client.ingest(records, id_field="id", text_fields=["body"])
    assert isinstance(result, IngestResult)
    assert result.records_new == 2
    request = route.calls[0].request
    assert request.headers["content-type"] == "application/x-ndjson"
    lines = request.content.decode().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "v1"


@respx.mock
def test_delete_returns_delete_result(base_url: str):
    respx.delete(f"{base_url}/ingest/cve-1").mock(
        return_value=httpx.Response(200, json=DELETE_FIXTURE)
    )
    client = FastRAGClient(base_url=base_url)
    result = client.delete("cve-1")
    assert isinstance(result, DeleteResult)
    assert result.deleted is True


@respx.mock
def test_stats_returns_dict(base_url: str):
    stats = {"corpus": "default", "entries": {"live": 10}}
    respx.get(f"{base_url}/stats").mock(
        return_value=httpx.Response(200, json=stats)
    )
    client = FastRAGClient(base_url=base_url)
    result = client.stats()
    assert result["entries"]["live"] == 10


@respx.mock
def test_corpora_returns_list(base_url: str):
    respx.get(f"{base_url}/corpora").mock(
        return_value=httpx.Response(200, json=CORPORA_FIXTURE)
    )
    client = FastRAGClient(base_url=base_url)
    result = client.corpora()
    assert len(result) == 1
    assert isinstance(result[0], CorpusInfo)
    assert result[0].name == "default"


@respx.mock
def test_health_returns_true(base_url: str):
    respx.get(f"{base_url}/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = FastRAGClient(base_url=base_url)
    assert client.health() is True


@respx.mock
def test_health_returns_false_on_error(base_url: str):
    respx.get(f"{base_url}/health").mock(side_effect=httpx.ConnectError("refused"))
    client = FastRAGClient(base_url=base_url)
    assert client.health() is False


@respx.mock
def test_auth_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(AuthenticationError) as exc_info:
        client.query("test")
    assert exc_info.value.status_code == 401


@respx.mock
def test_not_found_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(404, text="corpus not found")
    )
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(NotFoundError) as exc_info:
        client.query("test")
    assert exc_info.value.status_code == 404


@respx.mock
def test_validation_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(400, text="bad filter")
    )
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(ValidationError):
        client.query("test")


@respx.mock
def test_token_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = FastRAGClient(base_url=base_url, token="secret123")
    client.query("test")
    request = route.calls[0].request
    assert request.headers["x-fastrag-token"] == "secret123"


@respx.mock
def test_tenant_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = FastRAGClient(base_url=base_url, tenant_id="acme")
    client.query("test")
    request = route.calls[0].request
    assert request.headers["x-fastrag-tenant"] == "acme"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_client.py -v
```

Expected: `ImportError` — `FastRAGClient` doesn't exist.

- [ ] **Step 3: Implement `client.py`**

Create `src/fastrag_client/client.py`:

```python
"""Synchronous fastrag HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import (
    AuthenticationError,
    ConnectionError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)
from .filters import FilterExpr
from .models import BatchResult, CorpusInfo, DeleteResult, IngestResult, SearchHit

_STATUS_MAP: dict[int, type[FastRAGError]] = {
    400: ValidationError,
    401: AuthenticationError,
    404: NotFoundError,
    413: PayloadTooLargeError,
    500: ServerError,
    503: ServerError,
}


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        exc_cls = _STATUS_MAP.get(resp.status_code, FastRAGError)
        raise exc_cls(
            resp.text,
            status_code=resp.status_code,
            body=resp.text,
        )


def _build_headers(
    token: str | None,
    tenant_id: str | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["x-fastrag-token"] = token
    if tenant_id:
        headers["x-fastrag-tenant"] = tenant_id
    return headers


class FastRAGClient:
    """Synchronous client for fastrag's HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=_build_headers(token, tenant_id),
            timeout=timeout,
        )

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
        params: dict[str, Any] = {
            "q": q,
            "top_k": top_k,
            "corpus": corpus,
            "snippet_len": snippet_len,
        }
        if filter is not None:
            params["filter"] = str(filter)
        if fields is not None:
            params["fields"] = ",".join(fields)
        if rerank is not None:
            params["rerank"] = rerank
        if over_fetch is not None:
            params["over_fetch"] = over_fetch

        resp = self._client.get("/query", params=params)
        _raise_for_status(resp)
        return [SearchHit.model_validate(h) for h in resp.json()]

    def batch_query(
        self,
        queries: list[dict[str, Any]],
    ) -> list[BatchResult]:
        # Serialize any FilterExpr in query dicts
        serialized = []
        for q in queries:
            item = dict(q)
            if "filter" in item and isinstance(item["filter"], FilterExpr):
                item["filter"] = str(item["filter"])
            serialized.append(item)

        resp = self._client.post("/batch-query", json={"queries": serialized})
        _raise_for_status(resp)
        data = resp.json()
        return [BatchResult.model_validate(r) for r in data["results"]]

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
        params: dict[str, Any] = {
            "corpus": corpus,
            "id_field": id_field,
            "text_fields": ",".join(text_fields),
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        if metadata_fields:
            params["metadata_fields"] = ",".join(metadata_fields)
        if metadata_types:
            params["metadata_types"] = ",".join(f"{k}={v}" for k, v in metadata_types.items())
        if array_fields:
            params["array_fields"] = ",".join(array_fields)

        ndjson = "\n".join(json.dumps(r) for r in records) + "\n"
        resp = self._client.post(
            "/ingest",
            content=ndjson.encode(),
            headers={"content-type": "application/x-ndjson"},
            params=params,
        )
        _raise_for_status(resp)
        return IngestResult.model_validate(resp.json())

    def delete(self, id: str, *, corpus: str = "default") -> DeleteResult:
        resp = self._client.delete(f"/ingest/{id}", params={"corpus": corpus})
        _raise_for_status(resp)
        return DeleteResult.model_validate(resp.json())

    def stats(self, *, corpus: str = "default") -> dict[str, Any]:
        resp = self._client.get("/stats", params={"corpus": corpus})
        _raise_for_status(resp)
        return resp.json()

    def corpora(self) -> list[CorpusInfo]:
        resp = self._client.get("/corpora")
        _raise_for_status(resp)
        return [CorpusInfo.model_validate(c) for c in resp.json()["corpora"]]

    def health(self) -> bool:
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FastRAGClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_client.py -v
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/client.py tests/conftest.py tests/test_client.py
git commit -m "feat: add sync FastRAGClient with all 7 endpoints"
```

---

### Task 6: Async client

**Files:**
- Create: `src/fastrag_client/async_client.py`
- Create: `tests/test_async_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_async_client.py`:

```python
"""Tests for async AsyncFastRAGClient."""

import json

import httpx
import pytest
import respx

from fastrag_client.async_client import AsyncFastRAGClient
from fastrag_client.errors import AuthenticationError, NotFoundError
from fastrag_client.filters import F
from fastrag_client.models import (
    BatchResult,
    CorpusInfo,
    DeleteResult,
    IngestResult,
    SearchHit,
)

from .conftest import (
    BATCH_FIXTURE,
    CORPORA_FIXTURE,
    DELETE_FIXTURE,
    INGEST_FIXTURE,
    SEARCH_HIT_FIXTURE,
)


@respx.mock
async def test_query_returns_search_hits(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE])
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        hits = await client.query("SQL injection")
    assert len(hits) == 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].score == 0.87


@respx.mock
async def test_query_with_filter(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE])
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        await client.query("SQL injection", filter=F.severity == "HIGH")
    request = route.calls[0].request
    assert "filter=" in str(request.url)


@respx.mock
async def test_batch_query_returns_batch_results(base_url: str):
    respx.post(f"{base_url}/batch-query").mock(
        return_value=httpx.Response(200, json=BATCH_FIXTURE)
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        results = await client.batch_query([{"q": "SQL injection", "top_k": 3}])
    assert len(results) == 2
    assert isinstance(results[0], BatchResult)


@respx.mock
async def test_ingest_sends_ndjson(base_url: str):
    route = respx.post(f"{base_url}/ingest").mock(
        return_value=httpx.Response(200, json=INGEST_FIXTURE)
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        result = await client.ingest(
            [{"id": "v1", "body": "test"}], id_field="id", text_fields=["body"]
        )
    assert isinstance(result, IngestResult)
    request = route.calls[0].request
    assert request.headers["content-type"] == "application/x-ndjson"


@respx.mock
async def test_delete_returns_delete_result(base_url: str):
    respx.delete(f"{base_url}/ingest/cve-1").mock(
        return_value=httpx.Response(200, json=DELETE_FIXTURE)
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        result = await client.delete("cve-1")
    assert isinstance(result, DeleteResult)
    assert result.deleted is True


@respx.mock
async def test_stats_returns_dict(base_url: str):
    respx.get(f"{base_url}/stats").mock(
        return_value=httpx.Response(200, json={"corpus": "default"})
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        result = await client.stats()
    assert result["corpus"] == "default"


@respx.mock
async def test_corpora_returns_list(base_url: str):
    respx.get(f"{base_url}/corpora").mock(
        return_value=httpx.Response(200, json=CORPORA_FIXTURE)
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        result = await client.corpora()
    assert len(result) == 1
    assert isinstance(result[0], CorpusInfo)


@respx.mock
async def test_health_returns_true(base_url: str):
    respx.get(f"{base_url}/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        assert await client.health() is True


@respx.mock
async def test_health_returns_false_on_error(base_url: str):
    respx.get(f"{base_url}/health").mock(side_effect=httpx.ConnectError("refused"))
    async with AsyncFastRAGClient(base_url=base_url) as client:
        assert await client.health() is False


@respx.mock
async def test_auth_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(401, text="unauthorized")
    )
    async with AsyncFastRAGClient(base_url=base_url) as client:
        with pytest.raises(AuthenticationError):
            await client.query("test")


@respx.mock
async def test_token_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[])
    )
    async with AsyncFastRAGClient(base_url=base_url, token="secret") as client:
        await client.query("test")
    assert route.calls[0].request.headers["x-fastrag-token"] == "secret"


@respx.mock
async def test_tenant_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(
        return_value=httpx.Response(200, json=[])
    )
    async with AsyncFastRAGClient(base_url=base_url, tenant_id="acme") as client:
        await client.query("test")
    assert route.calls[0].request.headers["x-fastrag-tenant"] == "acme"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_async_client.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `async_client.py`**

Create `src/fastrag_client/async_client.py`:

```python
"""Asynchronous fastrag HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .client import _build_headers, _raise_for_status, _STATUS_MAP
from .filters import FilterExpr
from .models import BatchResult, CorpusInfo, DeleteResult, IngestResult, SearchHit


class AsyncFastRAGClient:
    """Asynchronous client for fastrag's HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=_build_headers(token, tenant_id),
            timeout=timeout,
        )

    async def query(
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
        params: dict[str, Any] = {
            "q": q,
            "top_k": top_k,
            "corpus": corpus,
            "snippet_len": snippet_len,
        }
        if filter is not None:
            params["filter"] = str(filter)
        if fields is not None:
            params["fields"] = ",".join(fields)
        if rerank is not None:
            params["rerank"] = rerank
        if over_fetch is not None:
            params["over_fetch"] = over_fetch

        resp = await self._client.get("/query", params=params)
        _raise_for_status(resp)
        return [SearchHit.model_validate(h) for h in resp.json()]

    async def batch_query(
        self,
        queries: list[dict[str, Any]],
    ) -> list[BatchResult]:
        serialized = []
        for q in queries:
            item = dict(q)
            if "filter" in item and isinstance(item["filter"], FilterExpr):
                item["filter"] = str(item["filter"])
            serialized.append(item)

        resp = await self._client.post("/batch-query", json={"queries": serialized})
        _raise_for_status(resp)
        data = resp.json()
        return [BatchResult.model_validate(r) for r in data["results"]]

    async def ingest(
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
        params: dict[str, Any] = {
            "corpus": corpus,
            "id_field": id_field,
            "text_fields": ",".join(text_fields),
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        if metadata_fields:
            params["metadata_fields"] = ",".join(metadata_fields)
        if metadata_types:
            params["metadata_types"] = ",".join(f"{k}={v}" for k, v in metadata_types.items())
        if array_fields:
            params["array_fields"] = ",".join(array_fields)

        ndjson = "\n".join(json.dumps(r) for r in records) + "\n"
        resp = await self._client.post(
            "/ingest",
            content=ndjson.encode(),
            headers={"content-type": "application/x-ndjson"},
            params=params,
        )
        _raise_for_status(resp)
        return IngestResult.model_validate(resp.json())

    async def delete(self, id: str, *, corpus: str = "default") -> DeleteResult:
        resp = await self._client.delete(f"/ingest/{id}", params={"corpus": corpus})
        _raise_for_status(resp)
        return DeleteResult.model_validate(resp.json())

    async def stats(self, *, corpus: str = "default") -> dict[str, Any]:
        resp = await self._client.get("/stats", params={"corpus": corpus})
        _raise_for_status(resp)
        return resp.json()

    async def corpora(self) -> list[CorpusInfo]:
        resp = await self._client.get("/corpora")
        _raise_for_status(resp)
        return [CorpusInfo.model_validate(c) for c in resp.json()["corpora"]]

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncFastRAGClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/github/fastrag-client-python
pytest tests/test_async_client.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/async_client.py tests/test_async_client.py
git commit -m "feat: add async AsyncFastRAGClient with all 7 endpoints"
```

---

### Task 7: Package exports + final checks

**Files:**
- Modify: `src/fastrag_client/__init__.py`

- [ ] **Step 1: Write `__init__.py` exports**

Update `src/fastrag_client/__init__.py`:

```python
"""Typed Python client for fastrag's HTTP API."""

from .async_client import AsyncFastRAGClient
from .client import FastRAGClient
from .errors import (
    AuthenticationError,
    ConnectionError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)
from .filters import F
from .models import (
    BatchResult,
    CorpusInfo,
    DeleteResult,
    IngestResult,
    SearchHit,
)

__all__ = [
    "AsyncFastRAGClient",
    "AuthenticationError",
    "BatchResult",
    "ConnectionError",
    "CorpusInfo",
    "DeleteResult",
    "F",
    "FastRAGClient",
    "FastRAGError",
    "IngestResult",
    "NotFoundError",
    "PayloadTooLargeError",
    "SearchHit",
    "ServerError",
    "ValidationError",
]
```

- [ ] **Step 2: Verify imports from top-level package**

```bash
cd ~/github/fastrag-client-python
python -c "from fastrag_client import FastRAGClient, AsyncFastRAGClient, F; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite**

```bash
cd ~/github/fastrag-client-python
pytest tests/ -v
```

Expected: all tests pass (4 + 8 + 11 + 16 + 12 = 51 tests).

- [ ] **Step 4: Run ruff**

```bash
cd ~/github/fastrag-client-python
ruff check src/ tests/
ruff format --check src/ tests/
```

Fix any issues.

- [ ] **Step 5: Commit**

```bash
cd ~/github/fastrag-client-python
git add src/fastrag_client/__init__.py
git commit -m "feat: add package exports and __all__"
```

---

### Task 8: Push and CI setup

- [ ] **Step 1: Create GitHub repo**

```bash
cd ~/github/fastrag-client-python
gh repo create crook3dfingers/fastrag-client-python --private --source=. --push
```

- [ ] **Step 2: Run final local checks**

```bash
cd ~/github/fastrag-client-python
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
```

All must pass.

- [ ] **Step 3: Push**

```bash
cd ~/github/fastrag-client-python
git push -u origin main
```

---

## Verification

```bash
cd ~/github/fastrag-client-python

# Full test suite
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format --check src/ tests/

# Import check
python -c "from fastrag_client import FastRAGClient, AsyncFastRAGClient, F; print('OK')"

# Manual smoke test (requires running fastrag server)
python -c "
from fastrag_client import FastRAGClient
client = FastRAGClient('http://localhost:8081')
print(client.health())
print(client.corpora())
"
```
