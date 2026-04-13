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
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE]))
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
    respx.post(f"{base_url}/batch-query").mock(return_value=httpx.Response(200, json=BATCH_FIXTURE))
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
    respx.get(f"{base_url}/stats").mock(return_value=httpx.Response(200, json=stats))
    client = FastRAGClient(base_url=base_url)
    result = client.stats()
    assert result["entries"]["live"] == 10


@respx.mock
def test_corpora_returns_list(base_url: str):
    respx.get(f"{base_url}/corpora").mock(return_value=httpx.Response(200, json=CORPORA_FIXTURE))
    client = FastRAGClient(base_url=base_url)
    result = client.corpora()
    assert len(result) == 1
    assert isinstance(result[0], CorpusInfo)
    assert result[0].name == "default"


@respx.mock
def test_health_returns_true(base_url: str):
    respx.get(f"{base_url}/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    client = FastRAGClient(base_url=base_url)
    assert client.health() is True


@respx.mock
def test_health_returns_false_on_error(base_url: str):
    respx.get(f"{base_url}/health").mock(side_effect=httpx.ConnectError("refused"))
    client = FastRAGClient(base_url=base_url)
    assert client.health() is False


@respx.mock
def test_auth_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(401, text="unauthorized"))
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(AuthenticationError) as exc_info:
        client.query("test")
    assert exc_info.value.status_code == 401


@respx.mock
def test_not_found_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(404, text="corpus not found"))
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(NotFoundError) as exc_info:
        client.query("test")
    assert exc_info.value.status_code == 404


@respx.mock
def test_validation_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(400, text="bad filter"))
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(ValidationError):
        client.query("test")


@respx.mock
def test_token_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[]))
    client = FastRAGClient(base_url=base_url, token="secret123")
    client.query("test")
    request = route.calls[0].request
    assert request.headers["x-fastrag-token"] == "secret123"


@respx.mock
def test_tenant_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[]))
    client = FastRAGClient(base_url=base_url, tenant_id="acme")
    client.query("test")
    request = route.calls[0].request
    assert request.headers["x-fastrag-tenant"] == "acme"
