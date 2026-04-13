"""Tests for async AsyncFastRAGClient."""

import httpx
import pytest
import respx

from fastrag_client.async_client import AsyncFastRAGClient
from fastrag_client.errors import AuthenticationError
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
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[SEARCH_HIT_FIXTURE]))
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
    respx.post(f"{base_url}/batch-query").mock(return_value=httpx.Response(200, json=BATCH_FIXTURE))
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
    respx.get(f"{base_url}/corpora").mock(return_value=httpx.Response(200, json=CORPORA_FIXTURE))
    async with AsyncFastRAGClient(base_url=base_url) as client:
        result = await client.corpora()
    assert len(result) == 1
    assert isinstance(result[0], CorpusInfo)


@respx.mock
async def test_health_returns_true(base_url: str):
    respx.get(f"{base_url}/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    async with AsyncFastRAGClient(base_url=base_url) as client:
        assert await client.health() is True


@respx.mock
async def test_health_returns_false_on_error(base_url: str):
    respx.get(f"{base_url}/health").mock(side_effect=httpx.ConnectError("refused"))
    async with AsyncFastRAGClient(base_url=base_url) as client:
        assert await client.health() is False


@respx.mock
async def test_auth_error_raises(base_url: str):
    respx.get(f"{base_url}/query").mock(return_value=httpx.Response(401, text="unauthorized"))
    async with AsyncFastRAGClient(base_url=base_url) as client:
        with pytest.raises(AuthenticationError):
            await client.query("test")


@respx.mock
async def test_token_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[]))
    async with AsyncFastRAGClient(base_url=base_url, token="secret") as client:
        await client.query("test")
    assert route.calls[0].request.headers["x-fastrag-token"] == "secret"


@respx.mock
async def test_tenant_header_sent(base_url: str):
    route = respx.get(f"{base_url}/query").mock(return_value=httpx.Response(200, json=[]))
    async with AsyncFastRAGClient(base_url=base_url, tenant_id="acme") as client:
        await client.query("test")
    assert route.calls[0].request.headers["x-fastrag-tenant"] == "acme"
