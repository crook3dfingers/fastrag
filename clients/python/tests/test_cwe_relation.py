"""Tests for FastRAGClient.cwe_relation (GET /cwe/relation)."""

from __future__ import annotations

import httpx
import pytest
import respx

from fastrag_client import FastRAGClient
from fastrag_client.errors import ValidationError
from fastrag_client.models import CweRelation


@respx.mock
def test_cwe_relation_both(base_url: str) -> None:
    route = respx.get(f"{base_url}/cwe/relation").mock(
        return_value=httpx.Response(
            200,
            json={"cwe_id": 89, "ancestors": [707, 943, 20], "descendants": [564]},
        )
    )
    client = FastRAGClient(base_url=base_url)
    rel = client.cwe_relation(89)
    assert isinstance(rel, CweRelation)
    assert rel.cwe_id == 89
    assert rel.ancestors == [707, 943, 20]
    assert rel.descendants == [564]
    url = str(route.calls[0].request.url)
    assert "cwe_id=89" in url
    assert "direction=both" in url


@respx.mock
def test_cwe_relation_ancestors_with_depth(base_url: str) -> None:
    route = respx.get(f"{base_url}/cwe/relation").mock(
        return_value=httpx.Response(
            200, json={"cwe_id": 89, "ancestors": [707, 943], "descendants": []}
        )
    )
    client = FastRAGClient(base_url=base_url)
    rel = client.cwe_relation(89, direction="ancestors", max_depth=1)
    assert rel.ancestors == [707, 943]
    assert rel.descendants == []
    url = str(route.calls[0].request.url)
    assert "direction=ancestors" in url
    assert "max_depth=1" in url


@respx.mock
def test_cwe_relation_accepts_prefixed_string(base_url: str) -> None:
    route = respx.get(f"{base_url}/cwe/relation").mock(
        return_value=httpx.Response(200, json={"cwe_id": 89, "ancestors": [], "descendants": []})
    )
    FastRAGClient(base_url=base_url).cwe_relation("CWE-89")
    assert "cwe_id=89" in str(route.calls[0].request.url)


@respx.mock
def test_cwe_relation_raises_on_400(base_url: str) -> None:
    respx.get(f"{base_url}/cwe/relation").mock(
        return_value=httpx.Response(400, json={"error": "invalid_cwe_id"})
    )
    with pytest.raises(ValidationError):
        FastRAGClient(base_url=base_url).cwe_relation(-1)
