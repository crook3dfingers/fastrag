"""Tests for FastRAGClient.get_cwe (thin lookup over GET /cwe/{id})."""

from __future__ import annotations

import httpx
import respx

from fastrag_client import FastRAGClient
from fastrag_client.models import SearchHit

_CWE_HIT: dict = {
    "score": 1.0,
    "chunk_text": "CWE-89: SQL Injection",
    "source_path": "cwe/89",
    "chunk_index": 0,
    "pages": [],
    "element_kinds": [],
    "metadata": {"cwe_id": 89, "parents": [707, 943]},
}


@respx.mock
def test_get_cwe_by_int(base_url: str) -> None:
    respx.get(f"{base_url}/cwe/89").mock(
        return_value=httpx.Response(200, json={"hits": [_CWE_HIT]})
    )
    client = FastRAGClient(base_url=base_url)
    rec = client.get_cwe(89)
    assert rec is not None
    assert isinstance(rec, SearchHit)
    assert rec.metadata["cwe_id"] == 89
    assert rec.metadata["parents"] == [707, 943]


@respx.mock
def test_get_cwe_by_prefixed_string(base_url: str) -> None:
    respx.get(f"{base_url}/cwe/CWE-89").mock(
        return_value=httpx.Response(200, json={"hits": [_CWE_HIT]})
    )
    client = FastRAGClient(base_url=base_url)
    rec = client.get_cwe("CWE-89")
    assert rec is not None
    assert rec.metadata["cwe_id"] == 89


@respx.mock
def test_get_cwe_none_on_404(base_url: str) -> None:
    respx.get(f"{base_url}/cwe/9999").mock(
        return_value=httpx.Response(404, json={"error": "cwe_not_found", "id": 9999})
    )
    assert FastRAGClient(base_url=base_url).get_cwe(9999) is None


@respx.mock
def test_get_cwe_none_on_empty_hits(base_url: str) -> None:
    respx.get(f"{base_url}/cwe/42").mock(return_value=httpx.Response(200, json={"hits": []}))
    assert FastRAGClient(base_url=base_url).get_cwe(42) is None
