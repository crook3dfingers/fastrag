"""Tests for FastRAGClient.get_cve (thin lookup over GET /cve/{id})."""

from __future__ import annotations

import httpx
import pytest
import respx

from fastrag_client import FastRAGClient
from fastrag_client.errors import ServerError
from fastrag_client.models import SearchHit

_CVE_HIT: dict = {
    "score": 1.0,
    "chunk_text": "Log4Shell: remote code execution via JNDI lookup in log4j2",
    "source_path": "cve/CVE-2021-44228",
    "chunk_index": 0,
    "pages": [],
    "element_kinds": [],
    "metadata": {"cve_id": "CVE-2021-44228", "cvss_score": 10.0},
}


@respx.mock
def test_get_cve_happy_path(base_url: str) -> None:
    respx.get(f"{base_url}/cve/CVE-2021-44228").mock(
        return_value=httpx.Response(200, json={"hits": [_CVE_HIT]})
    )
    client = FastRAGClient(base_url=base_url)
    rec = client.get_cve("CVE-2021-44228")
    assert rec is not None
    assert isinstance(rec, SearchHit)
    assert rec.score == 1.0
    assert rec.metadata["cve_id"] == "CVE-2021-44228"
    assert rec.metadata["cvss_score"] == 10.0


@respx.mock
def test_get_cve_returns_none_on_404(base_url: str) -> None:
    respx.get(f"{base_url}/cve/CVE-9999-0000").mock(
        return_value=httpx.Response(404, json={"error": "cve_not_found", "id": "CVE-9999-0000"})
    )
    client = FastRAGClient(base_url=base_url)
    assert client.get_cve("CVE-9999-0000") is None


@respx.mock
def test_get_cve_returns_none_on_empty_hits(base_url: str) -> None:
    respx.get(f"{base_url}/cve/CVE-0-0").mock(return_value=httpx.Response(200, json={"hits": []}))
    client = FastRAGClient(base_url=base_url)
    assert client.get_cve("CVE-0-0") is None


@respx.mock
def test_get_cve_raises_on_5xx(base_url: str) -> None:
    respx.get(f"{base_url}/cve/CVE-1").mock(
        return_value=httpx.Response(503, json={"error": "bundle_not_loaded"})
    )
    client = FastRAGClient(base_url=base_url)
    with pytest.raises(ServerError):
        client.get_cve("CVE-1")
