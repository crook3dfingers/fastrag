"""Tests for FastRAGClient.similar (POST /similar)."""

from __future__ import annotations

import json

import httpx
import respx

from fastrag_client import FastRAGClient
from fastrag_client.models import SimilarHit

_HIT: dict = {
    "id": "d1",
    "score": 0.95,
    "text": "similar chunk",
    "chunk_text": "similar chunk",
    "source_path": "a.md",
    "metadata": {"source_path": "a.md"},
}


@respx.mock
def test_similar_happy_path(base_url: str) -> None:
    respx.post(f"{base_url}/similar").mock(return_value=httpx.Response(200, json={"hits": [_HIT]}))
    client = FastRAGClient(base_url=base_url)
    hits = client.similar("query text", threshold=0.8, max_results=5)
    assert len(hits) == 1
    assert isinstance(hits[0], SimilarHit)
    assert hits[0].id == "d1"
    assert hits[0].score == 0.95


@respx.mock
def test_similar_sends_filter_and_corpora(base_url: str) -> None:
    route = respx.post(f"{base_url}/similar").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    FastRAGClient(base_url=base_url).similar(
        "q",
        threshold=0.7,
        corpora=["cve", "kev"],
        filter={"cve_id": "CVE-2021-44228"},
    )
    body = json.loads(route.calls[0].request.content)
    assert body["text"] == "q"
    assert body["threshold"] == 0.7
    assert body["max_results"] == 10
    assert body["corpora"] == ["cve", "kev"]
    assert body["filter"] == {"cve_id": "CVE-2021-44228"}
    assert "corpus" not in body


@respx.mock
def test_similar_sends_single_corpus(base_url: str) -> None:
    route = respx.post(f"{base_url}/similar").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    FastRAGClient(base_url=base_url).similar("q", threshold=0.5, corpus="cve")
    body = json.loads(route.calls[0].request.content)
    assert body["corpus"] == "cve"
    assert "corpora" not in body


@respx.mock
def test_similar_sends_fields_and_verify(base_url: str) -> None:
    route = respx.post(f"{base_url}/similar").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    FastRAGClient(base_url=base_url).similar(
        "q",
        threshold=0.5,
        fields=["score", "chunk_text"],
        verify={"method": "minhash", "jaccard_threshold": 0.85},
    )
    body = json.loads(route.calls[0].request.content)
    assert body["fields"] == "score,chunk_text"
    assert body["verify"] == {"method": "minhash", "jaccard_threshold": 0.85}


@respx.mock
def test_similar_empty_hits(base_url: str) -> None:
    respx.post(f"{base_url}/similar").mock(return_value=httpx.Response(200, json={"hits": []}))
    hits = FastRAGClient(base_url=base_url).similar("q", threshold=0.9)
    assert hits == []
