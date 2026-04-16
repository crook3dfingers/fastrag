"""Tests for FastRAGClient.ready (GET /ready probe)."""

from __future__ import annotations

import httpx
import pytest
import respx

from fastrag_client import FastRAGClient
from fastrag_client.errors import FastRAGError
from fastrag_client.models import ReadyStatus


@respx.mock
def test_ready_ok(base_url: str) -> None:
    respx.get(f"{base_url}/ready").mock(return_value=httpx.Response(200, json={"ready": True}))
    r = FastRAGClient(base_url=base_url).ready()
    assert isinstance(r, ReadyStatus)
    assert r.ok is True
    assert r.reasons == []


@respx.mock
def test_ready_503_returns_status_not_raise(base_url: str) -> None:
    respx.get(f"{base_url}/ready").mock(
        return_value=httpx.Response(503, json={"ready": False, "reasons": ["bundle_not_loaded"]})
    )
    r = FastRAGClient(base_url=base_url).ready()
    assert r.ok is False
    assert "bundle_not_loaded" in r.reasons


@respx.mock
def test_ready_raises_on_unexpected_status(base_url: str) -> None:
    respx.get(f"{base_url}/ready").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(FastRAGError):
        FastRAGClient(base_url=base_url).ready()
