"""Tests for FastRAGClient.reload_bundle (POST /admin/reload)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from fastrag_client import FastRAGClient
from fastrag_client.errors import (
    AuthenticationError,
    ConflictError,
    ValidationError,
)
from fastrag_client.models import ReloadResult


@respx.mock
def test_reload_bundle_happy_path(base_url: str) -> None:
    route = respx.post(f"{base_url}/admin/reload").mock(
        return_value=httpx.Response(
            200,
            json={
                "reloaded": True,
                "bundle_id": "fastrag-20260417",
                "previous_bundle_id": "fastrag-20260416",
            },
        )
    )
    r = FastRAGClient(base_url=base_url).reload_bundle("fastrag-20260417", admin_token="tok")
    assert isinstance(r, ReloadResult)
    assert r.reloaded is True
    assert r.bundle_id == "fastrag-20260417"
    assert r.previous_bundle_id == "fastrag-20260416"

    request = route.calls[0].request
    assert request.headers["x-fastrag-admin-token"] == "tok"
    assert json.loads(request.content)["bundle_path"] == "fastrag-20260417"


@respx.mock
def test_reload_bundle_uses_constructor_token(base_url: str) -> None:
    route = respx.post(f"{base_url}/admin/reload").mock(
        return_value=httpx.Response(
            200,
            json={"reloaded": True, "bundle_id": "b2", "previous_bundle_id": "b1"},
        )
    )
    client = FastRAGClient(base_url=base_url, admin_token="ctor-tok")
    client.reload_bundle("b2")
    assert route.calls[0].request.headers["x-fastrag-admin-token"] == "ctor-tok"


@respx.mock
def test_reload_bundle_unauthorized(base_url: str) -> None:
    respx.post(f"{base_url}/admin/reload").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    with pytest.raises(AuthenticationError):
        FastRAGClient(base_url=base_url).reload_bundle("x", admin_token="wrong")


@respx.mock
def test_reload_bundle_409_conflict(base_url: str) -> None:
    respx.post(f"{base_url}/admin/reload").mock(
        return_value=httpx.Response(409, json={"error": "reload_in_progress"})
    )
    with pytest.raises(ConflictError):
        FastRAGClient(base_url=base_url).reload_bundle("x", admin_token="tok")


@respx.mock
def test_reload_bundle_400_path_escape(base_url: str) -> None:
    respx.post(f"{base_url}/admin/reload").mock(
        return_value=httpx.Response(400, json={"error": "path_escape"})
    )
    with pytest.raises(ValidationError):
        FastRAGClient(base_url=base_url).reload_bundle("../escape", admin_token="tok")
