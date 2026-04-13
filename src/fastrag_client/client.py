"""Synchronous fastrag HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import (
    AuthenticationError,
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
