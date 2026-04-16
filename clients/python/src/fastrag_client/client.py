"""Synchronous fastrag HTTP client."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import (
    AuthenticationError,
    ConflictError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)
from .filters import FilterExpr
from .models import (
    BatchResult,
    CorpusInfo,
    CweRelation,
    DeleteResult,
    IngestResult,
    ReadyStatus,
    ReloadResult,
    SearchHit,
    SimilarHit,
)

_STATUS_MAP: dict[int, type[FastRAGError]] = {
    400: ValidationError,
    401: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
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
        admin_token: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._admin_token = admin_token
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

    def get_cve(self, cve_id: str) -> SearchHit | None:
        resp = self._client.get(f"/cve/{cve_id}")
        if resp.status_code == 404:
            return None
        _raise_for_status(resp)
        hits = resp.json().get("hits", [])
        if not hits:
            return None
        return SearchHit.model_validate(hits[0])

    def get_cwe(self, cwe_id: int | str) -> SearchHit | None:
        segment = str(cwe_id)
        resp = self._client.get(f"/cwe/{segment}")
        if resp.status_code == 404:
            return None
        _raise_for_status(resp)
        hits = resp.json().get("hits", [])
        if not hits:
            return None
        return SearchHit.model_validate(hits[0])

    def cwe_relation(
        self,
        cwe_id: int | str,
        *,
        direction: str = "both",
        max_depth: int | None = None,
    ) -> CweRelation:
        if isinstance(cwe_id, str):
            stripped = cwe_id.removeprefix("CWE-")
            cwe_id = int(stripped)
        params: dict[str, str] = {"cwe_id": str(cwe_id), "direction": direction}
        if max_depth is not None:
            params["max_depth"] = str(max_depth)
        resp = self._client.get("/cwe/relation", params=params)
        _raise_for_status(resp)
        return CweRelation.model_validate(resp.json())

    def ready(self) -> ReadyStatus:
        resp = self._client.get("/ready")
        if resp.status_code in (200, 503):
            body = resp.json()
            return ReadyStatus(
                ok=bool(body.get("ready")),
                reasons=list(body.get("reasons", [])),
            )
        _raise_for_status(resp)
        return ReadyStatus(ok=False, reasons=["unknown_status"])

    def reload_bundle(
        self,
        bundle_path: str,
        *,
        admin_token: str | None = None,
    ) -> ReloadResult:
        headers: dict[str, str] = {}
        token = admin_token or self._admin_token
        if token:
            headers["x-fastrag-admin-token"] = token
        resp = self._client.post(
            "/admin/reload",
            json={"bundle_path": bundle_path},
            headers=headers,
        )
        _raise_for_status(resp)
        return ReloadResult.model_validate(resp.json())

    def similar(
        self,
        text: str,
        threshold: float,
        *,
        max_results: int = 10,
        corpus: str | None = None,
        corpora: list[str] | None = None,
        filter: dict[str, Any] | str | None = None,
        fields: list[str] | None = None,
        verify: dict[str, Any] | None = None,
    ) -> list[SimilarHit]:
        body: dict[str, Any] = {
            "text": text,
            "threshold": threshold,
            "max_results": max_results,
        }
        if corpus is not None:
            body["corpus"] = corpus
        if corpora is not None:
            body["corpora"] = corpora
        if filter is not None:
            body["filter"] = filter
        if fields is not None:
            body["fields"] = ",".join(fields)
        if verify is not None:
            body["verify"] = verify
        resp = self._client.post("/similar", json=body)
        _raise_for_status(resp)
        data = resp.json()
        hits = data.get("hits", []) if isinstance(data, dict) else data
        return [SimilarHit.model_validate(h) for h in hits]

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
