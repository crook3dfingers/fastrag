"""Typed Python client for fastrag's HTTP API."""

from .async_client import AsyncFastRAGClient
from .client import FastRAGClient
from .errors import (
    AuthenticationError,
    ConflictError,
    ConnectionError,
    FastRAGError,
    NotFoundError,
    PayloadTooLargeError,
    ServerError,
    ValidationError,
)
from .filters import F
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

__all__ = [
    "AsyncFastRAGClient",
    "AuthenticationError",
    "BatchResult",
    "ConflictError",
    "ConnectionError",
    "CorpusInfo",
    "CweRelation",
    "DeleteResult",
    "F",
    "FastRAGClient",
    "FastRAGError",
    "IngestResult",
    "NotFoundError",
    "PayloadTooLargeError",
    "ReadyStatus",
    "ReloadResult",
    "SearchHit",
    "ServerError",
    "SimilarHit",
    "ValidationError",
]
