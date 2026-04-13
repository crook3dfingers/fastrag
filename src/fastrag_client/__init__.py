"""Typed Python client for fastrag's HTTP API."""

from .async_client import AsyncFastRAGClient
from .client import FastRAGClient
from .errors import (
    AuthenticationError,
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
    DeleteResult,
    IngestResult,
    SearchHit,
)

__all__ = [
    "AsyncFastRAGClient",
    "AuthenticationError",
    "BatchResult",
    "ConnectionError",
    "CorpusInfo",
    "DeleteResult",
    "F",
    "FastRAGClient",
    "FastRAGError",
    "IngestResult",
    "NotFoundError",
    "PayloadTooLargeError",
    "SearchHit",
    "ServerError",
    "ValidationError",
]
