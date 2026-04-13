"""Exception hierarchy for fastrag client."""


class FastRAGError(Exception):
    """Base exception for all fastrag client errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthenticationError(FastRAGError):
    """401 — missing or invalid token."""


class NotFoundError(FastRAGError):
    """404 — unknown corpus or record."""


class ValidationError(FastRAGError):
    """400 — bad filter, mixed field selectors, malformed JSON."""


class PayloadTooLargeError(FastRAGError):
    """413 — ingest body exceeds server limit."""


class ServerError(FastRAGError):
    """500/503 — server-side failure."""


class ConnectionError(FastRAGError):
    """Connection refused, timeout, DNS failure."""
