"""Shared test fixtures."""

import pytest

SEARCH_HIT_FIXTURE: dict = {
    "score": 0.87,
    "chunk_text": "SQL injection vulnerability allows remote code execution",
    "snippet": "<b>SQL</b> <b>injection</b> vulnerability",
    "source": {"id": "cve-1", "body": "SQL injection..."},
    "source_path": "cve-1",
    "chunk_index": 0,
    "metadata": {"severity": "HIGH"},
}

INGEST_FIXTURE: dict = {
    "corpus": "default",
    "records_new": 2,
    "records_updated": 0,
    "records_unchanged": 0,
    "chunks_added": 6,
}

DELETE_FIXTURE: dict = {
    "corpus": "default",
    "id": "cve-1",
    "deleted": True,
}

CORPORA_FIXTURE: dict = {
    "corpora": [
        {"name": "default", "path": "/data/corpus", "status": "ready"},
    ],
}

BATCH_FIXTURE: dict = {
    "results": [
        {"index": 0, "hits": [SEARCH_HIT_FIXTURE]},
        {"index": 1, "error": "corpus not found"},
    ],
}


@pytest.fixture
def base_url() -> str:
    return "http://testserver:8081"
