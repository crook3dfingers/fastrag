# Result Shaping (Snippets + Field Selection) — #49

## Goal

Reduce query response payload by adding highlighted snippets, exposing the original source JSON record, and letting callers select which fields to return. Saves LLM context tokens for downstream consumers (pentest-scribe, vams).

## Architecture

Three additions to the existing query path:

1. **Snippet generation** in the Store layer — Tantivy `SnippetGenerator` for BM25 queries (term-highlighted excerpts), truncated preview for dense-only queries.
2. **Source exposure** — thread `SearchHit.source` (already fetched by `Store::fetch_hits()`, currently discarded) through to `SearchHitDto`.
3. **Field selection** — post-filter on serialized JSON in the HTTP handler. Presentation concern, not corpus logic.

Backward compatible: all new fields are optional, field selection is opt-in. Existing clients see no change.

## Response Shape

Current `SearchHitDto` gains two fields:

```json
{
  "score": 0.87,
  "chunk_text": "SQL injection vulnerability in the login handler allows...",
  "snippet": "SQL <b>injection</b> <b>vulnerability</b> in the login...",
  "source": {"id": "cve-1", "body": "SQL injection...", "severity": "HIGH"},
  "source_path": "cve-1",
  "chunk_index": 0,
  "metadata": {"severity": "HIGH", "cvss": 9.8}
}
```

- **`snippet`** (`Option<String>`) — highlighted excerpt for BM25, truncated preview for dense, absent when `snippet_len=0` or snippet generation fails.
- **`source`** (`Option<serde_json::Value>`) — original JSON record stored at ingest time. `null` for file-based corpora that have no source JSON.

Both use `#[serde(skip_serializing_if = "Option::is_none")]`.

## Query Parameters

### `GET /query`

Existing params unchanged. New additions:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `snippet_len` | `usize` | `150` | Max characters for snippet. `0` disables snippet generation. |
| `fields` | `Option<String>` | `None` (all fields) | Comma-separated field projection. |

### `POST /batch-query`

Same two params added to each item in the batch request body. Applied per-result.

### Field Selection Syntax

When `fields` is present, only listed fields appear in the response.

- `fields=score,snippet,metadata` — include only these top-level keys
- `fields=source.severity,source.cvss` — include `source` but only with `severity` and `cvss` sub-keys
- `fields=-chunk_text` — include everything except `chunk_text`
- `fields=-source,-chunk_text` — exclude both

Positive and negative selectors cannot be mixed in the same request (400 error). When `fields` is absent, all fields are returned (backward compatible).

## Snippet Generation

### BM25 Path (Store-backed corpus)

Tantivy 0.22 provides `tantivy::SnippetGenerator`:

1. Parse query text with `QueryParser::for_index(index, vec![chunk_text_field])`.
2. Create `SnippetGenerator::create(&searcher, &query, chunk_text_field)`.
3. Call `snippet_generator.snippet_from_doc(&doc)` for each retrieved document.
4. The result contains `fragment()` (text with highlight markers) and `to_html()` (with `<b>` tags).

The `SnippetGenerator` is created once per query and reused across all hits.

**Implementation:** Add `TantivyStore::generate_snippets(query_text: &str, doc_addresses: &[DocAddress], max_chars: usize) -> Vec<Option<String>>` that returns a snippet for each document address. Returns `None` for documents where snippet generation fails.

### Dense-Only Path

No BM25 terms available for highlighting. Return `chunk_text[..snippet_len]` as a plain preview (no `<b>` tags). Truncate at a word boundary when possible.

### Snippet in SearchHitDto

The snippet is generated in the Store layer and threaded through `SearchHit` → `SearchHitDto`. The `scored_ids_to_dtos` function in `corpus/mod.rs` receives snippets alongside the scored IDs.

## Source Exposure

`Store::fetch_hits()` already extracts `source` (the `_source` field in Tantivy) and returns it in `SearchHit.source: Option<serde_json::Value>`. Currently, `scored_ids_to_dtos()` discards it.

Change: thread `SearchHit.source` through to `SearchHitDto.source`. No new Store logic needed.

For HNSW-only corpora (no Store), `source` remains `None`.

## Field Selection Implementation

Applied in the HTTP handler after serialization, before returning. A thin filter function:

```
fn apply_field_selection(hits: &mut Vec<serde_json::Value>, fields: &FieldSelection)
```

`FieldSelection` is an enum:
- `All` — no filtering (default)
- `Include(Vec<FieldPath>)` — keep only listed fields
- `Exclude(Vec<FieldPath>)` — remove listed fields

`FieldPath` is either a top-level key (`"score"`) or a dotted path (`"source.severity"`). For dotted paths, the parent object is preserved but pruned to only the selected sub-keys.

Parsing `fields` query param: if any field starts with `-`, treat all as excludes. If none start with `-`, treat all as includes. Mixed → 400.

## Files Changed

| File | Change |
|------|--------|
| `crates/fastrag-store/src/tantivy.rs` | Add `generate_snippets()` method using Tantivy SnippetGenerator |
| `crates/fastrag-store/src/lib.rs` | Add `Store::generate_snippets()` delegation |
| `crates/fastrag/src/corpus/mod.rs` | Add `snippet` + `source` to `SearchHitDto`; thread through `scored_ids_to_dtos` |
| `fastrag-cli/src/http.rs` | Add `snippet_len`, `fields` query params; `FieldSelection` parsing; `apply_field_selection` filter; wire snippet generation into query handlers |
| `fastrag-cli/tests/snippet_e2e.rs` | Integration tests |

## Testing

### Unit Tests

1. `TantivyStore::generate_snippets` — BM25 query returns highlighted text containing `<b>` tags
2. `generate_snippets` with empty query — returns None for each doc
3. `apply_field_selection` — include mode filters correctly
4. `apply_field_selection` — exclude mode filters correctly
5. `apply_field_selection` — dotted source paths prune sub-keys

### Integration Tests (`snippet_e2e.rs`)

1. **`snippet_returned_for_bm25_query`** — Ingest records, query with `snippet_len=100`, verify response includes `snippet` field with `<b>` tags.
2. **`source_exposed_in_response`** — Ingest NDJSON records, query, verify `source` contains the original JSON.
3. **`field_selection_include`** — Query with `fields=score,snippet`, verify only those fields present.
4. **`field_selection_exclude`** — Query with `fields=-chunk_text,-source`, verify those fields absent.
5. **`snippet_disabled_when_zero`** — Query with `snippet_len=0`, verify no `snippet` field.

## Out of Scope

- Custom highlight tags (hardcoded `<b>`/`</b>` for now)
- Snippet for dense queries beyond truncation (would need query term extraction from embedding)
- Pagination / cursor-based results
- Snippet caching
