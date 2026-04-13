# Corpus Statistics API — #51

## Goal

Expose a `GET /stats` HTTP endpoint that returns corpus health metrics — entry counts, field cardinality/ranges, disk usage, embedding model, chunking config, and timestamps — computed on request from Tantivy fast fields and manifest metadata.

## Architecture

Compute-on-request, no caching layer. Tantivy fast-field column scans are O(segments), not O(docs), so computation is single-digit milliseconds for typical corpora. The handler acquires the existing per-corpus read-lock (same concurrency model as query handlers).

Two corpus types are supported:
- **Store-backed** (has `schema.json`): full stats including per-field cardinality and numeric min/max via Tantivy fast fields.
- **HNSW-only** (legacy `index_path`): entry count from `HnswIndex::len()`, manifest metadata, but `fields` array is empty (no Tantivy index to scan).

## Endpoint

### `GET /stats?corpus=<name>`

Query parameter `corpus` defaults to `"default"` when omitted (same convention as `/query`).

**Response (HTTP 200):**

```json
{
  "corpus": "default",
  "entries": {
    "live": 1024,
    "tombstoned": 12
  },
  "chunks": 3072,
  "disk_bytes": 52428800,
  "embedding": {
    "model_id": "bge-small-en-v1.5",
    "dimensions": 384
  },
  "chunking": {
    "strategy": "recursive",
    "max_characters": 1000,
    "overlap": 200
  },
  "timestamps": {
    "created_unix": 1712966400,
    "last_indexed_unix": 1712980800
  },
  "fields": [
    {
      "name": "severity",
      "type": "text",
      "cardinality": 4
    },
    {
      "name": "cvss",
      "type": "numeric",
      "min": 3.1,
      "max": 9.8,
      "cardinality": 47
    }
  ]
}
```

**Error responses:**
- Unknown corpus → 404 `{"error": "corpus not found: nonexistent"}`
- Corpus dir exists but index is corrupted → 500

## Data Sources

| Field | Source | Cost |
|-------|--------|------|
| `entries.live` | `Store::live_count()` or `HnswIndex::live_count()` | O(1) |
| `entries.tombstoned` | `Store::tombstone_count()` or `HnswIndex::tombstone_count()` | O(1) |
| `chunks` | `CorpusManifest::chunk_count` | O(1) |
| `disk_bytes` | `fs::read_dir` + `fs::metadata` on corpus dir (non-recursive, few files) | O(files) |
| `embedding` | `CorpusManifest::identity` (`model_id`, `dim`) | O(1) |
| `chunking` | `CorpusManifest::chunking_strategy` | O(1) |
| `timestamps.created_unix` | `CorpusManifest::created_at_unix_seconds` | O(1) |
| `timestamps.last_indexed_unix` | `max(CorpusManifest::roots[].last_indexed_unix_seconds)` | O(roots) |
| `fields[].cardinality` | Tantivy segment fast-field column scan — count distinct values per segment, union across segments | O(segments) |
| `fields[].min/max` | Tantivy numeric fast-field column min/max per segment | O(segments) |
| `fields[].type` | `DynamicSchema::user_fields[].typed` | O(1) |

## Implementation Files

| File | Change |
|------|--------|
| `crates/fastrag-store/src/lib.rs` | Add `Store::field_stats() -> Vec<FieldStat>` method that walks Tantivy segments |
| `crates/fastrag-store/src/tantivy.rs` | Add `TantivyStore::field_stats(schema: &DynamicSchema) -> Vec<FieldStat>` — segment-level fast-field scan |
| `crates/fastrag/src/corpus/mod.rs` | Add `corpus_stats(corpus_dir) -> CorpusStats` function |
| `fastrag-cli/src/http.rs` | Add `GET /stats` handler, `StatsQueryParams`, route registration |
| `fastrag-cli/tests/stats_e2e.rs` | Integration tests |

## `FieldStat` Type

```rust
pub struct FieldStat {
    pub name: String,
    pub field_type: FieldStatType,  // Text or Numeric
    pub cardinality: u64,
}

pub enum FieldStatType {
    Text,
    Numeric { min: f64, max: f64 },
}
```

## Tantivy Fast-Field Scan

For each user field in the schema:
1. Get the Tantivy field handle from the schema.
2. For each segment reader from `searcher.segment_readers()`:
   - For numeric fields (`u64` or `f64`): read the fast-field column, iterate values, track min/max and insert into a `HashSet` for cardinality.
   - For text fields with `FAST`: use the `str_column` fast-field reader, iterate term ordinals, count distinct.
3. Union cardinality across segments (values may repeat across segments, so this is an approximation — acceptable for health monitoring).

The cardinality for text fields across segments is approximate (sum of per-segment distinct counts, not exact global distinct). This is acceptable for the "health check" use case. Exact global cardinality would require merging term dictionaries, which is O(terms) not O(segments).

## Concurrency

The handler acquires the per-corpus read-lock via `get_or_create_lock()` — the same lock used by query handlers. Stats computation is a read operation. Concurrent reads are allowed; writes (ingest/delete) wait for stats to finish.

The Tantivy fast-field scan happens in the async handler directly (no `spawn_blocking`). Fast-field column reads are memory-mapped and do not block — they're comparable to reading a few struct fields, not disk I/O.

## HNSW-Only Fallback

When a corpus has no `schema.json`:
1. Load via `HnswIndex::load()` (existing path).
2. Return `entries`, `chunks`, `disk_bytes`, `embedding`, `chunking`, `timestamps` from the manifest.
3. `fields` is an empty array.

Detection: `corpus_dir.join("schema.json").exists()`.

## Testing

### `fastrag-cli/tests/stats_e2e.rs`

1. **`stats_after_ingest`** — POST 2 NDJSON records with severity + cvss fields, GET /stats, assert: `entries.live == 2`, `fields` contains severity (text, cardinality > 0) and cvss (numeric, min/max present).

2. **`stats_empty_corpus`** — Register a corpus dir with no data, GET /stats, assert: `entries.live == 0`, `chunks == 0`, `fields` empty array.

3. **`stats_unknown_corpus_returns_404`** — GET /stats?corpus=nonexistent, assert 404.

4. **`stats_reflects_delete`** — Ingest 2 records, GET /stats (live == 2), DELETE one, GET /stats (live == 1, tombstoned == 1).

## Out of Scope

- ETag / conditional caching (no commit generation counter exists; add later if needed)
- Full value histograms / top-N distributions (deferred; cardinality + range is sufficient for health monitoring)
- CLI `corpus-stats` command (HTTP-only for now; CLI already has `corpus-info`)
