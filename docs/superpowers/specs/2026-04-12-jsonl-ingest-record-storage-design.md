# JSONL Ingest Engine + Record Storage Model

**Issue:** #41 — Phase 3 Step 1
**Date:** 2026-04-12

## Context

FastRAG only indexes files from disk. Tarmo tools (VAMS, scribe, storm) produce structured JSON records that need semantic search. The current storage model (`entries.bin` with `BTreeMap<String, String>` metadata) cannot support typed filters, full record round-trips, or upsert/delete. This change introduces JSONL ingest and replaces the storage layer with a unified Tantivy + HNSW architecture.

## Key Design Decisions

1. **Tantivy always-on** — every corpus gets Tantivy (tiered: core-only for file ingest, full dynamic schema for JSONL). The `hybrid` feature flag folds into default.
2. **HNSW stores vectors only** — `{id, vector}`. All text, metadata, and `_source` live in Tantivy. Query path: HNSW returns IDs → Tantivy lookup.
3. **File-based external IDs** — content-addressed: blake3 hash of `(source_path, chunk_index)`. JSONL uses `--id-field`.
4. **Dynamic schema per corpus** — core fields always present, user fields declared at ingest via `--metadata-fields`/`--metadata-types`. Persisted in `schema.json`. New fields can be added on subsequent ingests; type conflicts error out.
5. **`_source` per chunk, deduped at query time** — every chunk stores the full original JSON record. Results grouped by `_external_id` to deduplicate.
6. **No backward compatibility** — existing corpora must be re-indexed. `entries.bin` is deleted. No migration path.
7. **Tombstone-based HNSW deletion** — delete marks slots as dead; explicit `fastrag compact` rebuilds the graph.

## Architecture

### New Crate: `crates/fastrag-store/`

Owns unified persistence. Coordinates Tantivy (records/metadata/full-text) and HNSW (vectors).

```rust
pub struct Store {
    tantivy: TantivyStore,
    hnsw: HnswVectors,
    schema: DynamicSchema,
}
```

`fastrag-tantivy` merges into `fastrag-store`. The separate crate is deleted.

### Crate Dependency Graph

```
fastrag-store (NEW)
  ├── tantivy
  ├── fastrag-index    (slimmed: HnswVectors only)
  └── serde_json

fastrag (facade)
  ├── fastrag-store    (replaces fastrag-index + fastrag-tantivy)
  ├── fastrag-core
  ├── fastrag-embed
  └── fastrag-context  (optional)

fastrag-cli
  └── fastrag
```

### File Moves

| From | To |
|---|---|
| `fastrag-tantivy/src/lib.rs` | `fastrag-store/src/tantivy.rs` |
| `fastrag-index/src/entry.rs` | Slimmed to `VectorEntry {id, vector}` |
| `fastrag-index/src/hnsw.rs` | Keeps HNSW graph + tombstone set |
| (new) | `fastrag-store/src/schema.rs` — `DynamicSchema` |
| (new) | `fastrag-store/src/lib.rs` — `Store` facade |
| (new) | `fastrag/src/ingest/jsonl.rs` — JSONL parser |

## Data Model

### `VectorEntry` (slimmed, in `fastrag-index`)

```rust
pub struct VectorEntry {
    pub id: u64,
    pub vector: Vec<f32>,
}
```

### `DynamicSchema` (in `fastrag-store`, persisted as `schema.json`)

```rust
pub struct DynamicSchema {
    pub core_fields: CoreFields,
    pub user_fields: Vec<FieldDef>,
}

pub struct FieldDef {
    pub name: String,
    pub typed: TypedKind,
    pub indexed: bool,
    pub stored: bool,
    pub positions: bool,
}

pub enum TypedKind {
    String,
    Numeric,
    Bool,
    Date,
    Array(Box<TypedKind>),
}
```

### `TypedValue` (runtime values)

```rust
pub enum TypedValue {
    String(String),
    Numeric(f64),
    Bool(bool),
    Date(NaiveDate),
    Array(Vec<TypedValue>),
}
```

### Core Tantivy Fields (always present)

| Field | Tantivy Type | Flags | Purpose |
|---|---|---|---|
| `_id` | `u64` | INDEXED, STORED, FAST | Internal chunk ID, joins to HNSW |
| `_external_id` | `String` | INDEXED, STORED, FAST | Stable external ID for upsert/delete |
| `_content_hash` | `String` | STORED | blake3 of source content |
| `_chunk_index` | `u64` | STORED, FAST | Position within parent record |
| `_source_path` | `String` | STORED | Origin file or JSONL source |
| `_source` | `JsonObject` | STORED | Full original JSON record (null for file ingest) |
| `_chunk_text` | `Text` | INDEXED, STORED, POSITIONS | Chunk text for BM25 + snippets |

### User Field Type Mapping

| TypedKind | Tantivy Type | Flags |
|---|---|---|
| `String` | `Text` | INDEXED, STORED, FAST (keyword), POSITIONS |
| `Numeric` | `f64` | INDEXED, STORED, FAST |
| `Bool` | `u64` (0/1) | INDEXED, STORED, FAST |
| `Date` | `Date` | INDEXED, STORED, FAST |
| `Array(String)` | `Text` | INDEXED, STORED (multi-valued) |
| `Array(Numeric)` | `f64` | INDEXED, STORED, FAST (multi-valued) |

## JSONL Ingest Pipeline

### Configuration

```rust
pub struct JsonlIngestConfig {
    pub text_fields: Vec<String>,
    pub id_field: String,
    pub metadata_fields: Vec<String>,
    pub metadata_types: BTreeMap<String, TypedKind>,
    pub array_fields: Vec<String>,
}
```

### Per-Line Flow

1. **Parse**: `serde_json::from_str::<Value>(line)` — reject malformed lines with line number in error
2. **Extract external ID**: `record[id_field]` → `_external_id`. Error if missing.
3. **Content hash**: blake3 of raw JSON line bytes → `_content_hash`
4. **Upsert check**: Query Tantivy for `_external_id`. If `_content_hash` matches → skip. If differs → delete old (Tantivy + HNSW tombstone), insert new.
5. **Extract text**: Concatenate `text_fields` values with `\n\n` separator
6. **Type inference**: For fields in `metadata_fields` not in `metadata_types`, infer from JSON value type (`Number` → Numeric, `Bool` → Bool, ISO-8601 string → Date, `Array` → Array, else String). First record with a non-null value for a field sets that field's type; subsequent records must match. Null values are stored as Tantivy's default for the type.
7. **Store `_source`**: Full original `serde_json::Value` as stored-only JSON field
8. **Chunk**: Apply chunking strategy to concatenated text. Each chunk gets its own Tantivy doc + HNSW vector, sharing `_external_id` and `_source`.
9. **Embed**: `embedder.embed_passage_dyn()` on chunk texts
10. **Persist**: `Store::add_record()` writes Tantivy docs + HNSW vectors

### File-Based Ingest Adaptation

- External ID = blake3 hash of `(source_path, chunk_index)`
- `_source` = null
- Schema = core fields only (unless `--metadata` flags used)
- Same `Store` API as JSONL path

## Delete & Compaction

### Delete Flow

1. Look up `_external_id` in Tantivy → collect all `_id` values
2. Delete Tantivy docs by `_id`
3. Mark HNSW slots as tombstoned (`HashSet<u64>` persisted alongside `index.bin`)
4. Query path skips tombstoned IDs

### Compaction

- `fastrag compact --corpus ./corpus` rebuilds HNSW from live vectors only
- Triggered when `tombstone_count > 20%` of total entries (reported by `corpus-info`)
- No auto-compaction

## Query Path

### Dense Search

1. HNSW search → `Vec<(u64, f32)>` (id, distance)
2. Filter out tombstoned IDs
3. `Store::fetch_by_ids(ids)` → Tantivy doc lookup
4. Return `Vec<SearchHit>`

### Hybrid Search

1. Tantivy BM25 on `_chunk_text` → top-50
2. HNSW dense → top-50
3. RRF fusion (k=60)
4. Security ID exact lookup (unchanged)
5. Fetch records from Tantivy

### Typed Filtering

Post-filter first (HNSW over-fetch → Tantivy filter). Pre-filter optimization (Tantivy query → candidate set → HNSW re-score) deferred to a later step.

### Result Shape

```rust
pub struct SearchHit {
    pub external_id: String,
    pub score: f32,
    pub source: Option<serde_json::Value>,
    pub chunks: Vec<ChunkHit>,
}

pub struct ChunkHit {
    pub chunk_index: usize,
    pub chunk_text: String,
    pub score: f32,
}
```

Results deduplicated by `_external_id` — multiple chunks from the same record grouped into one `SearchHit`.

## CLI Surface

```bash
# JSONL ingest
fastrag index data.jsonl --format jsonl --corpus ./corpus \
  --text-fields title,description \
  --metadata-fields severity,cwe_id,cvss_score \
  --metadata-types cwe_id=numeric,cvss_score=numeric \
  --array-fields tags,affected_hosts \
  --id-field id

# File-based ingest (unchanged syntax, new storage backend)
fastrag index ./documents --corpus ./corpus

# Delete by external ID
fastrag delete --corpus ./corpus --id <external-id>

# Compact HNSW (purge tombstones)
fastrag compact --corpus ./corpus

# Corpus info (gains tombstone ratio + schema summary)
fastrag corpus-info --corpus ./corpus
```

## Critical Files to Modify

| File | Change |
|---|---|
| `crates/fastrag-store/src/lib.rs` | New: `Store` facade |
| `crates/fastrag-store/src/schema.rs` | New: `DynamicSchema`, `FieldDef`, `TypedKind` |
| `crates/fastrag-store/src/tantivy.rs` | Migrated + extended from `fastrag-tantivy` |
| `crates/fastrag-index/src/entry.rs` | Slim to `VectorEntry {id, vector}` |
| `crates/fastrag-index/src/hnsw.rs` | Add tombstone set |
| `crates/fastrag/src/ingest/jsonl.rs` | New: JSONL parser + field extraction |
| `crates/fastrag/src/corpus/mod.rs` | Rewrite to use `Store` API |
| `crates/fastrag/src/corpus/hybrid.rs` | Fold into default (Tantivy always-on) |
| `fastrag-cli/src/main.rs` | New commands: delete, compact. JSONL flags on index. |
| `fastrag-cli/src/http.rs` | Query path returns `SearchHit` |
| `crates/fastrag-mcp/src/lib.rs` | search_corpus returns `SearchHit` |
| Workspace `Cargo.toml` | Add `fastrag-store`, remove `fastrag-tantivy` |

## Verification

1. **Unit tests**: `DynamicSchema` merge/compatibility, `TypedValue` serde, JSONL line parsing, tombstone set operations
2. **Integration test**: Ingest JSONL fixture → query → verify `_source` round-trip, typed filter, dedup by `_external_id`
3. **Upsert test**: Ingest → modify record → re-ingest → verify old chunks gone, new chunks present
4. **Delete test**: Ingest → delete by ID → query returns no results → compact → HNSW rebuilt without deleted vectors
5. **File-based ingest test**: Existing file ingest through new `Store` path → query → verify results
6. **Schema evolution test**: Ingest with fields A,B → ingest with fields A,B,C → verify C added. Ingest with field A as different type → verify error.
7. **clippy + fmt**: `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check`
