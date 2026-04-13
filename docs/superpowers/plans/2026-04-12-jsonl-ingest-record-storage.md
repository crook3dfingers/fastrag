# JSONL Ingest Engine + Record Storage Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bincode-based storage layer with a unified Tantivy + HNSW architecture, add JSONL ingest with typed metadata, and support upsert/delete via stable external IDs.

**Architecture:** New `fastrag-store` crate owns all persistence — Tantivy for records/metadata/full-text, HNSW for vectors only. `fastrag-tantivy` merges into it. JSONL parsing lives in `crates/fastrag/src/ingest/jsonl.rs`. The `hybrid` feature flag is removed — Tantivy is always-on.

**Tech Stack:** Rust, tantivy 0.22, instant-distance 0.6.1, serde_json, blake3

**Spec:** `docs/superpowers/specs/2026-04-12-jsonl-ingest-record-storage-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `crates/fastrag-store/Cargo.toml` | Crate manifest — depends on tantivy, fastrag-index, serde_json, blake3, thiserror |
| `crates/fastrag-store/src/lib.rs` | `Store` facade — coordinates Tantivy + HNSW, public API for add/delete/query/save/load |
| `crates/fastrag-store/src/schema.rs` | `DynamicSchema`, `FieldDef`, `TypedKind`, `TypedValue` — schema definition + Tantivy field mapping |
| `crates/fastrag-store/src/tantivy.rs` | `TantivyStore` — Tantivy index wrapper with dynamic schema, `_source` storage, typed field writes/reads |
| `crates/fastrag-store/src/error.rs` | `StoreError` enum |
| `crates/fastrag/src/ingest/mod.rs` | Module declaration for ingest submodule |
| `crates/fastrag/src/ingest/jsonl.rs` | `JsonlIngestConfig`, JSONL line parser, field extraction, type inference |

### Modified Files

| File | Change |
|---|---|
| `Cargo.toml` (workspace) | Add `fastrag-store` member, remove `fastrag-tantivy` member |
| `crates/fastrag-index/src/entry.rs` | Replace `IndexEntry` with `VectorEntry {id, vector}`, remove `SearchHit` |
| `crates/fastrag-index/src/hnsw.rs` | Use `VectorEntry` instead of `IndexEntry`, add tombstone `HashSet<u64>`, tombstone-aware query |
| `crates/fastrag-index/src/lib.rs` | Update re-exports, update `VectorIndex` trait to use `VectorEntry` |
| `crates/fastrag/Cargo.toml` | Replace `fastrag-tantivy` dep with `fastrag-store`, remove `hybrid` feature, add `store` to `retrieval` |
| `crates/fastrag/src/corpus/mod.rs` | Rewrite `index_path_with_metadata` and query functions to use `Store` API |
| `crates/fastrag/src/corpus/hybrid.rs` | Delete — Tantivy is always-on, hybrid logic moves into `Store` |
| `crates/fastrag/src/lib.rs` | Add `pub mod ingest;` |
| `fastrag-cli/src/main.rs` or `fastrag-cli/src/args.rs` | Add `Delete`, `Compact` commands; add JSONL flags to `Index` |
| `fastrag-cli/src/http.rs` | Update query path to use new `SearchHit` from store |
| `crates/fastrag-mcp/src/lib.rs` | Update search_corpus to use new query API |
| `crates/fastrag-index/src/manifest.rs` | Bump version to 5, add `storage: "store"` field |

### Deleted Files

| File | Reason |
|---|---|
| `crates/fastrag-tantivy/` (entire crate) | Merged into `fastrag-store` |
| `crates/fastrag/src/corpus/hybrid.rs` | Tantivy always-on, logic absorbed by `Store` |

---

## Task 1: Create `fastrag-store` Crate Skeleton with `DynamicSchema`

**Files:**
- Create: `crates/fastrag-store/Cargo.toml`
- Create: `crates/fastrag-store/src/lib.rs`
- Create: `crates/fastrag-store/src/schema.rs`
- Create: `crates/fastrag-store/src/error.rs`
- Modify: `Cargo.toml` (workspace root, lines 3-27 members list)

- [ ] **Step 1: Write failing test for `TypedKind` and `TypedValue` serde round-trip**

Create `crates/fastrag-store/src/schema.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TypedKind {
    String,
    Numeric,
    Bool,
    Date,
    Array(Box<TypedKind>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypedValue {
    Bool(bool),
    Numeric(f64),
    String(String),
    Date(chrono::NaiveDate),
    Array(Vec<TypedValue>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub typed: TypedKind,
    pub indexed: bool,
    pub stored: bool,
    pub positions: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicSchema {
    pub user_fields: Vec<FieldDef>,
}

impl DynamicSchema {
    pub fn new() -> Self {
        Self {
            user_fields: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_kind_serde_round_trip() {
        let kinds = vec![
            TypedKind::String,
            TypedKind::Numeric,
            TypedKind::Bool,
            TypedKind::Date,
            TypedKind::Array(Box::new(TypedKind::String)),
        ];
        let json = serde_json::to_string(&kinds).unwrap();
        let parsed: Vec<TypedKind> = serde_json::from_str(&json).unwrap();
        assert_eq!(kinds, parsed);
    }

    #[test]
    fn typed_value_serde_round_trip() {
        let values = vec![
            TypedValue::String("high".into()),
            TypedValue::Numeric(9.8),
            TypedValue::Bool(true),
            TypedValue::Date(chrono::NaiveDate::from_ymd_opt(2026, 1, 15).unwrap()),
            TypedValue::Array(vec![TypedValue::String("a".into()), TypedValue::String("b".into())]),
        ];
        let json = serde_json::to_string(&values).unwrap();
        let parsed: Vec<TypedValue> = serde_json::from_str(&json).unwrap();
        assert_eq!(values, parsed);
    }
}
```

- [ ] **Step 2: Create `Cargo.toml` and wire into workspace**

Create `crates/fastrag-store/Cargo.toml`:

```toml
[package]
name = "fastrag-store"
version = "0.1.0"
edition = "2021"

[dependencies]
tantivy.workspace = true
fastrag-index.workspace = true
serde.workspace = true
serde_json.workspace = true
blake3.workspace = true
thiserror.workspace = true
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3"
```

Create `crates/fastrag-store/src/lib.rs`:

```rust
pub mod error;
pub mod schema;
```

Create `crates/fastrag-store/src/error.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("index error: {0}")]
    Index(#[from] fastrag_index::IndexError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema conflict: field '{field}' has type {existing:?} but got {incoming:?}")]
    SchemaConflict {
        field: String,
        existing: crate::schema::TypedKind,
        incoming: crate::schema::TypedKind,
    },
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("corrupt store: {0}")]
    Corrupt(String),
}

pub type StoreResult<T> = Result<T, StoreError>;
```

Add `fastrag-store` to workspace `Cargo.toml` members (after `fastrag-tantivy` line):

```toml
    "crates/fastrag-store",
```

Add workspace dependency:

```toml
fastrag-store = { path = "crates/fastrag-store", version = "0.1.0" }
```

- [ ] **Step 3: Run test to verify it passes**

Run: `cargo test -p fastrag-store`

Expected: 2 tests pass (`typed_kind_serde_round_trip`, `typed_value_serde_round_trip`)

- [ ] **Step 4: Write failing test for `DynamicSchema::merge()`**

Add to `crates/fastrag-store/src/schema.rs`:

```rust
impl DynamicSchema {
    // ... existing new() ...

    /// Merge incoming fields into the schema. New fields are added.
    /// Existing fields with matching types are kept. Type conflicts error.
    pub fn merge(&mut self, incoming: &[FieldDef]) -> Result<(), crate::error::StoreError> {
        for field in incoming {
            if let Some(existing) = self.user_fields.iter().find(|f| f.name == field.name) {
                if existing.typed != field.typed {
                    return Err(crate::error::StoreError::SchemaConflict {
                        field: field.name.clone(),
                        existing: existing.typed.clone(),
                        incoming: field.typed.clone(),
                    });
                }
            } else {
                self.user_fields.push(field.clone());
            }
        }
        Ok(())
    }
}
```

Add tests:

```rust
    #[test]
    fn schema_merge_adds_new_fields() {
        let mut schema = DynamicSchema::new();
        let fields = vec![
            FieldDef { name: "severity".into(), typed: TypedKind::String, indexed: true, stored: true, positions: false },
            FieldDef { name: "cvss_score".into(), typed: TypedKind::Numeric, indexed: true, stored: true, positions: false },
        ];
        schema.merge(&fields).unwrap();
        assert_eq!(schema.user_fields.len(), 2);
        assert_eq!(schema.user_fields[0].name, "severity");
        assert_eq!(schema.user_fields[1].name, "cvss_score");
    }

    #[test]
    fn schema_merge_allows_same_type() {
        let mut schema = DynamicSchema::new();
        let fields = vec![
            FieldDef { name: "severity".into(), typed: TypedKind::String, indexed: true, stored: true, positions: false },
        ];
        schema.merge(&fields).unwrap();
        schema.merge(&fields).unwrap(); // same field, same type — no error
        assert_eq!(schema.user_fields.len(), 1);
    }

    #[test]
    fn schema_merge_rejects_type_conflict() {
        let mut schema = DynamicSchema::new();
        let fields_str = vec![
            FieldDef { name: "score".into(), typed: TypedKind::String, indexed: true, stored: true, positions: false },
        ];
        let fields_num = vec![
            FieldDef { name: "score".into(), typed: TypedKind::Numeric, indexed: true, stored: true, positions: false },
        ];
        schema.merge(&fields_str).unwrap();
        let err = schema.merge(&fields_num).unwrap_err();
        assert!(err.to_string().contains("schema conflict"));
    }

    #[test]
    fn schema_serde_round_trip() {
        let mut schema = DynamicSchema::new();
        schema.merge(&[
            FieldDef { name: "tags".into(), typed: TypedKind::Array(Box::new(TypedKind::String)), indexed: true, stored: true, positions: false },
        ]).unwrap();
        let json = serde_json::to_string_pretty(&schema).unwrap();
        let parsed: DynamicSchema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, parsed);
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p fastrag-store`

Expected: 6 tests pass

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p fastrag-store -- -D warnings`

Expected: No warnings

- [ ] **Step 7: Commit**

```bash
git add crates/fastrag-store/ Cargo.toml
git commit -m "feat(store): fastrag-store crate skeleton with DynamicSchema + TypedValue

Closes #41 (partial)"
```

---

## Task 2: Slim `fastrag-index` to Vectors-Only

**Files:**
- Modify: `crates/fastrag-index/src/entry.rs`
- Modify: `crates/fastrag-index/src/hnsw.rs`
- Modify: `crates/fastrag-index/src/lib.rs`

- [ ] **Step 1: Write failing test for `VectorEntry`**

Replace contents of `crates/fastrag-index/src/entry.rs` with:

```rust
use serde::{Deserialize, Serialize};

/// Slim vector-only entry. All text, metadata, and _source live in Tantivy
/// (managed by fastrag-store).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// Dense search result: vector entry + cosine similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    pub id: u64,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_entry_serde_round_trip() {
        let entry = VectorEntry {
            id: 42,
            vector: vec![0.1, 0.2, 0.3],
        };
        let bytes = bincode::serialize(&entry).unwrap();
        let parsed: VectorEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(entry, parsed);
    }
}
```

- [ ] **Step 2: Run test — expect compilation errors everywhere**

Run: `cargo test -p fastrag-index`

Expected: Compilation failures in `hnsw.rs` and `lib.rs` referencing old `IndexEntry` and `SearchHit`.

- [ ] **Step 3: Update `HnswIndex` to use `VectorEntry` + add tombstones**

Rewrite `crates/fastrag-index/src/hnsw.rs`. Key changes:
- `entries: Vec<VectorEntry>` instead of `Vec<IndexEntry>`
- Add `tombstones: HashSet<u64>` field (serde default empty)
- `add()` takes `Vec<VectorEntry>`
- `query()` returns `Vec<VectorHit>` (id + score), skips tombstoned IDs
- `tombstone()` marks IDs as deleted without rebuilding graph
- `compact()` removes tombstoned entries and rebuilds graph
- `tombstone_count()` and `live_count()` for reporting
- Remove `entry_by_id()`, `entries()` accessors (records live in Tantivy now)
- `save()` persists tombstones alongside index.bin as `tombstones.bin`
- `load()` reads tombstones; accepts manifest version 5

```rust
use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use crate::entry::{VectorEntry, VectorHit};
use crate::error::{IndexError, IndexResult};
use crate::manifest::CorpusManifest;
use fastrag_embed::DynEmbedderTrait;

#[derive(Clone, Serialize, Deserialize)]
struct VectorPoint {
    vector: Vec<f32>,
}

impl Point for VectorPoint {
    fn distance(&self, other: &Self) -> f32 {
        euclidean_distance(&self.vector, &other.vector)
    }
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    dim: usize,
    manifest: CorpusManifest,
    entries: Vec<VectorEntry>,
    graph: HnswMap<VectorPoint, usize>,
    #[serde(default)]
    tombstones: HashSet<u64>,
}

impl HnswIndex {
    pub fn new(manifest: CorpusManifest) -> Self {
        let dim = manifest.identity.dim;
        let graph = Builder::default().build(Vec::<VectorPoint>::new(), Vec::<usize>::new());
        Self {
            dim,
            manifest,
            entries: Vec::new(),
            graph,
            tombstones: HashSet::new(),
        }
    }

    fn rebuild_graph(&mut self) {
        let points = self
            .entries
            .iter()
            .map(|entry| VectorPoint {
                vector: normalize(entry.vector.clone()),
            })
            .collect::<Vec<_>>();
        let values = (0..self.entries.len()).collect::<Vec<_>>();
        self.graph = Builder::default().build(points, values);
    }

    fn validate_vector(&self, vector: &[f32]) -> IndexResult<()> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    fn manifest_path(dir: &Path) -> PathBuf { dir.join("manifest.json") }
    fn index_path(dir: &Path) -> PathBuf { dir.join("index.bin") }
    fn entries_path(dir: &Path) -> PathBuf { dir.join("entries.bin") }
    fn tombstones_path(dir: &Path) -> PathBuf { dir.join("tombstones.bin") }

    pub fn manifest(&self) -> &CorpusManifest { &self.manifest }
    pub fn replace_manifest(&mut self, manifest: CorpusManifest) { self.manifest = manifest; }

    /// Mark IDs as tombstoned. Does not rebuild the graph.
    pub fn tombstone(&mut self, ids: &[u64]) {
        self.tombstones.extend(ids.iter());
    }

    /// Remove tombstoned entries and rebuild graph.
    pub fn compact(&mut self) {
        self.entries.retain(|e| !self.tombstones.contains(&e.id));
        self.tombstones.clear();
        self.manifest.chunk_count = self.entries.len();
        self.rebuild_graph();
    }

    pub fn tombstone_count(&self) -> usize { self.tombstones.len() }
    pub fn live_count(&self) -> usize { self.entries.len() - self.tombstones.len() }

    pub fn load(dir: &Path, embedder: &dyn DynEmbedderTrait) -> IndexResult<Self> {
        use fastrag_embed::{CANARY_COSINE_TOLERANCE, CANARY_TEXT, PassageText};

        let manifest_bytes = std::fs::read(Self::manifest_path(dir))
            .map_err(|_| IndexError::MissingCorpusFile { path: Self::manifest_path(dir) })?;
        let manifest: CorpusManifest = serde_json::from_slice(&manifest_bytes)?;

        if manifest.version != 5 {
            return Err(IndexError::UnsupportedSchema { got: manifest.version });
        }

        let live = embedder.identity();
        if live != manifest.identity {
            return Err(IndexError::IdentityMismatch {
                existing: manifest.identity.model_id.clone(),
                existing_dim: manifest.identity.dim,
                requested: live.model_id,
                requested_dim: live.dim,
            });
        }

        let reembedded = embedder
            .embed_passage_dyn(&[PassageText::new(CANARY_TEXT)])
            .map_err(|e| IndexError::CanaryEmbed(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| IndexError::CanaryEmbed("empty output".into()))?;

        let cosine = cosine_similarity(&reembedded, &manifest.canary.vector);
        if cosine < CANARY_COSINE_TOLERANCE {
            return Err(IndexError::CanaryMismatch { cosine, tolerance: CANARY_COSINE_TOLERANCE });
        }

        let graph_bytes = std::fs::read(Self::index_path(dir))
            .map_err(|_| IndexError::MissingCorpusFile { path: Self::index_path(dir) })?;
        let entries_bytes = std::fs::read(Self::entries_path(dir))
            .map_err(|_| IndexError::MissingCorpusFile { path: Self::entries_path(dir) })?;

        let graph: HnswMap<VectorPoint, usize> = bincode::deserialize(&graph_bytes)?;
        let entries: Vec<VectorEntry> = bincode::deserialize(&entries_bytes)?;

        let tombstones: HashSet<u64> = if Self::tombstones_path(dir).exists() {
            let bytes = std::fs::read(Self::tombstones_path(dir))?;
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            HashSet::new()
        };

        if manifest.identity.dim == 0 {
            return Err(IndexError::CorruptCorpus { message: "manifest dim cannot be 0".into() });
        }

        let dim = manifest.identity.dim;
        Ok(Self { dim, manifest, entries, graph, tombstones })
    }
}

impl crate::VectorIndex for HnswIndex {
    fn add(&mut self, entries: Vec<VectorEntry>) -> IndexResult<()> {
        for entry in &entries {
            self.validate_vector(&entry.vector)?;
        }
        self.entries.extend(entries.into_iter().map(|mut entry| {
            entry.vector = normalize(entry.vector);
            entry
        }));
        self.manifest.chunk_count = self.entries.len();
        self.rebuild_graph();
        Ok(())
    }

    fn query(&self, vector: &[f32], top_k: usize) -> IndexResult<Vec<VectorHit>> {
        self.validate_vector(vector)?;
        if top_k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_query = normalize(vector.to_vec());
        let query_point = VectorPoint { vector: normalized_query.clone() };
        let mut search = Search::default();
        let mut hits = Vec::new();

        // Over-fetch to account for tombstones
        let fetch = top_k + self.tombstones.len();
        for result in self.graph.search(&query_point, &mut search).take(fetch) {
            let entry = self.entries.get(*result.value)
                .ok_or_else(|| IndexError::CorruptCorpus {
                    message: format!("missing entry at index {}", result.value),
                })?;
            if self.tombstones.contains(&entry.id) {
                continue;
            }
            let score = cosine_similarity(&normalized_query, &result.point.vector);
            hits.push(VectorHit { id: entry.id, score });
            if hits.len() == top_k {
                break;
            }
        }

        hits.sort_by(|a, b| match b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal) {
            Ordering::Equal => a.id.cmp(&b.id),
            ord => ord,
        });
        Ok(hits)
    }

    fn save(&self, dir: &Path) -> IndexResult<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(Self::manifest_path(dir), serde_json::to_vec_pretty(&self.manifest)?)?;
        std::fs::write(Self::index_path(dir), bincode::serialize(&self.graph)?)?;
        std::fs::write(Self::entries_path(dir), bincode::serialize(&self.entries)?)?;
        if !self.tombstones.is_empty() {
            std::fs::write(Self::tombstones_path(dir), bincode::serialize(&self.tombstones)?)?;
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}
```

Keep the `normalize`, `euclidean_distance`, `cosine_similarity` utility functions unchanged at the bottom of the file.

- [ ] **Step 4: Update `lib.rs` re-exports and `VectorIndex` trait**

Replace `crates/fastrag-index/src/lib.rs`:

```rust
mod entry;
mod error;
mod hnsw;
mod manifest;

pub use entry::{VectorEntry, VectorHit};
pub use error::{IndexError, IndexResult};
pub use hnsw::HnswIndex;
pub use manifest::{
    ContextualizerManifest, CorpusManifest, FileEntry, ManifestChunkingStrategy, RootEntry,
};
pub use fastrag_core::ElementKind;

use std::path::Path;

pub trait VectorIndex {
    fn add(&mut self, entries: Vec<VectorEntry>) -> IndexResult<()>;
    fn query(&self, vector: &[f32], top_k: usize) -> IndexResult<Vec<VectorHit>>;
    fn save(&self, dir: &Path) -> IndexResult<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
```

- [ ] **Step 5: Update manifest version to 5**

In `crates/fastrag-index/src/manifest.rs`, change the `new()` function's version field from `4` to `5`.

- [ ] **Step 6: Run tests**

Run: `cargo test -p fastrag-index`

Expected: All `fastrag-index` tests pass. Other crates that depend on `IndexEntry` will break — that's expected, we fix them in later tasks.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p fastrag-index -- -D warnings`

Expected: No warnings

- [ ] **Step 8: Commit**

```bash
git add crates/fastrag-index/
git commit -m "refactor(index): slim to VectorEntry + tombstone support

IndexEntry replaced by VectorEntry {id, vector}. All text/metadata
moves to Tantivy (fastrag-store). HnswIndex gains tombstone set for
lazy deletion and compact() for graph rebuild."
```

---

## Task 3: Build `TantivyStore` with Dynamic Schema

**Files:**
- Create: `crates/fastrag-store/src/tantivy.rs`
- Modify: `crates/fastrag-store/src/lib.rs`
- Modify: `crates/fastrag-store/src/schema.rs`

- [ ] **Step 1: Write failing test for `TantivyStore` creation and document write/read**

Add to `crates/fastrag-store/src/tantivy.rs`:

```rust
use std::path::Path;
use tantivy::schema::{
    FAST, Field, INDEXED, NumericOptions, STORED, STRING, Schema, SchemaBuilder,
    TextFieldIndexing, TextOptions,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::error::{StoreError, StoreResult};
use crate::schema::{DynamicSchema, FieldDef, TypedKind, TypedValue};

/// Core fields always present in every corpus.
#[derive(Clone, Debug)]
pub struct CoreFields {
    pub id: Field,
    pub external_id: Field,
    pub content_hash: Field,
    pub chunk_index: Field,
    pub source_path: Field,
    pub source: Field,
    pub chunk_text: Field,
}

/// Handles for user-declared dynamic fields.
#[derive(Clone, Debug)]
pub struct UserFieldHandle {
    pub name: String,
    pub field: Field,
    pub typed: TypedKind,
}

pub struct TantivyStore {
    index: Index,
    reader: IndexReader,
    core: CoreFields,
    user_fields: Vec<UserFieldHandle>,
    schema: Schema,
}

impl TantivyStore {
    pub fn create(dir: &Path, dynamic_schema: &DynamicSchema) -> StoreResult<Self> {
        let (schema, core, user_fields) = build_tantivy_schema(dynamic_schema)?;
        let index = Index::create_in_dir(dir, schema.clone())?;
        let reader = index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self { index, reader, core, user_fields, schema })
    }

    pub fn open(dir: &Path, dynamic_schema: &DynamicSchema) -> StoreResult<Self> {
        let (schema, core, user_fields) = build_tantivy_schema(dynamic_schema)?;
        let index = Index::open_in_dir(dir)?;
        let reader = index.reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self { index, reader, core, user_fields, schema })
    }

    pub fn writer(&self) -> StoreResult<IndexWriter> {
        Ok(self.index.writer(50_000_000)?) // 50MB heap
    }

    pub fn core(&self) -> &CoreFields { &self.core }
    pub fn user_field(&self, name: &str) -> Option<&UserFieldHandle> {
        self.user_fields.iter().find(|f| f.name == name)
    }
}

fn build_tantivy_schema(
    dynamic_schema: &DynamicSchema,
) -> StoreResult<(Schema, CoreFields, Vec<UserFieldHandle>)> {
    let mut builder = SchemaBuilder::new();

    let id = builder.add_u64_field("_id", NumericOptions::default() | INDEXED | STORED | FAST);
    let external_id = builder.add_text_field("_external_id", STRING | STORED | FAST);
    let content_hash = builder.add_text_field("_content_hash", STORED);
    let chunk_index = builder.add_u64_field("_chunk_index", NumericOptions::default() | STORED | FAST);
    let source_path = builder.add_text_field("_source_path", STRING | STORED);
    let source = builder.add_text_field("_source", STORED); // stored-only JSON blob

    let text_opts = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    let chunk_text = builder.add_text_field("_chunk_text", text_opts);

    let core = CoreFields { id, external_id, content_hash, chunk_index, source_path, source, chunk_text };

    let mut user_fields = Vec::new();
    for field_def in &dynamic_schema.user_fields {
        let field = match &field_def.typed {
            TypedKind::String => {
                if field_def.positions {
                    let opts = TextOptions::default()
                        .set_indexing_options(
                            TextFieldIndexing::default()
                                .set_tokenizer("default")
                                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
                        )
                        .set_stored();
                    builder.add_text_field(&field_def.name, opts)
                } else {
                    builder.add_text_field(&field_def.name, STRING | STORED | FAST)
                }
            }
            TypedKind::Numeric => {
                builder.add_f64_field(&field_def.name, NumericOptions::default() | INDEXED | STORED | FAST)
            }
            TypedKind::Bool => {
                builder.add_u64_field(&field_def.name, NumericOptions::default() | INDEXED | STORED | FAST)
            }
            TypedKind::Date => {
                builder.add_date_field(&field_def.name, NumericOptions::default() | INDEXED | STORED | FAST)
            }
            TypedKind::Array(inner) => match inner.as_ref() {
                TypedKind::String => builder.add_text_field(&field_def.name, STRING | STORED),
                TypedKind::Numeric => builder.add_f64_field(&field_def.name, NumericOptions::default() | INDEXED | STORED | FAST),
                other => return Err(StoreError::Corrupt(format!("unsupported array element type: {other:?}"))),
            },
        };
        user_fields.push(UserFieldHandle {
            name: field_def.name.clone(),
            field,
            typed: field_def.typed.clone(),
        });
    }

    let schema = builder.build();
    Ok((schema, core, user_fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldDef;

    #[test]
    fn create_and_write_core_document() {
        let dir = tempfile::tempdir().unwrap();
        let tantivy_dir = dir.path().join("tantivy");
        std::fs::create_dir_all(&tantivy_dir).unwrap();

        let dyn_schema = DynamicSchema::new();
        let store = TantivyStore::create(&tantivy_dir, &dyn_schema).unwrap();

        let mut writer = store.writer().unwrap();
        writer.add_document(doc!(
            store.core().id => 1u64,
            store.core().external_id => "ext-001",
            store.core().content_hash => "abc123",
            store.core().chunk_index => 0u64,
            store.core().source_path => "test.jsonl",
            store.core().source => r#"{"title":"test"}"#,
            store.core().chunk_text => "This is a test chunk.",
        )).unwrap();
        writer.commit().unwrap();
        store.reader.reload().unwrap();

        let searcher = store.reader.searcher();
        assert_eq!(searcher.num_docs(), 1);
    }

    #[test]
    fn create_with_user_fields() {
        let dir = tempfile::tempdir().unwrap();
        let tantivy_dir = dir.path().join("tantivy");
        std::fs::create_dir_all(&tantivy_dir).unwrap();

        let mut dyn_schema = DynamicSchema::new();
        dyn_schema.merge(&[
            FieldDef { name: "severity".into(), typed: TypedKind::String, indexed: true, stored: true, positions: false },
            FieldDef { name: "cvss_score".into(), typed: TypedKind::Numeric, indexed: true, stored: true, positions: false },
        ]).unwrap();

        let store = TantivyStore::create(&tantivy_dir, &dyn_schema).unwrap();
        assert!(store.user_field("severity").is_some());
        assert!(store.user_field("cvss_score").is_some());
        assert!(store.user_field("nonexistent").is_none());
    }
}
```

- [ ] **Step 2: Update `lib.rs` to include tantivy module**

```rust
pub mod error;
pub mod schema;
pub mod tantivy;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p fastrag-store`

Expected: All tests pass (schema tests + tantivy tests)

- [ ] **Step 4: Add document fetch-by-IDs and delete-by-external-ID methods**

Add to `TantivyStore`:

```rust
    /// Fetch documents by their internal `_id` values.
    pub fn fetch_by_ids(&self, ids: &[u64]) -> StoreResult<Vec<TantivyDocument>> {
        let searcher = self.reader.searcher();
        let mut results = Vec::with_capacity(ids.len());
        for &id in ids {
            let term = tantivy::Term::from_field_u64(self.core.id, id);
            let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
            let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(1))?;
            if let Some((_score, addr)) = top_docs.first() {
                let doc: TantivyDocument = searcher.doc(*addr)?;
                results.push(doc);
            }
        }
        Ok(results)
    }

    /// Delete all documents with a given `_external_id`. Returns the `_id` values of deleted docs.
    pub fn delete_by_external_id(&self, writer: &mut IndexWriter, external_id: &str) -> StoreResult<Vec<u64>> {
        let searcher = self.reader.searcher();
        let term = tantivy::Term::from_field_text(self.core.external_id, external_id);
        let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(10_000))?;

        let mut deleted_ids = Vec::new();
        for (_score, addr) in &top_docs {
            let doc: TantivyDocument = searcher.doc(*addr)?;
            if let Some(id_val) = doc.get_first(self.core.id) {
                if let Some(id) = id_val.as_u64() {
                    deleted_ids.push(id);
                }
            }
        }

        writer.delete_term(tantivy::Term::from_field_text(self.core.external_id, external_id));
        Ok(deleted_ids)
    }

    /// Look up `_content_hash` for a given `_external_id`. Returns None if not found.
    pub fn content_hash_for(&self, external_id: &str) -> StoreResult<Option<String>> {
        let searcher = self.reader.searcher();
        let term = tantivy::Term::from_field_text(self.core.external_id, external_id);
        let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(1))?;

        if let Some((_score, addr)) = top_docs.first() {
            let doc: TantivyDocument = searcher.doc(*addr)?;
            if let Some(val) = doc.get_first(self.core.content_hash) {
                if let Some(s) = val.as_str() {
                    return Ok(Some(s.to_string()));
                }
            }
        }
        Ok(None)
    }

    /// BM25 full-text search on `_chunk_text`.
    pub fn bm25_search(&self, query_text: &str, top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        let searcher = self.reader.searcher();
        let query_parser = tantivy::query::QueryParser::for_index(&self.index, vec![self.core.chunk_text]);
        let query = query_parser.parse_query(query_text)
            .map_err(|e| StoreError::Corrupt(format!("query parse error: {e}")))?;
        let top_docs = searcher.search(&query, &tantivy::collector::TopDocs::with_limit(top_k))?;

        let mut hits = Vec::new();
        for (score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id_val) = doc.get_first(self.core.id) {
                if let Some(id) = id_val.as_u64() {
                    hits.push((id, score));
                }
            }
        }
        Ok(hits)
    }
```

Add test:

```rust
    #[test]
    fn write_fetch_delete_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let tantivy_dir = dir.path().join("tantivy");
        std::fs::create_dir_all(&tantivy_dir).unwrap();

        let dyn_schema = DynamicSchema::new();
        let store = TantivyStore::create(&tantivy_dir, &dyn_schema).unwrap();

        // Write two chunks for the same external ID
        let mut writer = store.writer().unwrap();
        writer.add_document(doc!(
            store.core().id => 1u64,
            store.core().external_id => "finding-001",
            store.core().content_hash => "hash1",
            store.core().chunk_index => 0u64,
            store.core().source_path => "data.jsonl",
            store.core().source => r#"{"id":"finding-001","title":"XSS"}"#,
            store.core().chunk_text => "Cross-site scripting found in login form.",
        )).unwrap();
        writer.add_document(doc!(
            store.core().id => 2u64,
            store.core().external_id => "finding-001",
            store.core().content_hash => "hash1",
            store.core().chunk_index => 1u64,
            store.core().source_path => "data.jsonl",
            store.core().source => r#"{"id":"finding-001","title":"XSS"}"#,
            store.core().chunk_text => "Remediation: encode output.",
        )).unwrap();
        writer.commit().unwrap();
        store.reader.reload().unwrap();

        // Fetch by IDs
        let docs = store.fetch_by_ids(&[1, 2]).unwrap();
        assert_eq!(docs.len(), 2);

        // Content hash lookup
        let hash = store.content_hash_for("finding-001").unwrap();
        assert_eq!(hash, Some("hash1".into()));

        // Delete by external ID
        let mut writer = store.writer().unwrap();
        let deleted = store.delete_by_external_id(&mut writer, "finding-001").unwrap();
        assert_eq!(deleted.len(), 2);
        writer.commit().unwrap();
        store.reader.reload().unwrap();

        let searcher = store.reader.searcher();
        assert_eq!(searcher.num_docs(), 0);
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p fastrag-store`

Expected: All tests pass

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p fastrag-store -- -D warnings`

Expected: No warnings

- [ ] **Step 7: Commit**

```bash
git add crates/fastrag-store/
git commit -m "feat(store): TantivyStore with dynamic schema, fetch, delete, BM25"
```

---

## Task 4: Build `Store` Facade

**Files:**
- Modify: `crates/fastrag-store/src/lib.rs`

- [ ] **Step 1: Write failing test for `Store` create/open/add/query cycle**

Replace `crates/fastrag-store/src/lib.rs`:

```rust
pub mod error;
pub mod schema;
pub mod tantivy;

use std::path::{Path, PathBuf};

use crate::error::{StoreError, StoreResult};
use crate::schema::DynamicSchema;
use crate::tantivy::TantivyStore;
use fastrag_index::HnswIndex;
use fastrag_index::manifest::CorpusManifest;

/// A record to persist: one chunk's worth of data.
pub struct ChunkRecord {
    pub id: u64,
    pub external_id: String,
    pub content_hash: String,
    pub chunk_index: usize,
    pub source_path: String,
    pub source_json: Option<String>,
    pub chunk_text: String,
    pub vector: Vec<f32>,
    pub user_fields: Vec<(String, crate::schema::TypedValue)>,
}

/// Result from a dense or hybrid search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub external_id: String,
    pub score: f32,
    pub source: Option<serde_json::Value>,
    pub chunks: Vec<ChunkHit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkHit {
    pub id: u64,
    pub chunk_index: usize,
    pub chunk_text: String,
    pub score: f32,
}

use serde::{Deserialize, Serialize};

pub struct Store {
    tantivy: TantivyStore,
    hnsw: HnswIndex,
    schema: DynamicSchema,
    corpus_dir: PathBuf,
}

impl Store {
    pub fn create(
        corpus_dir: &Path,
        manifest: CorpusManifest,
        schema: DynamicSchema,
    ) -> StoreResult<Self> {
        std::fs::create_dir_all(corpus_dir)?;
        let tantivy_dir = corpus_dir.join("tantivy_index");
        std::fs::create_dir_all(&tantivy_dir)?;

        let tantivy = TantivyStore::create(&tantivy_dir, &schema)?;
        let hnsw = HnswIndex::new(manifest);

        // Persist schema
        let schema_json = serde_json::to_vec_pretty(&schema)?;
        std::fs::write(corpus_dir.join("schema.json"), schema_json)?;

        Ok(Self { tantivy, hnsw, schema, corpus_dir: corpus_dir.to_path_buf() })
    }

    pub fn open(
        corpus_dir: &Path,
        embedder: &dyn fastrag_embed::DynEmbedderTrait,
    ) -> StoreResult<Self> {
        let schema_bytes = std::fs::read(corpus_dir.join("schema.json"))
            .map_err(|_| StoreError::Corrupt("missing schema.json".into()))?;
        let schema: DynamicSchema = serde_json::from_slice(&schema_bytes)?;

        let tantivy_dir = corpus_dir.join("tantivy_index");
        let tantivy = TantivyStore::open(&tantivy_dir, &schema)?;
        let hnsw = HnswIndex::load(corpus_dir, embedder)?;

        Ok(Self { tantivy, hnsw, schema, corpus_dir: corpus_dir.to_path_buf() })
    }

    pub fn schema(&self) -> &DynamicSchema { &self.schema }
    pub fn hnsw(&self) -> &HnswIndex { &self.hnsw }
    pub fn tantivy(&self) -> &TantivyStore { &self.tantivy }

    /// Merge new fields into the schema. Persists updated schema.json.
    pub fn evolve_schema(&mut self, fields: &[crate::schema::FieldDef]) -> StoreResult<()> {
        self.schema.merge(fields)?;
        let schema_json = serde_json::to_vec_pretty(&self.schema)?;
        std::fs::write(self.corpus_dir.join("schema.json"), schema_json)?;
        Ok(())
    }

    /// Add chunk records to both Tantivy and HNSW.
    pub fn add_records(&mut self, records: Vec<ChunkRecord>) -> StoreResult<()> {
        use ::tantivy::doc;

        let mut writer = self.tantivy.writer()?;
        let mut vector_entries = Vec::with_capacity(records.len());

        for rec in &records {
            let mut doc = doc!(
                self.tantivy.core().id => rec.id,
                self.tantivy.core().external_id => rec.external_id.clone(),
                self.tantivy.core().content_hash => rec.content_hash.clone(),
                self.tantivy.core().chunk_index => rec.chunk_index as u64,
                self.tantivy.core().source_path => rec.source_path.clone(),
                self.tantivy.core().chunk_text => rec.chunk_text.clone(),
            );

            if let Some(ref src) = rec.source_json {
                doc.add_text(self.tantivy.core().source, src);
            }

            for (field_name, value) in &rec.user_fields {
                if let Some(handle) = self.tantivy.user_field(field_name) {
                    add_typed_value_to_doc(&mut doc, handle.field, value);
                }
            }

            writer.add_document(doc)?;
            vector_entries.push(fastrag_index::VectorEntry {
                id: rec.id,
                vector: rec.vector.clone(),
            });
        }

        writer.commit()?;
        self.tantivy.reload()?;
        self.hnsw.add(vector_entries)?;
        Ok(())
    }

    /// Delete a record by external ID from both stores.
    pub fn delete_by_external_id(&mut self, external_id: &str) -> StoreResult<Vec<u64>> {
        let mut writer = self.tantivy.writer()?;
        let deleted_ids = self.tantivy.delete_by_external_id(&mut writer, external_id)?;
        writer.commit()?;
        self.tantivy.reload()?;
        self.hnsw.tombstone(&deleted_ids);
        Ok(deleted_ids)
    }

    /// Check content hash for upsert idempotency.
    pub fn content_hash_for(&self, external_id: &str) -> StoreResult<Option<String>> {
        self.tantivy.content_hash_for(external_id)
    }

    /// Dense vector search. Returns (id, score) pairs.
    pub fn query_dense(&self, vector: &[f32], top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        let hits = self.hnsw.query(vector, top_k)?;
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }

    /// BM25 full-text search. Returns (id, score) pairs.
    pub fn query_bm25(&self, query_text: &str, top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        self.tantivy.bm25_search(query_text, top_k)
    }

    /// Fetch full records by internal IDs, grouped by external_id.
    pub fn fetch_hits(&self, scored_ids: &[(u64, f32)]) -> StoreResult<Vec<SearchHit>> {
        let ids: Vec<u64> = scored_ids.iter().map(|(id, _)| *id).collect();
        let score_map: std::collections::HashMap<u64, f32> = scored_ids.iter().copied().collect();
        let docs = self.tantivy.fetch_by_ids(&ids)?;

        let mut groups: std::collections::HashMap<String, SearchHit> = std::collections::HashMap::new();

        for doc in docs {
            let ext_id = doc.get_first(self.tantivy.core().external_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let id = doc.get_first(self.tantivy.core().id)
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let chunk_text = doc.get_first(self.tantivy.core().chunk_text)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let chunk_index = doc.get_first(self.tantivy.core().chunk_index)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let source_str = doc.get_first(self.tantivy.core().source)
                .and_then(|v| v.as_str());
            let source: Option<serde_json::Value> = source_str
                .and_then(|s| serde_json::from_str(s).ok());
            let score = score_map.get(&id).copied().unwrap_or(0.0);

            let hit = groups.entry(ext_id.clone()).or_insert_with(|| SearchHit {
                external_id: ext_id,
                score,
                source,
                chunks: Vec::new(),
            });
            // Keep highest score
            if score > hit.score {
                hit.score = score;
            }
            hit.chunks.push(ChunkHit { id, chunk_index, chunk_text, score });
        }

        let mut results: Vec<SearchHit> = groups.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Compact HNSW (purge tombstones).
    pub fn compact(&mut self) {
        self.hnsw.compact();
    }

    pub fn tombstone_count(&self) -> usize { self.hnsw.tombstone_count() }
    pub fn live_count(&self) -> usize { self.hnsw.live_count() }

    /// Save HNSW to disk. Tantivy commits happen inline during add/delete.
    pub fn save(&self) -> StoreResult<()> {
        self.hnsw.save(&self.corpus_dir)?;
        Ok(())
    }

    pub fn replace_manifest(&mut self, manifest: CorpusManifest) {
        self.hnsw.replace_manifest(manifest);
    }

    pub fn manifest(&self) -> &CorpusManifest {
        self.hnsw.manifest()
    }
}

fn add_typed_value_to_doc(doc: &mut ::tantivy::TantivyDocument, field: ::tantivy::schema::Field, value: &crate::schema::TypedValue) {
    match value {
        crate::schema::TypedValue::String(s) => doc.add_text(field, s),
        crate::schema::TypedValue::Numeric(n) => doc.add_f64(field, *n),
        crate::schema::TypedValue::Bool(b) => doc.add_u64(field, if *b { 1 } else { 0 }),
        crate::schema::TypedValue::Date(d) => {
            let dt = d.and_hms_opt(0, 0, 0).unwrap();
            let tantivy_dt = ::tantivy::DateTime::from_timestamp_secs(dt.and_utc().timestamp());
            doc.add_date(field, tantivy_dt);
        }
        crate::schema::TypedValue::Array(arr) => {
            for item in arr {
                add_typed_value_to_doc(doc, field, item);
            }
        }
    }
}
```

Also add a `reload` method to `TantivyStore`:

```rust
    pub fn reload(&self) -> StoreResult<()> {
        self.reader.reload()?;
        Ok(())
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p fastrag-store`

Expected: All tests pass

- [ ] **Step 3: Write integration test for full Store round-trip**

Add test at bottom of `lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldDef, TypedKind, TypedValue};

    fn test_manifest() -> CorpusManifest {
        use fastrag_index::manifest::ManifestChunkingStrategy;
        use fastrag_embed::EmbedderIdentity;
        CorpusManifest::new(
            EmbedderIdentity { model_id: "test".into(), dim: 3, prefix_scheme_hash: 0 },
            vec![0.1, 0.2, 0.3], // canary vector
            ManifestChunkingStrategy::Basic { max_characters: 500, overlap: 0 },
        )
    }

    #[test]
    fn store_add_fetch_round_trip() {
        let dir = tempfile::tempdir().unwrap();

        let mut schema = DynamicSchema::new();
        schema.merge(&[
            FieldDef { name: "severity".into(), typed: TypedKind::String, indexed: true, stored: true, positions: false },
        ]).unwrap();

        let mut store = Store::create(dir.path(), test_manifest(), schema).unwrap();

        let records = vec![
            ChunkRecord {
                id: 1,
                external_id: "finding-001".into(),
                content_hash: "hash1".into(),
                chunk_index: 0,
                source_path: "data.jsonl".into(),
                source_json: Some(r#"{"id":"finding-001","title":"XSS"}"#.into()),
                chunk_text: "XSS in login form".into(),
                vector: vec![0.5, 0.3, 0.1],
                user_fields: vec![("severity".into(), TypedValue::String("high".into()))],
            },
            ChunkRecord {
                id: 2,
                external_id: "finding-001".into(),
                content_hash: "hash1".into(),
                chunk_index: 1,
                source_path: "data.jsonl".into(),
                source_json: Some(r#"{"id":"finding-001","title":"XSS"}"#.into()),
                chunk_text: "Encode all output".into(),
                vector: vec![0.2, 0.6, 0.4],
                user_fields: vec![("severity".into(), TypedValue::String("high".into()))],
            },
        ];

        store.add_records(records).unwrap();

        // Dense query
        let dense_hits = store.query_dense(&[0.5, 0.3, 0.1], 5).unwrap();
        assert!(!dense_hits.is_empty());

        // Fetch and verify dedup
        let results = store.fetch_hits(&dense_hits).unwrap();
        assert_eq!(results.len(), 1); // One external_id
        assert_eq!(results[0].external_id, "finding-001");
        assert_eq!(results[0].chunks.len(), 2);
        assert!(results[0].source.is_some());

        store.save().unwrap();
    }

    #[test]
    fn store_upsert_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let schema = DynamicSchema::new();
        let mut store = Store::create(dir.path(), test_manifest(), schema).unwrap();

        // Add
        store.add_records(vec![ChunkRecord {
            id: 1, external_id: "r1".into(), content_hash: "h1".into(),
            chunk_index: 0, source_path: "a.jsonl".into(), source_json: None,
            chunk_text: "hello".into(), vector: vec![0.1, 0.2, 0.3],
            user_fields: vec![],
        }]).unwrap();

        // Idempotency check
        let hash = store.content_hash_for("r1").unwrap();
        assert_eq!(hash, Some("h1".into()));

        // Delete
        let deleted = store.delete_by_external_id("r1").unwrap();
        assert_eq!(deleted, vec![1]);
        assert_eq!(store.tombstone_count(), 1);

        // Compact
        store.compact();
        assert_eq!(store.tombstone_count(), 0);
        assert_eq!(store.live_count(), 0);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p fastrag-store`

Expected: All tests pass

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p fastrag-store -- -D warnings`

Expected: No warnings

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag-store/
git commit -m "feat(store): Store facade with add/delete/query/fetch/compact"
```

---

## Task 5: JSONL Ingest Parser

**Files:**
- Create: `crates/fastrag/src/ingest/mod.rs`
- Create: `crates/fastrag/src/ingest/jsonl.rs`
- Modify: `crates/fastrag/src/lib.rs`

- [ ] **Step 1: Write failing test for JSONL line parsing**

Create `crates/fastrag/src/ingest/mod.rs`:

```rust
pub mod jsonl;
```

Create `crates/fastrag/src/ingest/jsonl.rs`:

```rust
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};

use fastrag_store::schema::{FieldDef, TypedKind, TypedValue};

#[derive(Debug, Clone)]
pub struct JsonlIngestConfig {
    pub text_fields: Vec<String>,
    pub id_field: String,
    pub metadata_fields: Vec<String>,
    pub metadata_types: BTreeMap<String, TypedKind>,
    pub array_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParsedRecord {
    pub external_id: String,
    pub content_hash: String,
    pub text: String,
    pub source_json: String,
    pub metadata: Vec<(String, TypedValue)>,
}

#[derive(Debug, thiserror::Error)]
pub enum JsonlError {
    #[error("line {line}: {message}")]
    ParseError { line: usize, message: String },
    #[error("line {line}: missing required field '{field}'")]
    MissingField { line: usize, field: String },
    #[error("line {line}: type mismatch for field '{field}': expected {expected:?}, got {got:?}")]
    TypeMismatch { line: usize, field: String, expected: TypedKind, got: TypedKind },
}

/// Infer TypedKind from a serde_json::Value.
pub fn infer_type(value: &serde_json::Value) -> Option<TypedKind> {
    match value {
        serde_json::Value::Bool(_) => Some(TypedKind::Bool),
        serde_json::Value::Number(_) => Some(TypedKind::Numeric),
        serde_json::Value::String(s) => {
            // Try ISO-8601 date
            if chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok() {
                Some(TypedKind::Date)
            } else {
                Some(TypedKind::String)
            }
        }
        serde_json::Value::Array(arr) => {
            let inner = arr.first().and_then(infer_type)?;
            Some(TypedKind::Array(Box::new(inner)))
        }
        serde_json::Value::Null => None,
        serde_json::Value::Object(_) => None,
    }
}

/// Convert a serde_json::Value to TypedValue given expected TypedKind.
pub fn to_typed_value(value: &serde_json::Value, kind: &TypedKind) -> Option<TypedValue> {
    match (value, kind) {
        (serde_json::Value::String(s), TypedKind::String) => Some(TypedValue::String(s.clone())),
        (serde_json::Value::Number(n), TypedKind::Numeric) => n.as_f64().map(TypedValue::Numeric),
        (serde_json::Value::Bool(b), TypedKind::Bool) => Some(TypedValue::Bool(*b)),
        (serde_json::Value::String(s), TypedKind::Date) => {
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok().map(TypedValue::Date)
        }
        (serde_json::Value::Array(arr), TypedKind::Array(inner)) => {
            let items: Option<Vec<TypedValue>> = arr.iter().map(|v| to_typed_value(v, inner)).collect();
            items.map(TypedValue::Array)
        }
        _ => None,
    }
}

/// Parse a JSONL stream into records. Infers types from first non-null values.
pub fn parse_jsonl<R: Read>(
    reader: R,
    config: &JsonlIngestConfig,
) -> Result<(Vec<ParsedRecord>, Vec<FieldDef>), JsonlError> {
    let buf = BufReader::new(reader);
    let mut records = Vec::new();
    let mut inferred_types: BTreeMap<String, TypedKind> = config.metadata_types.clone();
    let mut field_defs_built = false;
    let mut field_defs = Vec::new();

    for (line_idx, line_result) in buf.lines().enumerate() {
        let line_num = line_idx + 1;
        let line = line_result.map_err(|e| JsonlError::ParseError { line: line_num, message: e.to_string() })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| JsonlError::ParseError { line: line_num, message: e.to_string() })?;

        // Extract external ID
        let external_id = obj.get(&config.id_field)
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .ok_or_else(|| JsonlError::MissingField { line: line_num, field: config.id_field.clone() })?;

        // Content hash
        let content_hash = blake3::hash(trimmed.as_bytes()).to_hex().to_string();

        // Extract text
        let text_parts: Vec<&str> = config.text_fields.iter()
            .filter_map(|f| obj.get(f).and_then(|v| v.as_str()))
            .collect();
        let text = text_parts.join("\n\n");

        // Extract and type-check metadata
        let mut metadata = Vec::new();
        for field_name in &config.metadata_fields {
            let value = match obj.get(field_name) {
                Some(v) if !v.is_null() => v,
                _ => continue,
            };

            let is_array = config.array_fields.contains(field_name);
            let kind = if let Some(explicit) = inferred_types.get(field_name) {
                explicit.clone()
            } else if let Some(inferred) = infer_type(value) {
                let kind = if is_array && !matches!(inferred, TypedKind::Array(_)) {
                    TypedKind::Array(Box::new(inferred))
                } else {
                    inferred
                };
                inferred_types.insert(field_name.clone(), kind.clone());
                kind
            } else {
                continue;
            };

            if let Some(typed_val) = to_typed_value(value, &kind) {
                metadata.push((field_name.clone(), typed_val));
            }
        }

        // Build field defs once we have types for all fields
        if !field_defs_built && inferred_types.len() >= config.metadata_fields.iter()
            .filter(|f| obj.get(*f).map(|v| !v.is_null()).unwrap_or(false))
            .count()
        {
            for field_name in &config.metadata_fields {
                if let Some(kind) = inferred_types.get(field_name) {
                    field_defs.push(FieldDef {
                        name: field_name.clone(),
                        typed: kind.clone(),
                        indexed: true,
                        stored: true,
                        positions: matches!(kind, TypedKind::String),
                    });
                }
            }
            field_defs_built = true;
        }

        records.push(ParsedRecord {
            external_id,
            content_hash,
            text,
            source_json: trimmed.to_string(),
            metadata,
        });
    }

    // Build field defs if not yet built (e.g., all records processed)
    if !field_defs_built {
        for field_name in &config.metadata_fields {
            if let Some(kind) = inferred_types.get(field_name) {
                field_defs.push(FieldDef {
                    name: field_name.clone(),
                    typed: kind.clone(),
                    indexed: true,
                    stored: true,
                    positions: matches!(kind, TypedKind::String),
                });
            }
        }
    }

    Ok((records, field_defs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> JsonlIngestConfig {
        JsonlIngestConfig {
            text_fields: vec!["title".into(), "description".into()],
            id_field: "id".into(),
            metadata_fields: vec!["severity".into(), "cvss_score".into(), "tags".into()],
            metadata_types: BTreeMap::from([("cvss_score".into(), TypedKind::Numeric)]),
            array_fields: vec!["tags".into()],
        }
    }

    #[test]
    fn parse_single_record() {
        let input = r#"{"id":"f-001","title":"XSS","description":"Found XSS in login","severity":"high","cvss_score":7.5,"tags":["web","owasp"]}"#;
        let (records, fields) = parse_jsonl(input.as_bytes(), &sample_config()).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].external_id, "f-001");
        assert_eq!(records[0].text, "XSS\n\nFound XSS in login");
        assert!(!records[0].content_hash.is_empty());

        // Check metadata
        assert_eq!(records[0].metadata.len(), 3);

        // Check inferred field defs
        assert!(fields.iter().any(|f| f.name == "severity" && f.typed == TypedKind::String));
        assert!(fields.iter().any(|f| f.name == "cvss_score" && f.typed == TypedKind::Numeric));
        assert!(fields.iter().any(|f| f.name == "tags" && matches!(f.typed, TypedKind::Array(_))));
    }

    #[test]
    fn parse_multiple_records() {
        let input = concat!(
            r#"{"id":"1","title":"A","description":"Desc A","severity":"high","cvss_score":9.0,"tags":["a"]}"#, "\n",
            r#"{"id":"2","title":"B","description":"Desc B","severity":"low","cvss_score":2.0,"tags":["b"]}"#, "\n",
        );
        let (records, _) = parse_jsonl(input.as_bytes(), &sample_config()).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].external_id, "1");
        assert_eq!(records[1].external_id, "2");
    }

    #[test]
    fn missing_id_field_errors() {
        let input = r#"{"title":"no id"}"#;
        let err = parse_jsonl(input.as_bytes(), &sample_config()).unwrap_err();
        assert!(matches!(err, JsonlError::MissingField { line: 1, .. }));
    }

    #[test]
    fn malformed_json_errors() {
        let input = "not json at all";
        let err = parse_jsonl(input.as_bytes(), &sample_config()).unwrap_err();
        assert!(matches!(err, JsonlError::ParseError { line: 1, .. }));
    }

    #[test]
    fn blank_lines_skipped() {
        let input = concat!(
            r#"{"id":"1","title":"A","description":"D"}"#, "\n",
            "\n",
            r#"{"id":"2","title":"B","description":"D"}"#, "\n",
        );
        let config = JsonlIngestConfig {
            text_fields: vec!["title".into()],
            id_field: "id".into(),
            metadata_fields: vec![],
            metadata_types: BTreeMap::new(),
            array_fields: vec![],
        };
        let (records, _) = parse_jsonl(input.as_bytes(), &config).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn source_json_round_trip() {
        let input = r#"{"id":"1","title":"test","nested":{"a":1}}"#;
        let config = JsonlIngestConfig {
            text_fields: vec!["title".into()],
            id_field: "id".into(),
            metadata_fields: vec![],
            metadata_types: BTreeMap::new(),
            array_fields: vec![],
        };
        let (records, _) = parse_jsonl(input.as_bytes(), &config).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&records[0].source_json).unwrap();
        assert_eq!(parsed["nested"]["a"], 1);
    }
}
```

- [ ] **Step 2: Add `pub mod ingest;` to facade crate**

In `crates/fastrag/src/lib.rs`, add:

```rust
#[cfg(feature = "store")]
pub mod ingest;
```

- [ ] **Step 3: Update `crates/fastrag/Cargo.toml`**

Add dependencies and feature:

```toml
fastrag-store = { workspace = true, optional = true }
chrono = { version = "0.4", optional = true }
```

Add feature:

```toml
store = ["dep:fastrag-store", "dep:blake3", "dep:chrono", "retrieval"]
```

Update `retrieval` to not require `hybrid`:
```toml
retrieval = ["embedding", "index"]
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p fastrag --features store -- ingest`

Expected: All JSONL parser tests pass

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p fastrag --features store -- -D warnings`

Expected: No warnings

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/ingest/ crates/fastrag/src/lib.rs crates/fastrag/Cargo.toml
git commit -m "feat(ingest): JSONL parser with type inference and field extraction"
```

---

## Task 6: Wire JSONL Ingest into Corpus Operations

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs`

This is the largest change — `index_path_with_metadata` needs a JSONL path that uses `Store` instead of `HnswIndex` directly. The file-based path also needs to write through `Store`.

- [ ] **Step 1: Add `index_jsonl()` function to corpus/mod.rs**

Add a new public function alongside existing `index_path_with_metadata`:

```rust
#[cfg(feature = "store")]
pub fn index_jsonl(
    input: &Path,
    corpus_dir: &Path,
    chunking: &ChunkingStrategy,
    embedder: &dyn DynEmbedderTrait,
    config: &crate::ingest::jsonl::JsonlIngestConfig,
) -> Result<JsonlIndexStats, CorpusError> {
    use crate::ingest::jsonl::parse_jsonl;
    use fastrag_store::{Store, ChunkRecord};
    use fastrag_store::schema::DynamicSchema;
    use fastrag_embed::PassageText;

    // Parse JSONL
    let file = std::fs::File::open(input)
        .map_err(|e| CorpusError::Io(e))?;
    let (records, field_defs) = parse_jsonl(file, config)
        .map_err(|e| CorpusError::Other(e.to_string()))?;

    // Create or open store
    let mut store = if corpus_dir.join("schema.json").exists() {
        Store::open(corpus_dir, embedder)?
    } else {
        let canary_vec = embedder
            .embed_passage_dyn(&[PassageText::new(fastrag_embed::CANARY_TEXT)])
            .map_err(|e| CorpusError::Other(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| CorpusError::Other("empty canary".into()))?;

        let manifest = fastrag_index::CorpusManifest::new(
            embedder.identity(),
            canary_vec,
            chunking.to_manifest_strategy(),
        );
        let schema = DynamicSchema::new();
        Store::create(corpus_dir, manifest, schema)?
    };

    // Evolve schema with inferred fields
    if !field_defs.is_empty() {
        store.evolve_schema(&field_defs)?;
    }

    let mut chunk_records = Vec::new();
    let mut next_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut skipped = 0usize;
    let mut upserted = 0usize;

    for record in &records {
        // Upsert check
        if let Some(existing_hash) = store.content_hash_for(&record.external_id)? {
            if existing_hash == record.content_hash {
                skipped += 1;
                continue;
            }
            // Hash differs — delete old
            store.delete_by_external_id(&record.external_id)?;
            upserted += 1;
        }

        // Chunk the text
        let chunks = crate::chunk_text(&record.text, chunking);

        // Embed
        let passages: Vec<PassageText> = chunks.iter()
            .map(|c| PassageText::new(&c.text))
            .collect();
        let vectors = embedder.embed_passage_dyn(&passages)
            .map_err(|e| CorpusError::Other(e.to_string()))?;

        for (i, (chunk, vector)) in chunks.iter().zip(vectors.into_iter()).enumerate() {
            let id = next_id;
            next_id += 1;
            chunk_records.push(ChunkRecord {
                id,
                external_id: record.external_id.clone(),
                content_hash: record.content_hash.clone(),
                chunk_index: i,
                source_path: input.to_string_lossy().to_string(),
                source_json: Some(record.source_json.clone()),
                chunk_text: chunk.text.clone(),
                vector,
                user_fields: record.metadata.clone(),
            });
        }
    }

    if !chunk_records.is_empty() {
        let count = chunk_records.len();
        store.add_records(chunk_records)?;
        // Update manifest chunk count
        let mut manifest = store.manifest().clone();
        manifest.chunk_count = store.live_count();
        store.replace_manifest(manifest);
        store.save()?;
    }

    Ok(JsonlIndexStats {
        records_total: records.len(),
        records_skipped: skipped,
        records_upserted: upserted,
        records_new: records.len() - skipped - upserted,
        chunks_created: store.live_count(),
    })
}

#[cfg(feature = "store")]
#[derive(Debug, Clone)]
pub struct JsonlIndexStats {
    pub records_total: usize,
    pub records_skipped: usize,
    pub records_upserted: usize,
    pub records_new: usize,
    pub chunks_created: usize,
}
```

- [ ] **Step 2: Add `CorpusError` variant for store errors**

Add to the `CorpusError` enum:

```rust
#[cfg(feature = "store")]
#[error("store error: {0}")]
Store(#[from] fastrag_store::error::StoreError),
```

- [ ] **Step 3: Write integration test**

Create `tests/jsonl_ingest_e2e.rs` in the `crates/fastrag/tests/` directory:

```rust
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::io::Write;

use fastrag::ingest::jsonl::JsonlIngestConfig;
use fastrag::corpus::index_jsonl;
use fastrag::ChunkingStrategy;
use fastrag_embed::test_utils::FakeEmbedder;

fn config() -> JsonlIngestConfig {
    JsonlIngestConfig {
        text_fields: vec!["title".into(), "description".into()],
        id_field: "id".into(),
        metadata_fields: vec!["severity".into()],
        metadata_types: BTreeMap::new(),
        array_fields: vec![],
    }
}

#[test]
fn ingest_jsonl_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    let input = dir.path().join("data.jsonl");

    let mut f = std::fs::File::create(&input).unwrap();
    writeln!(f, r#"{{"id":"f1","title":"SQL Injection","description":"Found SQLi in search endpoint","severity":"critical"}}"#).unwrap();
    writeln!(f, r#"{{"id":"f2","title":"XSS","description":"Reflected XSS in profile page","severity":"high"}}"#).unwrap();

    let embedder = FakeEmbedder::new(3);
    let chunking = ChunkingStrategy::Basic { max_characters: 500, overlap: 0 };

    let stats = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
    assert_eq!(stats.records_total, 2);
    assert_eq!(stats.records_new, 2);
    assert_eq!(stats.records_skipped, 0);

    // Re-ingest same data — should skip
    let stats2 = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
    assert_eq!(stats2.records_skipped, 2);
    assert_eq!(stats2.records_new, 0);
}
```

Note: This test requires `FakeEmbedder` from `fastrag_embed::test_utils`. If that module doesn't exist, create a minimal fake embedder that returns fixed-dimension vectors. Check the existing test utilities in `fastrag-embed` first.

- [ ] **Step 4: Run tests**

Run: `cargo test -p fastrag --features store -- jsonl`

Expected: Integration test passes

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p fastrag --features store -- -D warnings`

Expected: No warnings

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs crates/fastrag/tests/
git commit -m "feat(corpus): index_jsonl with upsert idempotency and Store backend"
```

---

## Task 7: CLI Surface — Index JSONL, Delete, Compact Commands

**Files:**
- Modify: `fastrag-cli/src/args.rs`
- Modify: `fastrag-cli/src/main.rs`
- Modify: `fastrag-cli/Cargo.toml`

- [ ] **Step 1: Add JSONL flags to the Index command in `args.rs`**

Add to the `Index` variant (inside `#[cfg(feature = "retrieval")]`):

```rust
    /// Input format (auto-detect from extension, or specify explicitly)
    #[arg(long)]
    format: Option<String>,

    /// JSONL: fields to concatenate as embeddable text
    #[arg(long, value_delimiter = ',')]
    text_fields: Option<Vec<String>>,

    /// JSONL: field to use as stable external ID
    #[arg(long)]
    id_field: Option<String>,

    /// JSONL: fields to index as typed metadata
    #[arg(long, value_delimiter = ',')]
    metadata_fields: Option<Vec<String>>,

    /// JSONL: explicit type overrides (field=type, e.g. cvss_score=numeric)
    #[arg(long, value_delimiter = ',')]
    metadata_types: Option<Vec<String>>,

    /// JSONL: fields that contain arrays
    #[arg(long, value_delimiter = ',')]
    array_fields: Option<Vec<String>>,
```

- [ ] **Step 2: Add Delete and Compact commands**

Add to the `Command` enum:

```rust
    /// Delete a record by external ID from a corpus.
    #[cfg(feature = "store")]
    Delete {
        /// Path to corpus directory
        #[arg(long)]
        corpus: PathBuf,

        /// External ID of the record to delete
        #[arg(long)]
        id: String,
    },

    /// Compact corpus HNSW index (purge tombstones, rebuild graph).
    #[cfg(feature = "store")]
    Compact {
        /// Path to corpus directory
        #[arg(long)]
        corpus: PathBuf,
    },
```

- [ ] **Step 3: Wire handlers in `main.rs`**

Add match arms in the main command dispatch:

For the `Index` command, detect JSONL format and route to `index_jsonl`:

```rust
Command::Index { input, corpus, format, text_fields, id_field, metadata_fields, metadata_types, array_fields, .. } => {
    let is_jsonl = format.as_deref() == Some("jsonl")
        || input.extension().map(|e| e == "jsonl").unwrap_or(false);

    if is_jsonl {
        let config = fastrag::ingest::jsonl::JsonlIngestConfig {
            text_fields: text_fields.unwrap_or_default(),
            id_field: id_field.unwrap_or_else(|| "id".into()),
            metadata_fields: metadata_fields.unwrap_or_default(),
            metadata_types: parse_metadata_types(&metadata_types.unwrap_or_default()),
            array_fields: array_fields.unwrap_or_default(),
        };
        let embedder = embed_loader::load_for_write(&embedder_opts)?;
        let stats = fastrag::corpus::index_jsonl(&input, &corpus, &chunking, &*embedder, &config)?;
        println!("Indexed {} records ({} new, {} upserted, {} skipped), {} chunks",
            stats.records_total, stats.records_new, stats.records_upserted,
            stats.records_skipped, stats.chunks_created);
    } else {
        // Existing file-based index path
        // ...
    }
}
```

Add helper:

```rust
fn parse_metadata_types(types: &[String]) -> std::collections::BTreeMap<String, fastrag_store::schema::TypedKind> {
    let mut map = std::collections::BTreeMap::new();
    for entry in types {
        if let Some((k, v)) = entry.split_once('=') {
            let kind = match v {
                "string" => fastrag_store::schema::TypedKind::String,
                "numeric" => fastrag_store::schema::TypedKind::Numeric,
                "bool" => fastrag_store::schema::TypedKind::Bool,
                "date" => fastrag_store::schema::TypedKind::Date,
                _ => continue,
            };
            map.insert(k.to_string(), kind);
        }
    }
    map
}
```

Delete handler:

```rust
#[cfg(feature = "store")]
Command::Delete { corpus, id } => {
    let embedder = embed_loader::load_for_read(&corpus, &embedder_opts)?;
    let mut store = fastrag_store::Store::open(&corpus, &*embedder)?;
    let deleted = store.delete_by_external_id(&id)?;
    store.save()?;
    println!("Deleted {} chunks for external ID '{}'", deleted.len(), id);
}
```

Compact handler:

```rust
#[cfg(feature = "store")]
Command::Compact { corpus } => {
    let embedder = embed_loader::load_for_read(&corpus, &embedder_opts)?;
    let mut store = fastrag_store::Store::open(&corpus, &*embedder)?;
    let before = store.tombstone_count();
    store.compact();
    store.save()?;
    println!("Compacted: purged {} tombstones, {} live entries", before, store.live_count());
}
```

- [ ] **Step 4: Update `fastrag-cli/Cargo.toml`**

Add dependency:

```toml
fastrag-store = { workspace = true, optional = true }
```

Add feature:

```toml
store = ["dep:fastrag-store", "fastrag/store"]
```

Add `"store"` to the `default` feature list.

- [ ] **Step 5: Build and verify CLI help**

Run: `cargo build -p fastrag-cli`

Run: `cargo run -- index --help` — verify JSONL flags appear

Run: `cargo run -- delete --help` — verify delete command exists

Run: `cargo run -- compact --help` — verify compact command exists

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p fastrag-cli -- -D warnings`

Expected: No warnings

- [ ] **Step 7: Commit**

```bash
git add fastrag-cli/ crates/fastrag/
git commit -m "feat(cli): JSONL index flags, delete and compact commands"
```

---

## Task 8: Update Query Path and HTTP Server

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs` (query functions)
- Modify: `fastrag-cli/src/http.rs`

- [ ] **Step 1: Add Store-based query functions**

Add to `crates/fastrag/src/corpus/mod.rs`:

```rust
#[cfg(feature = "store")]
pub fn query_store(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
) -> Result<Vec<fastrag_store::SearchHit>, CorpusError> {
    use fastrag_embed::QueryText;

    let store = fastrag_store::Store::open(corpus_dir, embedder)?;

    let query_vec = embedder.embed_query_dyn(&[QueryText::new(query)])
        .map_err(|e| CorpusError::Other(e.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| CorpusError::Other("empty query embedding".into()))?;

    let dense_hits = store.query_dense(&query_vec, top_k)?;
    store.fetch_hits(&dense_hits)
        .map_err(|e| e.into())
}

#[cfg(feature = "store")]
pub fn query_store_hybrid(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
) -> Result<Vec<fastrag_store::SearchHit>, CorpusError> {
    use fastrag_embed::QueryText;

    let store = fastrag_store::Store::open(corpus_dir, embedder)?;

    let query_vec = embedder.embed_query_dyn(&[QueryText::new(query)])
        .map_err(|e| CorpusError::Other(e.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| CorpusError::Other("empty query embedding".into()))?;

    // Dense
    let dense_hits = store.query_dense(&query_vec, top_k * 2)?;
    // BM25
    let bm25_hits = store.query_bm25(query, top_k * 2)?;

    // RRF fusion (k=60)
    let fused = rrf_fuse(&dense_hits, &bm25_hits, top_k);

    store.fetch_hits(&fused)
        .map_err(|e| e.into())
}

#[cfg(feature = "store")]
fn rrf_fuse(dense: &[(u64, f32)], bm25: &[(u64, f32)], top_k: usize) -> Vec<(u64, f32)> {
    use std::collections::HashMap;
    let k = 60.0f32;
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for (rank, (id, _)) in dense.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }
    for (rank, (id, _)) in bm25.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (k + rank as f32 + 1.0);
    }

    let mut fused: Vec<(u64, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);
    fused
}
```

- [ ] **Step 2: Update HTTP query handler**

Update `fastrag-cli/src/http.rs` `run_query` to detect v5 corpora (presence of `schema.json`) and route through `query_store` or `query_store_hybrid`. Keep the existing path for legacy corpora during the transition period (will be removed once file-based ingest goes through Store too).

- [ ] **Step 3: Run tests**

Run: `cargo test --workspace --features store`

Expected: All tests pass

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --workspace --features store -- -D warnings`

Expected: No warnings

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/ fastrag-cli/src/http.rs
git commit -m "feat(query): Store-based query with RRF fusion and SearchHit grouping"
```

---

## Task 9: Update MCP Search and corpus-info

**Files:**
- Modify: `crates/fastrag-mcp/src/lib.rs`
- Modify: `crates/fastrag/src/corpus/mod.rs` (corpus_info)

- [ ] **Step 1: Update `search_corpus` MCP tool**

Update the MCP `search_corpus` handler to detect Store-based corpora and use `query_store()`. Return `SearchHit` JSON (with `_source` and grouped chunks) instead of flat chunk list.

- [ ] **Step 2: Update `corpus_info` to report Store metadata**

Add tombstone count, schema summary (field names + types), and storage type to the `CorpusInfo` struct returned by `corpus_info()`.

- [ ] **Step 3: Update CLI `corpus-info` display**

Add schema and tombstone info to the CLI output.

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace --features store`

Expected: All tests pass

- [ ] **Step 5: Run clippy**

Run: `cargo clippy --workspace --all-targets --features store -- -D warnings`

Expected: No warnings

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag-mcp/ crates/fastrag/src/corpus/ fastrag-cli/
git commit -m "feat(mcp,info): Store-aware search_corpus and corpus-info with schema display"
```

---

## Task 10: Remove `fastrag-tantivy` and `hybrid` Feature Flag

**Files:**
- Delete: `crates/fastrag-tantivy/` (entire directory)
- Modify: `Cargo.toml` (workspace) — remove member and workspace dep
- Modify: `crates/fastrag/Cargo.toml` — remove `fastrag-tantivy` dep and `hybrid` feature
- Modify: `fastrag-cli/Cargo.toml` — remove `hybrid` feature
- Modify: `crates/fastrag/src/corpus/hybrid.rs` — delete file
- Modify: any remaining references

- [ ] **Step 1: Remove crate directory**

Delete `crates/fastrag-tantivy/`.

- [ ] **Step 2: Remove from workspace `Cargo.toml`**

Remove `"crates/fastrag-tantivy"` from members list. Remove `fastrag-tantivy` from workspace dependencies.

- [ ] **Step 3: Remove from `crates/fastrag/Cargo.toml`**

Remove `fastrag-tantivy` optional dependency. Remove `hybrid` feature. Remove `hybrid.rs` module.

- [ ] **Step 4: Remove from `fastrag-cli/Cargo.toml`**

Remove `hybrid` from default features and feature definitions.

- [ ] **Step 5: Fix all compilation errors**

Grep for `hybrid` and `fastrag_tantivy` across the workspace. Fix any remaining references.

- [ ] **Step 6: Run full test suite**

Run: `cargo test --workspace --features store`

Expected: All tests pass

- [ ] **Step 7: Run full clippy**

Run: `cargo clippy --workspace --all-targets --features store -- -D warnings`

Expected: No warnings

- [ ] **Step 8: Run fmt**

Run: `cargo fmt --check`

Expected: No formatting issues

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove fastrag-tantivy crate and hybrid feature flag

Tantivy is now always-on via fastrag-store. The hybrid feature flag
and separate crate are no longer needed."
```

---

## Task 11: End-to-End Integration Test

**Files:**
- Create: `fastrag-cli/tests/jsonl_e2e.rs` (or `crates/fastrag/tests/jsonl_store_e2e.rs`)

- [ ] **Step 1: Write full round-trip integration test**

```rust
//! End-to-end: ingest JSONL → query → verify _source round-trip, typed filter, dedup.
#![cfg(feature = "store")]

use std::io::Write;

#[test]
fn jsonl_ingest_query_delete_compact() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("corpus");
    let input = dir.path().join("findings.jsonl");

    // Write test JSONL
    let mut f = std::fs::File::create(&input).unwrap();
    writeln!(f, r#"{{"id":"f1","title":"SQL Injection","description":"SQLi in /search","severity":"critical","cvss_score":9.8}}"#).unwrap();
    writeln!(f, r#"{{"id":"f2","title":"XSS","description":"Reflected XSS in /profile","severity":"high","cvss_score":7.5}}"#).unwrap();
    writeln!(f, r#"{{"id":"f3","title":"Info Disclosure","description":"Server version in headers","severity":"low","cvss_score":3.0}}"#).unwrap();

    // TODO: Use FakeEmbedder or real embedder depending on test environment
    // Index, query, verify _source, delete f2, compact, verify f2 gone
    // This test body depends on the exact API shape from prior tasks
}
```

The full test body will be written during implementation once the exact API is stable. The test must verify:
1. Ingest 3 JSONL records
2. Query returns results with `_source` intact
3. Re-ingest same data → all skipped (idempotent)
4. Modify f1 and re-ingest → f1 upserted, others skipped
5. Delete f2 → query no longer returns it
6. Compact → tombstones purged
7. `corpus-info` shows correct schema and counts

- [ ] **Step 2: Run test**

Run: `cargo test -p fastrag --features store -- jsonl_ingest_query_delete_compact`

Expected: PASS

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace --features store`

Expected: All pass

- [ ] **Step 4: Run full lint gate**

Run: `cargo clippy --workspace --all-targets --features store -- -D warnings && cargo fmt --check`

Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add fastrag-cli/tests/ crates/fastrag/tests/
git commit -m "test: end-to-end JSONL ingest, query, upsert, delete, compact

Closes #41"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end:

1. `cargo test --workspace` — all existing tests still pass
2. `cargo test --workspace --features store` — all new Store tests pass
3. `cargo clippy --workspace --all-targets --features store -- -D warnings` — clean
4. `cargo fmt --check` — clean
5. Manual smoke test:
   - Create a JSONL file with 3-5 records
   - `cargo run -- index test.jsonl --format jsonl --corpus /tmp/test-corpus --text-fields title,description --id-field id --metadata-fields severity`
   - `cargo run -- query "SQL injection" --corpus /tmp/test-corpus`
   - `cargo run -- corpus-info --corpus /tmp/test-corpus` — shows schema, chunk count
   - `cargo run -- delete --corpus /tmp/test-corpus --id f1`
   - `cargo run -- compact --corpus /tmp/test-corpus`
   - `cargo run -- corpus-info --corpus /tmp/test-corpus` — confirms deletion
