pub mod error;
pub mod schema;
pub mod tantivy;

use std::path::{Path, PathBuf};

use ::tantivy::TantivyDocument;
use ::tantivy::schema::{Field, Value};
use fastrag_index::{CorpusManifest, HnswIndex, VectorEntry, VectorIndex};
use serde::{Deserialize, Serialize};

use crate::error::StoreResult;
use crate::schema::{DynamicSchema, FieldDef, TypedValue};
use crate::tantivy::TantivyStore;

// ── string interning ─────────────────────────────────────────────────────────

/// Intern a heap-allocated string so that each unique value is leaked at most
/// once.  Subsequent calls with the same string return the previously-leaked
/// pointer without allocating.
fn intern_str(s: String) -> &'static str {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, &'static str>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(&cached) = map.get(&s) {
        return cached;
    }
    let leaked: &'static str = Box::leak(s.clone().into_boxed_str());
    map.insert(s, leaked);
    leaked
}

// ── public data types ─────────────────────────────────────────────────────────

/// A record to persist: one chunk's worth of data.
pub struct ChunkRecord {
    pub id: u64,
    pub external_id: String,
    pub content_hash: String,
    pub chunk_index: usize,
    pub source_path: String,
    /// Full original JSON, None for file-based ingest.
    pub source_json: Option<String>,
    pub chunk_text: String,
    pub vector: Vec<f32>,
    pub user_fields: Vec<(String, TypedValue)>,
}

/// Grouped search result — one per external_id.
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

// ── Store ─────────────────────────────────────────────────────────────────────

pub struct Store {
    tantivy: TantivyStore,
    hnsw: HnswIndex,
    schema: DynamicSchema,
    corpus_dir: PathBuf,
}

impl Store {
    fn schema_path(corpus_dir: &Path) -> PathBuf {
        corpus_dir.join("schema.json")
    }

    fn tantivy_dir(corpus_dir: &Path) -> PathBuf {
        corpus_dir.join("tantivy_index")
    }

    /// Create a new corpus directory with Tantivy + HNSW sub-stores.
    pub fn create(
        corpus_dir: &Path,
        manifest: CorpusManifest,
        schema: DynamicSchema,
    ) -> StoreResult<Self> {
        std::fs::create_dir_all(corpus_dir)?;
        let tantivy_dir = Self::tantivy_dir(corpus_dir);
        std::fs::create_dir_all(&tantivy_dir)?;

        let tantivy = TantivyStore::create(&tantivy_dir, &schema)?;
        let hnsw = HnswIndex::new(manifest);

        // Persist schema.json
        let schema_json = serde_json::to_vec_pretty(&schema)?;
        std::fs::write(Self::schema_path(corpus_dir), schema_json)?;

        Ok(Self {
            tantivy,
            hnsw,
            schema,
            corpus_dir: corpus_dir.to_path_buf(),
        })
    }

    /// Open an existing corpus directory.
    pub fn open(
        corpus_dir: &Path,
        embedder: &dyn fastrag_embed::DynEmbedderTrait,
    ) -> StoreResult<Self> {
        let schema: DynamicSchema = {
            let bytes = std::fs::read(Self::schema_path(corpus_dir))?;
            serde_json::from_slice(&bytes)?
        };

        let tantivy_dir = Self::tantivy_dir(corpus_dir);
        let tantivy = TantivyStore::open(&tantivy_dir, &schema)?;
        let hnsw = HnswIndex::load(corpus_dir, embedder)?;

        Ok(Self {
            tantivy,
            hnsw,
            schema,
            corpus_dir: corpus_dir.to_path_buf(),
        })
    }

    /// Open a corpus for querying with pre-computed vectors.
    ///
    /// Reads the manifest to construct a passthrough embedder that satisfies
    /// `HnswIndex::load`'s identity and canary checks without calling any real
    /// embedding model. Use this when the caller already holds pre-computed
    /// vectors and does not need `embed_query`.
    pub fn open_no_embedder(corpus_dir: &Path) -> StoreResult<Self> {
        use fastrag_embed::{EmbedderIdentity, PassageText, PrefixScheme, QueryText};
        use fastrag_index::CorpusManifest;

        // Read the manifest to extract identity + canary so we can construct
        // a NullEmbedder that passes HnswIndex::load's identity and canary checks.
        let manifest_bytes = std::fs::read(corpus_dir.join("manifest.json"))?;
        let manifest: CorpusManifest =
            serde_json::from_slice(&manifest_bytes).map_err(crate::error::StoreError::Json)?;

        let identity = manifest.identity.clone();
        let canary_vector = manifest.canary.vector.clone();

        // Leak the model_id string to satisfy the &'static str constraint on
        // DynEmbedderTrait::model_id(). This allocation lives for the duration
        // of the process but open_no_embedder is called infrequently.
        let model_id_static: &'static str = intern_str(identity.model_id.clone());

        struct ManifestPassthroughEmbedder {
            identity: EmbedderIdentity,
            canary_vector: Vec<f32>,
            model_id_static: &'static str,
        }

        impl fastrag_embed::DynEmbedderTrait for ManifestPassthroughEmbedder {
            fn model_id(&self) -> &'static str {
                self.model_id_static
            }

            fn dim(&self) -> usize {
                self.identity.dim
            }

            fn prefix_scheme(&self) -> PrefixScheme {
                PrefixScheme::NONE
            }

            fn prefix_scheme_hash(&self) -> u64 {
                self.identity.prefix_scheme_hash
            }

            fn identity(&self) -> EmbedderIdentity {
                self.identity.clone()
            }

            fn default_batch_size(&self) -> usize {
                64
            }

            fn embed_query_dyn(
                &self,
                _texts: &[QueryText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                Err(fastrag_embed::EmbedError::Http(
                    "ManifestPassthroughEmbedder: embed_query must not be called".into(),
                ))
            }

            fn embed_passage_dyn(
                &self,
                texts: &[PassageText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                // Return the stored canary vector for every input. HnswIndex::load
                // calls this exactly once with the canary text to verify drift.
                Ok(vec![self.canary_vector.clone(); texts.len()])
            }
        }

        let passthrough = ManifestPassthroughEmbedder {
            identity,
            canary_vector,
            model_id_static,
        };

        Self::open(corpus_dir, &passthrough)
    }

    pub fn schema(&self) -> &DynamicSchema {
        &self.schema
    }

    /// Merge new fields into schema.json.
    ///
    /// Note: Tantivy schema is fixed at creation time. evolve_schema only
    /// updates schema.json — the Tantivy index must be recreated if new
    /// fields need to be queryable.
    pub fn evolve_schema(&mut self, fields: &[FieldDef]) -> StoreResult<()> {
        for field in fields {
            self.schema.merge(field.clone())?;
        }
        let schema_json = serde_json::to_vec_pretty(&self.schema)?;
        std::fs::write(Self::schema_path(&self.corpus_dir), schema_json)?;
        Ok(())
    }

    /// Write records to Tantivy + HNSW in one transaction.
    pub fn add_records(&mut self, records: Vec<ChunkRecord>) -> StoreResult<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut writer = self.tantivy.writer()?;
        let core = self.tantivy.core().clone();

        let mut vector_entries: Vec<VectorEntry> = Vec::with_capacity(records.len());

        for record in records {
            let mut doc = TantivyDocument::default();
            doc.add_u64(core.id, record.id);
            doc.add_text(core.external_id, &record.external_id);
            doc.add_text(core.content_hash, &record.content_hash);
            doc.add_u64(core.chunk_index, record.chunk_index as u64);
            doc.add_text(core.source_path, &record.source_path);
            if let Some(ref src) = record.source_json {
                doc.add_text(core.source, src);
            } else {
                doc.add_text(core.source, "");
            }
            doc.add_text(core.chunk_text, &record.chunk_text);

            for (field_name, value) in &record.user_fields {
                if let Some(handle) = self.tantivy.user_field(field_name) {
                    let field = handle.field;
                    add_typed_value_to_doc(&mut doc, field, value);
                }
            }

            writer.add_document(doc)?;

            vector_entries.push(VectorEntry {
                id: record.id,
                vector: record.vector,
            });
        }

        writer.commit()?;
        drop(writer);
        self.tantivy.reload()?;

        self.hnsw.add(vector_entries)?;

        Ok(())
    }

    /// Delete all records with the given external_id from Tantivy + HNSW.
    ///
    /// Returns the internal IDs that were tombstoned.
    pub fn delete_by_external_id(&mut self, external_id: &str) -> StoreResult<Vec<u64>> {
        let mut writer = self.tantivy.writer()?;
        let deleted_ids = self
            .tantivy
            .delete_by_external_id(&mut writer, external_id)?;
        writer.commit()?;
        drop(writer);
        self.tantivy.reload()?;

        if !deleted_ids.is_empty() {
            self.hnsw.tombstone(&deleted_ids);
        }

        Ok(deleted_ids)
    }

    /// Fetch user-field metadata for the given `_id` values.
    ///
    /// Delegates to [`TantivyStore::fetch_metadata`].
    pub fn fetch_metadata(&self, ids: &[u64]) -> StoreResult<Vec<crate::tantivy::MetadataRow>> {
        self.tantivy.fetch_metadata(ids)
    }

    /// Return the content hash for an external_id, if present.
    pub fn content_hash_for(&self, external_id: &str) -> StoreResult<Option<String>> {
        self.tantivy.content_hash_for(external_id)
    }

    /// Dense vector search; returns (id, score) pairs.
    pub fn query_dense(&self, vector: &[f32], top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        let hits = self.hnsw.query(vector, top_k)?;
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }

    /// BM25 keyword search; returns (id, score) pairs.
    pub fn query_bm25(&self, query_text: &str, top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        self.tantivy.bm25_search(query_text, top_k)
    }

    /// Fetch and group Tantivy docs by external_id, building SearchHit structs.
    pub fn fetch_hits(&self, scored_ids: &[(u64, f32)]) -> StoreResult<Vec<SearchHit>> {
        if scored_ids.is_empty() {
            return Ok(vec![]);
        }

        let ids: Vec<u64> = scored_ids.iter().map(|(id, _)| *id).collect();
        let docs = self.tantivy.fetch_by_ids(&ids)?;
        let core = self.tantivy.core();

        // Build a score lookup
        let score_map: std::collections::HashMap<u64, f32> = scored_ids.iter().copied().collect();

        // Group by external_id
        let mut groups: std::collections::HashMap<
            String,
            (f32, Option<serde_json::Value>, Vec<ChunkHit>),
        > = std::collections::HashMap::new();

        for doc in &docs {
            let id = doc.get_first(core.id).and_then(|v| v.as_u64()).unwrap_or(0);
            let score = score_map.get(&id).copied().unwrap_or(0.0);

            let external_id = doc
                .get_first(core.external_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let chunk_index = doc
                .get_first(core.chunk_index)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let chunk_text = doc
                .get_first(core.chunk_text)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let source: Option<serde_json::Value> = doc
                .get_first(core.source)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str(s).ok());

            let chunk_hit = ChunkHit {
                id,
                chunk_index,
                chunk_text,
                score,
            };

            let entry = groups
                .entry(external_id)
                .or_insert_with(|| (score, source, Vec::new()));

            // Use highest score per group
            if score > entry.0 {
                entry.0 = score;
            }
            // Overwrite source if not yet set
            if entry.1.is_none() && chunk_hit.score >= 0.0 {
                // source is per-external_id, keep first non-None
            }
            entry.2.push(chunk_hit);
        }

        // Fix source: re-iterate to pick source from the doc that actually has it
        // (already captured above; the or_insert_with sets it from the first doc)

        let mut results: Vec<SearchHit> = groups
            .into_iter()
            .map(|(external_id, (score, source, chunks))| SearchHit {
                external_id,
                score,
                source,
                chunks,
            })
            .collect();

        // Sort deterministically by descending score, then external_id
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.external_id.cmp(&b.external_id))
        });

        Ok(results)
    }

    /// Purge tombstoned HNSW entries and rebuild the graph.
    pub fn compact(&mut self) {
        self.hnsw.compact();
    }

    pub fn tombstone_count(&self) -> usize {
        self.hnsw.tombstone_count()
    }

    pub fn live_count(&self) -> usize {
        self.hnsw.live_count()
    }

    /// Persist the HNSW index to corpus_dir.
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

// ── helper ────────────────────────────────────────────────────────────────────

fn add_typed_value_to_doc(doc: &mut TantivyDocument, field: Field, value: &TypedValue) {
    match value {
        TypedValue::String(s) => doc.add_text(field, s),
        TypedValue::Numeric(n) => doc.add_f64(field, *n),
        TypedValue::Bool(b) => doc.add_u64(field, if *b { 1 } else { 0 }),
        TypedValue::Date(d) => {
            let dt = d.and_hms_opt(0, 0, 0).unwrap();
            let tantivy_dt = ::tantivy::DateTime::from_timestamp_secs(dt.and_utc().timestamp());
            doc.add_date(field, tantivy_dt);
        }
        TypedValue::Array(arr) => {
            for item in arr {
                add_typed_value_to_doc(doc, field, item);
            }
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldDef, TypedKind};
    use fastrag_embed::{Canary, EmbedderIdentity, PrefixScheme};
    use fastrag_index::{CorpusManifest, ManifestChunkingStrategy};
    use tempfile::TempDir;

    fn test_manifest() -> CorpusManifest {
        CorpusManifest::new(
            EmbedderIdentity {
                model_id: "test/stub-3d-v1".into(),
                dim: 3,
                prefix_scheme_hash: PrefixScheme::NONE.hash(),
            },
            Canary {
                text_version: 1,
                vector: vec![1.0, 0.0, 0.0],
            },
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
        )
    }

    fn severity_schema() -> DynamicSchema {
        let mut s = DynamicSchema::new();
        s.merge(FieldDef {
            name: "severity".to_string(),
            typed: TypedKind::String,
            indexed: true,
            stored: true,
            positions: false,
        })
        .unwrap();
        s
    }

    fn make_chunk(id: u64, external_id: &str, chunk_index: usize, text: &str) -> ChunkRecord {
        ChunkRecord {
            id,
            external_id: external_id.to_string(),
            content_hash: "hash-test".to_string(),
            chunk_index,
            source_path: "/test/path".to_string(),
            source_json: Some(r#"{"origin":"test"}"#.to_string()),
            chunk_text: text.to_string(),
            vector: vec![1.0, 0.0, 0.0],
            user_fields: vec![(
                "severity".to_string(),
                TypedValue::String("high".to_string()),
            )],
        }
    }

    #[test]
    fn store_add_fetch_round_trip() {
        let dir = TempDir::new().unwrap();
        let mut store = Store::create(dir.path(), test_manifest(), severity_schema()).unwrap();

        let rec1 = make_chunk(1, "finding-001", 0, "First chunk of finding 001");
        let rec2 = make_chunk(2, "finding-001", 1, "Second chunk of finding 001");

        store.add_records(vec![rec1, rec2]).unwrap();

        let scored = store.query_dense(&[1.0, 0.0, 0.0], 10).unwrap();
        assert!(!scored.is_empty(), "dense query must return results");

        let hits = store.fetch_hits(&scored).unwrap();
        assert_eq!(hits.len(), 1, "expected 1 SearchHit grouped by external_id");

        let hit = &hits[0];
        assert_eq!(hit.external_id, "finding-001");
        assert_eq!(hit.chunks.len(), 2, "expected 2 chunks");
        assert!(hit.source.is_some(), "source must be present");
    }

    #[test]
    fn store_fetch_metadata_delegates_to_tantivy() {
        let dir = TempDir::new().unwrap();
        let mut schema = DynamicSchema::new();
        schema
            .merge(FieldDef {
                name: "severity".to_string(),
                typed: TypedKind::String,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        schema
            .merge(FieldDef {
                name: "cvss_score".to_string(),
                typed: TypedKind::Numeric,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();

        let mut store = Store::create(dir.path(), test_manifest(), schema).unwrap();

        let rec = ChunkRecord {
            id: 1,
            external_id: "vuln-001".to_string(),
            content_hash: "h1".to_string(),
            chunk_index: 0,
            source_path: "/test.txt".to_string(),
            source_json: None,
            chunk_text: "test".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            user_fields: vec![
                (
                    "severity".to_string(),
                    TypedValue::String("high".to_string()),
                ),
                ("cvss_score".to_string(), TypedValue::Numeric(7.5)),
            ],
        };

        store.add_records(vec![rec]).unwrap();

        let meta = store.fetch_metadata(&[1]).unwrap();
        assert_eq!(meta.len(), 1);

        let (id, fields) = &meta[0];
        assert_eq!(*id, 1);

        let map: std::collections::HashMap<&str, &TypedValue> =
            fields.iter().map(|(k, v)| (k.as_str(), v)).collect();

        assert_eq!(map["severity"], &TypedValue::String("high".to_string()));
        assert_eq!(map["cvss_score"], &TypedValue::Numeric(7.5));
    }

    #[test]
    fn store_upsert_and_delete() {
        let dir = TempDir::new().unwrap();
        let mut store = Store::create(dir.path(), test_manifest(), DynamicSchema::new()).unwrap();

        let rec = ChunkRecord {
            id: 10,
            external_id: "r1".to_string(),
            content_hash: "h1".to_string(),
            chunk_index: 0,
            source_path: "/data/r1.txt".to_string(),
            source_json: None,
            chunk_text: "content of r1".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            user_fields: vec![],
        };

        store.add_records(vec![rec]).unwrap();

        let hash = store.content_hash_for("r1").unwrap();
        assert_eq!(hash.as_deref(), Some("h1"), "content hash must be h1");

        let deleted = store.delete_by_external_id("r1").unwrap();
        assert_eq!(deleted.len(), 1, "expected 1 deleted id");
        assert_eq!(deleted[0], 10);

        assert_eq!(store.tombstone_count(), 1, "tombstone_count must be 1");

        store.compact();

        assert_eq!(
            store.tombstone_count(),
            0,
            "tombstone_count must be 0 after compact"
        );
        assert_eq!(store.live_count(), 0, "live_count must be 0 after compact");
    }
}
