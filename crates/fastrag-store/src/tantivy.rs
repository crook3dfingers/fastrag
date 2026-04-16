use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::{QueryParser, TermQuery};
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, NumericOptions, STORED, STRING, Schema, SchemaBuilder,
    TextFieldIndexing, TextOptions, Value,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use crate::error::{StoreError, StoreResult};
use crate::schema::{DynamicSchema, FieldDef, TypedKind, TypedValue};

/// Per-document metadata: internal id paired with typed field values.
pub type MetadataRow = (u64, Vec<(String, TypedValue)>);

/// Handles to the always-present core Tantivy fields.
#[derive(Debug, Clone)]
pub struct CoreFields {
    /// Auto-incrementing internal integer id.
    pub id: Field,
    /// Caller-supplied external identifier (e.g. file path + chunk index).
    pub external_id: Field,
    /// Content hash (blake3 hex) used for deduplication.
    pub content_hash: Field,
    /// Zero-based position of this chunk within its source document.
    pub chunk_index: Field,
    /// Path of the source file.
    pub source_path: Field,
    /// Full JSON blob of source metadata (stored only, not indexed).
    pub source: Field,
    /// The actual chunk text — indexed with positions for BM25 + snippets.
    pub chunk_text: Field,
}

/// Handle for a user-declared dynamic field.
#[derive(Debug, Clone)]
pub struct UserFieldHandle {
    pub name: String,
    pub field: Field,
    pub typed: TypedKind,
}

/// Per-field statistics computed from Tantivy fast-field columns.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldStat {
    pub name: String,
    pub field_type: FieldStatType,
    pub cardinality: u64,
}

/// Type-specific stat payload.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldStatType {
    Text,
    Numeric { min: f64, max: f64 },
}

/// Tantivy index wrapper with dynamic schema support.
pub struct TantivyStore {
    index: Index,
    reader: IndexReader,
    core: CoreFields,
    user_fields: Vec<UserFieldHandle>,
    // kept for reference; the schema is also accessible via index.schema()
    _schema: Schema,
}

/// Build a Tantivy [`Schema`] from a [`DynamicSchema`].
///
/// Returns the schema, core-field handles, and user-field handles.
pub fn build_tantivy_schema(
    dynamic_schema: &DynamicSchema,
) -> StoreResult<(Schema, CoreFields, Vec<UserFieldHandle>)> {
    let mut builder = SchemaBuilder::new();

    // --- core fields ---
    let id = builder.add_u64_field("_id", INDEXED | STORED | FAST);
    let external_id = builder.add_text_field("_external_id", STRING | STORED | FAST);
    let content_hash = builder.add_text_field("_content_hash", STORED);
    let chunk_index = builder.add_u64_field("_chunk_index", STORED | FAST);
    let source_path = builder.add_text_field("_source_path", STRING | STORED);
    let source = builder.add_text_field("_source", STORED);
    let chunk_text = {
        let indexing = TextFieldIndexing::default()
            .set_index_option(IndexRecordOption::WithFreqsAndPositions)
            .set_tokenizer("en_stem");
        let opts = TextOptions::default()
            .set_indexing_options(indexing)
            .set_stored();
        builder.add_text_field("_chunk_text", opts)
    };

    let core = CoreFields {
        id,
        external_id,
        content_hash,
        chunk_index,
        source_path,
        source,
        chunk_text,
    };

    // --- user-declared fields ---
    let mut user_handles: Vec<UserFieldHandle> = Vec::new();

    for field_def in &dynamic_schema.user_fields {
        let handle = build_user_field(&mut builder, field_def)?;
        user_handles.push(handle);
    }

    let schema = builder.build();
    Ok((schema, core, user_handles))
}

fn build_user_field(builder: &mut SchemaBuilder, fd: &FieldDef) -> StoreResult<UserFieldHandle> {
    let field = match &fd.typed {
        TypedKind::String => {
            if fd.positions {
                let indexing = TextFieldIndexing::default()
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions)
                    .set_tokenizer("en_stem");
                let opts = TextOptions::default()
                    .set_indexing_options(indexing)
                    .set_stored();
                builder.add_text_field(&fd.name, opts)
            } else {
                builder.add_text_field(&fd.name, STRING | STORED | FAST)
            }
        }
        TypedKind::Numeric => {
            let opts = NumericOptions::default() | INDEXED | STORED | FAST;
            builder.add_f64_field(&fd.name, opts)
        }
        TypedKind::Bool => {
            // Store bools as u64 (0/1) so they can be range-queried.
            let opts = NumericOptions::default() | INDEXED | STORED | FAST;
            builder.add_u64_field(&fd.name, opts)
        }
        TypedKind::Date => {
            let opts = tantivy::schema::DateOptions::default()
                .set_indexed()
                .set_stored()
                .set_fast();
            builder.add_date_field(&fd.name, opts)
        }
        TypedKind::Array => {
            // Arrays are stored as a multi-value text field (string elements)
            // or f64 field (numeric elements). Without element-type info in
            // FieldDef we default to STRING for ingest flexibility.
            builder.add_text_field(&fd.name, STRING | STORED)
        }
    };

    Ok(UserFieldHandle {
        name: fd.name.clone(),
        field,
        typed: fd.typed,
    })
}

// ── TantivyStore ────────────────────────────────────────────────────────────

impl TantivyStore {
    /// Create a brand-new Tantivy index in `dir`.
    pub fn create(dir: &Path, dynamic_schema: &DynamicSchema) -> StoreResult<Self> {
        let (schema, core, user_fields) = build_tantivy_schema(dynamic_schema)?;
        let index = Index::create_in_dir(dir, schema.clone())?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(StoreError::Tantivy)?;
        Ok(Self {
            index,
            reader,
            core,
            user_fields,
            _schema: schema,
        })
    }

    /// Open an existing Tantivy index from `dir`.
    pub fn open(dir: &Path, dynamic_schema: &DynamicSchema) -> StoreResult<Self> {
        let (schema, core, user_fields) = build_tantivy_schema(dynamic_schema)?;
        let index = Index::open_in_dir(dir)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(StoreError::Tantivy)?;
        Ok(Self {
            index,
            reader,
            core,
            user_fields,
            _schema: schema,
        })
    }

    /// Return a 50 MB index writer.
    pub fn writer(&self) -> StoreResult<IndexWriter> {
        Ok(self.index.writer(50_000_000)?)
    }

    /// Reload the reader so newly committed segments become visible.
    pub fn reload(&self) -> StoreResult<()> {
        self.reader.reload().map_err(StoreError::Tantivy)
    }

    /// Access the core field handles.
    pub fn core(&self) -> &CoreFields {
        &self.core
    }

    /// Look up a user-declared field handle by name.
    pub fn user_field(&self, name: &str) -> Option<&UserFieldHandle> {
        self.user_fields.iter().find(|h| h.name == name)
    }

    // ── query helpers ────────────────────────────────────────────────────────

    /// Find a single `_id` whose user field `field` equals the text value
    /// `value`. Supports [`TypedKind::String`] (non-tokenized exact match)
    /// and [`TypedKind::Numeric`]/[`TypedKind::Bool`] (via numeric parse).
    /// Returns `Ok(None)` if the field does not exist or the value has no
    /// match.
    ///
    /// Field types beyond those above (`Date`, `Array`, tokenized `String`
    /// with `positions=true`) are rejected with [`StoreError::Corrupt`] to
    /// avoid silently returning wrong answers.
    pub fn find_by_user_field(&self, field: &str, value: &str) -> StoreResult<Option<u64>> {
        let Some(handle) = self.user_field(field) else {
            return Ok(None);
        };
        let term = match handle.typed {
            TypedKind::String => Term::from_field_text(handle.field, value),
            TypedKind::Numeric => {
                let Ok(n) = value.parse::<f64>() else {
                    return Ok(None);
                };
                Term::from_field_f64(handle.field, n)
            }
            TypedKind::Bool => {
                let n: u64 = match value {
                    "true" | "1" => 1,
                    "false" | "0" => 0,
                    _ => return Ok(None),
                };
                Term::from_field_u64(handle.field, n)
            }
            TypedKind::Date | TypedKind::Array => {
                return Err(StoreError::Corrupt(format!(
                    "find_by_user_field does not support field type for {field}"
                )));
            }
        };
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let searcher = self.reader.searcher();
        let top = searcher.search(&query, &TopDocs::with_limit(1))?;
        for (_score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id_val) = doc.get_first(self.core.id).and_then(|v| v.as_u64()) {
                return Ok(Some(id_val));
            }
        }
        Ok(None)
    }

    /// Fetch documents by their `_id` values.
    pub fn fetch_by_ids(&self, ids: &[u64]) -> StoreResult<Vec<TantivyDocument>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let searcher = self.reader.searcher();
        let mut docs = Vec::with_capacity(ids.len());

        for &id_val in ids {
            let term = Term::from_field_u64(self.core.id, id_val);
            let query = TermQuery::new(term, IndexRecordOption::Basic);
            let top = searcher.search(&query, &TopDocs::with_limit(1))?;
            for (_score, addr) in top {
                let doc: TantivyDocument = searcher.doc(addr)?;
                docs.push(doc);
            }
        }
        Ok(docs)
    }

    /// Delete all documents with the given `_external_id`.
    ///
    /// Returns the `_id` values of the deleted documents.
    pub fn delete_by_external_id(
        &self,
        writer: &mut IndexWriter,
        external_id: &str,
    ) -> StoreResult<Vec<u64>> {
        // Collect ids before deletion so the caller can cascade to HNSW.
        let searcher = self.reader.searcher();
        let term = Term::from_field_text(self.core.external_id, external_id);
        let query = TermQuery::new(term.clone(), IndexRecordOption::Basic);
        let top = searcher.search(&query, &TopDocs::with_limit(1024))?;

        let mut deleted_ids = Vec::with_capacity(top.len());
        for (_score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id_val) = doc.get_first(self.core.id).and_then(|v| v.as_u64()) {
                deleted_ids.push(id_val);
            }
        }

        writer.delete_term(term);
        Ok(deleted_ids)
    }

    /// Return the `_content_hash` for the given `_external_id`, if it exists.
    pub fn content_hash_for(&self, external_id: &str) -> StoreResult<Option<String>> {
        let searcher = self.reader.searcher();
        let term = Term::from_field_text(self.core.external_id, external_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let top = searcher.search(&query, &TopDocs::with_limit(1))?;
        for (_score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(hash) = doc
                .get_first(self.core.content_hash)
                .and_then(|v| v.as_str())
            {
                return Ok(Some(hash.to_string()));
            }
        }
        Ok(None)
    }

    /// Fetch user-field metadata for the given `_id` values.
    ///
    /// Returns `(id, Vec<(field_name, TypedValue)>)` pairs. Fields with no
    /// stored value are silently skipped.
    pub fn fetch_metadata(&self, ids: &[u64]) -> StoreResult<Vec<MetadataRow>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let docs = self.fetch_by_ids(ids)?;
        let core = &self.core;
        let mut results = Vec::with_capacity(docs.len());

        for doc in &docs {
            let id = doc.get_first(core.id).and_then(|v| v.as_u64()).unwrap_or(0);

            let mut fields: Vec<(String, TypedValue)> = Vec::new();

            for handle in &self.user_fields {
                match handle.typed {
                    TypedKind::Array => {
                        let vals: Vec<TypedValue> = doc
                            .get_all(handle.field)
                            .filter_map(|v| v.as_str().map(|s| TypedValue::String(s.to_string())))
                            .collect();
                        if !vals.is_empty() {
                            fields.push((handle.name.clone(), TypedValue::Array(vals)));
                        }
                    }
                    TypedKind::String => {
                        if let Some(s) = doc.get_first(handle.field).and_then(|v| v.as_str()) {
                            fields.push((handle.name.clone(), TypedValue::String(s.to_string())));
                        }
                    }
                    TypedKind::Numeric => {
                        if let Some(n) = doc.get_first(handle.field).and_then(|v| v.as_f64()) {
                            fields.push((handle.name.clone(), TypedValue::Numeric(n)));
                        }
                    }
                    TypedKind::Bool => {
                        if let Some(u) = doc.get_first(handle.field).and_then(|v| v.as_u64()) {
                            fields.push((handle.name.clone(), TypedValue::Bool(u == 1)));
                        }
                    }
                    TypedKind::Date => {
                        if let Some(naive) = doc
                            .get_first(handle.field)
                            .and_then(|v| v.as_datetime())
                            .and_then(|dt| {
                                chrono::DateTime::from_timestamp(dt.into_timestamp_secs(), 0)
                                    .map(|d| d.date_naive())
                            })
                        {
                            fields.push((handle.name.clone(), TypedValue::Date(naive)));
                        }
                    }
                }
            }

            results.push((id, fields));
        }

        Ok(results)
    }

    /// BM25 search over `_chunk_text`.
    ///
    /// Returns `(id, score)` pairs for the top-`k` results.
    pub fn bm25_search(&self, query_text: &str, top_k: usize) -> StoreResult<Vec<(u64, f32)>> {
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.core.chunk_text]);
        let query = parser
            .parse_query(query_text)
            .map_err(|e| StoreError::Corrupt(format!("query parse error: {e}")))?;
        let top = searcher.search(&query, &TopDocs::with_limit(top_k))?;

        let mut results = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id_val) = doc.get_first(self.core.id).and_then(|v| v.as_u64()) {
                results.push((id_val, score));
            }
        }
        Ok(results)
    }

    /// Compute per-field statistics from Tantivy fast-field columns.
    ///
    /// Only `String` (non-tokenized/FAST) and `Numeric` fields are included.
    /// `Bool`, `Date`, and `Array` fields are skipped.
    pub fn field_stats(&self) -> Vec<FieldStat> {
        let searcher = self.reader.searcher();
        let mut stats = Vec::new();

        for handle in &self.user_fields {
            match handle.typed {
                TypedKind::String => {
                    // Only non-tokenized STRING fields have FAST flag;
                    // tokenized (positions=true) fields won't have a str column.
                    let mut cardinality: u64 = 0;
                    for seg_reader in searcher.segment_readers() {
                        if let Ok(Some(str_col)) = seg_reader.fast_fields().str(&handle.name) {
                            cardinality += str_col.num_terms() as u64;
                        }
                    }
                    stats.push(FieldStat {
                        name: handle.name.clone(),
                        field_type: FieldStatType::Text,
                        cardinality,
                    });
                }
                TypedKind::Numeric => {
                    let mut global_min = f64::INFINITY;
                    let mut global_max = f64::NEG_INFINITY;
                    let mut distinct = std::collections::BTreeSet::<u64>::new();
                    let mut has_values = false;

                    for seg_reader in searcher.segment_readers() {
                        if let Ok(Some(col)) =
                            seg_reader.fast_fields().column_opt::<f64>(&handle.name)
                        {
                            let seg_min = col.min_value();
                            let seg_max = col.max_value();
                            if seg_min <= seg_max {
                                has_values = true;
                                if seg_min < global_min {
                                    global_min = seg_min;
                                }
                                if seg_max > global_max {
                                    global_max = seg_max;
                                }
                            }
                            for doc_id in 0..seg_reader.max_doc() {
                                for val in col.values_for_doc(doc_id) {
                                    distinct.insert(val.to_bits());
                                }
                            }
                        }
                    }

                    let (min, max) = if has_values {
                        (global_min, global_max)
                    } else {
                        (0.0, 0.0)
                    };

                    stats.push(FieldStat {
                        name: handle.name.clone(),
                        field_type: FieldStatType::Numeric { min, max },
                        cardinality: distinct.len() as u64,
                    });
                }
                // Skip Bool, Date, Array fields.
                _ => {}
            }
        }

        stats
    }

    /// Generate highlighted snippets for the given internal IDs using Tantivy's
    /// SnippetGenerator. Returns one `Option<String>` per ID in input order.
    /// `None` if the document is not found or snippet generation fails.
    pub fn generate_snippets(
        &self,
        query_text: &str,
        ids: &[u64],
        max_chars: usize,
    ) -> Vec<Option<String>> {
        use tantivy::query::QueryParser;
        use tantivy::snippet::SnippetGenerator;

        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(searcher.index(), vec![self.core.chunk_text]);
        let query = match parser.parse_query(query_text) {
            Ok(q) => q,
            Err(_) => return vec![None; ids.len()],
        };

        let mut generator = match SnippetGenerator::create(&searcher, &*query, self.core.chunk_text)
        {
            Ok(g) => g,
            Err(_) => return vec![None; ids.len()],
        };
        generator.set_max_num_chars(max_chars);

        let mut results = Vec::with_capacity(ids.len());
        for &id_val in ids {
            let term = tantivy::Term::from_field_u64(self.core.id, id_val);
            let id_query =
                tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
            let top = match searcher.search(&id_query, &tantivy::collector::TopDocs::with_limit(1))
            {
                Ok(t) => t,
                Err(_) => {
                    results.push(None);
                    continue;
                }
            };
            if let Some((_score, addr)) = top.first() {
                match searcher.doc::<TantivyDocument>(*addr) {
                    Ok(doc) => {
                        let snippet = generator.snippet_from_doc(&doc);
                        let html = snippet.to_html();
                        if html.is_empty() {
                            results.push(None);
                        } else {
                            results.push(Some(html));
                        }
                    }
                    Err(_) => results.push(None),
                }
            } else {
                results.push(None);
            }
        }
        results
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{DynamicSchema, FieldDef, TypedKind};
    use tempfile::TempDir;

    fn empty_schema() -> DynamicSchema {
        DynamicSchema::new()
    }

    fn make_store(dir: &TempDir) -> TantivyStore {
        TantivyStore::create(dir.path(), &empty_schema()).expect("create store")
    }

    // ── test 1 ───────────────────────────────────────────────────────────────

    #[test]
    fn create_and_write_core_document() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        let core = store.core();

        let mut writer = store.writer().unwrap();
        let mut doc = TantivyDocument::default();
        doc.add_u64(core.id, 1);
        doc.add_text(core.external_id, "doc-001");
        doc.add_text(core.content_hash, "abc123");
        doc.add_u64(core.chunk_index, 0);
        doc.add_text(core.source_path, "/data/test.txt");
        doc.add_text(core.source, r#"{"origin":"test"}"#);
        doc.add_text(
            core.chunk_text,
            "The quick brown fox jumps over the lazy dog",
        );
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        store.reload().unwrap();

        let searcher = store.reader.searcher();
        assert_eq!(searcher.num_docs(), 1, "expected exactly one document");
    }

    // ── test 2 ───────────────────────────────────────────────────────────────

    #[test]
    fn create_with_user_fields() {
        let dir = TempDir::new().unwrap();
        let mut dyn_schema = DynamicSchema::new();
        dyn_schema
            .merge(FieldDef {
                name: "severity".to_string(),
                typed: TypedKind::String,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "cvss_score".to_string(),
                typed: TypedKind::Numeric,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();

        let store = TantivyStore::create(dir.path(), &dyn_schema).unwrap();

        let sev = store
            .user_field("severity")
            .expect("severity field must exist");
        assert_eq!(sev.name, "severity");
        assert_eq!(sev.typed, TypedKind::String);

        let cvss = store
            .user_field("cvss_score")
            .expect("cvss_score field must exist");
        assert_eq!(cvss.name, "cvss_score");
        assert_eq!(cvss.typed, TypedKind::Numeric);

        assert!(store.user_field("nonexistent").is_none());
    }

    // ── test: fetch_metadata ────────────────────────────────────────────────

    #[test]
    fn fetch_metadata_returns_typed_values() {
        let dir = TempDir::new().unwrap();

        let mut dyn_schema = DynamicSchema::new();
        dyn_schema
            .merge(FieldDef {
                name: "severity".to_string(),
                typed: TypedKind::String,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "cvss_score".to_string(),
                typed: TypedKind::Numeric,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "published".to_string(),
                typed: TypedKind::Bool,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "pub_date".to_string(),
                typed: TypedKind::Date,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "tags".to_string(),
                typed: TypedKind::Array,
                indexed: false,
                stored: true,
                positions: false,
            })
            .unwrap();

        let store = TantivyStore::create(dir.path(), &dyn_schema).unwrap();
        let core = store.core();

        let mut writer = store.writer().unwrap();
        let mut doc = TantivyDocument::default();
        doc.add_u64(core.id, 1);
        doc.add_text(core.external_id, "vuln-001");
        doc.add_text(core.content_hash, "abc");
        doc.add_u64(core.chunk_index, 0);
        doc.add_text(core.source_path, "/test.txt");
        doc.add_text(core.source, "{}");
        doc.add_text(core.chunk_text, "test chunk");

        // Add user field values
        let sev = store.user_field("severity").unwrap();
        doc.add_text(sev.field, "critical");

        let cvss = store.user_field("cvss_score").unwrap();
        doc.add_f64(cvss.field, 9.8);

        let pub_field = store.user_field("published").unwrap();
        doc.add_u64(pub_field.field, 1); // true

        let date_field = store.user_field("pub_date").unwrap();
        let dt = chrono::NaiveDate::from_ymd_opt(2024, 6, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let tantivy_dt = ::tantivy::DateTime::from_timestamp_secs(dt.and_utc().timestamp());
        doc.add_date(date_field.field, tantivy_dt);

        let tags_field = store.user_field("tags").unwrap();
        doc.add_text(tags_field.field, "rce");
        doc.add_text(tags_field.field, "network");

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        store.reload().unwrap();

        let meta = store.fetch_metadata(&[1]).unwrap();
        assert_eq!(meta.len(), 1, "expected 1 result");

        let (id, fields) = &meta[0];
        assert_eq!(*id, 1);

        use crate::schema::TypedValue;

        let map: std::collections::HashMap<&str, &TypedValue> =
            fields.iter().map(|(k, v)| (k.as_str(), v)).collect();

        assert_eq!(map["severity"], &TypedValue::String("critical".to_string()));
        assert_eq!(map["cvss_score"], &TypedValue::Numeric(9.8));
        assert_eq!(map["published"], &TypedValue::Bool(true));
        assert_eq!(
            map["pub_date"],
            &TypedValue::Date(chrono::NaiveDate::from_ymd_opt(2024, 6, 15).unwrap())
        );
        assert_eq!(
            map["tags"],
            &TypedValue::Array(vec![
                TypedValue::String("rce".to_string()),
                TypedValue::String("network".to_string()),
            ])
        );
    }

    // ── test: field_stats ────────────────────────────────────────────────────

    #[test]
    fn field_stats_returns_cardinality_and_range() {
        let dir = TempDir::new().unwrap();
        let mut dyn_schema = DynamicSchema::new();
        dyn_schema
            .merge(FieldDef {
                name: "severity".to_string(),
                typed: TypedKind::String,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();
        dyn_schema
            .merge(FieldDef {
                name: "cvss".to_string(),
                typed: TypedKind::Numeric,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();

        let store = TantivyStore::create(dir.path(), &dyn_schema).unwrap();
        let core = store.core();
        let sev = store.user_field("severity").unwrap().field;
        let cvss = store.user_field("cvss").unwrap().field;

        let mut writer = store.writer().unwrap();
        for (id, severity, score) in [(1u64, "HIGH", 9.8), (2, "LOW", 3.1), (3, "HIGH", 7.5)] {
            let mut doc = TantivyDocument::default();
            doc.add_u64(core.id, id);
            doc.add_text(core.external_id, format!("doc-{id}"));
            doc.add_text(core.content_hash, "hash");
            doc.add_u64(core.chunk_index, 0);
            doc.add_text(core.source_path, "/test.txt");
            doc.add_text(core.source, "{}");
            doc.add_text(core.chunk_text, "test");
            doc.add_text(sev, severity);
            doc.add_f64(cvss, score);
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        store.reload().unwrap();

        let stats = store.field_stats();
        assert_eq!(stats.len(), 2, "expected 2 field stats (severity + cvss)");

        let sev_stat = stats
            .iter()
            .find(|s| s.name == "severity")
            .expect("severity stat");
        assert!(
            sev_stat.cardinality >= 2,
            "severity cardinality should be >= 2 (HIGH, LOW), got {}",
            sev_stat.cardinality
        );
        assert_eq!(sev_stat.field_type, FieldStatType::Text);

        let cvss_stat = stats.iter().find(|s| s.name == "cvss").expect("cvss stat");
        assert!(
            cvss_stat.cardinality >= 3,
            "cvss cardinality should be >= 3, got {}",
            cvss_stat.cardinality
        );
        match &cvss_stat.field_type {
            FieldStatType::Numeric { min, max } => {
                assert!(
                    (*min - 3.1).abs() < 0.01,
                    "cvss min should be ~3.1, got {min}"
                );
                assert!(
                    (*max - 9.8).abs() < 0.01,
                    "cvss max should be ~9.8, got {max}"
                );
            }
            other => panic!("expected Numeric stat for cvss, got {other:?}"),
        }
    }

    #[test]
    fn field_stats_empty_index() {
        let dir = TempDir::new().unwrap();
        let mut dyn_schema = DynamicSchema::new();
        dyn_schema
            .merge(FieldDef {
                name: "severity".to_string(),
                typed: TypedKind::String,
                indexed: true,
                stored: true,
                positions: false,
            })
            .unwrap();

        let store = TantivyStore::create(dir.path(), &dyn_schema).unwrap();

        let stats = store.field_stats();
        assert_eq!(stats.len(), 1);
        let sev_stat = &stats[0];
        assert_eq!(sev_stat.name, "severity");
        assert_eq!(sev_stat.cardinality, 0);
        assert_eq!(sev_stat.field_type, FieldStatType::Text);
    }

    // ── test 3 ───────────────────────────────────────────────────────────────

    #[test]
    fn write_fetch_delete_round_trip() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        let core = store.core();

        let mut writer = store.writer().unwrap();

        // Write two chunks with the same external_id.
        for chunk_idx in 0u64..2 {
            let mut doc = TantivyDocument::default();
            doc.add_u64(core.id, 10 + chunk_idx);
            doc.add_text(core.external_id, "ext-42");
            doc.add_text(core.content_hash, "hash-xyz");
            doc.add_u64(core.chunk_index, chunk_idx);
            doc.add_text(core.source_path, "/data/multi.txt");
            doc.add_text(core.source, "{}");
            doc.add_text(core.chunk_text, format!("chunk number {chunk_idx}"));
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        drop(writer); // release the index lock before acquiring a new writer
        store.reload().unwrap();

        // fetch_by_ids must return both docs.
        let fetched = store.fetch_by_ids(&[10, 11]).unwrap();
        assert_eq!(fetched.len(), 2, "expected 2 fetched docs");

        // content_hash_for must return the hash.
        let hash = store.content_hash_for("ext-42").unwrap();
        assert_eq!(hash.as_deref(), Some("hash-xyz"));

        // delete_by_external_id must report 2 deleted ids and remove the docs.
        let mut writer2 = store.writer().unwrap();
        let deleted = store.delete_by_external_id(&mut writer2, "ext-42").unwrap();
        assert_eq!(deleted.len(), 2, "expected 2 deleted ids");
        assert!(deleted.contains(&10));
        assert!(deleted.contains(&11));
        writer2.commit().unwrap();
        store.reload().unwrap();

        let searcher = store.reader.searcher();
        assert_eq!(searcher.num_docs(), 0, "all docs must be deleted");
    }

    #[test]
    fn generate_snippets_highlights_matching_terms() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        let core = store.core();

        let mut writer = store.writer().unwrap();
        let mut doc = TantivyDocument::default();
        doc.add_u64(core.id, 1);
        doc.add_text(core.external_id, "doc-1");
        doc.add_text(core.content_hash, "h1");
        doc.add_u64(core.chunk_index, 0);
        doc.add_text(core.source_path, "/t.txt");
        doc.add_text(core.source, "{}");
        doc.add_text(
            core.chunk_text,
            "The SQL injection vulnerability allows remote code execution on the server",
        );
        writer.add_document(doc).unwrap();
        writer.commit().unwrap();
        store.reload().unwrap();

        let snippets = store.generate_snippets("SQL injection", &[1], 150);
        assert_eq!(snippets.len(), 1);
        let snippet = snippets[0].as_ref().expect("snippet should be Some");
        assert!(
            snippet.contains("<b>"),
            "snippet should contain highlight tags: {snippet}"
        );
    }

    #[test]
    fn generate_snippets_returns_none_for_missing_doc() {
        let dir = TempDir::new().unwrap();
        let store = make_store(&dir);
        let snippets = store.generate_snippets("test", &[999], 150);
        assert_eq!(snippets.len(), 1);
        assert!(snippets[0].is_none(), "missing doc should produce None");
    }
}
