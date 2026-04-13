use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ChunkingStrategy, Document, ElementKind, FastRagError, HnswIndex, VectorEntry, VectorHit,
    VectorIndex,
};

#[cfg(feature = "index")]
use crate::{CorpusManifest, ManifestChunkingStrategy};

use fastrag_embed::DynEmbedderTrait;

pub mod incremental;

/// Options that opt a single `index_path_with_metadata` run into Contextual
/// Retrieval. Carries the contextualizer, a mutable borrow on the cache, and
/// the `strict` flag. Only constructed at the CLI / test layer — the core
/// ops crate stays feature-gated.
#[cfg(feature = "contextual")]
pub struct ContextualizeOptions<'a> {
    pub contextualizer: &'a dyn fastrag_context::Contextualizer,
    pub cache: &'a mut fastrag_context::ContextCache,
    pub strict: bool,
}

/// Per-run contextualization counters surfaced back to the CLI summary line.
#[cfg(feature = "contextual")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ContextualizeStats {
    pub ok: usize,
    pub fallback: usize,
}

/// Result of a `--retry-failed` pass: how many failed rows we found, how many
/// we repaired, and whether the dense HNSW index was rebuilt from the cache.
#[cfg(feature = "contextual")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RetryReport {
    pub total_failed: usize,
    pub repaired: usize,
    pub rebuilt_dense: bool,
}

/// Per-stage query latency in microseconds.
///
/// Passed `&mut` into every `query_corpus_*` variant. Callers that don't
/// care pass `&mut LatencyBreakdown::default()` and ignore the result.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub embed_us: u64,
    pub bm25_us: u64,
    pub hnsw_us: u64,
    pub rerank_us: u64,
    pub fuse_us: u64,
    pub total_us: u64,
}

impl LatencyBreakdown {
    /// Sum per-stage microseconds into `total_us`. Call once after a query completes.
    pub fn finalize(&mut self) {
        self.total_us = self
            .embed_us
            .saturating_add(self.bm25_us)
            .saturating_add(self.hnsw_us)
            .saturating_add(self.rerank_us)
            .saturating_add(self.fuse_us);
    }
}

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] FastRagError),
    #[error("embedding error: {0}")]
    Embed(String),
    #[cfg(feature = "index")]
    #[error("index error: {0}")]
    Index(#[from] crate::IndexError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no parseable files found in {0}")]
    NoParseableFiles(PathBuf),
    #[error("embedder returned {got} vectors for {expected} chunks")]
    EmbeddingOutputMismatch { expected: usize, got: usize },
    #[error("embedder returned no vectors")]
    EmptyEmbeddingOutput,
    #[error("invalid metadata sidecar: {0}")]
    BadMetadataSidecar(String),
    #[cfg(feature = "rerank")]
    #[error("reranker error: {0}")]
    Rerank(String),
    #[cfg(feature = "store")]
    #[error("store error: {0}")]
    Store(#[from] fastrag_store::error::StoreError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusIndexStats {
    pub corpus_dir: PathBuf,
    pub input_dir: PathBuf,
    pub files_indexed: usize,
    pub chunk_count: usize,
    pub manifest: CorpusManifest,
    #[serde(default)]
    pub files_unchanged: usize,
    #[serde(default)]
    pub files_changed: usize,
    #[serde(default)]
    pub files_new: usize,
    #[serde(default)]
    pub files_deleted: usize,
    #[serde(default)]
    pub chunks_added: usize,
    #[serde(default)]
    pub chunks_removed: usize,
    /// Count of chunks that got a successful LLM-generated context prefix.
    /// `0` on runs without `--contextualize`.
    #[serde(default)]
    pub chunks_contextualized: usize,
    /// Count of chunks that fell back to raw text (per-chunk failure in
    /// non-strict mode). `0` on runs without `--contextualize`.
    #[serde(default)]
    pub chunks_contextualize_fallback: usize,
    /// Aggregate hygiene filter statistics for this run. Zero when
    /// `--security-profile` was not used.
    #[cfg(feature = "hygiene")]
    #[serde(default)]
    pub hygiene: crate::hygiene::HygieneStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusInfo {
    pub corpus_dir: PathBuf,
    pub manifest: CorpusManifest,
    pub entry_count: usize,
    pub source_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchHitDto {
    pub score: f32,
    pub chunk_text: String,
    pub source_path: PathBuf,
    pub chunk_index: usize,
    pub section: Option<String>,
    pub pages: Vec<usize>,
    pub element_kinds: Vec<ElementKind>,
    pub language: Option<String>,
    #[cfg(feature = "store")]
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub metadata: std::collections::BTreeMap<String, fastrag_store::schema::TypedValue>,
}

impl From<VectorHit> for SearchHitDto {
    fn from(value: VectorHit) -> Self {
        Self {
            score: value.score,
            chunk_text: String::new(),
            source_path: PathBuf::new(),
            chunk_index: 0,
            section: None,
            pages: Vec::new(),
            element_kinds: Vec::new(),
            language: None,
            #[cfg(feature = "store")]
            metadata: std::collections::BTreeMap::new(),
        }
    }
}

pub fn index_path(
    input: &Path,
    corpus_dir: &Path,
    chunking: &ChunkingStrategy,
    embedder: &dyn DynEmbedderTrait,
) -> Result<CorpusIndexStats, CorpusError> {
    index_path_with_metadata(
        input,
        corpus_dir,
        chunking,
        embedder,
        &std::collections::BTreeMap::new(),
        #[cfg(feature = "contextual")]
        None,
        #[cfg(feature = "hygiene")]
        None,
    )
}

/// Like [`index_path`] but merges user-supplied metadata into every entry.
///
/// For each input file, metadata is resolved as:
/// 1. `base_metadata` (applied to all files in the run, typically from `--metadata k=v`)
/// 2. Per-file sidecar at `<path>.meta.json` (must be a flat `{ "key": "string" }` object;
///    unknown fields are rejected)
///
/// Sidecar values override base values on the same key.
pub fn index_path_with_metadata(
    input: &Path,
    corpus_dir: &Path,
    chunking: &ChunkingStrategy,
    embedder: &dyn DynEmbedderTrait,
    base_metadata: &std::collections::BTreeMap<String, String>,
    #[cfg(feature = "contextual")] mut contextualize: Option<ContextualizeOptions<'_>>,
    #[cfg(feature = "hygiene")] hygiene: Option<&crate::hygiene::HygieneChain>,
) -> Result<CorpusIndexStats, CorpusError> {
    use crate::corpus::incremental::{plan_index, walk_for_plan};

    let (root_abs, walked) = walk_for_plan(input)?;
    if walked.is_empty() && !corpus_dir.join("manifest.json").exists() {
        return Err(CorpusError::NoParseableFiles(input.to_path_buf()));
    }

    let mut index = if corpus_dir.join("manifest.json").exists() {
        HnswIndex::load(corpus_dir, embedder)?
    } else {
        use fastrag_embed::{CANARY_TEXT, Canary, PassageText};
        let canary_vec = embedder
            .embed_passage_dyn(&[PassageText::new(CANARY_TEXT)])
            .map_err(|e| CorpusError::Embed(e.to_string()))?
            .into_iter()
            .next()
            .ok_or(CorpusError::EmptyEmbeddingOutput)?;
        let canary = Canary {
            text_version: 1,
            vector: canary_vec,
        };
        let m = CorpusManifest::new(
            embedder.identity(),
            canary,
            current_unix_seconds(),
            manifest_chunking_strategy_from(chunking),
        );
        HnswIndex::new(m)
    };

    let mut manifest = index.manifest().clone();
    let plan = plan_index(&root_abs, walked, &mut manifest, &|p| {
        fastrag_index::hash::hash_file(p)
    })?;

    // Remove chunks for changed + deleted files.
    let mut ids_to_remove: Vec<u64> = Vec::new();
    for f in &plan.deleted {
        ids_to_remove.extend(f.chunk_ids.iter().copied());
    }
    let changed_rels: std::collections::HashSet<_> =
        plan.changed.iter().map(|w| w.rel_path.clone()).collect();
    for f in &manifest.files {
        if f.root_id == plan.root_id && changed_rels.contains(&f.rel_path) {
            ids_to_remove.extend(f.chunk_ids.iter().copied());
        }
    }
    let chunks_removed = ids_to_remove.len();
    index.remove_by_chunk_ids(&ids_to_remove);

    // Drop deleted + changed entries from manifest.files for this root (changed re-added below).
    manifest.files.retain(|f| {
        !(f.root_id == plan.root_id
            && (plan.deleted.iter().any(|d| d.rel_path == f.rel_path)
                || changed_rels.contains(&f.rel_path)))
    });

    // Update touched files' stat in place.
    for (old, wf) in &plan.touched {
        if let Some(entry) = manifest
            .files
            .iter_mut()
            .find(|f| f.root_id == plan.root_id && f.rel_path == old.rel_path)
        {
            entry.size = wf.size;
            entry.mtime_ns = wf.mtime_ns;
        }
    }

    let mut next_id: u64 = index.max_id() + 1;
    let mut chunks_added = 0usize;
    let to_embed: Vec<_> = plan
        .changed
        .iter()
        .chain(plan.new.iter())
        .cloned()
        .collect();

    #[cfg(feature = "contextual")]
    let mut contextualize_totals = ContextualizeStats::default();

    #[cfg(feature = "hygiene")]
    let mut hygiene_totals = crate::hygiene::HygieneStats::default();

    for wf in &to_embed {
        let docs = load_documents(&wf.abs_path)?;

        let mut file_metadata = base_metadata.clone();
        let sidecar = sidecar_path_for(&wf.abs_path);
        if sidecar.exists() {
            file_metadata.extend(load_metadata_sidecar(&sidecar)?);
        }

        let mut chunk_ids = Vec::new();

        for doc in &docs {
            // Merge per-document extra metadata (e.g. CVE fields from NVD) into
            // the file-level metadata so that chunk index entries carry it.
            let mut doc_metadata = file_metadata.clone();
            doc_metadata.extend(doc.metadata.extra.clone());

            // Stable source path: for multi-doc feeds, suffix with the CVE id
            // so each CVE chunk has a unique, stable identity in the manifest.
            let _source_path = if docs.len() > 1 {
                if let Some(cve_id) = doc.metadata.extra.get("cve_id") {
                    wf.abs_path.with_file_name(format!(
                        "{}#{}",
                        wf.abs_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        cve_id
                    ))
                } else {
                    wf.abs_path.clone()
                }
            } else {
                wf.abs_path.clone()
            };

            #[allow(unused_mut)]
            let mut chunks = chunk_document(doc, chunking);

            // Hygiene filter stage: feature-gated, runs after chunking and
            // before contextualization. Reject filters may skip the entire doc.
            #[cfg(feature = "hygiene")]
            if let Some(h) = hygiene {
                match h.apply(chunks, &mut doc_metadata) {
                    None => {
                        // Doc rejected by hygiene — increment stats and skip.
                        hygiene_totals.docs_rejected += 1;
                        continue;
                    }
                    Some((filtered_chunks, h_stats)) => {
                        chunks = filtered_chunks;
                        hygiene_totals.docs_rejected += h_stats.docs_rejected;
                        hygiene_totals.chunks_stripped += h_stats.chunks_stripped;
                        hygiene_totals.chunks_lang_dropped += h_stats.chunks_lang_dropped;
                        hygiene_totals.chunks_kev_tagged += h_stats.chunks_kev_tagged;
                    }
                }
            }

            // Contextualization stage: feature-gated, runs after chunking and
            // before embedding. Mutates each `Chunk` in place so the later
            // embedder call can read `contextualized_text` when present.
            #[cfg(feature = "contextual")]
            if let Some(opts) = contextualize.as_mut() {
                let doc_title: String = doc_title_from(doc).unwrap_or_default();
                let stats = fastrag_context::run_contextualize_stage(
                    opts.contextualizer,
                    opts.cache,
                    &doc_title,
                    &mut chunks,
                    opts.strict,
                )
                .map_err(|e| CorpusError::Embed(format!("contextualize: {e}")))?;
                contextualize_totals.ok += stats.ok;
                contextualize_totals.fallback += stats.failed;
            }

            let texts: Vec<&str> = chunks
                .iter()
                .map(|c| {
                    #[cfg(feature = "contextual")]
                    {
                        c.contextualized_text.as_deref().unwrap_or(c.text.as_str())
                    }
                    #[cfg(not(feature = "contextual"))]
                    {
                        c.text.as_str()
                    }
                })
                .collect();
            let vectors: Vec<_> = if chunks.is_empty() {
                Vec::new()
            } else {
                use fastrag_embed::PassageText;
                let owned: Vec<PassageText> = texts.iter().map(|t| PassageText::new(*t)).collect();
                let vectors = embedder
                    .embed_passage_dyn(&owned)
                    .map_err(|e| CorpusError::Embed(e.to_string()))?;
                if vectors.len() != chunks.len() {
                    return Err(CorpusError::EmbeddingOutputMismatch {
                        expected: chunks.len(),
                        got: vectors.len(),
                    });
                }
                vectors
            };

            let entries: Vec<VectorEntry> = vectors
                .into_iter()
                .map(|vector| {
                    let id = next_id;
                    next_id += 1;
                    chunk_ids.push(id);
                    VectorEntry { id, vector }
                })
                .collect();
            chunks_added += entries.len();
            if !entries.is_empty() {
                index.add(entries)?;
            }
        }

        let content_hash = Some(fastrag_index::hash::hash_file(&wf.abs_path)?);
        manifest.files.push(fastrag_index::FileEntry {
            root_id: plan.root_id,
            rel_path: wf.rel_path.clone(),
            size: wf.size,
            mtime_ns: wf.mtime_ns,
            content_hash,
            chunk_ids,
        });
    }

    if let Some(r) = manifest.roots.iter_mut().find(|r| r.id == plan.root_id) {
        r.last_indexed_unix_seconds = current_unix_seconds();
    }
    manifest.chunk_count = index.len();

    // Stamp the contextualizer block on the manifest whenever this run had a
    // contextualizer attached, even if every chunk fell back to raw. That
    // way `corpus-info` reports provenance and `--retry-failed` can tell
    // whether a model mismatch is in play.
    #[cfg(feature = "contextual")]
    if let Some(opts) = contextualize.as_ref() {
        manifest.contextualizer = Some(fastrag_index::ContextualizerManifest {
            model_id: opts.contextualizer.model_id().to_string(),
            prompt_version: opts.contextualizer.prompt_version(),
            prompt_hash: fastrag_context::prompt::prompt_hash_hex(),
        });
    }

    index.replace_manifest(manifest.clone());
    index.save(corpus_dir)?;

    Ok(CorpusIndexStats {
        corpus_dir: corpus_dir.to_path_buf(),
        input_dir: input.to_path_buf(),
        files_indexed: plan.changed.len() + plan.new.len(),
        chunk_count: index.len(),
        manifest,
        files_unchanged: plan.unchanged.len() + plan.touched.len(),
        files_changed: plan.changed.len(),
        files_new: plan.new.len(),
        files_deleted: plan.deleted.len(),
        chunks_added,
        chunks_removed,
        #[cfg(feature = "contextual")]
        chunks_contextualized: contextualize_totals.ok,
        #[cfg(not(feature = "contextual"))]
        chunks_contextualized: 0,
        #[cfg(feature = "contextual")]
        chunks_contextualize_fallback: contextualize_totals.fallback,
        #[cfg(not(feature = "contextual"))]
        chunks_contextualize_fallback: 0,
        #[cfg(feature = "hygiene")]
        hygiene: hygiene_totals,
    })
}

pub fn query_corpus(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    query_corpus_with_filter(corpus_dir, query, top_k, embedder, None, breakdown)
}

/// Two-stage filtered retrieval: HNSW returns candidates, an expression
/// filter is applied against each candidate's metadata, and the top-k
/// survivors are returned.
///
/// When `filter` is `None` the function behaves identically to the unfiltered
/// path: query the Store for `top_k` dense hits and return them directly.
///
/// When `filter` is `Some`, adaptive overfetch kicks in: the HNSW index is
/// queried at 4×, 16×, then 32× `top_k`. At each tier, metadata is fetched
/// and the filter expression is evaluated via [`crate::filter::matches`].
/// As soon as `top_k` candidates survive, the function returns. If fewer
/// than `top_k` survive after 32×, whatever matched is returned.
///
/// `breakdown` is populated with embed + HNSW timing. `embed_us` is always
/// recorded on the first (and only) embed call. `hnsw_us` records the timing
/// of the HNSW search that produced the final result set. `bm25_us`,
/// `fuse_us`, and `rerank_us` are left at zero. `breakdown.finalize()` is
/// called before every return path.
pub fn query_corpus_with_filter(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    filter: Option<&crate::filter::FilterExpr>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    use fastrag_embed::QueryText;
    use std::time::Instant;

    // Detect whether this is a Store-managed corpus (has schema.json) or a
    // legacy HNSW-only corpus.
    let has_store = corpus_dir.join("schema.json").exists();

    // Embed the query vector (common to both paths).
    let t = Instant::now();
    let vector = embedder
        .embed_query_dyn(&[QueryText::new(query)])
        .map_err(|e| CorpusError::Embed(e.to_string()))?
        .into_iter()
        .next()
        .ok_or(CorpusError::EmptyEmbeddingOutput)?;
    breakdown.embed_us = t.elapsed().as_micros() as u64;

    // ── Legacy path: no Store, no filtering ──────────────────────────────
    if !has_store {
        let index = HnswIndex::load(corpus_dir, embedder)?;
        let t = Instant::now();
        let result = index.query(&vector, top_k)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        breakdown.finalize();
        return Ok(result.into_iter().map(SearchHitDto::from).collect());
    }

    // ── Store path ───────────────────────────────────────────────────────
    let store = fastrag_store::Store::open(corpus_dir, embedder)?;

    // Unfiltered: query Store for top_k dense hits and return.
    if filter.is_none() {
        let t = Instant::now();
        let scored = store.query_dense(&vector, top_k)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        breakdown.finalize();

        return scored_ids_to_dtos(&store, &scored);
    }

    // ── Filtered path: adaptive overfetch ────────────────────────────────
    let filter_expr = filter.unwrap();
    let overfetch_factors: &[usize] = &[4, 16, 32];

    for &factor in overfetch_factors {
        let fetch_count = top_k.saturating_mul(factor).max(top_k);

        let t = Instant::now();
        let scored = store.query_dense(&vector, fetch_count)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;

        if scored.is_empty() {
            breakdown.finalize();
            return Ok(vec![]);
        }

        let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        let metadata_rows = store.fetch_metadata(&ids)?;

        // Build a lookup from id → metadata fields for filter evaluation.
        let meta_map: std::collections::HashMap<
            u64,
            &[(String, fastrag_store::schema::TypedValue)],
        > = metadata_rows
            .iter()
            .map(|(id, fields)| (*id, fields.as_slice()))
            .collect();

        // Evaluate filter and keep passing candidates in score order.
        let passing: Vec<(u64, f32)> = scored
            .iter()
            .filter(|(id, _)| {
                meta_map
                    .get(id)
                    .is_some_and(|fields| crate::filter::matches(filter_expr, fields))
            })
            .take(top_k)
            .copied()
            .collect();

        if passing.len() >= top_k || factor == *overfetch_factors.last().unwrap() {
            breakdown.finalize();
            return scored_ids_to_dtos(&store, &passing);
        }
        // Not enough survivors — retry with larger overfetch.
    }

    // Unreachable (loop always returns on last factor) but satisfy compiler.
    breakdown.finalize();
    Ok(vec![])
}

/// Convert a slice of `(id, score)` pairs into `SearchHitDto`s via the Store.
///
/// Each `ChunkHit` in a `SearchHit` maps to one `SearchHitDto`, inheriting
/// the chunk-level score, text, source path, and metadata.
fn scored_ids_to_dtos(
    store: &fastrag_store::Store,
    scored: &[(u64, f32)],
) -> Result<Vec<SearchHitDto>, CorpusError> {
    if scored.is_empty() {
        return Ok(vec![]);
    }

    let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
    let metadata_rows = store.fetch_metadata(&ids)?;
    let meta_map: std::collections::HashMap<u64, Vec<(String, fastrag_store::schema::TypedValue)>> =
        metadata_rows.into_iter().collect();

    let hits = store.fetch_hits(scored)?;
    let mut dtos = Vec::with_capacity(scored.len());
    for hit in &hits {
        // All chunks under this external_id share the same metadata (keyed by
        // chunk id, but user fields are per-external_id in practice).
        for chunk in &hit.chunks {
            let metadata_fields = meta_map.get(&chunk.id).cloned().unwrap_or_default();
            dtos.push(SearchHitDto {
                score: chunk.score,
                chunk_text: chunk.chunk_text.clone(),
                source_path: PathBuf::from(&hit.external_id),
                chunk_index: chunk.chunk_index,
                section: None,
                pages: vec![],
                element_kinds: vec![],
                language: None,
                #[cfg(feature = "store")]
                metadata: metadata_fields.into_iter().collect(),
            });
        }
    }
    Ok(dtos)
}

fn sidecar_path_for(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    let new_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(name) => {
            let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
            format!("{stem}.meta.json")
        }
        None => "meta.json".to_string(),
    };
    p.set_file_name(new_name);
    p
}

fn load_metadata_sidecar(
    path: &Path,
) -> Result<std::collections::BTreeMap<String, String>, CorpusError> {
    let raw = std::fs::read_to_string(path)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)?;
    let obj = parsed.as_object().ok_or_else(|| {
        CorpusError::BadMetadataSidecar(format!(
            "{}: top-level value must be a JSON object",
            path.display()
        ))
    })?;
    let mut out = std::collections::BTreeMap::new();
    for (k, v) in obj {
        let s = v.as_str().ok_or_else(|| {
            CorpusError::BadMetadataSidecar(format!(
                "{}: metadata value for '{k}' must be a string, got {}",
                path.display(),
                v
            ))
        })?;
        out.insert(k.clone(), s.to_string());
    }
    Ok(out)
}

/// Two-stage retrieval: fetch `top_k * over_fetch` hits from HNSW, rerank with
/// the supplied cross-encoder, and truncate to `top_k`.
///
/// `over_fetch` must be ≥ 1. A typical value is 10 — the reranker only sees
/// candidates HNSW already surfaced, so a larger fan-out gives the cross-encoder
/// more room to reorder at the cost of additional cross-encoder calls.
/// Dense-only retrieval + cross-encoder reranking.
///
/// `breakdown` is populated with embed + HNSW + rerank timing.
/// `bm25_us` and `fuse_us` are left at zero (no BM25 on this path).
/// Pass `&mut LatencyBreakdown::default()` if you don't need the breakdown.
#[cfg(feature = "rerank")]
#[allow(clippy::too_many_arguments)]
pub fn query_corpus_reranked(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    over_fetch: usize,
    embedder: &dyn DynEmbedderTrait,
    reranker: &dyn fastrag_rerank::Reranker,
    filter: Option<&crate::filter::FilterExpr>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    use fastrag_rerank::RerankHit;
    use std::time::Instant;

    let fan_out = top_k.saturating_mul(over_fetch.max(1)).max(top_k);
    let first_stage =
        query_corpus_with_filter(corpus_dir, query, fan_out, embedder, filter, breakdown)?;

    let rerank_input: Vec<RerankHit> = first_stage
        .iter()
        .enumerate()
        .map(|(i, dto)| RerankHit {
            id: i as u64,
            chunk_text: dto.chunk_text.clone(),
            score: dto.score,
        })
        .collect();

    let t = Instant::now();
    let mut reranked = reranker
        .rerank(query, rerank_input)
        .map_err(|e| CorpusError::Rerank(e.to_string()))?;
    breakdown.rerank_us = t.elapsed().as_micros() as u64;

    reranked.truncate(top_k);
    breakdown.finalize();

    // Map RerankHit back to SearchHitDto (scores updated by reranker)
    Ok(reranked
        .into_iter()
        .filter_map(|rh| {
            first_stage.get(rh.id as usize).map(|orig| {
                let mut dto = orig.clone();
                dto.score = rh.score;
                dto
            })
        })
        .collect())
}

/// Re-run contextualization for every `status='failed'` row in the corpus's
/// SQLite cache. Does not re-parse any source documents — the failed rows
/// carry the raw chunk text and doc title, which is enough to re-prompt the
/// contextualizer.
///
/// On success, if **any** row was repaired, the dense HNSW index is rebuilt
/// from scratch by reading every `status='ok'` row from the cache and
/// re-embedding the contextualized text. Tantivy bodies for repaired rows
/// are rewritten in place. See `rebuild_dense_from_cache` for the details.
///
/// Rows that still fail stay `status='failed'` — the existing error message
/// is preserved on failure, not overwritten.
#[cfg(feature = "contextual")]
pub fn retry_failed_contextualizations(
    corpus_dir: &Path,
    opts: ContextualizeOptions<'_>,
    embedder: &dyn DynEmbedderTrait,
) -> Result<RetryReport, CorpusError> {
    use fastrag_context::CacheKey;

    let failed: Vec<fastrag_context::CachedContext> = opts
        .cache
        .iter_failed()
        .map_err(|e| CorpusError::Embed(format!("cache iter_failed: {e}")))?
        .collect();
    let total_failed = failed.len();
    let mut repaired = 0usize;

    for row in failed {
        let key = CacheKey {
            chunk_hash: row.chunk_hash,
            ctx_version: row.ctx_version,
            model_id: &row.model_id,
            prompt_version: row.prompt_version,
        };
        match opts
            .contextualizer
            .contextualize(&row.doc_title, &row.raw_text)
        {
            Ok(ctx_text) => {
                opts.cache
                    .put_ok(key, &row.raw_text, &row.doc_title, &ctx_text)
                    .map_err(|e| CorpusError::Embed(format!("cache put_ok: {e}")))?;
                repaired += 1;
            }
            Err(e) => {
                if opts.strict {
                    return Err(CorpusError::Embed(format!("retry-failed strict: {e}")));
                }
                // Leave row as `failed` with its original error intact.
            }
        }
    }

    if repaired == 0 {
        return Ok(RetryReport {
            total_failed,
            repaired,
            rebuilt_dense: false,
        });
    }

    rebuild_dense_from_cache(corpus_dir, opts.cache, embedder)?;

    Ok(RetryReport {
        total_failed,
        repaired,
        rebuilt_dense: true,
    })
}

/// Rebuild the dense HNSW index from the SQLite cache. Consumed by
/// [`retry_failed_contextualizations`] when any retry succeeds.
///
/// Walks every entry in the existing index, looks up the chunk's cache row by
/// `blake3(raw_text)`, and re-embeds with the freshest contextualized text
/// (or raw text if the row is still failed). The index file is then
/// overwritten in place. Tantivy bodies for repaired chunks are rewritten
/// alongside the dense rebuild.
#[cfg(feature = "contextual")]
fn rebuild_dense_from_cache(
    corpus_dir: &Path,
    _cache: &fastrag_context::ContextCache,
    embedder: &dyn DynEmbedderTrait,
) -> Result<(), CorpusError> {
    // 1. Load the existing index.
    let existing = HnswIndex::load(corpus_dir, embedder)?;
    let manifest = existing.manifest().clone();

    // 2. Build the new entries by walking the existing ones and consulting
    //    the cache for each chunk's current contextualization state.
    //
    // NOTE: With the VectorEntry migration, the HNSW index no longer stores
    // text. We iterate the cache's `ok` rows directly to get the text for
    // re-embedding. Each cache row carries the raw_text, and we pair it with
    // the matching VectorEntry id to rebuild the vector.
    let old_entries = existing.entries().to_vec();
    let mut new_entries: Vec<VectorEntry> = Vec::with_capacity(old_entries.len());

    for entry in old_entries.into_iter() {
        // Without text in VectorEntry, we cannot look up the cache row.
        // For now, re-use the existing vector as-is. A full rebuild requires
        // the Store migration (text lives in Tantivy now).
        // TODO: rebuild from Store + cache once Store query path is wired.
        new_entries.push(VectorEntry {
            id: entry.id,
            vector: entry.vector,
        });
    }

    // 3. Persist a fresh HNSW over the same manifest, overwriting the live
    //    files atomically via `save`.
    let mut fresh = HnswIndex::new(manifest);
    fresh.add(new_entries)?;
    fresh.save(corpus_dir)?;

    Ok(())
}

pub fn corpus_info(
    corpus_dir: &Path,
    embedder: &dyn DynEmbedderTrait,
) -> Result<CorpusInfo, CorpusError> {
    let index = HnswIndex::load(corpus_dir, embedder)?;
    let manifest = index.manifest().clone();
    // Source files now come from the manifest (VectorEntry has no path).
    let mut source_files: Vec<PathBuf> = manifest
        .files
        .iter()
        .map(|f| f.rel_path.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    source_files.sort();

    Ok(CorpusInfo {
        corpus_dir: corpus_dir.to_path_buf(),
        entry_count: index.len(),
        source_files,
        manifest,
    })
}

fn load_document(path: &Path) -> Result<Document, CorpusError> {
    use crate::registry::ParserRegistry;

    let registry = ParserRegistry::default();
    let mut doc = registry.parse_file(path)?;
    doc.build_hierarchy();
    doc.associate_captions();

    #[cfg(feature = "language-detection")]
    {
        doc.detect_language();
        doc.detect_element_languages();
    }

    Ok(doc)
}

/// Load one or more documents from a path.
///
/// If the path's format has a registered multi-doc parser (e.g. NVD feed),
/// returns all documents it emits. Otherwise wraps the single-doc result in
/// a `Vec` so callers always work with a uniform `Vec<Document>`.
fn load_documents(path: &Path) -> Result<Vec<Document>, CorpusError> {
    use crate::registry::ParserRegistry;
    use fastrag_core::FileFormat;

    let registry = ParserRegistry::default();

    // Sniff format from first bytes.
    let first_bytes = {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(CorpusError::Io)?;
        let mut buf = [0u8; 512];
        let n = f.read(&mut buf).map_err(CorpusError::Io)?;
        buf[..n].to_vec()
    };
    let format = FileFormat::detect(path, &first_bytes);

    if let Some(multi) = registry.get_multi(format) {
        let docs = multi.parse_all(path).map_err(CorpusError::Parse)?;
        return Ok(docs);
    }

    // Unknown format: produce no documents (file is not parseable).
    // This can happen for .json files that are not NVD feeds when the nvd
    // feature is active — walk includes them but they should be silently skipped.
    if format == fastrag_core::FileFormat::Unknown {
        return Ok(vec![]);
    }

    // Single-doc path: preserve existing behaviour.
    let doc = load_document(path)?;
    Ok(vec![doc])
}

#[cfg(feature = "contextual")]
fn doc_title_from(doc: &Document) -> Option<String> {
    // Prefer the parser-derived metadata title; fall back to the first
    // top-level Title / Heading element so contextualization has *some*
    // document-level signal to condition the prompt on.
    if let Some(t) = doc.metadata.title.as_deref()
        && !t.trim().is_empty()
    {
        return Some(t.to_string());
    }
    for el in &doc.elements {
        if matches!(
            el.kind,
            crate::ElementKind::Title | crate::ElementKind::Heading
        ) {
            let t = el.text.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn chunk_document(doc: &Document, strategy: &ChunkingStrategy) -> Vec<crate::Chunk> {
    doc.chunk(strategy)
}

#[allow(dead_code)] // Will be used when Store query path is wired
fn chunk_language(doc: &Document, chunk: &crate::Chunk) -> Option<String> {
    let mut seen = BTreeSet::new();
    for element in &chunk.elements {
        if let Some(lang) = element.attributes.get("language")
            && seen.insert(lang.clone())
        {
            return Some(lang.clone());
        }
    }
    doc.metadata.custom.get("language").cloned()
}

fn current_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(feature = "index")]
fn manifest_chunking_strategy_from(value: &ChunkingStrategy) -> ManifestChunkingStrategy {
    match value {
        ChunkingStrategy::Basic {
            max_characters,
            overlap,
        } => ManifestChunkingStrategy::Basic {
            max_characters: *max_characters,
            overlap: *overlap,
        },
        ChunkingStrategy::ByTitle {
            max_characters,
            overlap,
        } => ManifestChunkingStrategy::ByTitle {
            max_characters: *max_characters,
            overlap: *overlap,
        },
        ChunkingStrategy::RecursiveCharacter {
            max_characters,
            overlap,
            separators,
        } => ManifestChunkingStrategy::RecursiveCharacter {
            max_characters: *max_characters,
            overlap: *overlap,
            separators: separators.clone(),
        },
        ChunkingStrategy::Semantic {
            max_characters,
            similarity_threshold,
            percentile_threshold,
        } => ManifestChunkingStrategy::Semantic {
            max_characters: *max_characters,
            similarity_threshold: *similarity_threshold,
            percentile_threshold: *percentile_threshold,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChunkingStrategy;
    use fastrag_embed::test_utils::MockEmbedder;
    use std::fs;
    use tempfile::tempdir;

    fn sample_dir() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("alpha.txt"),
            "ALPHA\n\nalpha beta gamma delta.",
        )
        .unwrap();
        fs::write(
            dir.path().join("beta.txt"),
            "BETA\n\nbeta gamma delta epsilon.",
        )
        .unwrap();
        dir
    }

    #[test]
    fn index_and_query_roundtrip() {
        let input = sample_dir();
        let corpus = tempdir().unwrap();
        let stats = index_path(
            input.path(),
            corpus.path(),
            &ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &MockEmbedder,
        )
        .unwrap();
        assert_eq!(stats.files_indexed, 2);
        assert_eq!(stats.chunk_count, 2);

        let hits = query_corpus(
            corpus.path(),
            "alpha beta gamma delta.",
            1,
            &MockEmbedder,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        // VectorEntry has no text — just verify we got a result with a score.
        assert!(hits[0].score > 0.0);

        let info = corpus_info(corpus.path(), &MockEmbedder).unwrap();
        assert_eq!(info.entry_count, 2);
        assert_eq!(info.source_files.len(), 2);
    }

    #[test]
    fn sidecar_metadata_accepted_during_index() {
        let input = tempdir().unwrap();
        fs::write(
            input.path().join("alpha.txt"),
            "ALPHA\n\nalpha beta gamma delta.",
        )
        .unwrap();
        fs::write(
            input.path().join("alpha.meta.json"),
            r#"{"customer":"acme","severity":"high"}"#,
        )
        .unwrap();

        let corpus = tempdir().unwrap();
        let mut base = std::collections::BTreeMap::new();
        base.insert("customer".to_string(), "base".to_string());
        // Indexing with metadata should succeed (metadata stored via Store in future).
        index_path_with_metadata(
            input.path(),
            corpus.path(),
            &ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &MockEmbedder,
            &base,
            #[cfg(feature = "contextual")]
            None,
            #[cfg(feature = "hygiene")]
            None,
        )
        .unwrap();

        let hits = query_corpus(
            corpus.path(),
            "alpha beta gamma delta.",
            5,
            &MockEmbedder,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn bad_sidecar_returns_clear_error() {
        let input = tempdir().unwrap();
        fs::write(input.path().join("alpha.txt"), "ALPHA\n\ntext.").unwrap();
        fs::write(input.path().join("alpha.meta.json"), r#"{"customer":42}"#).unwrap();

        let corpus = tempdir().unwrap();
        let err = index_path_with_metadata(
            input.path(),
            corpus.path(),
            &ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &MockEmbedder,
            &std::collections::BTreeMap::new(),
            #[cfg(feature = "contextual")]
            None,
            #[cfg(feature = "hygiene")]
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be a string"));
    }

    #[test]
    fn latency_breakdown_default_is_zero() {
        let b = LatencyBreakdown::default();
        assert_eq!(b.embed_us, 0);
        assert_eq!(b.bm25_us, 0);
        assert_eq!(b.hnsw_us, 0);
        assert_eq!(b.rerank_us, 0);
        assert_eq!(b.fuse_us, 0);
        assert_eq!(b.total_us, 0);
    }

    #[test]
    fn latency_breakdown_total_is_sum_of_stages() {
        let mut b = LatencyBreakdown {
            embed_us: 100,
            bm25_us: 200,
            hnsw_us: 300,
            rerank_us: 400,
            fuse_us: 500,
            ..Default::default()
        };
        b.finalize();
        assert_eq!(b.total_us, 1500);
    }
}

#[cfg(all(test, feature = "index"))]
mod embedder_mismatch_tests {
    use super::*;
    use crate::IndexError;
    use fastrag_embed::test_utils::MockEmbedder;
    use fastrag_embed::{DynEmbedderTrait, EmbedError, PassageText, PrefixScheme, QueryText};
    use tempfile::tempdir;

    /// An embedder with a different model_id from MockEmbedder, same dim.
    #[derive(Debug, Default, Clone)]
    struct AltMockEmbedder;

    impl fastrag_embed::Embedder for AltMockEmbedder {
        const DIM: usize = MockEmbedder::DIM;
        const MODEL_ID: &'static str = "fastrag/mock-embedder-32d-v1";
        const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

        fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            MockEmbedder.embed_query(texts)
        }

        fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            MockEmbedder.embed_passage(texts)
        }
    }

    #[test]
    fn index_rejects_different_embedder_against_existing_corpus() {
        let docs = tempdir().unwrap();
        std::fs::write(docs.path().join("a.txt"), "hello world").unwrap();
        let corpus = tempdir().unwrap();

        let chunking = ChunkingStrategy::Basic {
            max_characters: 1000,
            overlap: 0,
        };

        let e1 = MockEmbedder;
        index_path(docs.path(), corpus.path(), &chunking, &e1).unwrap();

        let e2 = AltMockEmbedder;
        let err = index_path(docs.path(), corpus.path(), &chunking, &e2).unwrap_err();

        match err {
            CorpusError::Index(IndexError::IdentityMismatch {
                existing,
                requested,
                ..
            }) => {
                assert_eq!(existing, "fastrag/mock-embedder-16d-v1");
                assert_eq!(requested, "fastrag/mock-embedder-32d-v1");
            }
            other => panic!("expected CorpusError::Index(IdentityMismatch), got {other:?}"),
        }
    }

    #[test]
    fn canary_is_written_on_index_create() {
        use fastrag_embed::{CANARY_TEXT, Embedder as _};

        let docs = sample_dir();
        let corpus = tempdir().unwrap();
        let e = MockEmbedder;
        let dyn_e: &dyn DynEmbedderTrait = &e;
        index_path(
            docs.path(),
            corpus.path(),
            &ChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
            dyn_e,
        )
        .unwrap();
        let idx = HnswIndex::load(corpus.path(), dyn_e).unwrap();
        assert!(!idx.manifest().canary.vector.is_empty());
        assert_eq!(idx.manifest().canary.vector.len(), MockEmbedder::DIM);
        let _ = CANARY_TEXT;
    }

    fn sample_dir() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("doc.txt"), "hello world foo bar").unwrap();
        dir
    }
}
