use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ChunkingStrategy, Document, ElementKind, FastRagError, HnswIndex, IndexEntry, SearchHit,
    VectorIndex,
};

#[cfg(feature = "index")]
use crate::{CorpusManifest, ManifestChunkingStrategy};

use fastrag_embed::DynEmbedderTrait;

#[cfg(feature = "hybrid")]
pub mod hybrid;
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
    #[cfg(feature = "hybrid")]
    #[error("tantivy error: {0}")]
    Tantivy(#[from] fastrag_tantivy::TantivyIndexError),
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
}

impl From<SearchHit> for SearchHitDto {
    fn from(value: SearchHit) -> Self {
        Self {
            score: value.score,
            chunk_text: value.entry.chunk_text,
            source_path: value.entry.source_path,
            chunk_index: value.entry.chunk_index,
            section: value.entry.section,
            pages: value.entry.pages,
            element_kinds: value.entry.element_kinds,
            language: value.entry.language,
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

    #[cfg(feature = "hybrid")]
    let tantivy = if fastrag_tantivy::TantivyIndex::exists(corpus_dir) {
        Some(fastrag_tantivy::TantivyIndex::open(corpus_dir)?)
    } else {
        Some(fastrag_tantivy::TantivyIndex::create(corpus_dir)?)
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
    #[cfg(feature = "hybrid")]
    if let Some(ref tantivy) = tantivy
        && !ids_to_remove.is_empty()
    {
        tantivy.delete_by_ids(&ids_to_remove)?;
    }

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

    let mut next_id: u64 = index.entries().iter().map(|e| e.id).max().unwrap_or(0) + 1;
    let mut chunks_added = 0usize;
    let to_embed: Vec<_> = plan
        .changed
        .iter()
        .chain(plan.new.iter())
        .cloned()
        .collect();

    #[cfg(feature = "contextual")]
    let mut contextualize_totals = ContextualizeStats::default();

    for wf in &to_embed {
        let doc = load_document(&wf.abs_path)?;
        #[allow(unused_mut)]
        let mut chunks = chunk_document(&doc, chunking);

        // Contextualization stage: feature-gated, runs after chunking and
        // before embedding. Mutates each `Chunk` in place so the later
        // embedder call can read `contextualized_text` when present.
        #[cfg(feature = "contextual")]
        if let Some(opts) = contextualize.as_mut() {
            let doc_title: String = doc_title_from(&doc).unwrap_or_default();
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

        let mut file_metadata = base_metadata.clone();
        let sidecar = sidecar_path_for(&wf.abs_path);
        if sidecar.exists() {
            file_metadata.extend(load_metadata_sidecar(&sidecar)?);
        }

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        let entries: Vec<IndexEntry> = chunks
            .into_iter()
            .zip(vectors.into_iter())
            .map(|(chunk, vector)| {
                let id = next_id;
                next_id += 1;
                chunk_ids.push(id);
                // When contextualization produced a prefixed form, the
                // indexed body is that prefixed text and the display/raw
                // text is preserved separately. Otherwise chunk_text holds
                // raw text directly and display_text is None.
                #[allow(unused_mut)]
                let mut display_text: Option<String> = None;
                #[allow(unused_mut)]
                let mut chunk_text: String = chunk.text.clone();
                #[cfg(feature = "contextual")]
                if let Some(ctx) = chunk.contextualized_text.as_ref() {
                    display_text = Some(chunk.text.clone());
                    chunk_text = ctx.clone();
                }
                IndexEntry {
                    id,
                    vector,
                    chunk_text,
                    source_path: wf.abs_path.clone(),
                    chunk_index: chunk.index,
                    section: chunk.section.clone(),
                    element_kinds: chunk.elements.iter().map(|e| e.kind.clone()).collect(),
                    pages: chunk
                        .elements
                        .iter()
                        .filter_map(|e| e.page)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    language: chunk_language(&doc, &chunk),
                    metadata: file_metadata.clone(),
                    display_text,
                }
            })
            .collect();
        chunks_added += entries.len();
        if !entries.is_empty() {
            #[cfg(feature = "hybrid")]
            if let Some(ref tantivy) = tantivy {
                tantivy.add_entries(&entries)?;
            }
            index.add(entries)?;
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
    manifest.chunk_count = index.entries().len();

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
        chunk_count: index.entries().len(),
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
    })
}

pub fn query_corpus(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
    query_corpus_with_filter(
        corpus_dir,
        query,
        top_k,
        embedder,
        &std::collections::BTreeMap::new(),
        breakdown,
    )
}

/// Two-stage filtered retrieval: HNSW returns `top_k * over_fetch` candidates,
/// an equality filter is applied, and the top-k survivors are returned.
///
/// The over-fetch factor starts at 4×; if the filter eliminates every hit, the
/// search is retried once with 16× over-fetch before giving up and returning
/// an empty vec. An empty filter short-circuits the retry loop.
/// Two-stage filtered retrieval: HNSW returns `top_k * over_fetch` candidates,
/// an equality filter is applied, and the top-k survivors are returned.
///
/// The over-fetch factor starts at 4×; if the filter eliminates every hit, the
/// search is retried once with 16× over-fetch before giving up and returning
/// an empty vec. An empty filter short-circuits the retry loop.
///
/// `breakdown` is populated with embed + HNSW timing. `embed_us` is always
/// recorded on the first (and only) embed call. `hnsw_us` records the timing
/// of the HNSW call that ultimately returns results (first non-empty iteration,
/// or the final iteration). `bm25_us`, `fuse_us`, and `rerank_us` are left at
/// zero. Call `breakdown.finalize()` is called before every return path.
pub fn query_corpus_with_filter(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    filter: &std::collections::BTreeMap<String, String>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
    use fastrag_embed::QueryText;
    use std::time::Instant;

    let index = HnswIndex::load(corpus_dir, embedder)?;

    let t = Instant::now();
    let vector = embedder
        .embed_query_dyn(&[QueryText::new(query)])
        .map_err(|e| CorpusError::Embed(e.to_string()))?
        .into_iter()
        .next()
        .ok_or(CorpusError::EmptyEmbeddingOutput)?;
    breakdown.embed_us = t.elapsed().as_micros() as u64;

    if filter.is_empty() {
        let t = Instant::now();
        let result = index.query(&vector, top_k)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        breakdown.finalize();
        return Ok(result);
    }

    // Adaptive over-fetch + post-filter. embed_us is set once above.
    // hnsw_us records the iteration that ultimately returns results
    // (first non-empty), or the final retry if all iterations are empty.
    for factor in [4usize, 16] {
        let fan_out = top_k.saturating_mul(factor).max(top_k);
        let t = Instant::now();
        let hits = index.query(&vector, fan_out)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        let filtered: Vec<SearchHit> = hits
            .into_iter()
            .filter(|h| h.entry.matches_filter(filter))
            .take(top_k)
            .collect();
        if !filtered.is_empty() {
            breakdown.finalize();
            return Ok(filtered);
        }
    }
    breakdown.finalize();
    Ok(Vec::new())
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
    filter: &std::collections::BTreeMap<String, String>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
    use std::time::Instant;

    let fan_out = top_k.saturating_mul(over_fetch.max(1)).max(top_k);
    let first_stage =
        query_corpus_with_filter(corpus_dir, query, fan_out, embedder, filter, breakdown)?;

    let t = Instant::now();
    let mut reranked = reranker
        .rerank(query, first_stage)
        .map_err(|e| CorpusError::Rerank(e.to_string()))?;
    breakdown.rerank_us = t.elapsed().as_micros() as u64;

    reranked.truncate(top_k);
    // Re-finalize: query_corpus_with_filter already called finalize() which set
    // total_us without rerank_us. We overwrite here to include it.
    breakdown.finalize();
    Ok(reranked)
}

/// Hybrid BM25 + dense retrieval with RRF fusion and post-filtering.
///
/// `breakdown` is populated with per-stage microsecond timings. Pass
/// `&mut LatencyBreakdown::default()` if you don't need the breakdown.
#[cfg(feature = "hybrid")]
pub fn query_corpus_hybrid(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn DynEmbedderTrait,
    filter: &std::collections::BTreeMap<String, String>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
    use fastrag_embed::QueryText;
    use std::time::Instant;

    let hybrid_index = hybrid::HybridIndex::load(corpus_dir, embedder)?;

    let t = Instant::now();
    let vector = embedder
        .embed_query_dyn(&[QueryText::new(query)])
        .map_err(|e| CorpusError::Embed(e.to_string()))?
        .into_iter()
        .next()
        .ok_or(CorpusError::EmptyEmbeddingOutput)?;
    breakdown.embed_us = t.elapsed().as_micros() as u64;

    if filter.is_empty() {
        let result = hybrid_index.query_hybrid_timed(query, &vector, top_k, breakdown)?;
        breakdown.finalize();
        return Ok(result);
    }

    // Adaptive over-fetch + post-filter (same strategy as dense-only path).
    // Only the first iteration populates the breakdown; subsequent retries
    // overwrite it — acceptable because the eval harness exercises the
    // no-filter path and filter retries are rare in practice.
    for factor in [4usize, 16] {
        let fan_out = top_k.saturating_mul(factor).max(top_k);
        let hits = hybrid_index.query_hybrid_timed(query, &vector, fan_out, breakdown)?;
        let filtered: Vec<SearchHit> = hits
            .into_iter()
            .filter(|h| h.entry.matches_filter(filter))
            .take(top_k)
            .collect();
        if !filtered.is_empty() {
            breakdown.finalize();
            return Ok(filtered);
        }
    }
    breakdown.finalize();
    Ok(Vec::new())
}

/// Hybrid retrieval + cross-encoder reranking.
///
/// `breakdown` is populated with per-stage microsecond timings including
/// `rerank_us`. Pass `&mut LatencyBreakdown::default()` if you don't need it.
#[cfg(all(feature = "hybrid", feature = "rerank"))]
#[allow(clippy::too_many_arguments)]
pub fn query_corpus_hybrid_reranked(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    over_fetch: usize,
    embedder: &dyn DynEmbedderTrait,
    reranker: &dyn fastrag_rerank::Reranker,
    filter: &std::collections::BTreeMap<String, String>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHit>, CorpusError> {
    use std::time::Instant;

    let fan_out = top_k.saturating_mul(over_fetch.max(1)).max(top_k);
    let first_stage = query_corpus_hybrid(corpus_dir, query, fan_out, embedder, filter, breakdown)?;

    let t = Instant::now();
    let mut reranked = reranker
        .rerank(query, first_stage)
        .map_err(|e| CorpusError::Rerank(e.to_string()))?;
    breakdown.rerank_us = t.elapsed().as_micros() as u64;

    reranked.truncate(top_k);
    breakdown.finalize();
    Ok(reranked)
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
    cache: &fastrag_context::ContextCache,
    embedder: &dyn DynEmbedderTrait,
) -> Result<(), CorpusError> {
    use fastrag_context::{CTX_VERSION, CacheKey, CacheStatus};
    use fastrag_embed::PassageText;

    // 1. Load the existing index. We need its entries (for ids + metadata)
    //    and its manifest (to construct the fresh HNSW).
    let existing = HnswIndex::load(corpus_dir, embedder)?;
    let manifest = existing.manifest().clone();
    let contextualizer_meta = manifest.contextualizer.as_ref().ok_or_else(|| {
        CorpusError::Embed(
            "rebuild_dense_from_cache called on a corpus with no contextualizer manifest entry"
                .to_string(),
        )
    })?;
    let model_id = contextualizer_meta.model_id.clone();
    let prompt_version = contextualizer_meta.prompt_version;

    // 2. Build the new entries by walking the existing ones and consulting
    //    the cache for each chunk's current contextualization state.
    let old_entries = existing.entries().to_vec();
    let mut new_entries: Vec<IndexEntry> = Vec::with_capacity(old_entries.len());
    for entry in old_entries.into_iter() {
        // Raw text source: display_text (when contextualization had been
        // applied) or chunk_text (when the chunk fell back to raw — in which
        // case display_text is None).
        let raw_text = entry
            .display_text
            .clone()
            .unwrap_or_else(|| entry.chunk_text.clone());
        let chunk_hash: [u8; 32] = *blake3::hash(raw_text.as_bytes()).as_bytes();
        let key = CacheKey {
            chunk_hash,
            ctx_version: CTX_VERSION,
            model_id: &model_id,
            prompt_version,
        };
        let cache_row = cache
            .get(key)
            .map_err(|e| CorpusError::Embed(format!("cache.get during rebuild: {e}")))?;
        let (new_chunk_text, new_display_text) = match cache_row {
            Some(row) if row.status == CacheStatus::Ok => match row.context_text {
                Some(ctx) => (format!("{ctx}\n\n{raw_text}"), Some(raw_text.clone())),
                None => (raw_text.clone(), None),
            },
            _ => (raw_text.clone(), None),
        };

        let vector = embedder
            .embed_passage_dyn(&[PassageText::new(&new_chunk_text)])
            .map_err(|e| CorpusError::Embed(format!("embed during rebuild: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| CorpusError::Embed("embedder returned no vector".to_string()))?;

        new_entries.push(IndexEntry {
            id: entry.id,
            vector,
            chunk_text: new_chunk_text,
            source_path: entry.source_path,
            chunk_index: entry.chunk_index,
            section: entry.section,
            element_kinds: entry.element_kinds,
            pages: entry.pages,
            language: entry.language,
            metadata: entry.metadata,
            display_text: new_display_text,
        });
    }

    // 3. Persist a fresh HNSW over the same manifest, overwriting the live
    //    files atomically via `save`.
    let mut fresh = HnswIndex::new(manifest);
    fresh.add(new_entries.clone())?;
    fresh.save(corpus_dir)?;

    // 4. Rewrite Tantivy bodies for the same chunks. Delete-by-id then
    //    re-add with the new chunk_text. Skipped silently if the corpus has
    //    no Tantivy index (compiled without the `hybrid` feature, or the
    //    sidecar dir is missing).
    #[cfg(feature = "hybrid")]
    {
        if fastrag_tantivy::TantivyIndex::exists(corpus_dir) {
            let tantivy = fastrag_tantivy::TantivyIndex::open(corpus_dir)
                .map_err(|e| CorpusError::Embed(format!("tantivy open during rebuild: {e}")))?;
            let ids: Vec<u64> = new_entries.iter().map(|e| e.id).collect();
            tantivy
                .delete_by_ids(&ids)
                .map_err(|e| CorpusError::Embed(format!("tantivy delete during rebuild: {e}")))?;
            tantivy
                .add_entries(&new_entries)
                .map_err(|e| CorpusError::Embed(format!("tantivy add during rebuild: {e}")))?;
        }
    }

    Ok(())
}

pub fn corpus_info(
    corpus_dir: &Path,
    embedder: &dyn DynEmbedderTrait,
) -> Result<CorpusInfo, CorpusError> {
    let index = HnswIndex::load(corpus_dir, embedder)?;
    let mut source_files = index
        .entries()
        .iter()
        .map(|entry| entry.source_path.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    source_files.sort();

    Ok(CorpusInfo {
        corpus_dir: corpus_dir.to_path_buf(),
        manifest: index.manifest().clone(),
        entry_count: index.len(),
        source_files,
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
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "alpha.txt");

        let info = corpus_info(corpus.path(), &MockEmbedder).unwrap();
        assert_eq!(info.entry_count, 2);
        assert_eq!(info.source_files.len(), 2);
    }

    #[test]
    fn sidecar_overrides_base_metadata_and_filters_queries() {
        let input = tempdir().unwrap();
        fs::write(
            input.path().join("alpha.txt"),
            "ALPHA\n\nalpha beta gamma delta.",
        )
        .unwrap();
        fs::write(
            input.path().join("beta.txt"),
            "BETA\n\nbeta gamma delta epsilon.",
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
        )
        .unwrap();

        let unfiltered = query_corpus(
            corpus.path(),
            "beta gamma delta",
            5,
            &MockEmbedder,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert_eq!(unfiltered.len(), 2);

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "acme".to_string());
        let hits = query_corpus_with_filter(
            corpus.path(),
            "beta gamma delta",
            5,
            &MockEmbedder,
            &f,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "alpha.txt");
        assert_eq!(
            hits[0].entry.metadata.get("severity").map(String::as_str),
            Some("high")
        );

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "base".to_string());
        let hits = query_corpus_with_filter(
            corpus.path(),
            "beta gamma delta",
            5,
            &MockEmbedder,
            &f,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "beta.txt");

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "nobody".to_string());
        let hits = query_corpus_with_filter(
            corpus.path(),
            "beta gamma delta",
            5,
            &MockEmbedder,
            &f,
            &mut LatencyBreakdown::default(),
        )
        .unwrap();
        assert!(hits.is_empty());
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
