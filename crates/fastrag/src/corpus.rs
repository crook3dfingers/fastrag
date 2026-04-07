use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ops::collect_files;
use crate::{
    ChunkingStrategy, Document, ElementKind, FastRagError, HnswIndex, IndexEntry, SearchHit,
    VectorIndex,
};

#[cfg(feature = "embedding")]
use crate::Embedder;
#[cfg(feature = "index")]
use crate::{CorpusManifest, ManifestChunkingStrategy};

#[derive(Debug, Error)]
pub enum CorpusError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] FastRagError),
    #[cfg(feature = "embedding")]
    #[error("embedding error: {0}")]
    Embed(#[from] crate::EmbedderError),
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusIndexStats {
    pub corpus_dir: PathBuf,
    pub input_dir: PathBuf,
    pub files_indexed: usize,
    pub chunk_count: usize,
    pub manifest: CorpusManifest,
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
    embedder: &dyn Embedder,
) -> Result<CorpusIndexStats, CorpusError> {
    index_path_with_metadata(
        input,
        corpus_dir,
        chunking,
        embedder,
        &std::collections::BTreeMap::new(),
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
    embedder: &dyn Embedder,
    base_metadata: &std::collections::BTreeMap<String, String>,
) -> Result<CorpusIndexStats, CorpusError> {
    let mut files = if input.is_file() {
        vec![input.to_path_buf()]
    } else {
        collect_files(input)
    };
    files.sort();
    if files.is_empty() {
        return Err(CorpusError::NoParseableFiles(input.to_path_buf()));
    }

    let mut manifest = CorpusManifest::new(
        embedder.model_id().to_string(),
        embedder.dim(),
        current_unix_seconds(),
        manifest_chunking_strategy_from(chunking),
    );
    let mut index = HnswIndex::new(embedder.dim(), manifest.clone());
    index.set_manifest_model_id(embedder.model_id());

    let mut next_id: u64 = 1;
    let mut total_chunks = 0usize;

    for path in &files {
        let doc = load_document(path)?;
        let chunks = chunk_document(&doc, chunking);
        total_chunks += chunks.len();

        // Merge base metadata + per-file sidecar (sidecar wins on conflict).
        let mut file_metadata = base_metadata.clone();
        let sidecar_path = sidecar_path_for(path);
        if sidecar_path.exists() {
            let sidecar = load_metadata_sidecar(&sidecar_path)?;
            file_metadata.extend(sidecar);
        }

        let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
        let vectors = embedder.embed(&texts)?;
        if vectors.len() != chunks.len() {
            return Err(CorpusError::EmbeddingOutputMismatch {
                expected: chunks.len(),
                got: vectors.len(),
            });
        }

        let entries = chunks
            .into_iter()
            .zip(vectors.into_iter())
            .map(|(chunk, vector)| IndexEntry {
                id: next_id,
                vector,
                chunk_text: chunk.text.clone(),
                source_path: path.to_path_buf(),
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
            })
            .collect::<Vec<_>>();

        next_id += entries.len() as u64;
        index.add(entries)?;
    }

    manifest.chunk_count = total_chunks;
    index.save(corpus_dir)?;

    Ok(CorpusIndexStats {
        corpus_dir: corpus_dir.to_path_buf(),
        input_dir: input.to_path_buf(),
        files_indexed: files.len(),
        chunk_count: total_chunks,
        manifest,
    })
}

pub fn query_corpus(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn Embedder,
) -> Result<Vec<SearchHit>, CorpusError> {
    query_corpus_with_filter(
        corpus_dir,
        query,
        top_k,
        embedder,
        &std::collections::BTreeMap::new(),
    )
}

/// Two-stage filtered retrieval: HNSW returns `top_k * over_fetch` candidates,
/// an equality filter is applied, and the top-k survivors are returned.
///
/// The over-fetch factor starts at 4×; if the filter eliminates every hit, the
/// search is retried once with 16× over-fetch before giving up and returning
/// an empty vec. An empty filter short-circuits the retry loop.
pub fn query_corpus_with_filter(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    embedder: &dyn Embedder,
    filter: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<SearchHit>, CorpusError> {
    let index = HnswIndex::load(corpus_dir)?;
    let mut vectors = embedder.embed(&[query])?;
    let vector = vectors.pop().ok_or(CorpusError::EmptyEmbeddingOutput)?;

    if filter.is_empty() {
        return Ok(index.query(&vector, top_k)?);
    }

    for factor in [4usize, 16] {
        let fan_out = top_k.saturating_mul(factor).max(top_k);
        let hits = index.query(&vector, fan_out)?;
        let filtered: Vec<SearchHit> = hits
            .into_iter()
            .filter(|h| h.entry.matches_filter(filter))
            .take(top_k)
            .collect();
        if !filtered.is_empty() {
            return Ok(filtered);
        }
    }
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
#[cfg(feature = "rerank")]
pub fn query_corpus_reranked(
    corpus_dir: &Path,
    query: &str,
    top_k: usize,
    over_fetch: usize,
    embedder: &dyn Embedder,
    reranker: &dyn fastrag_rerank::Reranker,
) -> Result<Vec<SearchHit>, CorpusError> {
    let fan_out = top_k.saturating_mul(over_fetch.max(1)).max(top_k);
    let first_stage = query_corpus(corpus_dir, query, fan_out, embedder)?;
    let mut reranked = reranker
        .rerank(query, first_stage)
        .map_err(|e| CorpusError::Rerank(e.to_string()))?;
    reranked.truncate(top_k);
    Ok(reranked)
}

pub fn corpus_info(corpus_dir: &Path) -> Result<CorpusInfo, CorpusError> {
    let index = HnswIndex::load(corpus_dir)?;
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

        let hits =
            query_corpus(corpus.path(), "alpha beta gamma delta.", 1, &MockEmbedder).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "alpha.txt");

        let info = corpus_info(corpus.path()).unwrap();
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
        )
        .unwrap();

        let unfiltered = query_corpus(corpus.path(), "beta gamma delta", 5, &MockEmbedder).unwrap();
        assert_eq!(unfiltered.len(), 2);

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "acme".to_string());
        let hits =
            query_corpus_with_filter(corpus.path(), "beta gamma delta", 5, &MockEmbedder, &f)
                .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "alpha.txt");
        assert_eq!(
            hits[0].entry.metadata.get("severity").map(String::as_str),
            Some("high")
        );

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "base".to_string());
        let hits =
            query_corpus_with_filter(corpus.path(), "beta gamma delta", 5, &MockEmbedder, &f)
                .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.source_path.file_name().unwrap(), "beta.txt");

        let mut f = std::collections::BTreeMap::new();
        f.insert("customer".to_string(), "nobody".to_string());
        let hits =
            query_corpus_with_filter(corpus.path(), "beta gamma delta", 5, &MockEmbedder, &f)
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
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be a string"));
    }
}
