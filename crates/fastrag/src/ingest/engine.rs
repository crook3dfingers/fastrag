use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use fastrag_core::{ChunkingStrategy, Document, Element, ElementKind, FileFormat, Metadata};
use fastrag_embed::{CANARY_TEXT, Canary, DynEmbedderTrait, PassageText};
use fastrag_index::{CorpusManifest, ManifestChunkingStrategy};
use fastrag_store::ChunkRecord;
use fastrag_store::schema::DynamicSchema;

use crate::ingest::jsonl::{JsonlIngestConfig, parse_jsonl};

// ── public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct JsonlIndexStats {
    pub records_total: usize,
    pub records_skipped: usize,
    pub records_upserted: usize,
    pub records_new: usize,
    pub chunks_created: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("store error: {0}")]
    Store(#[from] fastrag_store::error::StoreError),
    #[error("index error: {0}")]
    Index(#[from] fastrag_index::IndexError),
    #[error("jsonl error: {0}")]
    Jsonl(#[from] crate::ingest::jsonl::JsonlError),
    #[error("embed error: {0}")]
    Embed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// ── main function ─────────────────────────────────────────────────────────────

pub fn index_jsonl(
    input: &Path,
    corpus_dir: &Path,
    chunking: &ChunkingStrategy,
    embedder: &dyn DynEmbedderTrait,
    config: &JsonlIngestConfig,
) -> Result<JsonlIndexStats, IngestError> {
    // 1. Parse JSONL
    let file = std::fs::File::open(input)?;
    let (records, field_defs) = parse_jsonl(file, config)?;

    // 2. Open or create Store
    let schema_path = corpus_dir.join("schema.json");
    let mut store = if schema_path.exists() {
        fastrag_store::Store::open(corpus_dir, embedder)?
        // Note: evolve_schema is intentionally skipped on re-open. Tantivy's
        // schema is fixed at creation time; new columns require a full rebuild.
    } else {
        let canary_vec = embedder
            .embed_passage_dyn(&[PassageText::new(CANARY_TEXT)])
            .map_err(|e| IngestError::Embed(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| IngestError::Embed("empty canary output".into()))?;

        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut manifest = CorpusManifest::new(
            embedder.identity(),
            Canary {
                text_version: 1,
                vector: canary_vec,
            },
            now_secs,
            chunking_to_manifest(chunking),
        );
        if let Some(field) = config.cwe_field.clone() {
            manifest.cwe_field = Some(field);
            manifest.cwe_taxonomy_version =
                Some(fastrag_cwe::data::embedded().version().to_string());
        }

        // Seed the schema with all fields seen in this batch so Tantivy
        // creates the necessary columns at index-creation time.
        let mut initial_schema = DynamicSchema::new();
        for fd in &field_defs {
            initial_schema.merge(fd.clone())?;
        }

        fastrag_store::Store::create(corpus_dir, manifest, initial_schema)?
    };

    // 4. Per-record upsert logic
    let mut stats = JsonlIndexStats {
        records_total: records.len(),
        records_skipped: 0,
        records_upserted: 0,
        records_new: 0,
        chunks_created: 0,
    };

    // Monotonic ID seed: nanos since epoch, incremented per chunk
    let mut id_counter = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let source_path = input.to_string_lossy().into_owned();

    for record in &records {
        let existing_hash = store.content_hash_for(&record.external_id)?;
        match existing_hash.as_deref() {
            Some(h) if h == record.content_hash => {
                stats.records_skipped += 1;
                continue;
            }
            Some(_) => {
                store.delete_by_external_id(&record.external_id)?;
                stats.records_upserted += 1;
            }
            None => {
                stats.records_new += 1;
            }
        }

        // Build a Document with a single Paragraph element and chunk it
        let doc = Document {
            metadata: Metadata::new(FileFormat::Text),
            elements: vec![Element::new(ElementKind::Paragraph, &record.text)],
        };
        let chunks = doc.chunk(chunking);

        if chunks.is_empty() {
            continue;
        }

        // Embed all chunks in a batch
        let passages: Vec<PassageText> = chunks.iter().map(|c| PassageText::new(&c.text)).collect();

        let vectors = embedder
            .embed_passage_dyn(&passages)
            .map_err(|e| IngestError::Embed(e.to_string()))?;

        let chunk_records: Vec<ChunkRecord> = chunks
            .iter()
            .zip(vectors.into_iter())
            .enumerate()
            .map(|(i, (chunk, vector))| {
                let id = id_counter;
                id_counter = id_counter.wrapping_add(1);
                ChunkRecord {
                    id,
                    external_id: record.external_id.clone(),
                    content_hash: record.content_hash.clone(),
                    chunk_index: i,
                    source_path: source_path.clone(),
                    source_json: Some(record.source_json.clone()),
                    chunk_text: chunk.text.clone(),
                    vector,
                    user_fields: record.metadata.clone(),
                }
            })
            .collect();

        stats.chunks_created += chunk_records.len();
        store.add_records(chunk_records)?;
    }

    // 5. Save HNSW index
    store.save()?;

    Ok(stats)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn chunking_to_manifest(strategy: &ChunkingStrategy) -> ManifestChunkingStrategy {
    match strategy {
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

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_embed::test_utils::MockEmbedder;
    use std::collections::BTreeMap;
    use std::io::Write;

    fn config() -> JsonlIngestConfig {
        JsonlIngestConfig {
            text_fields: vec!["title".into(), "description".into()],
            id_field: "id".into(),
            metadata_fields: vec!["severity".into()],
            metadata_types: BTreeMap::new(),
            array_fields: vec![],
            cwe_field: None,
        }
    }

    #[test]
    fn ingest_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("corpus");
        let input = dir.path().join("data.jsonl");

        let mut f = std::fs::File::create(&input).unwrap();
        writeln!(f, r#"{{"id":"f1","title":"SQL Injection","description":"Found SQLi in search endpoint","severity":"critical"}}"#).unwrap();
        writeln!(f, r#"{{"id":"f2","title":"XSS","description":"Reflected XSS in profile page","severity":"high"}}"#).unwrap();

        let embedder = MockEmbedder;
        let chunking = ChunkingStrategy::Basic {
            max_characters: 500,
            overlap: 0,
        };

        let stats = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
        assert_eq!(stats.records_total, 2);
        assert_eq!(stats.records_new, 2);
        assert_eq!(stats.records_skipped, 0);

        // Re-ingest same data — should skip both
        let stats2 = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
        assert_eq!(stats2.records_skipped, 2);
        assert_eq!(stats2.records_new, 0);
    }

    #[test]
    fn ingest_upsert_on_change() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("corpus");
        let input = dir.path().join("data.jsonl");

        // Initial ingest
        {
            let mut f = std::fs::File::create(&input).unwrap();
            writeln!(f, r#"{{"id":"f1","title":"Bug","description":"Original description","severity":"low"}}"#).unwrap();
        }
        let embedder = MockEmbedder;
        let chunking = ChunkingStrategy::Basic {
            max_characters: 500,
            overlap: 0,
        };
        let stats = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
        assert_eq!(stats.records_new, 1);

        // Change content and re-ingest
        {
            let mut f = std::fs::File::create(&input).unwrap();
            writeln!(f, r#"{{"id":"f1","title":"Bug","description":"Updated description","severity":"high"}}"#).unwrap();
        }
        let stats2 = index_jsonl(&input, &corpus, &chunking, &embedder, &config()).unwrap();
        assert_eq!(stats2.records_upserted, 1);
        assert_eq!(stats2.records_new, 0);
        assert_eq!(stats2.records_skipped, 0);
    }
}
