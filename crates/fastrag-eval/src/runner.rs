use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use hdrhistogram::Histogram;
use memory_stats::memory_stats;

use crate::dataset::{EvalDataset, EvalDocument};
use crate::error::{EvalError, EvalResult};
use crate::metrics::{hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k};
use crate::report::{EvalReport, LatencyStats, MemoryStats};

use fastrag_core::ChunkingStrategy;
use fastrag_embed::{CANARY_TEXT, Canary, DynEmbedderTrait, PassageText, QueryText};
use fastrag_index::{CorpusManifest, HnswIndex, IndexEntry, ManifestChunkingStrategy, VectorIndex};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkRecord {
    doc_id: String,
    title: Option<String>,
    text: String,
    chunk_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct QueryGroundTruth {
    relevant_docs: HashSet<String>,
    qrels: HashMap<String, u32>,
}

pub struct Runner<'a> {
    embedder: &'a dyn DynEmbedderTrait,
    chunking: ChunkingStrategy,
    dataset: &'a EvalDataset,
    top_k: usize,
}

impl<'a> Runner<'a> {
    pub fn new(
        embedder: &'a dyn DynEmbedderTrait,
        chunking: ChunkingStrategy,
        dataset: &'a EvalDataset,
        top_k: usize,
    ) -> Self {
        Self {
            embedder,
            chunking,
            dataset,
            top_k,
        }
    }

    pub fn run(&self) -> EvalResult<EvalReport> {
        let mut memory_peak = sample_current_rss_bytes();
        let build_started = Instant::now();
        let index = index_documents_with_memory(
            self.dataset,
            self.embedder,
            &self.chunking,
            &mut memory_peak,
        )?;
        let build_time_ms = build_started.elapsed().as_millis() as u64;

        let gt_by_query = self.build_ground_truth();
        let mut histogram = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3)
            .map_err(|err| EvalError::Histogram(err.to_string()))?;
        let mut metrics = AggregatedMetrics::default();

        for query in &self.dataset.queries {
            let query_started = Instant::now();
            let query_vec = self
                .embedder
                .embed_query_dyn(&[QueryText::new(query.text.as_str())])?;
            let query_vec = query_vec.into_iter().next().ok_or_else(|| {
                EvalError::MalformedDataset("embedder returned no vectors".to_string())
            })?;
            let hits = index.query(&query_vec, self.top_k)?;
            let elapsed_us = query_started.elapsed().as_micros() as u64;
            let elapsed_us = elapsed_us.max(1);
            histogram
                .record(elapsed_us)
                .map_err(|err| EvalError::Histogram(err.to_string()))?;
            memory_peak = memory_peak.max(sample_current_rss_bytes());

            let retrieved = unique_doc_ids(
                &hits
                    .iter()
                    .map(|hit| hit.entry.source_path.to_string_lossy().to_string())
                    .collect::<Vec<_>>(),
            );
            let ground_truth = gt_by_query.get(&query.id).cloned().unwrap_or_default();
            metrics.add_sample(
                &retrieved,
                &ground_truth.relevant_docs,
                &ground_truth.qrels,
                self.top_k,
            );
        }

        let current_rss_bytes = sample_current_rss_bytes();
        memory_peak = memory_peak.max(current_rss_bytes);

        let latency = latency_from_histogram(&histogram);
        let metrics = metrics.finish(self.dataset.queries.len().max(1) as f64);

        Ok(EvalReport {
            dataset: self.dataset.name.clone(),
            embedder: self.embedder.model_id().to_string(),
            chunking: chunking_label(&self.chunking),
            metrics,
            latency,
            memory: MemoryStats {
                peak_rss_bytes: memory_peak,
                current_rss_bytes,
            },
            build_time_ms,
            run_at_unix: current_unix_seconds(),
            top_k: self.top_k,
            git_rev: EvalReport::current_git_rev(),
            fastrag_version: EvalReport::current_fastrag_version(),
        })
    }

    fn build_ground_truth(&self) -> HashMap<String, QueryGroundTruth> {
        let mut map: HashMap<String, QueryGroundTruth> = HashMap::new();
        for qrel in &self.dataset.qrels {
            let entry = map
                .entry(qrel.query_id.clone())
                .or_insert_with(|| QueryGroundTruth {
                    relevant_docs: HashSet::new(),
                    qrels: HashMap::new(),
                });
            if qrel.relevance > 0 {
                entry.relevant_docs.insert(qrel.doc_id.clone());
            }
            entry.qrels.insert(qrel.doc_id.clone(), qrel.relevance);
        }
        map
    }
}

#[derive(Default)]
struct AggregatedMetrics {
    recall_at_1: f64,
    recall_at_5: f64,
    recall_at_10: f64,
    recall_at_20: f64,
    mrr_at_10: f64,
    ndcg_at_10: f64,
    hit_rate_at_10: f64,
}

impl AggregatedMetrics {
    fn add_sample(
        &mut self,
        retrieved: &[String],
        relevant: &HashSet<String>,
        qrels: &HashMap<String, u32>,
        _top_k: usize,
    ) {
        self.recall_at_1 += recall_at_k(retrieved, relevant, 1);
        self.recall_at_5 += recall_at_k(retrieved, relevant, 5);
        self.recall_at_10 += recall_at_k(retrieved, relevant, 10);
        self.recall_at_20 += recall_at_k(retrieved, relevant, 20);
        self.mrr_at_10 += mrr_at_k(retrieved, relevant, 10);
        self.ndcg_at_10 += ndcg_at_k(retrieved, qrels, 10);
        self.hit_rate_at_10 += hit_rate_at_k(retrieved, relevant, 10);
    }

    fn finish(self, query_count: f64) -> HashMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("hit_rate@10".to_string(), self.hit_rate_at_10 / query_count);
        metrics.insert("mrr@10".to_string(), self.mrr_at_10 / query_count);
        metrics.insert("ndcg@10".to_string(), self.ndcg_at_10 / query_count);
        metrics.insert("recall@1".to_string(), self.recall_at_1 / query_count);
        metrics.insert("recall@5".to_string(), self.recall_at_5 / query_count);
        metrics.insert("recall@10".to_string(), self.recall_at_10 / query_count);
        metrics.insert("recall@20".to_string(), self.recall_at_20 / query_count);
        metrics.into_iter().collect()
    }
}

pub fn index_documents(
    dataset: &EvalDataset,
    embedder: &dyn DynEmbedderTrait,
    chunking: &ChunkingStrategy,
) -> EvalResult<HnswIndex> {
    let mut memory_peak = sample_current_rss_bytes();
    index_documents_with_memory(dataset, embedder, chunking, &mut memory_peak)
}

fn index_documents_with_memory(
    dataset: &EvalDataset,
    embedder: &dyn DynEmbedderTrait,
    chunking: &ChunkingStrategy,
    memory_peak: &mut u64,
) -> EvalResult<HnswIndex> {
    let canary_vec = embedder
        .embed_passage_dyn(&[PassageText::new(CANARY_TEXT)])
        .map_err(|e| EvalError::MalformedDataset(e.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| EvalError::MalformedDataset("canary embed returned no vector".into()))?;
    let canary = Canary {
        text_version: 1,
        vector: canary_vec,
    };
    let manifest = CorpusManifest::new(
        embedder.identity(),
        canary,
        current_unix_seconds(),
        manifest_chunking_strategy(chunking)?,
    );
    let mut index = HnswIndex::new(manifest);
    let mut next_id = 1u64;
    let mut chunk_records = Vec::new();

    for document in &dataset.documents {
        let chunks = chunk_document(document, chunking)?;
        for chunk in chunks {
            chunk_records.push(ChunkRecord {
                doc_id: document.id.clone(),
                title: document.title.clone(),
                text: chunk.text,
                chunk_index: chunk.chunk_index,
            });
        }
    }

    // Sort chunks by text length before batching so each batch contains similarly-sized
    // inputs. Tokenizers use BatchLongest padding → a mixed batch pads every row to the
    // longest row. NFCorpus mixes 30-token titles with 512-token abstracts; sorting drops
    // padded-token count by ~3x and delivers a proportional speedup on pure-CPU candle.
    // The HNSW index is order-independent (verified via crates/fastrag-index/src/hnsw.rs:
    // rebuild_graph uses vector proximity, not insertion order), so we keep the sorted
    // order all the way through index insertion — no un-sort step needed.
    chunk_records.sort_by_key(|c| c.text.len());

    let texts = chunk_records
        .iter()
        .map(|chunk| PassageText::new(chunk.text.as_str()))
        .collect::<Vec<_>>();
    // Embedders like BGE materialize a (batch, seq, hidden) tensor, so embedding the
    // entire corpus in one shot blows RAM on real BEIR datasets. Delegate batching to the
    // DynEmbedderTrait so each model can pick its own safe size.
    let batch = embedder.default_batch_size();
    let mut vectors = Vec::with_capacity(texts.len());
    for slice in texts.chunks(batch) {
        let batch_vectors = embedder.embed_passage_dyn(slice)?;
        vectors.extend(batch_vectors);
        *memory_peak = (*memory_peak).max(sample_current_rss_bytes());
    }
    if vectors.len() != chunk_records.len() {
        return Err(EvalError::MalformedDataset(format!(
            "embedder returned {} vectors for {} chunks",
            vectors.len(),
            chunk_records.len()
        )));
    }

    let mut entries = Vec::with_capacity(chunk_records.len());
    for (chunk, vector) in chunk_records.into_iter().zip(vectors.into_iter()) {
        entries.push(IndexEntry {
            id: next_id,
            vector,
            chunk_text: chunk.text,
            source_path: PathBuf::from(chunk.doc_id),
            chunk_index: chunk.chunk_index,
            section: chunk.title,
            element_kinds: Vec::new(),
            pages: Vec::new(),
            language: None,
            metadata: std::collections::BTreeMap::new(),
        });
        next_id += 1;
        *memory_peak = (*memory_peak).max(sample_current_rss_bytes());
    }

    index.add(entries)?;
    *memory_peak = (*memory_peak).max(sample_current_rss_bytes());
    Ok(index)
}

fn chunk_document(
    document: &EvalDocument,
    chunking: &ChunkingStrategy,
) -> EvalResult<Vec<ChunkRecord>> {
    match chunking {
        ChunkingStrategy::Basic { max_characters, .. } => {
            Ok(split_text(document, *max_characters, false))
        }
        ChunkingStrategy::ByTitle { max_characters, .. } => {
            Ok(split_text(document, *max_characters, true))
        }
        other => Err(EvalError::UnsupportedChunkingStrategy(chunking_name(other))),
    }
}

fn split_text(
    document: &EvalDocument,
    max_characters: usize,
    prefix_title: bool,
) -> Vec<ChunkRecord> {
    let mut chunks = Vec::new();
    let base_text = if prefix_title {
        match &document.title {
            Some(title) if !title.is_empty() => format!("{}\n\n{}", title, document.text),
            _ => document.text.clone(),
        }
    } else {
        document.text.clone()
    };

    let chars: Vec<char> = base_text.chars().collect();
    if chars.len() <= max_characters {
        chunks.push(ChunkRecord {
            doc_id: document.id.clone(),
            title: document.title.clone(),
            text: base_text,
            chunk_index: 0,
        });
        return chunks;
    }

    let mut start = 0usize;
    let mut chunk_index = 0usize;
    while start < chars.len() {
        let end = (start + max_characters).min(chars.len());
        let text: String = chars[start..end].iter().collect();
        chunks.push(ChunkRecord {
            doc_id: document.id.clone(),
            title: document.title.clone(),
            text,
            chunk_index,
        });
        chunk_index += 1;
        start = end;
    }
    chunks
}

fn unique_doc_ids(retrieved: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for doc_id in retrieved {
        if seen.insert(doc_id.clone()) {
            out.push(doc_id.clone());
        }
    }
    out
}

fn manifest_chunking_strategy(chunking: &ChunkingStrategy) -> EvalResult<ManifestChunkingStrategy> {
    Ok(match chunking {
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
    })
}

fn chunking_label(chunking: &ChunkingStrategy) -> String {
    match chunking {
        ChunkingStrategy::Basic {
            max_characters,
            overlap,
        } => format!(
            "basic(max_characters={}, overlap={})",
            max_characters, overlap
        ),
        ChunkingStrategy::ByTitle {
            max_characters,
            overlap,
        } => format!(
            "by-title(max_characters={}, overlap={})",
            max_characters, overlap
        ),
        ChunkingStrategy::RecursiveCharacter {
            max_characters,
            overlap,
            separators,
        } => format!(
            "recursive(max_characters={}, overlap={}, separators={:?})",
            max_characters, overlap, separators
        ),
        ChunkingStrategy::Semantic {
            max_characters,
            similarity_threshold,
            percentile_threshold,
        } => format!(
            "semantic(max_characters={}, similarity_threshold={:?}, percentile_threshold={:?})",
            max_characters, similarity_threshold, percentile_threshold
        ),
    }
}

fn latency_from_histogram(histogram: &Histogram<u64>) -> LatencyStats {
    if histogram.is_empty() {
        return LatencyStats {
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            mean_ms: 0.0,
            count: 0,
        };
    }
    LatencyStats {
        p50_ms: histogram.value_at_quantile(0.50) as f64 / 1000.0,
        p95_ms: histogram.value_at_quantile(0.95) as f64 / 1000.0,
        p99_ms: histogram.value_at_quantile(0.99) as f64 / 1000.0,
        mean_ms: histogram.mean() / 1000.0,
        count: histogram.len(),
    }
}

fn sample_current_rss_bytes() -> u64 {
    memory_stats()
        .map(|stats| stats.physical_mem as u64)
        .unwrap_or(0)
}

fn current_unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn chunking_name(chunking: &ChunkingStrategy) -> String {
    match chunking {
        ChunkingStrategy::Basic { .. } => "basic".to_string(),
        ChunkingStrategy::ByTitle { .. } => "by-title".to_string(),
        ChunkingStrategy::RecursiveCharacter { .. } => "recursive".to_string(),
        ChunkingStrategy::Semantic { .. } => "semantic".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_embed::test_utils::MockEmbedder;

    fn tiny_dataset() -> EvalDataset {
        crate::dataset::EvalDataset::load(PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny.json"
        )))
        .unwrap()
    }

    #[test]
    fn runner_with_mock_embedder_deterministic() {
        let dataset = tiny_dataset();
        let embedder = MockEmbedder;
        let a = Runner::new(
            &embedder,
            ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &dataset,
            10,
        )
        .run()
        .unwrap();
        let b = Runner::new(
            &embedder,
            ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &dataset,
            10,
        )
        .run()
        .unwrap();
        assert_eq!(a.metrics, b.metrics);
    }

    #[test]
    fn split_text_handles_multibyte_chars() {
        // Pre-#25 the chunker sliced on byte indices and panicked on multibyte UTF-8
        // (e.g. NFCorpus contains the U+2009 thin-space). Splitting must always land on
        // a char boundary.
        let document = EvalDocument {
            id: "doc1".to_string(),
            title: None,
            text: "a\u{2009}".repeat(600), // 600 multibyte units, well over 1000 bytes
        };
        let chunks = split_text(&document, 1000, false);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.text.is_char_boundary(0));
            assert!(chunk.text.is_char_boundary(chunk.text.len()));
        }
        let rejoined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(rejoined, document.text);
    }

    #[test]
    fn runner_sorts_chunks_by_length_before_embedding() {
        // Proves that sorted-length batching is active: a CountingEmbedder records each
        // batch's min/max text length; with sort active, max-length grows monotonically
        // across batches and within any batch the spread is bounded by the batch size.
        use crate::dataset::{EvalDocument, Qrel};
        use std::sync::Mutex;

        struct LengthSpyEmbedder {
            spread_per_batch: Mutex<Vec<(usize, usize)>>,
        }
        impl fastrag_embed::Embedder for LengthSpyEmbedder {
            const DIM: usize = 4;
            const MODEL_ID: &'static str = "test/length-spy";
            const PREFIX_SCHEME: fastrag_embed::PrefixScheme = fastrag_embed::PrefixScheme::NONE;
            fn embed_query(
                &self,
                texts: &[fastrag_embed::QueryText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                let lens: Vec<usize> = texts.iter().map(|t| t.as_str().len()).collect();
                let (lo, hi) = (*lens.iter().min().unwrap(), *lens.iter().max().unwrap());
                self.spread_per_batch.lock().unwrap().push((lo, hi));
                Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
            }
            fn embed_passage(
                &self,
                texts: &[fastrag_embed::PassageText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                let lens: Vec<usize> = texts.iter().map(|t| t.as_str().len()).collect();
                let (lo, hi) = (
                    *lens.iter().min().unwrap_or(&0),
                    *lens.iter().max().unwrap_or(&0),
                );
                self.spread_per_batch.lock().unwrap().push((lo, hi));
                Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
            }
            fn default_batch_size(&self) -> usize {
                4
            }
        }

        let dataset = EvalDataset {
            name: "sort-test".to_string(),
            documents: vec![
                EvalDocument {
                    id: "a".to_string(),
                    title: None,
                    text: "x".repeat(500),
                },
                EvalDocument {
                    id: "b".to_string(),
                    title: None,
                    text: "y".repeat(10),
                },
                EvalDocument {
                    id: "c".to_string(),
                    title: None,
                    text: "z".repeat(250),
                },
                EvalDocument {
                    id: "d".to_string(),
                    title: None,
                    text: "q".repeat(50),
                },
                EvalDocument {
                    id: "e".to_string(),
                    title: None,
                    text: "p".repeat(400),
                },
            ],
            queries: Vec::new(),
            qrels: Vec::<Qrel>::new(),
        };
        let embedder = LengthSpyEmbedder {
            spread_per_batch: Default::default(),
        };
        Runner::new(
            &embedder,
            ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &dataset,
            10,
        )
        .run()
        .unwrap();
        let batches = embedder.spread_per_batch.lock().unwrap().clone();
        // Batch 0 is the canary passage (a single fixed text). Batches 1..N are the
        // document chunks sorted by length. With batch_size=4 and 5 docs:
        //   batch 1: [10, 50, 250, 400]
        //   batch 2: [500]
        // Without sort the first doc-batch would mix 500+10.
        assert_eq!(batches.len(), 3, "expected canary batch + 2 doc batches");
        let (first_lo, first_hi) = batches[1];
        let (second_lo, _second_hi) = batches[2];
        assert_eq!(first_lo, 10);
        assert_eq!(first_hi, 400);
        assert_eq!(second_lo, 500);
    }

    #[test]
    fn runner_recall_above_zero_on_tiny_dataset() {
        let dataset = tiny_dataset();
        let embedder = MockEmbedder;
        let report = Runner::new(
            &embedder,
            ChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
            &dataset,
            10,
        )
        .run()
        .unwrap();
        assert!(report.metrics["recall@10"] > 0.0);
    }
}
