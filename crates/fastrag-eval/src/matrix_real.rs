//! Real CorpusDriver implementation backed by fastrag Store query paths.
//!
//! Pre-loads both Stores (contextualized + raw) once at construction time
//! rather than re-opening them per query. This avoids repeated canary
//! verification embeds, cutting embedder HTTP calls in half.
//!
//! Variants map onto retrieval pipelines as follows:
//! - `Primary`, `NoContextual`: hybrid (BM25+dense RRF) + reranker
//! - `NoRerank`:                 hybrid only
//! - `DenseOnly`:                dense HNSW + reranker (no BM25)
//! - `TemporalOn`:               hybrid with post-fusion exponential decay +
//!   reranker. Uses `published_date` with a 2-year half-life and a 0.3
//!   weight floor.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use std::time::Duration;

use chrono::Utc;
use fastrag::corpus::LatencyBreakdown;
use fastrag::corpus::hybrid::{BlendMode, HybridOpts, TemporalOpts, query_hybrid};
use fastrag_embed::{DynEmbedderTrait, QueryText};
use fastrag_rerank::{RerankHit, Reranker};
use fastrag_store::Store;

use crate::error::EvalError;
use crate::matrix::{ConfigVariant, CorpusDriver};

/// Cross-encoder score keyed by (is_raw_store, question, doc_id).
/// The reranker score for (query_text, doc_text) is variant-independent,
/// so Primary / DenseOnly / TemporalOn share cached scores on ctx_store,
/// and NoContextual has its own namespace on raw_store.
type RerankCache = Mutex<HashMap<(bool, String, u64), f32>>;

pub struct RealCorpusDriver<'a> {
    ctx_store: Store,
    raw_store: Store,
    pub embedder: &'a dyn DynEmbedderTrait,
    pub reranker: &'a dyn Reranker,
    rerank_cache: RerankCache,
}

impl<'a> RealCorpusDriver<'a> {
    /// Load both corpus Stores up-front. The canary verification embed runs
    /// once per corpus here, not once per query.
    pub fn load(
        ctx_corpus: &Path,
        raw_corpus: &Path,
        embedder: &'a dyn DynEmbedderTrait,
        reranker: &'a dyn Reranker,
    ) -> Result<Self, EvalError> {
        let ctx_store = Store::open(ctx_corpus, embedder)
            .map_err(|e| EvalError::Runner(format!("load ctx corpus: {e}")))?;
        let raw_store = Store::open(raw_corpus, embedder)
            .map_err(|e| EvalError::Runner(format!("load raw corpus: {e}")))?;
        Ok(Self {
            ctx_store,
            raw_store,
            embedder,
            reranker,
            rerank_cache: Mutex::new(HashMap::new()),
        })
    }
}

impl CorpusDriver for RealCorpusDriver<'_> {
    fn embed_queries(&self, questions: &[&str]) -> Result<Vec<Vec<f32>>, EvalError> {
        let qt: Vec<QueryText> = questions.iter().map(|q| QueryText::new(*q)).collect();
        self.embedder
            .embed_query_dyn(&qt)
            .map_err(|e| EvalError::Runner(format!("batch embed: {e}")))
    }

    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
        query_vector: &[f32],
        top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError> {
        let store = match variant {
            ConfigVariant::NoContextual => &self.raw_store,
            _ => &self.ctx_store,
        };

        let over_fetch = top_k * 3;

        // ── First stage: candidate retrieval ────────────────────────────────
        let scored: Vec<(u64, f32)> = match variant {
            ConfigVariant::Primary | ConfigVariant::NoContextual | ConfigVariant::NoRerank => {
                let opts = HybridOpts {
                    enabled: true,
                    rrf_k: 60,
                    overfetch_factor: 3,
                    temporal: None,
                };
                let fused =
                    query_hybrid(store, question, query_vector, over_fetch, &opts, breakdown)
                        .map_err(|e| EvalError::Runner(format!("hybrid search: {e}")))?;
                fused.into_iter().map(|s| (s.id, s.score)).collect()
            }
            ConfigVariant::DenseOnly => {
                let t = Instant::now();
                let out = store
                    .query_dense(query_vector, over_fetch)
                    .map_err(|e| EvalError::Runner(format!("dense search: {e}")))?;
                breakdown.hnsw_us = t.elapsed().as_micros() as u64;
                out
            }
            ConfigVariant::TemporalOn => {
                let opts = HybridOpts {
                    enabled: true,
                    rrf_k: 60,
                    overfetch_factor: 3,
                    temporal: Some(TemporalOpts {
                        date_field: "published_date".to_string(),
                        halflife: Duration::from_secs(730 * 86_400),
                        weight_floor: 0.3,
                        dateless_prior: 1.0,
                        blend: BlendMode::Multiplicative,
                        now: Utc::now(),
                    }),
                };
                let fused =
                    query_hybrid(store, question, query_vector, over_fetch, &opts, breakdown)
                        .map_err(|e| EvalError::Runner(format!("hybrid+decay search: {e}")))?;
                fused.into_iter().map(|s| (s.id, s.score)).collect()
            }
        };

        if scored.is_empty() {
            breakdown.finalize();
            return Ok(vec![]);
        }

        // ── Hydrate chunk text from the Store ───────────────────────────────
        let hits = store
            .fetch_hits(&scored)
            .map_err(|e| EvalError::Runner(format!("fetch_hits: {e}")))?;
        let mut text_map: HashMap<u64, String> = HashMap::with_capacity(scored.len());
        for h in &hits {
            for c in &h.chunks {
                text_map.insert(c.id, c.chunk_text.clone());
            }
        }

        // ── Second stage: optional rerank ───────────────────────────────────
        let needs_rerank = matches!(
            variant,
            ConfigVariant::Primary
                | ConfigVariant::NoContextual
                | ConfigVariant::DenseOnly
                | ConfigVariant::TemporalOn
        );

        let mut ordered_texts: Vec<String> = if needs_rerank {
            let is_raw = matches!(variant, ConfigVariant::NoContextual);
            let rerank_input: Vec<RerankHit> = scored
                .iter()
                .filter_map(|(id, score)| {
                    text_map.get(id).map(|text| RerankHit {
                        id: *id,
                        chunk_text: text.clone(),
                        score: *score,
                    })
                })
                .collect();

            let mut cached: Vec<(String, f32)> = Vec::new();
            let mut misses: Vec<RerankHit> = Vec::with_capacity(rerank_input.len());
            {
                let cache = self.rerank_cache.lock().expect("rerank cache poisoned");
                for hit in rerank_input {
                    let key = (is_raw, question.to_string(), hit.id);
                    if let Some(&score) = cache.get(&key) {
                        cached.push((hit.chunk_text, score));
                    } else {
                        misses.push(hit);
                    }
                }
            }

            let t = Instant::now();
            let reranked_misses = if misses.is_empty() {
                Vec::new()
            } else {
                let out = self
                    .reranker
                    .rerank(question, misses)
                    .map_err(|e| EvalError::Runner(format!("rerank: {e}")))?;
                let mut cache = self.rerank_cache.lock().expect("rerank cache poisoned");
                for rh in &out {
                    cache.insert((is_raw, question.to_string(), rh.id), rh.score);
                }
                out
            };
            breakdown.rerank_us = t.elapsed().as_micros() as u64;

            let mut all: Vec<(String, f32)> = cached;
            all.extend(
                reranked_misses
                    .into_iter()
                    .map(|rh| (rh.chunk_text, rh.score)),
            );
            all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            all.into_iter().map(|(text, _)| text).collect()
        } else {
            scored
                .iter()
                .filter_map(|(id, _)| text_map.get(id).cloned())
                .collect()
        };

        ordered_texts.truncate(top_k);
        breakdown.finalize();
        Ok(ordered_texts)
    }
}
