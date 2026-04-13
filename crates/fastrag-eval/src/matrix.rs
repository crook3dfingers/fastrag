//! 4-variant config matrix runner and report types.
//!
//! `ConfigVariant` represents the four ablation conditions evaluated together.
//! `run_matrix` drives a `CorpusDriver` over every variant and every gold-set
//! question, collecting per-stage histograms and hit-rate metrics.

use std::process::Command;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::gold_set::{GoldSet, score_entry};
use fastrag::corpus::LatencyBreakdown;

// ─── Variant enum ─────────────────────────────────────────────────────────────

/// The four ablation conditions evaluated by the matrix runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigVariant {
    /// Full stack: hybrid (BM25 + dense) + reranker + contextual corpus.
    Primary,
    /// Hybrid retrieval only — no reranker pass.
    NoRerank,
    /// Full stack on non-contextual corpus.
    NoContextual,
    /// Dense-only retrieval (HNSW) + reranker, no BM25 fusion.
    DenseOnly,
}

impl ConfigVariant {
    /// All four variants in canonical order.
    pub fn all() -> [ConfigVariant; 4] {
        [
            ConfigVariant::Primary,
            ConfigVariant::NoRerank,
            ConfigVariant::NoContextual,
            ConfigVariant::DenseOnly,
        ]
    }

    /// Parse a variant from its stable label string.
    pub fn from_label(s: &str) -> Option<ConfigVariant> {
        match s {
            "primary" => Some(ConfigVariant::Primary),
            "no_rerank" => Some(ConfigVariant::NoRerank),
            "no_contextual" => Some(ConfigVariant::NoContextual),
            "dense_only" => Some(ConfigVariant::DenseOnly),
            _ => None,
        }
    }

    /// Stable label used in JSON reports and log output.
    pub fn label(&self) -> &'static str {
        match self {
            ConfigVariant::Primary => "primary",
            ConfigVariant::NoRerank => "no_rerank",
            ConfigVariant::NoContextual => "no_contextual",
            ConfigVariant::DenseOnly => "dense_only",
        }
    }
}

// ─── Report types ─────────────────────────────────────────────────────────────

/// Latency percentiles in microseconds for a single stage/variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

/// Per-stage latency percentiles for a variant run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub total: Percentiles,
    pub embed: Percentiles,
    pub bm25: Percentiles,
    pub hnsw: Percentiles,
    pub rerank: Percentiles,
    pub fuse: Percentiles,
}

/// Result for a single gold-set question under one variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResult {
    pub id: String,
    pub hit_at_1: bool,
    pub hit_at_5: bool,
    pub hit_at_10: bool,
    pub reciprocal_rank: f64,
    pub missing_cve_ids: Vec<String>,
    pub missing_terms: Vec<String>,
    pub latency_us: LatencyBreakdown,
}

/// Aggregated metrics for one variant over the whole gold set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantReport {
    pub variant: ConfigVariant,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub hit_at_10: f64,
    pub mrr_at_10: f64,
    pub latency: LatencyPercentiles,
    pub per_question: Vec<QuestionResult>,
}

/// Top-level matrix evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatrixReport {
    pub schema_version: u32,
    pub git_rev: String,
    pub captured_at: String,
    pub runs: Vec<VariantReport>,
    /// hit@5(Primary) − hit@5(NoRerank): marginal value of reranking.
    pub rerank_delta: f64,
    /// hit@5(Primary) − hit@5(NoContextual): marginal value of contextualisation.
    pub contextual_delta: f64,
    /// hit@5(Primary) − hit@5(DenseOnly): marginal value of BM25 fusion.
    pub hybrid_delta: f64,
}

impl MatrixReport {
    /// hit@5 rate for the named variant, or `None` if not present in `runs`.
    pub fn hit5(&self, variant: ConfigVariant) -> Option<f64> {
        self.runs
            .iter()
            .find(|r| r.variant == variant)
            .map(|r| r.hit_at_5)
    }
}

// ─── CorpusDriver trait ────────────────────────────────────────────────────────

/// Abstraction over the real retrieval stack, mockable in tests.
pub trait CorpusDriver {
    /// Embed all query texts in a single batch. Returns one vector per input.
    ///
    /// Called once before the variant loop so embeddings are computed once and
    /// reused across all four variants — query embeddings are corpus-independent.
    fn embed_queries(&self, questions: &[&str]) -> Result<Vec<Vec<f32>>, EvalError>;

    /// Run retrieval + optional reranking using a pre-computed query vector.
    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
        query_vector: &[f32],
        top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError>;
}

// ─── Orchestrator ─────────────────────────────────────────────────────────────

/// Maximum histogram value: 60 seconds in microseconds.
const HIST_MAX_US: u64 = 60_000_000;

fn new_hist() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, HIST_MAX_US, 3).expect("histogram bounds are valid")
}

fn percentiles(h: &Histogram<u64>) -> Percentiles {
    Percentiles {
        p50_us: h.value_at_quantile(0.50),
        p95_us: h.value_at_quantile(0.95),
        p99_us: h.value_at_quantile(0.99),
    }
}

fn git_rev() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

/// Run all four variants over every entry in `gold_set` and return a
/// `MatrixReport` with per-question and aggregate metrics.
pub fn run_matrix<D: CorpusDriver>(
    driver: &D,
    gold_set: &GoldSet,
    top_k: usize,
    variants: Option<&[ConfigVariant]>,
) -> Result<MatrixReport, EvalError> {
    // ── Batch-embed all queries once ─────────────────────────────────────
    // Query embeddings are corpus-independent, so one batch covers all four
    // variants. This reduces embed HTTP calls from 4×N to 1 and eliminates
    // the idle-connection churn that plagued the per-query approach.
    let questions: Vec<&str> = gold_set
        .entries
        .iter()
        .map(|e| e.question.as_str())
        .collect();
    let n = questions.len();
    eprintln!("[eval] embedding {n} queries in one batch…");
    let t_batch = std::time::Instant::now();
    let vectors = driver.embed_queries(&questions)?;
    let batch_embed_us = t_batch.elapsed().as_micros() as u64;
    let per_query_embed_us = if n > 0 { batch_embed_us / n as u64 } else { 0 };
    eprintln!("[eval] batch embed done in {batch_embed_us} µs ({per_query_embed_us} µs/query)");

    if vectors.len() != n {
        return Err(EvalError::Runner(format!(
            "embed_queries returned {} vectors for {} questions",
            vectors.len(),
            n
        )));
    }

    let mut variant_reports: Vec<VariantReport> = Vec::with_capacity(4);

    let active: Vec<ConfigVariant> =
        variants.map_or_else(|| ConfigVariant::all().to_vec(), |v| v.to_vec());

    for variant in &active {
        // Per-stage histograms for this variant.
        let mut h_total = new_hist();
        let mut h_embed = new_hist();
        let mut h_bm25 = new_hist();
        let mut h_hnsw = new_hist();
        let mut h_rerank = new_hist();
        let mut h_fuse = new_hist();

        let mut per_question: Vec<QuestionResult> = Vec::with_capacity(gold_set.entries.len());

        for (qi, (entry, vector)) in gold_set.entries.iter().zip(&vectors).enumerate() {
            eprintln!(
                "[eval] {variant:?} query {}/{}: {}",
                qi + 1,
                gold_set.entries.len(),
                &entry.question[..entry.question.len().min(60)]
            );
            // Amortize the batch embed time across individual queries.
            let mut bd = LatencyBreakdown {
                embed_us: per_query_embed_us,
                ..Default::default()
            };
            let chunks = driver
                .query(*variant, &entry.question, vector, top_k, &mut bd)
                .map_err(|e| EvalError::MatrixVariant {
                    variant: *variant,
                    source: Box::new(e),
                })?;

            // Record stage latencies (avoid 0 — histogram lower bound is 1).
            h_total.record(bd.total_us.max(1)).ok();
            h_embed.record(bd.embed_us.max(1)).ok();
            h_bm25.record(bd.bm25_us.max(1)).ok();
            h_hnsw.record(bd.hnsw_us.max(1)).ok();
            h_rerank.record(bd.rerank_us.max(1)).ok();
            h_fuse.record(bd.fuse_us.max(1)).ok();

            let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let score = score_entry(entry, &chunk_refs);

            per_question.push(QuestionResult {
                id: entry.id.clone(),
                hit_at_1: score.hit_at_1,
                hit_at_5: score.hit_at_5,
                hit_at_10: score.hit_at_10,
                reciprocal_rank: score.reciprocal_rank,
                missing_cve_ids: score.missing_cve_ids,
                missing_terms: score.missing_terms,
                latency_us: bd,
            });
        }

        let n = per_question.len() as f64;
        let hit_at_1 = if n > 0.0 {
            per_question.iter().filter(|q| q.hit_at_1).count() as f64 / n
        } else {
            0.0
        };
        let hit_at_5 = if n > 0.0 {
            per_question.iter().filter(|q| q.hit_at_5).count() as f64 / n
        } else {
            0.0
        };
        let hit_at_10 = if n > 0.0 {
            per_question.iter().filter(|q| q.hit_at_10).count() as f64 / n
        } else {
            0.0
        };
        let mrr_at_10 = if n > 0.0 {
            per_question.iter().map(|q| q.reciprocal_rank).sum::<f64>() / n
        } else {
            0.0
        };

        variant_reports.push(VariantReport {
            variant: *variant,
            hit_at_1,
            hit_at_5,
            hit_at_10,
            mrr_at_10,
            latency: LatencyPercentiles {
                total: percentiles(&h_total),
                embed: percentiles(&h_embed),
                bm25: percentiles(&h_bm25),
                hnsw: percentiles(&h_hnsw),
                rerank: percentiles(&h_rerank),
                fuse: percentiles(&h_fuse),
            },
            per_question,
        });
    }

    let hit5 = |v: ConfigVariant| -> f64 {
        variant_reports
            .iter()
            .find(|r| r.variant == v)
            .map_or(0.0, |r| r.hit_at_5)
    };

    let primary_h5 = hit5(ConfigVariant::Primary);
    let rerank_delta = primary_h5 - hit5(ConfigVariant::NoRerank);
    let contextual_delta = primary_h5 - hit5(ConfigVariant::NoContextual);
    let hybrid_delta = primary_h5 - hit5(ConfigVariant::DenseOnly);

    let rev = git_rev().unwrap_or_else(|| "unknown".to_string());
    let captured_at = chrono::Utc::now().to_rfc3339();

    Ok(MatrixReport {
        schema_version: 1,
        git_rev: rev,
        captured_at,
        runs: variant_reports,
        rerank_delta,
        contextual_delta,
        hybrid_delta,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_variant_all_returns_four_in_order() {
        let all = ConfigVariant::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], ConfigVariant::Primary);
        assert_eq!(all[3], ConfigVariant::DenseOnly);
    }

    #[test]
    fn config_variant_labels_are_stable() {
        assert_eq!(ConfigVariant::Primary.label(), "primary");
        assert_eq!(ConfigVariant::NoRerank.label(), "no_rerank");
        assert_eq!(ConfigVariant::NoContextual.label(), "no_contextual");
        assert_eq!(ConfigVariant::DenseOnly.label(), "dense_only");
    }

    #[test]
    fn from_label_round_trips_all_variants() {
        for v in ConfigVariant::all() {
            assert_eq!(ConfigVariant::from_label(v.label()), Some(v));
        }
        assert_eq!(ConfigVariant::from_label("bogus"), None);
    }
}
