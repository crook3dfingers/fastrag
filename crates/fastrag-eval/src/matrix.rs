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
    fn query(
        &self,
        variant: ConfigVariant,
        question: &str,
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
) -> Result<MatrixReport, EvalError> {
    let mut variant_reports: Vec<VariantReport> = Vec::with_capacity(4);

    for variant in ConfigVariant::all() {
        // Per-stage histograms for this variant.
        let mut h_total = new_hist();
        let mut h_embed = new_hist();
        let mut h_bm25 = new_hist();
        let mut h_hnsw = new_hist();
        let mut h_rerank = new_hist();
        let mut h_fuse = new_hist();

        let mut per_question: Vec<QuestionResult> = Vec::with_capacity(gold_set.entries.len());

        for entry in &gold_set.entries {
            let mut bd = LatencyBreakdown::default();
            let chunks = driver
                .query(variant, &entry.question, top_k, &mut bd)
                .map_err(|e| EvalError::MatrixVariant {
                    variant,
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
            variant,
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
}
