//! Hybrid retrieval (BM25 + dense RRF) with optional post-fusion temporal decay.
//!
//! Called from `query_corpus_with_filter_opts` when `QueryOpts::hybrid.enabled`
//! is set. Keeps the pure-function pieces (`decay_factor`, `apply_decay`)
//! separate from the I/O-bound `query_hybrid` so they can be unit-tested in
//! isolation.

#![allow(unused_imports)]

use std::time::Duration;

use chrono::{DateTime, NaiveDate, Utc};

use super::CorpusError;
use fastrag_index::fusion::{ScoredId, rrf_fuse};

#[derive(Debug, Clone)]
pub struct HybridOpts {
    pub enabled: bool,
    pub rrf_k: u32,
    pub overfetch_factor: usize,
    pub temporal: Option<TemporalOpts>,
}

impl Default for HybridOpts {
    fn default() -> Self {
        Self {
            enabled: false,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TemporalOpts {
    pub date_field: String,
    pub halflife: Duration,
    pub weight_floor: f32,
    pub dateless_prior: f32,
    pub blend: BlendMode,
    pub now: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Multiplicative,
    Additive,
}

/// Multiplicative decay factor in `[weight_floor, 1.0]`, or `dateless_prior`
/// when `age_days` is `None`. `halflife_days` must be > 0.
///
/// ```text
/// factor = alpha + (1 - alpha) * exp(-ln(2) * age_days / halflife)
/// ```
pub fn decay_factor(
    age_days: Option<f32>,
    halflife_days: f32,
    alpha: f32,
    dateless_prior: f32,
    _blend: BlendMode,
) -> f32 {
    match age_days {
        None => dateless_prior,
        Some(a) => {
            let a = a.max(0.0);
            let ln2: f32 = std::f32::consts::LN_2;
            alpha + (1.0 - alpha) * (-ln2 * a / halflife_days).exp()
        }
    }
}

#[cfg(test)]
mod decay_factor_tests {
    use super::*;

    #[test]
    fn age_zero_returns_one() {
        let f = decay_factor(Some(0.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn age_equal_halflife_returns_midpoint() {
        // alpha + (1-alpha)*0.5 = 0.3 + 0.35 = 0.65
        let f = decay_factor(Some(30.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 0.65).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn very_old_approaches_alpha_floor() {
        let f = decay_factor(Some(10_000.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!(f > 0.299 && f < 0.301, "got {f}");
    }

    #[test]
    fn alpha_one_disables_decay() {
        let f = decay_factor(Some(9999.0), 30.0, 1.0, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn dateless_returns_prior() {
        let f = decay_factor(None, 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 0.5).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn dateless_prior_independent_of_halflife() {
        let a = decay_factor(None, 30.0, 0.3, 0.42, BlendMode::Multiplicative);
        let b = decay_factor(None, 9000.0, 0.3, 0.42, BlendMode::Multiplicative);
        assert_eq!(a, b);
    }

    #[test]
    fn negative_age_clamps_to_one() {
        // Future-dated docs (negative age) treated as "today".
        let f = decay_factor(Some(-5.0), 30.0, 0.3, 0.5, BlendMode::Multiplicative);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }
}
