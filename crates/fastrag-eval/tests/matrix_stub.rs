//! Integration tests for `run_matrix` using a deterministic stub driver.

use fastrag::corpus::LatencyBreakdown;
use fastrag_eval::EvalError;
use fastrag_eval::gold_set::{Axes, GoldSet, GoldSetEntry, Style, TemporalIntent};
use fastrag_eval::matrix::{ConfigVariant, CorpusDriver, run_matrix};

/// Stub driver with fully deterministic per-stage latencies and chunk returns.
///
/// For all variants except `DenseOnly`, returns a chunk containing "libfoo" and
/// "CVE-2024-12345". For `DenseOnly` it returns an irrelevant chunk so the
/// hybrid_delta test can observe a meaningful positive delta.
struct StubDriver;

impl CorpusDriver for StubDriver {
    fn embed_queries(&self, questions: &[&str]) -> Result<Vec<Vec<f32>>, EvalError> {
        Ok(questions.iter().map(|_| vec![0.0_f32; 4]).collect())
    }

    fn query(
        &self,
        variant: ConfigVariant,
        _question: &str,
        _query_vector: &[f32],
        _top_k: usize,
        breakdown: &mut LatencyBreakdown,
    ) -> Result<Vec<String>, EvalError> {
        // Deterministic per-stage latencies so histograms always have non-zero values.
        breakdown.embed_us = 100;
        breakdown.bm25_us = 50;
        breakdown.hnsw_us = 80;
        breakdown.rerank_us = match variant {
            ConfigVariant::NoRerank => 0,
            _ => 200,
        };
        breakdown.fuse_us = 30;
        breakdown.finalize();

        let chunk = match variant {
            ConfigVariant::DenseOnly => "unrelated content without the target".to_string(),
            _ => "advisory for libfoo mentions CVE-2024-12345".to_string(),
        };
        Ok(vec![chunk])
    }
}

fn single_entry_gold_set() -> GoldSet {
    GoldSet {
        version: 1,
        entries: vec![GoldSetEntry {
            id: "q001".into(),
            question: "Is there an RCE in libfoo?".into(),
            must_contain_cve_ids: vec!["CVE-2024-12345".into()],
            must_contain_terms: vec!["libfoo".into()],
            notes: None,
            axes: Axes {
                style: Style::Identifier,
                temporal_intent: TemporalIntent::Neutral,
            },
        }],
    }
}

fn multi_entry_gold_set() -> GoldSet {
    GoldSet {
        version: 1,
        entries: vec![
            GoldSetEntry {
                id: "q001".into(),
                question: "Is there an RCE in libfoo?".into(),
                must_contain_cve_ids: vec!["CVE-2024-12345".into()],
                must_contain_terms: vec!["libfoo".into()],
                notes: None,
                axes: Axes {
                    style: Style::Identifier,
                    temporal_intent: TemporalIntent::Neutral,
                },
            },
            GoldSetEntry {
                id: "q002".into(),
                question: "What CVE affects libfoo?".into(),
                must_contain_cve_ids: vec!["CVE-2024-12345".into()],
                must_contain_terms: vec![],
                notes: None,
                axes: Axes {
                    style: Style::Identifier,
                    temporal_intent: TemporalIntent::Neutral,
                },
            },
            GoldSetEntry {
                id: "q003".into(),
                question: "Any libfoo vulnerabilities?".into(),
                must_contain_cve_ids: vec![],
                must_contain_terms: vec!["libfoo".into()],
                notes: None,
                axes: Axes {
                    style: Style::Conceptual,
                    temporal_intent: TemporalIntent::Neutral,
                },
            },
        ],
    }
}

#[test]
fn run_matrix_executes_all_five_variants_in_order() {
    let gs = single_entry_gold_set();
    let report = run_matrix(&StubDriver, &gs, 5, None).expect("run_matrix should succeed");
    assert_eq!(report.runs.len(), 5);
    assert_eq!(report.runs[0].variant, ConfigVariant::Primary);
    assert_eq!(report.runs[1].variant, ConfigVariant::NoRerank);
    assert_eq!(report.runs[2].variant, ConfigVariant::NoContextual);
    assert_eq!(report.runs[3].variant, ConfigVariant::DenseOnly);
    assert_eq!(report.runs[4].variant, ConfigVariant::TemporalOn);
}

#[test]
fn run_matrix_temporal_on_runs_without_errors() {
    let gs = single_entry_gold_set();
    let report = run_matrix(&StubDriver, &gs, 5, Some(&[ConfigVariant::TemporalOn]))
        .expect("run_matrix should succeed");
    assert_eq!(report.runs.len(), 1);
    assert_eq!(report.runs[0].variant, ConfigVariant::TemporalOn);
}

#[test]
fn run_matrix_records_every_stage_histogram() {
    let gs = single_entry_gold_set();
    let report = run_matrix(&StubDriver, &gs, 5, None).expect("run_matrix should succeed");
    for variant_report in &report.runs {
        assert!(
            variant_report.latency.total.p50_us > 0,
            "{:?}: total p50_us should be > 0",
            variant_report.variant
        );
        assert!(
            variant_report.latency.embed.p50_us > 0,
            "{:?}: embed p50_us should be > 0",
            variant_report.variant
        );
        assert!(
            variant_report.latency.bm25.p50_us > 0,
            "{:?}: bm25 p50_us should be > 0",
            variant_report.variant
        );
        assert!(
            variant_report.latency.hnsw.p50_us > 0,
            "{:?}: hnsw p50_us should be > 0",
            variant_report.variant
        );
    }
}

#[test]
fn run_matrix_per_question_count_matches_entries() {
    let gs = multi_entry_gold_set();
    let report = run_matrix(&StubDriver, &gs, 5, None).expect("run_matrix should succeed");
    for variant_report in &report.runs {
        assert_eq!(
            variant_report.per_question.len(),
            3,
            "{:?}: expected 3 per_question entries",
            variant_report.variant
        );
    }
}

#[test]
fn run_matrix_hybrid_delta_positive_when_dense_misses() {
    // StubDriver returns a hit for Primary but a miss for DenseOnly.
    let gs = single_entry_gold_set();
    let report = run_matrix(&StubDriver, &gs, 5, None).expect("run_matrix should succeed");
    assert!(
        report.hybrid_delta > 0.0,
        "hybrid_delta should be positive when DenseOnly misses; got {}",
        report.hybrid_delta
    );
}

#[test]
fn run_matrix_with_variant_filter_runs_only_selected() {
    let gs = single_entry_gold_set();
    let filter = [ConfigVariant::Primary, ConfigVariant::NoRerank];
    let report = run_matrix(&StubDriver, &gs, 5, Some(&filter)).expect("run_matrix should succeed");
    assert_eq!(report.runs.len(), 2);
    assert_eq!(report.runs[0].variant, ConfigVariant::Primary);
    assert_eq!(report.runs[1].variant, ConfigVariant::NoRerank);
}

#[test]
fn run_matrix_truncated_gold_set_limits_per_question() {
    let mut gs = multi_entry_gold_set();
    assert_eq!(gs.entries.len(), 3);
    gs.entries.truncate(2);
    let report = run_matrix(&StubDriver, &gs, 5, None).expect("run_matrix should succeed");
    for variant_report in &report.runs {
        assert_eq!(
            variant_report.per_question.len(),
            2,
            "{:?}: expected 2 per_question entries after truncation",
            variant_report.variant
        );
    }
}
