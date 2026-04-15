//! Checked-in baseline + slack gate for eval regressions.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::EvalError;
use crate::matrix::{ConfigVariant, MatrixReport};

pub const DEFAULT_SLACK: f64 = 0.02;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub schema_version: u32,
    pub git_rev: String,
    pub captured_at: String,
    pub runs: Vec<VariantBaseline>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantBaseline {
    pub variant: ConfigVariant,
    pub hit_at_5: f64,
    pub mrr_at_10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    pub variant: ConfigVariant,
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
    pub delta: f64,
    pub slack: f64,
}

#[derive(Debug, Default)]
pub struct BaselineDiff {
    pub regressions: Vec<Regression>,
}

impl BaselineDiff {
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }

    pub fn render_report(&self) -> String {
        if self.regressions.is_empty() {
            return "## Baseline OK — no regressions\n".into();
        }
        let mut out = format!("## Baseline regressions ({})\n", self.regressions.len());
        for r in &self.regressions {
            let pct = ((r.current - r.baseline) / r.baseline) * 100.0;
            out.push_str(&format!(
                "- {:?} {}: {:.4} → {:.4} ({:+.2}%, slack ±{:.0}%)\n",
                r.variant,
                r.metric,
                r.baseline,
                r.current,
                pct,
                r.slack * 100.0,
            ));
        }
        out
    }
}

pub fn load_baseline(path: &Path) -> Result<Baseline, EvalError> {
    let bytes = std::fs::read(path).map_err(EvalError::from)?;
    serde_json::from_slice(&bytes).map_err(|e| EvalError::BaselineLoad {
        path: path.to_path_buf(),
        source: e,
    })
}

pub fn diff(report: &MatrixReport, baseline: &Baseline) -> Result<BaselineDiff, EvalError> {
    if report.schema_version != baseline.schema_version {
        return Err(EvalError::BaselineSchemaMismatch {
            baseline_version: baseline.schema_version,
            report_version: report.schema_version,
        });
    }

    let mut regressions = Vec::new();
    for base in &baseline.runs {
        let Some(run) = report.runs.iter().find(|r| r.variant == base.variant) else {
            eprintln!(
                "[baseline] skipping {:?} — not in current run",
                base.variant
            );
            continue;
        };

        check(
            &mut regressions,
            base.variant,
            "hit@5",
            base.hit_at_5,
            run.hit_at_5,
        );
        check(
            &mut regressions,
            base.variant,
            "MRR@10",
            base.mrr_at_10,
            run.mrr_at_10,
        );
    }
    Ok(BaselineDiff { regressions })
}

fn check(
    out: &mut Vec<Regression>,
    variant: ConfigVariant,
    metric: &'static str,
    baseline: f64,
    current: f64,
) {
    let threshold = baseline * (1.0 - DEFAULT_SLACK);
    if current < threshold {
        out.push(Regression {
            variant,
            metric,
            baseline,
            current,
            delta: current - baseline,
            slack: DEFAULT_SLACK,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::*;

    fn mk_report(primary_hit5: f64, primary_mrr: f64) -> MatrixReport {
        let zero_pct = LatencyPercentiles {
            total: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
            embed: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
            bm25: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
            hnsw: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
            rerank: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
            fuse: Percentiles {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
            },
        };
        MatrixReport {
            schema_version: 1,
            git_rev: "x".into(),
            captured_at: "x".into(),
            runs: vec![VariantReport {
                variant: ConfigVariant::Primary,
                hit_at_1: 0.0,
                hit_at_5: primary_hit5,
                hit_at_10: 0.0,
                mrr_at_10: primary_mrr,
                latency: zero_pct,
                per_question: vec![],
                buckets: Default::default(),
            }],
            rerank_delta: 0.0,
            contextual_delta: 0.0,
            hybrid_delta: 0.0,
        }
    }

    fn mk_baseline(primary_hit5: f64, primary_mrr: f64) -> Baseline {
        Baseline {
            schema_version: 1,
            git_rev: "x".into(),
            captured_at: "x".into(),
            runs: vec![VariantBaseline {
                variant: ConfigVariant::Primary,
                hit_at_5: primary_hit5,
                mrr_at_10: primary_mrr,
            }],
        }
    }

    #[test]
    fn exact_match_has_no_regressions() {
        let d = diff(&mk_report(0.82, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert!(!d.has_regressions());
    }

    #[test]
    fn exactly_two_percent_drop_passes_at_boundary() {
        // threshold = 0.82 * 0.98 = 0.8036
        // 0.8036 meets the threshold (>= comparison internally is `<` so we need > threshold)
        let d = diff(&mk_report(0.8036, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert!(
            !d.has_regressions(),
            "boundary should pass, got: {:?}",
            d.regressions
        );
    }

    #[test]
    fn just_past_two_percent_drop_is_a_regression() {
        let d = diff(&mk_report(0.80, 0.71), &mk_baseline(0.82, 0.71)).unwrap();
        assert_eq!(d.regressions.len(), 1);
        assert_eq!(d.regressions[0].metric, "hit@5");
    }

    #[test]
    fn schema_mismatch_fails_hard() {
        let mut r = mk_report(0.82, 0.71);
        r.schema_version = 2;
        let err = diff(&r, &mk_baseline(0.82, 0.71)).unwrap_err();
        assert!(format!("{err}").contains("schema"));
    }

    #[test]
    fn render_report_no_regressions_is_ok_line() {
        let d = BaselineDiff::default();
        assert!(d.render_report().contains("Baseline OK"));
    }

    #[test]
    fn partial_report_skips_missing_variants() {
        // Baseline has Primary + NoRerank; report only has Primary.
        let report = mk_report(0.82, 0.71);
        let baseline = Baseline {
            schema_version: 1,
            git_rev: "x".into(),
            captured_at: "x".into(),
            runs: vec![
                VariantBaseline {
                    variant: ConfigVariant::Primary,
                    hit_at_5: 0.82,
                    mrr_at_10: 0.71,
                },
                VariantBaseline {
                    variant: ConfigVariant::NoRerank,
                    hit_at_5: 0.75,
                    mrr_at_10: 0.65,
                },
            ],
        };
        let d = diff(&report, &baseline).expect("should not error on missing variant");
        assert!(!d.has_regressions());
    }

    #[test]
    fn render_report_with_regression_names_variant_and_metric() {
        let d = diff(&mk_report(0.79, 0.60), &mk_baseline(0.82, 0.71)).unwrap();
        let out = d.render_report();
        assert!(out.contains("Primary"));
        assert!(out.contains("hit@5"));
        assert!(out.contains("MRR@10"));
    }
}
