//! Per-axis bucket aggregates computed from a variant's `per_question` list.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::gold_set::{Axes, GoldSet, GoldSetEntry};
use crate::matrix::QuestionResult;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketMetrics {
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub hit_at_10: f64,
    pub mrr_at_10: f64,
    pub n: usize,
}

/// Compute per-axis buckets from a variant's per-question results, keyed by
/// axis name then axis value string (snake_case). Empty buckets are omitted.
pub fn compute_buckets(
    per_question: &[QuestionResult],
    gold: &GoldSet,
) -> BTreeMap<String, BTreeMap<String, BucketMetrics>> {
    let by_id: std::collections::HashMap<&str, &GoldSetEntry> =
        gold.entries.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut groups: BTreeMap<(&'static str, String), Vec<&QuestionResult>> = BTreeMap::new();
    for q in per_question {
        let Some(entry) = by_id.get(q.id.as_str()) else {
            continue;
        };
        let Axes {
            style,
            temporal_intent,
        } = entry.axes;
        let style_key = serde_json::to_value(style)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let ti_key = serde_json::to_value(temporal_intent)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        groups.entry(("style", style_key)).or_default().push(q);
        groups
            .entry(("temporal_intent", ti_key))
            .or_default()
            .push(q);
    }
    let mut out: BTreeMap<String, BTreeMap<String, BucketMetrics>> = BTreeMap::new();
    for ((axis, value), results) in groups {
        let n = results.len();
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let m = BucketMetrics {
            hit_at_1: results.iter().filter(|q| q.hit_at_1).count() as f64 / nf,
            hit_at_5: results.iter().filter(|q| q.hit_at_5).count() as f64 / nf,
            hit_at_10: results.iter().filter(|q| q.hit_at_10).count() as f64 / nf,
            mrr_at_10: results.iter().map(|q| q.reciprocal_rank).sum::<f64>() / nf,
            n,
        };
        out.entry(axis.to_string()).or_default().insert(value, m);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gold_set::{Axes, GoldSet, GoldSetEntry, Style, TemporalIntent};
    use crate::matrix::QuestionResult;
    use fastrag::corpus::LatencyBreakdown;

    fn entry(id: &str, style: Style, ti: TemporalIntent) -> GoldSetEntry {
        GoldSetEntry {
            id: id.into(),
            question: "q".into(),
            must_contain_cve_ids: vec![],
            must_contain_terms: vec!["x".into()],
            notes: None,
            axes: Axes {
                style,
                temporal_intent: ti,
            },
        }
    }

    fn qr(id: &str, h1: bool, h5: bool, rr: f64) -> QuestionResult {
        QuestionResult {
            id: id.into(),
            hit_at_1: h1,
            hit_at_5: h5,
            hit_at_10: h5,
            reciprocal_rank: rr,
            missing_cve_ids: vec![],
            missing_terms: vec![],
            latency_us: LatencyBreakdown::default(),
        }
    }

    #[test]
    fn computes_per_axis_aggregates() {
        let gold = GoldSet {
            version: 1,
            entries: vec![
                entry("a", Style::Identifier, TemporalIntent::Neutral),
                entry("b", Style::Identifier, TemporalIntent::Historical),
                entry("c", Style::Conceptual, TemporalIntent::Neutral),
            ],
        };
        let per_q = vec![
            qr("a", true, true, 1.0),
            qr("b", false, true, 0.5),
            qr("c", false, false, 0.0),
        ];
        let buckets = compute_buckets(&per_q, &gold);
        let style = buckets.get("style").unwrap();
        let ident = style.get("identifier").unwrap();
        assert_eq!(ident.n, 2);
        assert!((ident.hit_at_1 - 0.5).abs() < 1e-9);
        assert!((ident.hit_at_5 - 1.0).abs() < 1e-9);
        assert!((ident.mrr_at_10 - 0.75).abs() < 1e-9);
        let conc = style.get("conceptual").unwrap();
        assert_eq!(conc.n, 1);
        assert_eq!(conc.hit_at_5, 0.0);

        let ti = buckets.get("temporal_intent").unwrap();
        assert_eq!(ti.get("neutral").unwrap().n, 2);
        assert_eq!(ti.get("historical").unwrap().n, 1);
    }

    #[test]
    fn empty_buckets_are_omitted() {
        let gold = GoldSet {
            version: 1,
            entries: vec![entry("a", Style::Identifier, TemporalIntent::Neutral)],
        };
        let per_q = vec![qr("a", true, true, 1.0)];
        let buckets = compute_buckets(&per_q, &gold);
        let style = buckets.get("style").unwrap();
        assert!(style.get("conceptual").is_none());
        assert!(style.get("mixed").is_none());
    }

    #[test]
    fn unknown_question_id_is_skipped() {
        let gold = GoldSet {
            version: 1,
            entries: vec![entry("a", Style::Identifier, TemporalIntent::Neutral)],
        };
        let per_q = vec![qr("unknown", true, true, 1.0), qr("a", false, true, 0.5)];
        let buckets = compute_buckets(&per_q, &gold);
        let style = buckets.get("style").unwrap();
        assert_eq!(style.get("identifier").unwrap().n, 1);
    }
}
