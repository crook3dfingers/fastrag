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
    pub date_fields: Vec<String>,
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

/// Apply decay to every `ScoredId`. `dates` must be the same length as `fused`
/// (index-aligned). `None` entries get the dateless prior.
///
/// Returns a new vector sorted by descending final score.
pub fn apply_decay(
    fused: &[ScoredId],
    dates: &[Option<NaiveDate>],
    opts: &TemporalOpts,
) -> Vec<ScoredId> {
    assert_eq!(
        fused.len(),
        dates.len(),
        "apply_decay: fused and dates must be index-aligned"
    );
    if fused.is_empty() {
        return Vec::new();
    }

    let halflife_days = (opts.halflife.as_secs_f32() / 86_400.0).max(f32::EPSILON);
    let now_date = opts.now.date_naive();
    let alpha = opts.weight_floor;

    let mut out: Vec<ScoredId> = match opts.blend {
        BlendMode::Multiplicative => fused
            .iter()
            .zip(dates.iter())
            .map(|(hit, date)| {
                let age = date.map(|d| (now_date - d).num_days() as f32);
                let factor =
                    decay_factor(age, halflife_days, alpha, opts.dateless_prior, opts.blend);
                ScoredId {
                    id: hit.id,
                    score: hit.score * factor,
                }
            })
            .collect(),
        BlendMode::Additive => {
            // Min-max normalize RRF scores within the candidate set.
            // When all candidates have identical RRF score, every normalized
            // score is 0.5 (neutral) so the decay term decides ordering.
            let max = fused
                .iter()
                .map(|s| s.score)
                .fold(f32::NEG_INFINITY, f32::max);
            let min = fused.iter().map(|s| s.score).fold(f32::INFINITY, f32::min);
            let span = (max - min).max(f32::EPSILON);
            let normalize = |s: f32| -> f32 {
                if (max - min).abs() < f32::EPSILON {
                    0.5
                } else {
                    (s - min) / span
                }
            };
            let ln2: f32 = std::f32::consts::LN_2;
            fused
                .iter()
                .zip(dates.iter())
                .map(|(hit, date)| {
                    let norm_rrf = normalize(hit.score);
                    let decay_term = match date {
                        None => opts.dateless_prior,
                        Some(d) => {
                            let age = ((now_date - *d).num_days() as f32).max(0.0);
                            (-ln2 * age / halflife_days).exp()
                        }
                    };
                    let final_score = alpha * norm_rrf + (1.0 - alpha) * decay_term;
                    ScoredId {
                        id: hit.id,
                        score: final_score,
                    }
                })
                .collect()
        }
    };

    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod apply_decay_tests {
    use super::*;
    use chrono::TimeZone;

    fn opts(halflife_days: u64, alpha: f32, prior: f32) -> TemporalOpts {
        TemporalOpts {
            date_fields: vec!["published_date".into()],
            halflife: Duration::from_secs(halflife_days * 86_400),
            weight_floor: alpha,
            dateless_prior: prior,
            blend: BlendMode::Multiplicative,
            now: Utc.with_ymd_and_hms(2026, 4, 14, 0, 0, 0).unwrap(),
        }
    }

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn uniform_age_preserves_order() {
        let fused = vec![
            ScoredId { id: 1, score: 0.9 },
            ScoredId { id: 2, score: 0.8 },
            ScoredId { id: 3, score: 0.7 },
        ];
        let same = Some(ymd(2026, 4, 1));
        let out = apply_decay(&fused, &[same, same, same], &opts(30, 0.3, 0.5));
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn fresh_outranks_equally_relevant_stale() {
        // Both have rrf_score 0.8; one is today, one is 10 years old.
        let fused = vec![
            ScoredId { id: 1, score: 0.8 }, // stale
            ScoredId { id: 2, score: 0.8 }, // fresh
        ];
        let dates = vec![Some(ymd(2016, 4, 14)), Some(ymd(2026, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts(30, 0.3, 0.5));
        assert_eq!(out[0].id, 2, "fresh must rank first; got {out:?}");
        assert_eq!(out[1].id, 1);
    }

    #[test]
    fn dateless_interleaves_at_neutral_prior() {
        // halflife=30d, alpha=0.3, prior=0.5.
        // id=1 fresh -> factor=1.0 -> 1.0
        // id=2 dateless -> factor=0.5 -> 0.5
        // id=3 very stale (5y) -> factor→0.3 -> 0.3
        let fused = vec![
            ScoredId { id: 1, score: 1.0 },
            ScoredId { id: 2, score: 1.0 },
            ScoredId { id: 3, score: 1.0 },
        ];
        let dates = vec![Some(ymd(2026, 4, 14)), None, Some(ymd(2021, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts(30, 0.3, 0.5));
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = apply_decay(&[], &[], &opts(30, 0.3, 0.5));
        assert!(out.is_empty());
    }

    fn opts_additive(halflife_days: u64, alpha: f32, prior: f32) -> TemporalOpts {
        let mut o = opts(halflife_days, alpha, prior);
        o.blend = BlendMode::Additive;
        o
    }

    #[test]
    fn additive_fresh_beats_stale_with_equal_relevance() {
        // All rrf identical → normalized to 0.5. Decay term dominates.
        let fused = vec![
            ScoredId { id: 1, score: 0.8 }, // stale
            ScoredId { id: 2, score: 0.8 }, // fresh
        ];
        let dates = vec![Some(ymd(2016, 4, 14)), Some(ymd(2026, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts_additive(30, 0.3, 0.5));
        assert_eq!(out[0].id, 2, "fresh must rank first in additive mode");
        assert_eq!(out[1].id, 1);
    }

    #[test]
    fn additive_high_relevance_can_outrank_fresh_but_less_relevant() {
        // alpha=0.7 — semantic weight heavy. Stale-but-relevant beats fresh-but-weak.
        // id=1: rrf=1.0 (norm=1.0), stale → final = 0.7*1.0 + 0.3*~0 = 0.70
        // id=2: rrf=0.1 (norm=0.0), fresh → final = 0.7*0.0 + 0.3*1.0 = 0.30
        let fused = vec![
            ScoredId { id: 1, score: 1.0 },
            ScoredId { id: 2, score: 0.1 },
        ];
        let dates = vec![Some(ymd(2016, 4, 14)), Some(ymd(2026, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts_additive(30, 0.7, 0.5));
        assert_eq!(
            out[0].id, 1,
            "high-relevance stale beats low-relevance fresh under alpha=0.7"
        );
    }

    #[test]
    fn additive_dateless_uses_prior_as_decay_term() {
        // rrf all equal → normalized to 0.5.
        // id=1 fresh: final = 0.5*0.5 + 0.5*1.0 = 0.75
        // id=2 dateless (prior=0.5): final = 0.5*0.5 + 0.5*0.5 = 0.50
        // id=3 stale (~decay≈0): final = 0.5*0.5 + 0.5*~0 = 0.25
        let fused = vec![
            ScoredId { id: 1, score: 1.0 },
            ScoredId { id: 2, score: 1.0 },
            ScoredId { id: 3, score: 1.0 },
        ];
        let dates = vec![Some(ymd(2026, 4, 14)), None, Some(ymd(2016, 4, 14))];
        let out = apply_decay(&fused, &dates, &opts_additive(30, 0.5, 0.5));
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    }
}

/// Extract a `NaiveDate` for one row of metadata by locating the named field.
/// Returns `None` when the field is absent or the value isn't a `Date`.
pub fn extract_date(
    fields: &[(String, fastrag_store::schema::TypedValue)],
    date_field: &str,
) -> Option<NaiveDate> {
    fields.iter().find_map(|(k, v)| {
        if k == date_field {
            match v {
                fastrag_store::schema::TypedValue::Date(d) => Some(*d),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// Try each field name in `date_fields` in order, returning the first `Date`
/// value found. Enables coalesce semantics: `["last_modified", "published_date"]`
/// picks `last_modified` when present, falls back to `published_date`.
pub fn extract_date_coalesce(
    fields: &[(String, fastrag_store::schema::TypedValue)],
    date_fields: &[String],
) -> Option<NaiveDate> {
    date_fields
        .iter()
        .find_map(|name| extract_date(fields, name))
}

#[cfg(test)]
mod extract_date_tests {
    use super::*;
    use fastrag_store::schema::TypedValue;

    fn field(name: &str, v: TypedValue) -> (String, TypedValue) {
        (name.to_string(), v)
    }

    #[test]
    fn returns_date_when_field_present() {
        let rows = vec![
            field("other", TypedValue::String("x".into())),
            field(
                "published_date",
                TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
            ),
        ];
        let d = extract_date(&rows, "published_date");
        assert_eq!(d, NaiveDate::from_ymd_opt(2024, 6, 1));
    }

    #[test]
    fn returns_none_when_field_missing() {
        let rows = vec![field("other", TypedValue::String("x".into()))];
        assert_eq!(extract_date(&rows, "published_date"), None);
    }

    #[test]
    fn returns_none_when_field_wrong_type() {
        let rows = vec![field(
            "published_date",
            TypedValue::String("2024-06-01".into()),
        )];
        assert_eq!(extract_date(&rows, "published_date"), None);
    }

    #[test]
    fn coalesce_returns_first_present_field() {
        let rows = vec![
            field(
                "last_modified",
                TypedValue::Date(NaiveDate::from_ymd_opt(2025, 1, 15).unwrap()),
            ),
            field(
                "published_date",
                TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
            ),
        ];
        let d = extract_date_coalesce(&rows, &["last_modified".into(), "published_date".into()]);
        assert_eq!(d, NaiveDate::from_ymd_opt(2025, 1, 15));
    }

    #[test]
    fn coalesce_falls_through_missing_to_second() {
        let rows = vec![field(
            "published_date",
            TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
        )];
        let d = extract_date_coalesce(&rows, &["last_modified".into(), "published_date".into()]);
        assert_eq!(d, NaiveDate::from_ymd_opt(2024, 6, 1));
    }

    #[test]
    fn coalesce_returns_none_when_no_field_matches() {
        let rows = vec![field("other", TypedValue::String("x".into()))];
        let d = extract_date_coalesce(&rows, &["last_modified".into(), "published_date".into()]);
        assert_eq!(d, None);
    }

    #[test]
    fn coalesce_single_field_matches_extract_date() {
        let rows = vec![field(
            "published_date",
            TypedValue::Date(NaiveDate::from_ymd_opt(2024, 6, 1).unwrap()),
        )];
        assert_eq!(
            extract_date_coalesce(&rows, &["published_date".into()]),
            extract_date(&rows, "published_date"),
        );
    }
}

/// Build [`HybridOpts`] from flat flag values. Shared entry point for CLI, HTTP,
/// and MCP surfaces. `blend_str` must be either `"multiplicative"` or `"additive"`.
/// `halflife_str` is parsed with `humantime::parse_duration` (e.g. `"30d"`, `"1y"`).
/// Returns a user-facing `String` error on validation failure.
#[allow(clippy::too_many_arguments)]
pub fn build_hybrid_opts_from_parts(
    hybrid: bool,
    rrf_k: u32,
    rrf_overfetch: usize,
    time_decay_field: Option<String>,
    time_decay_halflife: &str,
    time_decay_weight: f32,
    time_decay_dateless_prior: f32,
    time_decay_blend: &str,
) -> Result<HybridOpts, String> {
    // Reject decay-knob changes without a date field.
    let has_decay_params_without_field = time_decay_field.is_none()
        && (time_decay_halflife != "30d"
            || time_decay_weight != 0.3
            || time_decay_dateless_prior != 0.5);
    if has_decay_params_without_field {
        return Err(
            "time_decay_halflife / _weight / _dateless_prior require time_decay_field".to_string(),
        );
    }

    // Validate blend string.
    let blend = match time_decay_blend {
        "multiplicative" => BlendMode::Multiplicative,
        "additive" => BlendMode::Additive,
        other => {
            return Err(format!(
                "time_decay_blend: expected 'multiplicative' or 'additive', got {other:?}"
            ));
        }
    };

    // Parse halflife + build TemporalOpts only if a date field was supplied.
    let temporal = if let Some(field) = time_decay_field {
        let halflife = humantime::parse_duration(time_decay_halflife)
            .map_err(|e| format!("time_decay_halflife: {e}"))?;
        Some(TemporalOpts {
            date_fields: field.split(',').map(|s| s.trim().to_string()).collect(),
            halflife,
            weight_floor: time_decay_weight,
            dateless_prior: time_decay_dateless_prior,
            blend,
            now: chrono::Utc::now(),
        })
    } else {
        None
    };

    let enabled = hybrid || temporal.is_some();
    Ok(HybridOpts {
        enabled,
        rrf_k,
        overfetch_factor: rrf_overfetch,
        temporal,
    })
}

#[cfg(test)]
mod build_hybrid_opts_tests {
    use super::*;

    #[test]
    fn build_hybrid_opts_defaults_dense_only() {
        let opts =
            build_hybrid_opts_from_parts(false, 60, 4, None, "30d", 0.3, 0.5, "multiplicative")
                .unwrap();
        assert!(
            !opts.enabled,
            "enabled must be false for dense-only defaults"
        );
        assert!(opts.temporal.is_none(), "temporal must be None");
        assert_eq!(opts.rrf_k, 60);
        assert_eq!(opts.overfetch_factor, 4);
    }

    #[test]
    fn build_hybrid_opts_bad_blend_errors() {
        let result = build_hybrid_opts_from_parts(false, 60, 4, None, "30d", 0.3, 0.5, "bogus");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("expected 'multiplicative' or 'additive'"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn build_hybrid_opts_halflife_without_field_errors() {
        let result =
            build_hybrid_opts_from_parts(false, 60, 4, None, "7d", 0.3, 0.5, "multiplicative");
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("require time_decay_field"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn build_hybrid_opts_full_additive() {
        let opts = build_hybrid_opts_from_parts(
            true,
            60,
            4,
            Some("published_date".to_string()),
            "14d",
            0.4,
            0.6,
            "additive",
        )
        .unwrap();
        assert!(opts.enabled, "enabled must be true when hybrid=true");
        let temporal = opts.temporal.expect("temporal must be Some");
        assert_eq!(temporal.blend, BlendMode::Additive);
        assert_eq!(
            temporal.halflife,
            std::time::Duration::from_secs(14 * 86400)
        );
        assert_eq!(temporal.date_fields, vec!["published_date".to_string()]);
        assert!(
            (temporal.weight_floor - 0.4).abs() < 1e-6,
            "weight_floor mismatch"
        );
        assert!(
            (temporal.dateless_prior - 0.6).abs() < 1e-6,
            "dateless_prior mismatch"
        );
    }

    #[test]
    fn build_hybrid_opts_comma_separated_date_fields() {
        let opts = build_hybrid_opts_from_parts(
            false,
            60,
            4,
            Some("last_modified, published_date".to_string()),
            "30d",
            0.3,
            0.5,
            "multiplicative",
        )
        .unwrap();
        let temporal = opts.temporal.expect("temporal must be Some");
        assert_eq!(
            temporal.date_fields,
            vec!["last_modified".to_string(), "published_date".to_string()]
        );
    }
}

/// BM25 + dense candidate fetch, unweighted RRF, optional post-fusion decay.
///
/// Fetches `overfetch_factor * top_k` from each retriever (minimum `top_k`),
/// fuses via RRF(k=rrf_k), optionally applies temporal decay, sorts, truncates
/// to `top_k`.
///
/// Timing: populates `breakdown.bm25_us`, `breakdown.hnsw_us`, `breakdown.fuse_us`.
/// Caller is responsible for `embed_us` and `breakdown.finalize()`.
#[cfg(feature = "store")]
#[allow(clippy::too_many_arguments)]
pub fn query_hybrid(
    store: &fastrag_store::Store,
    query: &str,
    vector: &[f32],
    top_k: usize,
    opts: &HybridOpts,
    breakdown: &mut crate::corpus::LatencyBreakdown,
) -> Result<Vec<ScoredId>, CorpusError> {
    use std::time::Instant;

    let fetch_count = top_k
        .saturating_mul(opts.overfetch_factor.max(1))
        .max(top_k);

    let t = Instant::now();
    let bm25 = store.query_bm25(query, fetch_count)?;
    breakdown.bm25_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    let dense = store.query_dense(vector, fetch_count)?;
    breakdown.hnsw_us = t.elapsed().as_micros() as u64;

    let bm25_sids: Vec<ScoredId> = bm25
        .iter()
        .map(|(id, score)| ScoredId {
            id: *id,
            score: *score,
        })
        .collect();
    let dense_sids: Vec<ScoredId> = dense
        .iter()
        .map(|(id, score)| ScoredId {
            id: *id,
            score: *score,
        })
        .collect();

    let t = Instant::now();
    let mut fused = rrf_fuse(&[&bm25_sids, &dense_sids], opts.rrf_k);
    breakdown.fuse_us = t.elapsed().as_micros() as u64;

    if let Some(temp) = &opts.temporal {
        let ids: Vec<u64> = fused.iter().map(|s| s.id).collect();
        let rows = store.fetch_metadata(&ids)?;
        let row_map: std::collections::HashMap<
            u64,
            Vec<(String, fastrag_store::schema::TypedValue)>,
        > = rows.into_iter().collect();
        let dates: Vec<Option<NaiveDate>> = fused
            .iter()
            .map(|s| {
                row_map
                    .get(&s.id)
                    .and_then(|f| extract_date_coalesce(f, &temp.date_fields))
            })
            .collect();
        fused = apply_decay(&fused, &dates, temp);
    }

    fused.truncate(top_k);
    Ok(fused)
}

#[cfg(all(test, feature = "store"))]
mod query_hybrid_tests {
    use super::*;
    use crate::corpus::LatencyBreakdown;
    use fastrag_embed::{Canary, EmbedderIdentity, PrefixScheme};
    use fastrag_index::{CorpusManifest, ManifestChunkingStrategy};
    use fastrag_store::schema::DynamicSchema;
    use fastrag_store::{ChunkRecord, Store};
    use tempfile::TempDir;

    fn test_manifest() -> CorpusManifest {
        CorpusManifest::new(
            EmbedderIdentity {
                model_id: "test/stub-3d-v1".into(),
                dim: 3,
                prefix_scheme_hash: PrefixScheme::NONE.hash(),
            },
            Canary {
                text_version: 1,
                vector: vec![1.0, 0.0, 0.0],
            },
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
        )
    }

    fn chunk(id: u64, text: &str, vector: Vec<f32>) -> ChunkRecord {
        ChunkRecord {
            id,
            external_id: format!("ext-{id}"),
            content_hash: format!("hash-{id}"),
            chunk_index: 0,
            source_path: format!("/src/{id}.md"),
            source_json: None,
            chunk_text: text.to_string(),
            vector,
            user_fields: vec![],
        }
    }

    fn fixture() -> (TempDir, Store) {
        let dir = TempDir::new().unwrap();
        let mut store = Store::create(dir.path(), test_manifest(), DynamicSchema::new()).unwrap();
        store
            .add_records(vec![
                chunk(1, "alpha beta gamma", vec![1.0, 0.0, 0.0]),
                chunk(2, "delta epsilon zeta", vec![0.0, 1.0, 0.0]),
                chunk(3, "alpha delta eta", vec![0.7, 0.7, 0.0]),
            ])
            .unwrap();
        (dir, store)
    }

    #[test]
    fn hybrid_reorders_vs_dense_only() {
        let (_dir, store) = fixture();
        // Dense-only on qvec=[1,0,0] would rank id=1 first (exact match).
        // BM25 on "delta eta" matches id=3 (both tokens) and id=2 (only
        // "delta"); id=1 is absent from BM25. Fused winner must be id=3.
        let qvec = vec![1.0, 0.0, 0.0];

        let mut bd = LatencyBreakdown::default();
        let opts = HybridOpts {
            enabled: true,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: None,
        };
        let out = query_hybrid(&store, "delta eta", &qvec, 3, &opts, &mut bd).unwrap();

        assert_eq!(out.len(), 3, "should return 3 fused ids");
        assert_eq!(out[0].id, 3, "lexical overlap winner first; got {out:?}");
    }

    #[test]
    fn temporal_option_runs_decay_branch() {
        let (_dir, store) = fixture();
        let qvec = vec![1.0, 0.0, 0.0];

        let mut bd = LatencyBreakdown::default();
        let opts = HybridOpts {
            enabled: true,
            rrf_k: 60,
            overfetch_factor: 4,
            temporal: Some(TemporalOpts {
                date_fields: vec!["published_date".into()],
                halflife: Duration::from_secs(30 * 86400),
                weight_floor: 0.3,
                dateless_prior: 0.5,
                blend: BlendMode::Multiplicative,
                now: Utc::now(),
            }),
        };
        let out = query_hybrid(&store, "alpha", &qvec, 3, &opts, &mut bd).unwrap();
        assert!(!out.is_empty(), "decay branch returned empty");
    }
}
