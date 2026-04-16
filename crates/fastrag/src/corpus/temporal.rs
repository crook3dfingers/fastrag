//! Per-query temporal decay policy: API types, detector trait, and
//! late-stage injection wrapper.
//!
//! See `docs/superpowers/specs/2026-04-16-query-conditional-temporal-decay-design.md`.

use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strength {
    Light,
    Medium,
    Strong,
}

impl Strength {
    pub fn halflife(self) -> Duration {
        match self {
            Strength::Light => Duration::from_secs(365 * 86_400),
            Strength::Medium => Duration::from_secs(180 * 86_400),
            Strength::Strong => Duration::from_secs(60 * 86_400),
        }
    }

    pub fn weight_floor(self) -> f32 {
        match self {
            Strength::Light => 0.75,
            Strength::Medium => 0.60,
            Strength::Strong => 0.45,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPolicy {
    #[default]
    Auto,
    Off,
    FavorRecent(Strength),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalIntent {
    Historical,
    Neutral,
    RecencySeeking,
}

pub trait TemporalDetector: Send + Sync {
    fn detect(&self, query: &str) -> TemporalPolicy;
}

pub struct OracleDetector {
    intent: Option<TemporalIntent>,
}

impl OracleDetector {
    pub fn new(intent: Option<TemporalIntent>) -> Self {
        Self { intent }
    }
}

impl TemporalDetector for OracleDetector {
    fn detect(&self, _query: &str) -> TemporalPolicy {
        match self.intent {
            Some(TemporalIntent::RecencySeeking) => TemporalPolicy::FavorRecent(Strength::Medium),
            _ => TemporalPolicy::Off,
        }
    }
}

#[cfg(test)]
mod policy_serde_tests {
    use super::*;

    #[test]
    fn auto_serializes_as_string() {
        let v = TemporalPolicy::Auto;
        assert_eq!(serde_json::to_string(&v).unwrap(), "\"auto\"");
    }

    #[test]
    fn off_serializes_as_string() {
        let v = TemporalPolicy::Off;
        assert_eq!(serde_json::to_string(&v).unwrap(), "\"off\"");
    }

    #[test]
    fn favor_recent_serializes_as_tagged_object() {
        let v = TemporalPolicy::FavorRecent(Strength::Medium);
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"favor_recent":"medium"}"#
        );
    }

    #[test]
    fn auto_is_default() {
        let v: TemporalPolicy = Default::default();
        assert!(matches!(v, TemporalPolicy::Auto));
    }

    #[test]
    fn deserialize_round_trip() {
        for p in [
            TemporalPolicy::Auto,
            TemporalPolicy::Off,
            TemporalPolicy::FavorRecent(Strength::Light),
            TemporalPolicy::FavorRecent(Strength::Medium),
            TemporalPolicy::FavorRecent(Strength::Strong),
        ] {
            let s = serde_json::to_string(&p).unwrap();
            let back: TemporalPolicy = serde_json::from_str(&s).unwrap();
            assert_eq!(p, back);
        }
    }
}

#[cfg(test)]
mod strength_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn light_constants() {
        assert_eq!(
            Strength::Light.halflife(),
            Duration::from_secs(365 * 86_400)
        );
        assert!((Strength::Light.weight_floor() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn medium_constants() {
        assert_eq!(
            Strength::Medium.halflife(),
            Duration::from_secs(180 * 86_400)
        );
        assert!((Strength::Medium.weight_floor() - 0.60).abs() < 1e-6);
    }

    #[test]
    fn strong_constants() {
        assert_eq!(
            Strength::Strong.halflife(),
            Duration::from_secs(60 * 86_400)
        );
        assert!((Strength::Strong.weight_floor() - 0.45).abs() < 1e-6);
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;

    #[test]
    fn recency_seeking_routes_to_medium_favor_recent() {
        let d = OracleDetector::new(Some(TemporalIntent::RecencySeeking));
        match d.detect("any query text") {
            TemporalPolicy::FavorRecent(Strength::Medium) => {}
            other => panic!("expected FavorRecent(Medium), got {other:?}"),
        }
    }

    #[test]
    fn neutral_routes_to_off() {
        let d = OracleDetector::new(Some(TemporalIntent::Neutral));
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }

    #[test]
    fn historical_routes_to_off() {
        let d = OracleDetector::new(Some(TemporalIntent::Historical));
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }

    #[test]
    fn missing_intent_routes_to_off() {
        let d = OracleDetector::new(None);
        assert_eq!(d.detect("any query"), TemporalPolicy::Off);
    }
}
