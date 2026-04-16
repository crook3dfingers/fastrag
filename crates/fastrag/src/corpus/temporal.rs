//! Per-query temporal decay policy: API types, detector trait, and
//! late-stage injection wrapper.
//!
//! See `docs/superpowers/specs/2026-04-16-query-conditional-temporal-decay-design.md`.

use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
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

// ---------------------------------------------------------------------------
// AbstainingRegexDetector
// ---------------------------------------------------------------------------

fn keyword_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\b(latest|newest|current(?:ly)?|newer)\b(?:\W+\w+){0,6}\W+(advisory|exploit|bypass|cve|disclosure|vulnerabilit(?:y|ies)|patch|guidance|kev|mitigation|poc)\b",
        )
        .unwrap()
    })
}

fn recent_plus_noun_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)\brecent(?:ly)?\b(?:\W+\w+){0,4}\W+(advisory|exploit|bypass|cve|disclosure|vulnerabilit(?:y|ies)|patch|guidance)\b",
        )
        .unwrap()
    })
}

fn still_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\bstill\s+(exploited|in\s+kev|vulnerable|unpatched)\b").unwrap()
    })
}

fn as_of_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)\bas\s+of\s+(today|now|this\s+(week|month))\b").unwrap())
}

fn this_week_month_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)\bthis\s+(week|month)\b").unwrap())
}

fn year_2026_plus_noun_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"(?i)(?:\b2026\b(?:\W+\w+){0,4}\W+(cve|vulnerabilit(?:y|ies)|advisory|disclosure|exploit|poc|mitigation|patch)\b|\b(cve|vulnerabilit(?:y|ies)|advisory|disclosure|exploit|poc|mitigation|patch)\b(?:\W+\w+){0,4}\W+\b2026\b)",
        )
        .unwrap()
    })
}

/// Recognises high-precision recency signals and returns
/// `FavorRecent(Medium)` on a hit, `Off` otherwise. Never emits `Auto` or
/// a historical policy — historical queries are simply `Off`.
pub struct AbstainingRegexDetector;

impl AbstainingRegexDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AbstainingRegexDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalDetector for AbstainingRegexDetector {
    fn detect(&self, query: &str) -> TemporalPolicy {
        // Mask CVE-YYYY-NNNN identifiers before applying the year pattern so
        // that bare identifiers like `CVE-2026-0001` do not trip the detector.
        // The other patterns are intentionally run on the raw query.
        let masked = mask_cve_identifiers(query);
        let fires = keyword_re().is_match(query)
            || recent_plus_noun_re().is_match(query)
            || still_re().is_match(query)
            || as_of_re().is_match(query)
            || this_week_month_re().is_match(query)
            || year_2026_plus_noun_re().is_match(&masked);
        if fires {
            TemporalPolicy::FavorRecent(Strength::Medium)
        } else {
            TemporalPolicy::Off
        }
    }
}

fn mask_cve_identifiers(query: &str) -> String {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)\bCVE-\d{4}-\d{4,}\b").unwrap())
        .replace_all(query, "___")
        .into_owned()
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
mod regex_positive_tests {
    use super::*;

    fn fires(query: &str) -> bool {
        matches!(
            AbstainingRegexDetector::new().detect(query),
            TemporalPolicy::FavorRecent(Strength::Medium)
        )
    }

    #[test]
    fn latest_keyword() {
        assert!(fires("latest Log4j advisory"));
        assert!(fires("newest CVE in KEV"));
        assert!(fires("current mitigation for Shellshock"));
        assert!(fires("newer bypass for PrintNightmare"));
    }

    #[test]
    fn recent_plus_security_noun() {
        assert!(fires("recent advisory for PyYAML"));
        assert!(fires("recently disclosed CVE in libxml2"));
        assert!(fires("recent patch for sudoedit"));
    }

    #[test]
    fn still_exploited_family() {
        assert!(fires("is Heartbleed still exploited in 2026"));
        assert!(fires("CVE-2021-44228 still in KEV"));
        assert!(fires("still unpatched on Ubuntu LTS"));
    }

    #[test]
    fn as_of_now_variants() {
        assert!(fires("as of today, what is the fix"));
        assert!(fires("as of now any known exploits"));
        assert!(fires("as of this week is it patched"));
        assert!(fires("as of this month how bad is it"));
    }

    #[test]
    fn this_week_this_month() {
        assert!(fires("what dropped this week"));
        assert!(fires("advisories this month"));
    }

    #[test]
    fn current_year_plus_security_noun() {
        assert!(fires("2026 CVE for libsqlite"));
        assert!(fires("2026 advisory NVD"));
        assert!(fires("2026 mitigation guidance"));
    }
}

#[cfg(test)]
mod regex_negative_tests {
    use super::*;

    fn abstains(query: &str) -> bool {
        matches!(
            AbstainingRegexDetector::new().detect(query),
            TemporalPolicy::Off
        )
    }

    #[test]
    fn historical_queries_abstain() {
        assert!(abstains("describe CVE-2014-0160"));
        assert!(abstains("as of 2014 how did Shellshock work"));
        assert!(abstains("in 2021 what was Log4Shell"));
        assert!(abstains("back in 2017 the WannaCry worm"));
    }

    #[test]
    fn neutral_queries_abstain() {
        assert!(abstains("explain Kerberoasting"));
        assert!(abstains("what is NTLM relay"));
        assert!(abstains("how does ASLR work"));
    }

    #[test]
    fn bare_cve_identifier_abstains() {
        assert!(abstains("CVE-2026-0001"));
        assert!(abstains("CVE-2014-6271"));
    }

    #[test]
    fn bare_2026_abstains_without_noun() {
        assert!(abstains("port 2026 scan"));
        assert!(abstains("version 2026"));
        assert!(abstains("2026"));
    }

    #[test]
    fn false_friends_abstain() {
        // `current` without a security noun within 6 tokens
        assert!(abstains("current user guide"));
        assert!(abstains("current working directory"));
        // `latest` unrelated to freshness / security
        assert!(abstains("latest attempt to install X"));
        assert!(abstains("what is the latest episode about"));
        // bare `recent` / `recently` without a following security noun
        assert!(abstains("recently I was thinking"));
        assert!(abstains("a recent commit"));
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
