//! BoilerplateStripper: regex-driven strip pass for NVD description noise.
//!
//! Patterns stripped (in order):
//!   1. NVD reject/dispute markers: `** REJECT **` and `** DISPUTED **`
//!      (NVD 2.0 API inserts these prefixes on affected records)
//!   2. CPE 2.3 URI literals: `cpe:2.3:[aho*]:...` — noisy in descriptions
//!   3. Standalone URL lines: lines that are only a URL (http/https/ftp)
//!   4. NVD legal notice block: the standard "NOTE: ..." trailer pattern

use std::collections::BTreeMap;
use std::sync::OnceLock;

use regex::Regex;

use super::ChunkFilter;

static REJECT_MARKER: OnceLock<Regex> = OnceLock::new();
static DISPUTED_MARKER: OnceLock<Regex> = OnceLock::new();
static CPE_URI: OnceLock<Regex> = OnceLock::new();
static URL_ONLY_LINE: OnceLock<Regex> = OnceLock::new();
static LEGAL_NOTICE: OnceLock<Regex> = OnceLock::new();

fn reject_marker() -> &'static Regex {
    REJECT_MARKER.get_or_init(|| Regex::new(r"\*\*\s*REJECT\s*\*\*").unwrap())
}
fn disputed_marker() -> &'static Regex {
    DISPUTED_MARKER.get_or_init(|| Regex::new(r"\*\*\s*DISPUTED\s*\*\*").unwrap())
}
fn cpe_uri() -> &'static Regex {
    CPE_URI.get_or_init(|| Regex::new(r"cpe:2\.3:[aho\*]:[^\s]+").unwrap())
}
fn url_only_line() -> &'static Regex {
    URL_ONLY_LINE.get_or_init(|| Regex::new(r"(?m)^\s*https?://\S+\s*$").unwrap())
}
fn legal_notice() -> &'static Regex {
    LEGAL_NOTICE
        .get_or_init(|| Regex::new(r"(?i)NOTE:\s+Links?\s+are\s+provided.*?(\n|$)").unwrap())
}

/// Strips NVD boilerplate from chunk text.
pub struct BoilerplateStripper;

impl ChunkFilter for BoilerplateStripper {
    fn apply(&self, text: &str, _metadata: &BTreeMap<String, String>) -> String {
        let s = reject_marker().replace_all(text, "");
        let s = disputed_marker().replace_all(&s, "");
        let s = cpe_uri().replace_all(&s, "");
        let s = url_only_line().replace_all(&s, "");
        let s = legal_notice().replace_all(&s, "");
        // Collapse runs of blank lines left by stripping.
        s.lines()
            .fold((String::new(), false), |(mut acc, was_blank), line| {
                let blank = line.trim().is_empty();
                if blank && was_blank {
                    (acc, true)
                } else {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(line);
                    (acc, blank)
                }
            })
            .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(text: &str) -> String {
        BoilerplateStripper.apply(text, &BTreeMap::new())
    }

    // --- Positive cases (should be removed) ---

    #[test]
    fn strips_reject_marker() {
        let input = "** REJECT ** This CVE was issued in error.";
        let out = strip(input);
        assert!(
            !out.contains("REJECT"),
            "marker must be removed; got: {out}"
        );
        assert!(
            out.contains("This CVE was issued in error."),
            "prose must survive"
        );
    }

    #[test]
    fn strips_disputed_marker() {
        let input = "** DISPUTED ** The vendor disputes the severity.";
        let out = strip(input);
        assert!(!out.contains("DISPUTED"));
        assert!(out.contains("The vendor disputes the severity."));
    }

    #[test]
    fn strips_cpe_uri() {
        let input = "Affects cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:* and cpe:2.3:o:vendor:product:1.0:*:*:*:*:*:*:*.";
        let out = strip(input);
        assert!(!out.contains("cpe:2.3:"), "CPE URIs must be removed");
        assert!(out.contains("Affects"), "surrounding text must survive");
    }

    #[test]
    fn strips_standalone_url_line() {
        let input = "See the advisory.\nhttps://example.com/advisory/2024-001\nFor details.";
        let out = strip(input);
        assert!(!out.contains("https://"), "URL-only line must be removed");
        assert!(out.contains("See the advisory."));
        assert!(out.contains("For details."));
    }

    // --- Negative cases (must NOT be modified) ---

    #[test]
    fn preserves_normal_prose() {
        let input = "A heap-based buffer overflow allows remote code execution via crafted input.";
        let out = strip(input);
        assert_eq!(out.trim(), input.trim());
    }

    #[test]
    fn preserves_url_embedded_in_sentence() {
        let input = "See https://example.com/cve for more details on the vulnerability.";
        let out = strip(input);
        // URL is part of a sentence — the line is not URL-only, must survive.
        assert!(out.contains("https://example.com/cve"));
    }

    #[test]
    fn preserves_cve_id_references() {
        let input = "This is related to CVE-2021-44228 (Log4Shell).";
        let out = strip(input);
        assert!(out.contains("CVE-2021-44228"));
    }
}
