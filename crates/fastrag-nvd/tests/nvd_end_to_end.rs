//! Integration test: parse the 5-CVE fixture into Documents via MultiDocParser.

use fastrag_core::MultiDocParser;
use fastrag_nvd::NvdFeedParser;
use std::path::Path;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/nvd_slice.json")
}

#[test]
fn emits_five_documents_from_fixture() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).expect("parse must succeed");
    assert_eq!(docs.len(), 5, "expected 5 docs, got {}", docs.len());
}

#[test]
fn each_document_has_cve_id_in_extra() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    for doc in &docs {
        assert!(
            doc.metadata.extra.contains_key("cve_id"),
            "doc {:?} missing cve_id in extra",
            doc.metadata.title
        );
    }
}

#[test]
fn rejected_cve_metadata_preserves_status() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    let rejected = docs
        .iter()
        .find(|d| d.metadata.extra.get("cve_id").map(String::as_str) == Some("CVE-2024-10001"))
        .expect("CVE-2024-10001 must be present");
    assert_eq!(
        rejected
            .metadata
            .extra
            .get("vuln_status")
            .map(String::as_str),
        Some("Rejected")
    );
}

#[test]
fn disputed_cve_metadata_preserves_status() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    let disputed = docs
        .iter()
        .find(|d| d.metadata.extra.get("cve_id").map(String::as_str) == Some("CVE-2024-10002"))
        .expect("CVE-2024-10002 must be present");
    assert_eq!(
        disputed
            .metadata
            .extra
            .get("vuln_status")
            .map(String::as_str),
        Some("Disputed")
    );
}

#[test]
fn modified_cve_metadata_preserves_status() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    let modified = docs
        .iter()
        .find(|d| d.metadata.extra.get("cve_id").map(String::as_str) == Some("CVE-2023-44487"))
        .expect("CVE-2023-44487 must be present");
    assert_eq!(
        modified
            .metadata
            .extra
            .get("vuln_status")
            .map(String::as_str),
        Some("Modified")
    );
}

#[test]
fn log4shell_has_apache_vendor_in_metadata() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    let log4shell = docs
        .iter()
        .find(|d| d.metadata.extra.get("cve_id").map(String::as_str) == Some("CVE-2021-44228"))
        .expect("CVE-2021-44228 must be present");
    assert_eq!(
        log4shell
            .metadata
            .extra
            .get("cpe_vendor")
            .map(String::as_str),
        Some("apache")
    );
    assert_eq!(
        log4shell
            .metadata
            .extra
            .get("cpe_product")
            .map(String::as_str),
        Some("log4j")
    );
}

#[test]
fn status_distribution_all_four_statuses_present() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    let statuses: std::collections::HashSet<&str> = docs
        .iter()
        .filter_map(|d| d.metadata.extra.get("vuln_status").map(String::as_str))
        .collect();
    assert!(statuses.contains("Analyzed"), "Analyzed missing");
    assert!(statuses.contains("Rejected"), "Rejected missing");
    assert!(statuses.contains("Disputed"), "Disputed missing");
    assert!(statuses.contains("Modified"), "Modified missing");
}
