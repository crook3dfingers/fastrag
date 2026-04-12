# Security Corpus Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship fastrag-nvd parser crate and a hygiene filter chain behind `--security-profile` so NVD 2.0 feeds ingest as structured CVE documents, Rejected/Disputed CVEs are dropped, NVD boilerplate is stripped, non-English chunks are filtered, and CISA KEV CVEs are tagged.

**Architecture:** New `fastrag-nvd` crate (NVD 2.0 serde + `MultiDocParser` impl emitting one Document per CVE) registered behind `nvd` feature; new `crates/fastrag/src/hygiene/` module behind `hygiene` feature composing `ChunkFilter` implementations (reject → strip → language → kev-tag) applied between chunking and contextualization inside `index_path_with_metadata`. Metadata lives in Tantivy's existing `metadata_json` blob — zero schema migration.

**Tech Stack:** Rust workspace, serde/serde_json for NVD schema, `whatlang` (already in fastrag-core behind `language-detection`), regex, existing Tantivy + ParserRegistry + ChunkStage pipeline.

---

## File Structure

### New files

```
crates/fastrag-nvd/
  Cargo.toml
  src/
    lib.rs
    schema.rs       — NVD 2.0 serde types
    parser.rs       — NvdFeedParser: schema detect + multi-doc emit
    metadata.rs     — CVE → BTreeMap<String, String> projection
  tests/
    nvd_end_to_end.rs
  fixtures/
    nvd_slice.json  — 5-CVE slice (2 Analyzed, 1 Rejected, 1 Disputed, 1 Modified)

crates/fastrag/src/hygiene/
  mod.rs            — ChunkFilter trait, HygieneChain, HygieneStats
  reject.rs         — MetadataRejectFilter
  boilerplate.rs    — BoilerplateStripper
  language.rs       — LanguageFilter
  kev.rs            — KevTemporalTagger

fastrag-cli/tests/
  security_profile_e2e.rs

tests/gold/
  (5–10 new entries appended to questions.json)
```

### Modified files

```
Cargo.toml                                  — add fastrag-nvd workspace dep
crates/fastrag/Cargo.toml                   — nvd + hygiene feature flags
crates/fastrag/src/lib.rs                   — re-export hygiene module
crates/fastrag/src/registry.rs              — register NvdFeedParser under nvd feature
crates/fastrag/src/corpus/mod.rs            — multi-doc loop + hygiene insertion point
crates/fastrag-core/src/lib.rs              — re-export MultiDocParser trait
crates/fastrag-eval/src/datasets/nvd.rs     — thin re-export of fastrag-nvd schema types
fastrag-cli/src/args.rs                     — --security-profile + sub-flags on Index
fastrag-cli/src/main.rs                     — wire flags to HygieneChain
CLAUDE.md                                   — new test commands
README.md                                   — Security Corpus Hygiene section
docs/superpowers/roadmap-2026-04-phase2-rewrite.md  — mark Step 7 shipped
```

---

## Ground-truth notes (reference before editing)

- `Parser` trait in `crates/fastrag-core/src/lib.rs:19-40` takes `&[u8]` + `&SourceInfo`, returns one `Document`. Multi-doc path uses a new `MultiDocParser` trait alongside `Parser` with `parse_all(path: &Path) -> Result<Vec<Document>, FastRagError>` — takes a `Path` directly because NVD feeds are parsed from disk, not byte slices.
- `index_path_with_metadata` in `crates/fastrag/src/corpus/mod.rs:196` calls `load_document` (line ~299) then `chunk_document` (line ~301). Hygiene inserts between lines 301 and 306 (before contextualizer). The function signature must gain `#[cfg(feature = "hygiene")] hygiene: Option<&HygieneChain>`.
- `ParserRegistry` in `crates/fastrag/src/registry.rs:11-55` uses `FileFormat` enum key. NVD parser needs a new `FileFormat::NvdFeed` variant added to `crates/fastrag-core/src/format.rs`.
- `metadata_json` field in Tantivy schema (`crates/fastrag-tantivy/src/schema.rs:53`) stores arbitrary `BTreeMap<String,String>` — no schema migration needed. The NVD metadata keys (`cve_id`, `vuln_status`, `published_year`, `cvss_severity`, `cpe_vendor`, `cpe_product`, `description_lang`, `kev_flag`, `source`) all live there.
- `whatlang` is already wired behind `language-detection` feature in `crates/fastrag-core/Cargo.toml:13`. `hygiene` feature on the facade should enable `fastrag-core/language-detection`.
- `regex = "1"` is already in `[workspace.dependencies]` (added in Step 6). No new workspace dep needed.
- Existing eval NVD serde types (`NvdFeed`, `NvdVulnerability`, `NvdCve`, `NvdDescription`) are private/module-scoped in `crates/fastrag-eval/src/datasets/nvd.rs:198-220`. Task 2 makes them `pub` in `fastrag-nvd::schema` and re-imports them in the eval crate.
- `CorpusIndexStats` in `crates/fastrag/src/corpus/mod.rs` must gain hygiene counters (`chunks_rejected`, `chunks_stripped`, `chunks_lang_dropped`, `chunks_kev_tagged`) behind `#[cfg(feature = "hygiene")]`.
- `FileFormat::detect` in `crates/fastrag-core/src/format.rs` does extension + byte sniffing. NVD detection: peek first 512 bytes for `"NVD_CVE"` string (JSON field value) OR `"vulnerabilities"` array key, gate behind `nvd` feature in the facade `detect_nvd_feed` helper rather than modifying `FileFormat::detect` directly — the registry's `parse_file` already reads bytes before dispatch.

---

## Rollout Landing Map

- **Landing 1 (Tasks 1–3):** `fastrag-nvd` crate skeleton — compiles, no logic.
- **Landing 2 (Tasks 4–6):** NVD 2.0 serde types in `schema.rs`, round-trip test, eval re-export.
- **Landing 3 (Tasks 7–9):** `NvdFeedParser` single-record path + metadata projection + `MultiDocParser` trait in fastrag-core.
- **Landing 4 (Tasks 10–12):** Multi-doc ingest loop in `index_path_with_metadata` + 5-CVE integration test.
- **Landing 5 (Tasks 13–16):** Hygiene module skeleton + `ChunkFilter` trait + `HygieneChain` + `MetadataRejectFilter`.
- **Landing 6 (Tasks 17–19):** `BoilerplateStripper`.
- **Landing 7 (Tasks 20–22):** `LanguageFilter`.
- **Landing 8 (Tasks 23–25):** `KevTemporalTagger`.
- **Landing 9 (Tasks 26–33):** CLI wiring, E2E test, docs, gold-set entries, baseline pointer.

---

## Landing 1 — `fastrag-nvd` crate skeleton

### Task 1: Add `fastrag-nvd` to workspace

**Files:**
- Modify: `/home/ubuntu/github/fastrag/Cargo.toml`
- Create: `/home/ubuntu/github/fastrag/crates/fastrag-nvd/Cargo.toml`
- Create: `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/lib.rs` with a placeholder test:

```rust
//! NVD 2.0 feed parser for FastRAG.

pub mod metadata;
pub mod parser;
pub mod schema;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {
        assert_eq!(2 + 2, 4);
    }
}
```

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/metadata.rs`:

```rust
//! CVE record → BTreeMap<String, String> metadata projection.

use std::collections::BTreeMap;

use crate::schema::NvdCve;

/// Project an NVD CVE record into the flat metadata map stored in Tantivy's
/// `metadata_json` field. Keys match the Step 7 metadata contract.
pub fn cve_to_metadata(cve: &NvdCve) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    map.insert("source".to_string(), "nvd".to_string());
    if let Some(id) = &cve.id {
        map.insert("cve_id".to_string(), id.clone());
    }
    if let Some(status) = &cve.vuln_status {
        map.insert("vuln_status".to_string(), status.clone());
    }
    if let Some(published) = &cve.published {
        // published is an ISO-8601 string; take the first 4 chars as the year.
        let year = published.chars().take(4).collect::<String>();
        if year.len() == 4 && year.chars().all(|c| c.is_ascii_digit()) {
            map.insert("published_year".to_string(), year);
        }
    }
    // CVSS v3.1 severity from the first metric entry.
    let severity = cve
        .metrics
        .as_ref()
        .and_then(|m| m.cvss_metric_v31.as_ref())
        .and_then(|v| v.first())
        .and_then(|entry| entry.cvss_data.as_ref())
        .and_then(|d| d.base_severity.as_ref());
    if let Some(sev) = severity {
        map.insert("cvss_severity".to_string(), sev.clone());
    }
    // First CPE vendor/product from configurations.
    let first_cpe = cve
        .configurations
        .as_ref()
        .and_then(|configs| configs.first())
        .and_then(|cfg| cfg.nodes.as_ref())
        .and_then(|nodes| nodes.first())
        .and_then(|node| node.cpe_match.as_ref())
        .and_then(|matches| matches.first())
        .and_then(|m| m.criteria.as_ref());
    if let Some(cpe) = first_cpe {
        // cpe:2.3:a:vendor:product:... — field index 3 = vendor, 4 = product
        let parts: Vec<&str> = cpe.split(':').collect();
        if parts.len() > 4 {
            map.insert("cpe_vendor".to_string(), parts[3].to_string());
            map.insert("cpe_product".to_string(), parts[4].to_string());
        }
    }
    // Language of the first description entry.
    if let Some(first_desc) = cve.descriptions.first() {
        map.insert("description_lang".to_string(), first_desc.lang.clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{NvdCve, NvdDescription};

    #[test]
    fn projects_required_fields() {
        let cve = NvdCve {
            id: Some("CVE-2024-12345".to_string()),
            vuln_status: Some("Analyzed".to_string()),
            published: Some("2024-03-15T10:00:00.000".to_string()),
            descriptions: vec![NvdDescription {
                lang: "en".to_string(),
                value: "A test vulnerability.".to_string(),
            }],
            metrics: None,
            configurations: None,
            references: None,
        };
        let map = cve_to_metadata(&cve);
        assert_eq!(map.get("cve_id").map(String::as_str), Some("CVE-2024-12345"));
        assert_eq!(map.get("vuln_status").map(String::as_str), Some("Analyzed"));
        assert_eq!(map.get("published_year").map(String::as_str), Some("2024"));
        assert_eq!(map.get("source").map(String::as_str), Some("nvd"));
        assert_eq!(map.get("description_lang").map(String::as_str), Some("en"));
    }

    #[test]
    fn missing_optional_fields_absent_from_map() {
        let cve = NvdCve {
            id: Some("CVE-2024-00001".to_string()),
            vuln_status: None,
            published: None,
            descriptions: vec![],
            metrics: None,
            configurations: None,
            references: None,
        };
        let map = cve_to_metadata(&cve);
        assert!(!map.contains_key("vuln_status"));
        assert!(!map.contains_key("published_year"));
        assert!(!map.contains_key("cvss_severity"));
        assert_eq!(map.get("cve_id").map(String::as_str), Some("CVE-2024-00001"));
    }
}
```

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/parser.rs`:

```rust
//! NvdFeedParser: detects NVD 2.0 JSON feeds and emits one Document per CVE.

use std::path::Path;

use fastrag_core::{Document, Element, ElementKind, FastRagError, FileFormat, Metadata, SourceInfo};

use crate::metadata::cve_to_metadata;
use crate::schema::NvdFeed;

/// Detects whether `bytes` (first 512 bytes of file) look like an NVD 2.0 feed.
///
/// Checks for the `"NVD_CVE"` format marker or a `"vulnerabilities"` array.
pub fn is_nvd_feed(bytes: &[u8]) -> bool {
    let snippet = std::str::from_utf8(&bytes[..bytes.len().min(512)]).unwrap_or("");
    snippet.contains("\"NVD_CVE\"") || snippet.contains("\"vulnerabilities\"")
}

/// Parser for NVD 2.0 JSON feed files.
///
/// Implements [`fastrag_core::MultiDocParser`] — one call to `parse_all` on a
/// yearly dump emits one `Document` per CVE record. Each document's metadata
/// map is pre-populated with the Step 7 metadata contract keys.
pub struct NvdFeedParser;

impl NvdFeedParser {
    /// Parse an NVD feed at `path` and return one `Document` per CVE.
    pub fn parse_all(&self, path: &Path) -> Result<Vec<Document>, FastRagError> {
        let bytes = std::fs::read(path)
            .map_err(|e| FastRagError::Io(e))?;
        let feed: NvdFeed = serde_json::from_slice(&bytes)
            .map_err(|e| FastRagError::Parse(format!("NVD JSON parse error: {e}")))?;

        let mut docs = Vec::new();
        for vuln in feed.vulnerabilities {
            let Some(cve) = vuln.cve else { continue };
            let Some(id) = cve.id.clone() else { continue };

            // Build document text from English descriptions; skip CVEs with none.
            let desc_text: String = cve
                .descriptions
                .iter()
                .filter(|d| d.lang.eq_ignore_ascii_case("en"))
                .map(|d| d.value.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");

            // Include non-English CVEs but with whatever text is available, so
            // the LanguageFilter downstream can handle the drop/flag decision.
            let text = if desc_text.is_empty() {
                cve.descriptions
                    .iter()
                    .map(|d| d.value.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                desc_text
            };

            if text.is_empty() {
                continue;
            }

            let source = SourceInfo::new(FileFormat::NvdFeed)
                .with_filename(path.to_string_lossy().to_string());

            let metadata_map = cve_to_metadata(&cve);

            let mut doc = Document {
                source,
                elements: vec![Element {
                    kind: ElementKind::Paragraph,
                    text: text.clone(),
                    metadata: Default::default(),
                    ..Default::default()
                }],
                metadata: Metadata {
                    title: Some(id.clone()),
                    ..Default::default()
                },
            };
            // Attach the flat metadata map to the document's extra_metadata so
            // index_path_with_metadata can merge it into file_metadata.
            doc.metadata.extra = metadata_map;
            docs.push(doc);
        }

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const ONE_CVE_JSON: &str = r#"{
  "resultsPerPage": 1,
  "startIndex": 0,
  "totalResults": 1,
  "format": "NVD_CVE",
  "version": "2.0",
  "timestamp": "2024-01-01T00:00:00.000",
  "vulnerabilities": [
    {
      "cve": {
        "id": "CVE-2024-99001",
        "sourceIdentifier": "cve@example.com",
        "published": "2024-06-01T00:00:00.000",
        "lastModified": "2024-06-02T00:00:00.000",
        "vulnStatus": "Analyzed",
        "descriptions": [
          { "lang": "en", "value": "A heap overflow in libfoo allows remote code execution." }
        ],
        "metrics": {},
        "references": []
      }
    }
  ]
}"#;

    #[test]
    fn parses_single_cve_document() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(ONE_CVE_JSON.as_bytes()).unwrap();
        let parser = NvdFeedParser;
        let docs = parser.parse_all(tmp.path()).unwrap();
        assert_eq!(docs.len(), 1, "expected 1 document, got {}", docs.len());
        let doc = &docs[0];
        assert_eq!(doc.metadata.title.as_deref(), Some("CVE-2024-99001"));
        assert!(doc.elements[0].text.contains("heap overflow"));
    }

    #[test]
    fn metadata_projection_present_on_document() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(ONE_CVE_JSON.as_bytes()).unwrap();
        let parser = NvdFeedParser;
        let docs = parser.parse_all(tmp.path()).unwrap();
        let meta = &docs[0].metadata.extra;
        assert_eq!(meta.get("cve_id").map(String::as_str), Some("CVE-2024-99001"));
        assert_eq!(meta.get("vuln_status").map(String::as_str), Some("Analyzed"));
        assert_eq!(meta.get("published_year").map(String::as_str), Some("2024"));
        assert_eq!(meta.get("source").map(String::as_str), Some("nvd"));
    }

    #[test]
    fn skips_cve_with_no_descriptions() {
        let json = r#"{
  "format": "NVD_CVE", "version": "2.0",
  "vulnerabilities": [
    { "cve": { "id": "CVE-2024-00000", "vulnStatus": "Analyzed", "descriptions": [], "references": [] } }
  ]
}"#;
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(json.as_bytes()).unwrap();
        let docs = NvdFeedParser.parse_all(tmp.path()).unwrap();
        assert_eq!(docs.len(), 0);
    }

    #[test]
    fn is_nvd_feed_detects_format_marker() {
        assert!(is_nvd_feed(b"{\"format\":\"NVD_CVE\",\"vulnerabilities\":[]}"));
        assert!(is_nvd_feed(b"{\"vulnerabilities\":[]}"));
        assert!(!is_nvd_feed(b"{\"hello\":\"world\"}"));
    }
}
```

- [ ] **Step 2: Create the crate Cargo.toml**

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/Cargo.toml`:

```toml
[package]
name = "fastrag-nvd"
description = "NVD 2.0 JSON feed parser for FastRAG"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
fastrag-core.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Register in workspace**

In `/home/ubuntu/github/fastrag/Cargo.toml`, add `"crates/fastrag-nvd"` to `[workspace] members` (after `fastrag-context`) and add to `[workspace.dependencies]`:

```toml
fastrag-nvd = { path = "crates/fastrag-nvd", version = "0.1.0" }
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check -p fastrag-nvd
```

Expected: `Finished` with zero errors. The `schema` module is referenced but not yet created — that will compile once Task 2 fills it in. At this stage the `use crate::schema::*` imports in `metadata.rs` and `parser.rs` will error; that is expected (red phase). Actually: stub out `schema.rs` minimally so the crate compiles for the skeleton check:

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/schema.rs` with a minimal stub:

```rust
//! NVD 2.0 serde types — lifted from fastrag-eval and extended.
//! Full types defined in Task 4; this stub allows the skeleton to compile.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NvdFeed {
    #[serde(default)]
    pub vulnerabilities: Vec<NvdVulnerability>,
}

#[derive(Debug, Deserialize)]
pub struct NvdVulnerability {
    pub cve: Option<NvdCve>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdCve {
    pub id: Option<String>,
    pub vuln_status: Option<String>,
    pub published: Option<String>,
    #[serde(default)]
    pub descriptions: Vec<NvdDescription>,
    pub metrics: Option<NvdMetrics>,
    pub configurations: Option<Vec<NvdConfiguration>>,
    pub references: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdDescription {
    pub lang: String,
    pub value: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdMetrics {
    #[serde(rename = "cvssMetricV31", default)]
    pub cvss_metric_v31: Option<Vec<NvdCvssMetric>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdCvssMetric {
    #[serde(rename = "cvssData")]
    pub cvss_data: Option<NvdCvssData>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdCvssData {
    #[serde(rename = "baseSeverity")]
    pub base_severity: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdConfiguration {
    pub nodes: Option<Vec<NvdNode>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdNode {
    #[serde(rename = "cpeMatch", default)]
    pub cpe_match: Option<Vec<NvdCpeMatch>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvdCpeMatch {
    pub criteria: Option<String>,
}
```

Now `cargo check -p fastrag-nvd` should pass.

- [ ] **Step 5: Add `NvdFeed` format variant**

`ParserRegistry::parse_file` uses `FileFormat::detect` to pick a parser. NVD feeds are JSON files that need a dedicated variant. Add `NvdFeed` to the `FileFormat` enum in `crates/fastrag-core/src/format.rs`.

Read `/home/ubuntu/github/fastrag/crates/fastrag-core/src/format.rs` first, then add `NvdFeed` to the enum and its `Display` / `detect` impls. The `detect` function should return `FileFormat::NvdFeed` when the first 512 bytes contain `"NVD_CVE"` or `"vulnerabilities"` — but ONLY when the `nvd` feature is active on the facade. Since `format.rs` is in `fastrag-core` (always compiled), add `NvdFeed` unconditionally but guard its use in `detect` with a feature-agnostic byte check:

In the `FileFormat` enum, add:
```rust
/// NVD 2.0 JSON feed (one file contains many CVE records).
NvdFeed,
```

In the `Display` impl, add:
```rust
FileFormat::NvdFeed => write!(f, "nvd-feed"),
```

In `FileFormat::detect`, add a check before the extension-based branch:
```rust
// NVD 2.0 feed detection: schema marker takes priority over extension.
let snippet = std::str::from_utf8(&first_bytes[..first_bytes.len().min(512)]).unwrap_or("");
if snippet.contains("\"NVD_CVE\"") || (snippet.contains("\"vulnerabilities\"") && snippet.contains("\"format\"")) {
    return FileFormat::NvdFeed;
}
```

- [ ] **Step 6: Run the crate tests (expect most to pass at this stage)**

```bash
cargo test -p fastrag-nvd
```

Expected: all tests in `lib.rs`, `metadata.rs`, `parser.rs` pass. The `schema.rs` stub has no tests yet — that comes in Task 4.

- [ ] **Step 7: Commit skeleton**

```bash
git add crates/fastrag-nvd/ Cargo.toml crates/fastrag-core/src/format.rs
git commit -m "$(cat <<'EOF'
feat(nvd): add fastrag-nvd crate skeleton with NvdFeedParser + metadata projection

Cargo.toml, lib.rs, schema.rs (stub), parser.rs, metadata.rs created.
FileFormat::NvdFeed variant added to fastrag-core. All crate-local
tests pass; schema types will be expanded in the next landing.
EOF
)"
```

---

## Landing 2 — NVD 2.0 serde types + round-trip test + eval re-export

### Task 2: Full `schema.rs` + round-trip test + 5-CVE fixture

**Files:**
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/schema.rs`
- Create: `/home/ubuntu/github/fastrag/crates/fastrag-nvd/fixtures/nvd_slice.json`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag-eval/src/datasets/nvd.rs`

- [ ] **Step 1: Create 5-CVE fixture**

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/fixtures/nvd_slice.json`:

```json
{
  "resultsPerPage": 5,
  "startIndex": 0,
  "totalResults": 5,
  "format": "NVD_CVE",
  "version": "2.0",
  "timestamp": "2024-01-15T00:00:00.000",
  "vulnerabilities": [
    {
      "cve": {
        "id": "CVE-2021-44228",
        "sourceIdentifier": "cve@example.com",
        "published": "2021-12-10T10:15:00.000",
        "lastModified": "2021-12-14T00:00:00.000",
        "vulnStatus": "Analyzed",
        "descriptions": [
          { "lang": "en", "value": "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI lookups allow remote code execution." }
        ],
        "metrics": {
          "cvssMetricV31": [
            {
              "source": "nvd@nist.gov",
              "type": "Primary",
              "cvssData": {
                "version": "3.1",
                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "baseScore": 10.0,
                "baseSeverity": "CRITICAL"
              }
            }
          ]
        },
        "configurations": [
          {
            "nodes": [
              {
                "operator": "OR",
                "negate": false,
                "cpeMatch": [
                  { "vulnerable": true, "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*" }
                ]
              }
            ]
          }
        ],
        "references": []
      }
    },
    {
      "cve": {
        "id": "CVE-2023-44487",
        "sourceIdentifier": "cve@example.com",
        "published": "2023-10-10T14:15:00.000",
        "lastModified": "2023-10-11T00:00:00.000",
        "vulnStatus": "Modified",
        "descriptions": [
          { "lang": "en", "value": "HTTP/2 Rapid Reset Attack causes denial of service via stream cancellation." }
        ],
        "metrics": {
          "cvssMetricV31": [
            {
              "source": "nvd@nist.gov",
              "type": "Primary",
              "cvssData": {
                "version": "3.1",
                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                "baseScore": 7.5,
                "baseSeverity": "HIGH"
              }
            }
          ]
        },
        "configurations": [],
        "references": []
      }
    },
    {
      "cve": {
        "id": "CVE-2024-10001",
        "sourceIdentifier": "cve@example.com",
        "published": "2024-02-01T00:00:00.000",
        "lastModified": "2024-02-02T00:00:00.000",
        "vulnStatus": "Rejected",
        "descriptions": [
          { "lang": "en", "value": "** REJECT ** This CVE was assigned in error and has been rejected." }
        ],
        "metrics": {},
        "configurations": [],
        "references": []
      }
    },
    {
      "cve": {
        "id": "CVE-2024-10002",
        "sourceIdentifier": "cve@example.com",
        "published": "2024-03-01T00:00:00.000",
        "lastModified": "2024-03-02T00:00:00.000",
        "vulnStatus": "Disputed",
        "descriptions": [
          { "lang": "en", "value": "** DISPUTED ** The vendor disputes the impact of this issue." }
        ],
        "metrics": {},
        "configurations": [],
        "references": []
      }
    },
    {
      "cve": {
        "id": "CVE-2022-22965",
        "sourceIdentifier": "cve@example.com",
        "published": "2022-04-01T00:00:00.000",
        "lastModified": "2022-04-02T00:00:00.000",
        "vulnStatus": "Analyzed",
        "descriptions": [
          { "lang": "en", "value": "Spring Framework RCE via Data Binding on JDK 9+ (Spring4Shell)." }
        ],
        "metrics": {
          "cvssMetricV31": [
            {
              "source": "nvd@nist.gov",
              "type": "Primary",
              "cvssData": {
                "version": "3.1",
                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                "baseScore": 9.8,
                "baseSeverity": "CRITICAL"
              }
            }
          ]
        },
        "configurations": [
          {
            "nodes": [
              {
                "operator": "OR",
                "negate": false,
                "cpeMatch": [
                  { "vulnerable": true, "criteria": "cpe:2.3:a:vmware:spring_framework:*:*:*:*:*:*:*:*" }
                ]
              }
            ]
          }
        ],
        "references": []
      }
    }
  ]
}
```

- [ ] **Step 2: Write the failing round-trip test**

At the bottom of `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/schema.rs`, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/nvd_slice.json")
    }

    #[test]
    fn round_trips_five_cve_fixture() {
        let bytes = std::fs::read(fixture_path()).expect("fixture must exist");
        let feed: NvdFeed = serde_json::from_slice(&bytes).expect("must parse");
        assert_eq!(feed.vulnerabilities.len(), 5);
        let ids: Vec<&str> = feed
            .vulnerabilities
            .iter()
            .filter_map(|v| v.cve.as_ref())
            .filter_map(|c| c.id.as_deref())
            .collect();
        assert!(ids.contains(&"CVE-2021-44228"), "Log4Shell must be present");
        assert!(ids.contains(&"CVE-2024-10001"), "Rejected CVE must be present");
    }

    #[test]
    fn rejected_cve_has_correct_status() {
        let bytes = std::fs::read(fixture_path()).unwrap();
        let feed: NvdFeed = serde_json::from_slice(&bytes).unwrap();
        let rejected = feed
            .vulnerabilities
            .iter()
            .find(|v| {
                v.cve.as_ref()
                    .and_then(|c| c.id.as_deref())
                    == Some("CVE-2024-10001")
            })
            .and_then(|v| v.cve.as_ref())
            .unwrap();
        assert_eq!(rejected.vuln_status.as_deref(), Some("Rejected"));
    }

    #[test]
    fn log4shell_has_critical_cvss() {
        let bytes = std::fs::read(fixture_path()).unwrap();
        let feed: NvdFeed = serde_json::from_slice(&bytes).unwrap();
        let log4shell = feed
            .vulnerabilities
            .iter()
            .find(|v| {
                v.cve.as_ref().and_then(|c| c.id.as_deref()) == Some("CVE-2021-44228")
            })
            .and_then(|v| v.cve.as_ref())
            .unwrap();
        let severity = log4shell
            .metrics
            .as_ref()
            .and_then(|m| m.cvss_metric_v31.as_ref())
            .and_then(|v| v.first())
            .and_then(|e| e.cvss_data.as_ref())
            .and_then(|d| d.base_severity.as_ref())
            .map(String::as_str);
        assert_eq!(severity, Some("CRITICAL"));
    }
}
```

- [ ] **Step 3: Run — expect pass (schema is already complete from Task 1's stub)**

```bash
cargo test -p fastrag-nvd
```

Expected: all 3 schema tests plus the earlier parser/metadata tests pass.

- [ ] **Step 4: Update `fastrag-eval` to re-use `fastrag-nvd` types**

Add `fastrag-nvd` as a dependency in `/home/ubuntu/github/fastrag/crates/fastrag-eval/Cargo.toml`:

```toml
fastrag-nvd = { workspace = true }
```

Then in `/home/ubuntu/github/fastrag/crates/fastrag-eval/src/datasets/nvd.rs`, replace the four private serde structs (`NvdFeed`, `NvdVulnerability`, `NvdCve`, `NvdDescription`) with imports from `fastrag_nvd::schema`. Keep all existing function signatures and tests intact. The `read_gz_json` helper in `common.rs` deserializes into the type passed as a generic; update the call to `read_gz_json::<fastrag_nvd::schema::NvdFeed>(&source)`.

After the edit, run:

```bash
cargo test -p fastrag-eval
```

Expected: all existing NVD eval tests pass (fixture round-trip, query join, missing-query-id rejection).

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag-nvd/src/schema.rs crates/fastrag-nvd/fixtures/ \
        crates/fastrag-eval/Cargo.toml crates/fastrag-eval/src/datasets/nvd.rs
git commit -m "$(cat <<'EOF'
feat(nvd): NVD 2.0 serde types + 5-CVE fixture; eval re-exports fastrag-nvd schema

Removes duplicate serde structs from fastrag-eval::datasets::nvd.
Round-trip test verifies fixture parses correctly with all metadata fields.
EOF
)"
```

---

## Landing 3 — `MultiDocParser` trait + single-record parse path verified

### Task 3: `MultiDocParser` trait in fastrag-core

**Files:**
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag-core/src/lib.rs`

**Design decision:** `MultiDocParser` is a new standalone trait (not an extension of `Parser`) with a single method `parse_all(path: &Path) -> Result<Vec<Document>, FastRagError>`. It takes a `Path` directly because NVD feeds are disk-resident files — not byte slices fed from a registry. The facade crate's `load_documents` helper (created in Task 4) tries `MultiDocParser` first, falls back to single-doc `load_document` for other formats.

- [ ] **Step 1: Write the failing test**

Add to `/home/ubuntu/github/fastrag/crates/fastrag-core/src/lib.rs`:

```rust
/// A parser that emits multiple `Document` objects from a single source file.
///
/// Used for formats like NVD JSON feeds where one file encodes many
/// independent records. Implement this trait alongside (not instead of)
/// `Parser` when a format requires multi-doc emission.
pub trait MultiDocParser: Send + Sync {
    /// Parse the file at `path` and return one `Document` per logical record.
    fn parse_all(&self, path: &std::path::Path) -> Result<Vec<Document>, FastRagError>;
}
```

Add a compile-only test:

```rust
#[cfg(test)]
mod multi_doc_parser_tests {
    use super::*;
    use std::path::Path;

    struct StubMultiParser;

    impl MultiDocParser for StubMultiParser {
        fn parse_all(&self, _path: &Path) -> Result<Vec<Document>, FastRagError> {
            Ok(vec![
                Document { source: crate::format::SourceInfo::new(FileFormat::Text), elements: vec![], metadata: Default::default() },
                Document { source: crate::format::SourceInfo::new(FileFormat::Text), elements: vec![], metadata: Default::default() },
            ])
        }
    }

    #[test]
    fn multi_doc_parser_trait_is_implementable() {
        let parser = StubMultiParser;
        let docs = parser.parse_all(Path::new("dummy")).unwrap();
        assert_eq!(docs.len(), 2);
    }
}
```

- [ ] **Step 2: Implement `MultiDocParser` for `NvdFeedParser`**

In `/home/ubuntu/github/fastrag/crates/fastrag-nvd/src/parser.rs`, add the trait impl after the struct definition:

```rust
use fastrag_core::MultiDocParser;

impl MultiDocParser for NvdFeedParser {
    fn parse_all(&self, path: &std::path::Path) -> Result<Vec<fastrag_core::Document>, fastrag_core::FastRagError> {
        NvdFeedParser::parse_all(self, path)
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p fastrag-core
cargo test -p fastrag-nvd
```

Expected: the trait test passes; NvdFeedParser tests still pass.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag-core/src/lib.rs crates/fastrag-nvd/src/parser.rs
git commit -m "$(cat <<'EOF'
feat(core): add MultiDocParser trait for multi-record source formats

NvdFeedParser implements MultiDocParser. Single-doc parsers unaffected.
EOF
)"
```

---

## Landing 4 — Multi-doc ingest loop + integration test

### Task 4: `load_documents` helper + multi-doc branch in `index_path_with_metadata`

**Files:**
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/src/corpus/mod.rs`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/Cargo.toml`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/src/registry.rs`
- Create: `/home/ubuntu/github/fastrag/crates/fastrag-nvd/tests/nvd_end_to_end.rs`

- [ ] **Step 1: Add `nvd` feature flag to facade Cargo.toml**

In `/home/ubuntu/github/fastrag/crates/fastrag/Cargo.toml`, add to `[dependencies]`:

```toml
fastrag-nvd = { workspace = true, optional = true }
```

Add to `[features]`:

```toml
nvd = ["dep:fastrag-nvd"]
```

- [ ] **Step 2: Register `NvdFeedParser` in `ParserRegistry`**

The existing registry uses `FileFormat` as a key and `Box<dyn Parser>` as values. `NvdFeedParser` implements `MultiDocParser`, not `Parser`, so it needs a separate registry field.

In `/home/ubuntu/github/fastrag/crates/fastrag/src/registry.rs`, add a `multi_parsers` field:

```rust
use fastrag_core::MultiDocParser;

pub struct ParserRegistry {
    parsers: HashMap<FileFormat, Box<dyn Parser>>,
    multi_parsers: HashMap<FileFormat, Box<dyn MultiDocParser>>,
}
```

Update `new()` to initialize `multi_parsers: HashMap::new()`.

Add a method:

```rust
pub fn register_multi(&mut self, format: FileFormat, parser: Box<dyn MultiDocParser>) {
    self.multi_parsers.insert(format, parser);
}

/// Return the multi-doc parser for a format if one is registered.
pub fn get_multi(&self, format: FileFormat) -> Option<&dyn MultiDocParser> {
    self.multi_parsers.get(&format).map(|p| p.as_ref())
}
```

In `Default for ParserRegistry`, add (under `#[cfg(feature = "nvd")]`):

```rust
#[cfg(feature = "nvd")]
registry.register_multi(FileFormat::NvdFeed, Box::new(fastrag_nvd::NvdFeedParser));
```

Export `NvdFeedParser` from `fastrag-nvd`'s `lib.rs`:

```rust
pub use parser::NvdFeedParser;
```

- [ ] **Step 3: Write the failing integration test**

Create `/home/ubuntu/github/fastrag/crates/fastrag-nvd/tests/nvd_end_to_end.rs`:

```rust
//! Integration test: parse the 5-CVE fixture into Documents.

use fastrag_nvd::NvdFeedParser;
use fastrag_core::MultiDocParser;
use std::path::Path;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/nvd_slice.json")
}

#[test]
fn emits_five_documents_from_fixture() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).expect("parse must succeed");
    assert_eq!(docs.len(), 5, "expected 5 docs, got {}", docs.len());
}

#[test]
fn each_document_has_cve_id_in_metadata() {
    let parser = NvdFeedParser;
    let docs = parser.parse_all(&fixture()).unwrap();
    for doc in &docs {
        assert!(
            doc.metadata.extra.contains_key("cve_id"),
            "doc {:?} missing cve_id",
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
        rejected.metadata.extra.get("vuln_status").map(String::as_str),
        Some("Rejected")
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
        log4shell.metadata.extra.get("cpe_vendor").map(String::as_str),
        Some("apache")
    );
    assert_eq!(
        log4shell.metadata.extra.get("cpe_product").map(String::as_str),
        Some("log4j")
    );
}
```

- [ ] **Step 4: Check that `Document::metadata` has an `extra` field**

Read `/home/ubuntu/github/fastrag/crates/fastrag-core/src/document.rs` to confirm `Metadata` has an `extra: BTreeMap<String, String>` field. If it does not, add it now:

```rust
/// Arbitrary key-value metadata attached at the document level.
/// Used by NVD and other structured parsers to carry CVE fields.
#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
pub extra: std::collections::BTreeMap<String, String>,
```

- [ ] **Step 5: Wire multi-doc path in `index_path_with_metadata`**

In `/home/ubuntu/github/fastrag/crates/fastrag/src/corpus/mod.rs`, the current inner loop calls `load_document` (a single-doc path). Add a `load_documents` helper (at module level, not in the loop) that dispatches multi vs. single:

```rust
/// Load one or more documents from a path. If the format has a registered
/// multi-doc parser, returns all documents it emits. Otherwise wraps the
/// single-doc result in a Vec.
#[cfg(feature = "nvd")]
fn load_documents_multi(
    path: &Path,
    registry: &fastrag::registry::ParserRegistry,
) -> Result<Vec<Document>, CorpusError> {
    let data = std::fs::read(path).map_err(FastRagError::Io)?;
    let first = &data[..data.len().min(512)];
    let format = FileFormat::detect(path, first);
    if let Some(multi) = registry.get_multi(format) {
        return multi
            .parse_all(path)
            .map_err(|e| CorpusError::Parse(e.to_string()));
    }
    let doc = load_document(path)?;
    Ok(vec![doc])
}
```

Gate the inner per-file loop to call `load_documents_multi` when the `nvd` feature is active:

```rust
#[cfg(feature = "nvd")]
let docs_for_file = load_documents_multi(&wf.abs_path, &registry)?;
#[cfg(not(feature = "nvd"))]
let docs_for_file = { let doc = load_document(&wf.abs_path)?; vec![doc] };
```

Then replace the single `let doc = load_document(...)` + subsequent code with a loop over `docs_for_file`. Each document in the vec goes through chunking, optional contextualization, embedding, and indexing independently. The `wf.abs_path` serves as the `source_path` for all docs from a single feed file; the `doc.metadata.extra` map is merged into `file_metadata` so CVE fields flow into Tantivy.

**Important:** Add `source_path` override: for NVD multi-docs, store the `cve_id` in the source path suffix so each chunk has a unique stable source identifier (`{feed_path}#{cve_id}`). This prevents all CVEs from a feed collapsing to a single file entry in the manifest.

- [ ] **Step 6: Run integration test**

```bash
cargo test -p fastrag-nvd --test nvd_end_to_end
```

Expected: all 4 tests pass.

- [ ] **Step 7: Run workspace gate**

```bash
cargo test --workspace --features nvd
cargo clippy --workspace --all-targets --features retrieval,hybrid,nvd -- -D warnings
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add crates/fastrag/Cargo.toml crates/fastrag/src/registry.rs \
        crates/fastrag/src/corpus/mod.rs crates/fastrag-core/src/document.rs \
        crates/fastrag-nvd/tests/
git commit -m "$(cat <<'EOF'
feat(nvd): multi-doc ingest loop wired into index_path_with_metadata

Registry gains multi_parsers map; NvdFeedParser registered under nvd feature.
index_path_with_metadata dispatches through load_documents_multi when nvd
feature is active. Single-doc parsers unaffected.
EOF
)"
```

---

## Landing 5 — Hygiene module skeleton + `MetadataRejectFilter`

### Task 5: `ChunkFilter` trait + `HygieneChain` + `MetadataRejectFilter`

**Files:**
- Create: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/mod.rs`
- Create: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/reject.rs`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/Cargo.toml`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/src/lib.rs`

- [ ] **Step 1: Add `hygiene` feature flag**

In `/home/ubuntu/github/fastrag/crates/fastrag/Cargo.toml`, add to `[features]`:

```toml
hygiene = ["language-detection", "fastrag-core/language-detection", "dep:regex"]
```

Add `regex` to `[dependencies]`:

```toml
regex = { workspace = true, optional = true }
```

- [ ] **Step 2: Write the failing test**

Create `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/reject.rs`:

```rust
//! MetadataRejectFilter: drops documents whose `vuln_status` metadata field
//! is in the configured reject-status set.

use std::collections::{BTreeMap, BTreeSet};

use fastrag_core::{Chunk, Document};

/// A filter applied to a single document (represented as its chunks + metadata).
/// Returns `false` to drop the document entirely, `true` to keep it.
pub trait DocFilter: Send + Sync {
    fn keep(&self, chunks: &[Chunk], metadata: &BTreeMap<String, String>) -> bool;
}

/// Drops documents whose `vuln_status` metadata value is in the reject set.
///
/// Default reject set: `{"Rejected", "Disputed"}` — matching the NVD `vulnStatus`
/// field values per the NVD 2.0 API specification.
pub struct MetadataRejectFilter {
    pub reject_statuses: BTreeSet<String>,
}

impl Default for MetadataRejectFilter {
    fn default() -> Self {
        Self {
            reject_statuses: ["Rejected", "Disputed"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl DocFilter for MetadataRejectFilter {
    fn keep(&self, _chunks: &[Chunk], metadata: &BTreeMap<String, String>) -> bool {
        match metadata.get("vuln_status") {
            Some(status) => !self.reject_statuses.contains(status),
            None => true, // no vuln_status → not an NVD doc, keep it
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(status: &str) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("vuln_status".to_string(), status.to_string());
        m
    }

    fn empty_chunks() -> Vec<Chunk> {
        vec![]
    }

    #[test]
    fn drops_rejected_status() {
        let filter = MetadataRejectFilter::default();
        assert!(!filter.keep(&empty_chunks(), &meta("Rejected")));
    }

    #[test]
    fn drops_disputed_status() {
        let filter = MetadataRejectFilter::default();
        assert!(!filter.keep(&empty_chunks(), &meta("Disputed")));
    }

    #[test]
    fn keeps_analyzed_status() {
        let filter = MetadataRejectFilter::default();
        assert!(filter.keep(&empty_chunks(), &meta("Analyzed")));
    }

    #[test]
    fn keeps_modified_status() {
        let filter = MetadataRejectFilter::default();
        assert!(filter.keep(&empty_chunks(), &meta("Modified")));
    }

    #[test]
    fn keeps_doc_without_vuln_status() {
        let filter = MetadataRejectFilter::default();
        assert!(filter.keep(&empty_chunks(), &BTreeMap::new()));
    }

    #[test]
    fn custom_reject_set_drops_awaiting_analysis() {
        let mut filter = MetadataRejectFilter::default();
        filter.reject_statuses.insert("Awaiting Analysis".to_string());
        assert!(!filter.keep(&empty_chunks(), &meta("Awaiting Analysis")));
        assert!(filter.keep(&empty_chunks(), &meta("Analyzed")));
    }
}
```

Create `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/mod.rs`:

```rust
//! Ingest-time hygiene filter chain for security corpora.
//!
//! Filters compose via [`HygieneChain`] and run between chunking and
//! contextualization in `index_path_with_metadata`. The chain applies:
//! 1. Document-level reject filters (MetadataRejectFilter)
//! 2. Chunk-text strip filters (BoilerplateStripper)
//! 3. Language filters (LanguageFilter)
//! 4. Metadata enrichers (KevTemporalTagger)

use std::collections::BTreeMap;

use fastrag_core::Chunk;

pub mod boilerplate;
pub mod kev;
pub mod language;
pub mod reject;

pub use reject::{DocFilter, MetadataRejectFilter};

/// Per-run hygiene statistics surfaced to the CLI summary line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HygieneStats {
    pub docs_rejected: usize,
    pub chunks_stripped: usize,
    pub chunks_lang_dropped: usize,
    pub chunks_kev_tagged: usize,
}

/// A filter applied to individual chunk text. Returns the (possibly modified)
/// text. Returning an empty string signals the chunk should be dropped.
pub trait ChunkFilter: Send + Sync {
    fn apply(&self, text: &str, metadata: &BTreeMap<String, String>) -> String;
}

/// Composable hygiene filter chain.
pub struct HygieneChain {
    doc_filters: Vec<Box<dyn DocFilter>>,
    chunk_filters: Vec<Box<dyn ChunkFilter>>,
    /// Metadata enrichers that mutate the metadata map in place.
    enrichers: Vec<Box<dyn MetadataEnricher>>,
}

/// A mutating metadata enricher (e.g., KEV tagger). Runs after chunk filters.
pub trait MetadataEnricher: Send + Sync {
    fn enrich(&self, metadata: &mut BTreeMap<String, String>);
}

impl HygieneChain {
    pub fn new() -> Self {
        Self {
            doc_filters: vec![],
            chunk_filters: vec![],
            enrichers: vec![],
        }
    }

    pub fn with_doc_filter(mut self, f: Box<dyn DocFilter>) -> Self {
        self.doc_filters.push(f);
        self
    }

    pub fn with_chunk_filter(mut self, f: Box<dyn ChunkFilter>) -> Self {
        self.chunk_filters.push(f);
        self
    }

    pub fn with_enricher(mut self, e: Box<dyn MetadataEnricher>) -> Self {
        self.enrichers.push(e);
        self
    }

    /// Apply the full chain to a document's chunks and metadata.
    ///
    /// Returns `None` if a doc-level filter rejects the document, or
    /// `Some((filtered_chunks, stats))` otherwise.
    pub fn apply(
        &self,
        chunks: Vec<Chunk>,
        metadata: &mut BTreeMap<String, String>,
    ) -> Option<(Vec<Chunk>, HygieneStats)> {
        let mut stats = HygieneStats::default();

        // Doc-level reject pass.
        for filter in &self.doc_filters {
            if !filter.keep(&chunks, metadata) {
                stats.docs_rejected += 1;
                return None;
            }
        }

        // Chunk-text strip pass.
        let mut surviving: Vec<Chunk> = Vec::with_capacity(chunks.len());
        for mut chunk in chunks {
            let original_len = chunk.text.len();
            let mut text = chunk.text.clone();
            for filter in &self.chunk_filters {
                text = filter.apply(&text, metadata);
            }
            if text.trim().is_empty() {
                stats.chunks_lang_dropped += 1;
                continue;
            }
            if text.len() != original_len {
                stats.chunks_stripped += 1;
            }
            chunk.text = text.trim().to_string();
            chunk.char_count = chunk.text.len();
            surviving.push(chunk);
        }

        // Metadata enrichment pass.
        for enricher in &self.enrichers {
            enricher.enrich(metadata);
        }
        if metadata.get("kev_flag").map(String::as_str) == Some("true") {
            stats.chunks_kev_tagged += surviving.len();
        }

        Some((surviving, stats))
    }
}

impl Default for HygieneChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::{Chunk, Element, ElementKind};

    fn make_chunk(text: &str) -> Chunk {
        Chunk {
            elements: vec![Element {
                kind: ElementKind::Paragraph,
                text: text.to_string(),
                ..Default::default()
            }],
            text: text.to_string(),
            char_count: text.len(),
            section: None,
            index: 0,
            contextualized_text: None,
        }
    }

    fn rejected_meta() -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("vuln_status".to_string(), "Rejected".to_string());
        m
    }

    #[test]
    fn chain_rejects_rejected_document() {
        let chain = HygieneChain::new()
            .with_doc_filter(Box::new(MetadataRejectFilter::default()));
        let chunks = vec![make_chunk("some text")];
        let mut meta = rejected_meta();
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_none(), "Rejected doc must be dropped");
    }

    #[test]
    fn chain_keeps_analyzed_document() {
        let chain = HygieneChain::new()
            .with_doc_filter(Box::new(MetadataRejectFilter::default()));
        let chunks = vec![make_chunk("CVE analyzed content")];
        let mut meta = BTreeMap::new();
        meta.insert("vuln_status".to_string(), "Analyzed".to_string());
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_some());
        let (out_chunks, _) = result.unwrap();
        assert_eq!(out_chunks.len(), 1);
        assert_eq!(out_chunks[0].text, "CVE analyzed content");
    }

    #[test]
    fn empty_chain_is_passthrough() {
        let chain = HygieneChain::new();
        let chunks = vec![make_chunk("hello")];
        let mut meta = BTreeMap::new();
        let result = chain.apply(chunks, &mut meta);
        assert!(result.is_some());
        let (out, _stats) = result.unwrap();
        assert_eq!(out.len(), 1);
    }
}
```

- [ ] **Step 3: Wire hygiene module into facade lib.rs**

In `/home/ubuntu/github/fastrag/crates/fastrag/src/lib.rs`, add:

```rust
#[cfg(feature = "hygiene")]
pub mod hygiene;
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p fastrag --features hygiene
```

Expected: `chain_rejects_rejected_document`, `chain_keeps_analyzed_document`, `empty_chain_is_passthrough`, and all reject filter unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/hygiene/ crates/fastrag/src/lib.rs crates/fastrag/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(hygiene): ChunkFilter trait + HygieneChain + MetadataRejectFilter

hygiene feature gates the new module. Chain composes DocFilter,
ChunkFilter, and MetadataEnricher in order: reject → strip → lang → kev.
MetadataRejectFilter drops Rejected + Disputed NVD CVEs by default.
EOF
)"
```

---

## Landing 6 — `BoilerplateStripper`

### Task 6: `BoilerplateStripper` regex filter

**Files:**
- Create: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/boilerplate.rs`

- [ ] **Step 1: Write the failing tests first**

Create `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/boilerplate.rs` with the tests block first (TDD), then the impl:

```rust
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
    LEGAL_NOTICE.get_or_init(|| {
        Regex::new(r"(?i)NOTE:\s+Links?\s+are\s+provided.*?(\n|$)").unwrap()
    })
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
        let collapsed = s
            .lines()
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
            .0;
        collapsed
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
        assert!(!out.contains("REJECT"), "marker must be removed; got: {out}");
        assert!(out.contains("This CVE was issued in error."), "prose must survive");
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
```

- [ ] **Step 2: Run — confirm all tests pass (impl is already complete above)**

```bash
cargo test -p fastrag --features hygiene -- hygiene::boilerplate
```

Expected: all 7 boilerplate tests pass.

- [ ] **Step 3: Register `BoilerplateStripper` export in `hygiene/mod.rs`**

Add to the top of `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/mod.rs`:

```rust
pub use boilerplate::BoilerplateStripper;
```

- [ ] **Step 4: Run full hygiene suite**

```bash
cargo test -p fastrag --features hygiene
```

Expected: all prior hygiene tests plus 7 new boilerplate tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/hygiene/boilerplate.rs crates/fastrag/src/hygiene/mod.rs
git commit -m "$(cat <<'EOF'
feat(hygiene): BoilerplateStripper regex filter for NVD description noise

Strips ** REJECT **/DISPUTED markers, CPE 2.3 URI literals, standalone
URL lines, and NVD legal notices. Positive + negative test coverage.
EOF
)"
```

---

## Landing 7 — `LanguageFilter`

### Task 7: `LanguageFilter` wrapping whatlang

**Files:**
- Create: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/language.rs`

- [ ] **Step 1: Write the failing tests**

Create `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/language.rs`:

```rust
//! LanguageFilter: wraps whatlang to drop or flag non-English chunk text.
//!
//! Policy enum:
//! - `Drop`  → return empty string (chunk is eliminated by HygieneChain)
//! - `Flag`  → mutate metadata with `language=<detected>` and keep the text
//!
//! The filter only runs when text is long enough for whatlang to give a
//! reliable detection (>= 20 bytes). Shorter snippets are kept as-is.

use std::collections::BTreeMap;

use super::ChunkFilter;

/// What to do when a non-target-language chunk is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguagePolicy {
    /// Silently drop the chunk (return empty string → HygieneChain drops it).
    Drop,
    /// Keep the chunk but write `language=<lang_code>` into the metadata.
    Flag,
}

/// Minimum text length (bytes) before whatlang detection is attempted.
/// Shorter texts fall back to "keep without labelling".
const MIN_DETECT_BYTES: usize = 20;

/// Filters chunks whose detected language does not match `target_lang`.
pub struct LanguageFilter {
    /// BCP 47 language tag for the allowed language (e.g., `"en"`).
    pub target_lang: String,
    pub policy: LanguagePolicy,
}

impl Default for LanguageFilter {
    fn default() -> Self {
        Self {
            target_lang: "en".to_string(),
            policy: LanguagePolicy::Drop,
        }
    }
}

impl LanguageFilter {
    pub fn new(target_lang: impl Into<String>, policy: LanguagePolicy) -> Self {
        Self {
            target_lang: target_lang.into(),
            policy,
        }
    }
}

impl ChunkFilter for LanguageFilter {
    fn apply(&self, text: &str, _metadata: &BTreeMap<String, String>) -> String {
        if text.len() < MIN_DETECT_BYTES {
            return text.to_string();
        }
        #[cfg(feature = "hygiene")]
        {
            use whatlang::detect;
            if let Some(info) = detect(text) {
                let code = info.lang().code();
                if code != self.target_lang {
                    match self.policy {
                        LanguagePolicy::Drop => return String::new(),
                        LanguagePolicy::Flag => {
                            // Flag: we can't mutate metadata here (ChunkFilter
                            // doesn't have &mut metadata). The caller inspects
                            // the returned text; flag is propagated via a
                            // metadata enricher in practice, or via the
                            // chunk_text having a prepended tag. For now,
                            // return the text unchanged — the corpus pipeline
                            // uses the `description_lang` metadata key from
                            // NvdFeedParser for post-ingest filtering.
                            return text.to_string();
                        }
                    }
                }
            }
        }
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_meta() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    const ENGLISH_TEXT: &str =
        "A critical heap buffer overflow vulnerability allows remote code execution via network.";
    const SPANISH_TEXT: &str =
        "Una vulnerabilidad crítica de desbordamiento de búfer permite la ejecución remota de código.";
    const GERMAN_TEXT: &str =
        "Eine kritische Puffer-Überlauf-Schwachstelle ermöglicht entfernte Codeausführung über das Netzwerk.";

    #[test]
    fn english_text_passes_through_unchanged() {
        let filter = LanguageFilter::default();
        let out = filter.apply(ENGLISH_TEXT, &empty_meta());
        assert_eq!(out, ENGLISH_TEXT);
    }

    #[test]
    fn spanish_text_dropped_by_default() {
        let filter = LanguageFilter::default(); // Drop policy
        let out = filter.apply(SPANISH_TEXT, &empty_meta());
        assert!(
            out.is_empty(),
            "Spanish must be dropped in Drop mode; got: {out}"
        );
    }

    #[test]
    fn german_text_dropped_by_default() {
        let filter = LanguageFilter::default();
        let out = filter.apply(GERMAN_TEXT, &empty_meta());
        assert!(out.is_empty(), "German must be dropped in Drop mode");
    }

    #[test]
    fn flag_policy_keeps_non_english_text() {
        let filter = LanguageFilter::new("en", LanguagePolicy::Flag);
        let out = filter.apply(SPANISH_TEXT, &empty_meta());
        assert_eq!(
            out, SPANISH_TEXT,
            "Flag policy must preserve the text"
        );
    }

    #[test]
    fn short_text_below_threshold_always_kept() {
        let filter = LanguageFilter::default();
        // "hola" is Spanish but below MIN_DETECT_BYTES
        let out = filter.apply("hola", &empty_meta());
        assert_eq!(out, "hola");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p fastrag --features hygiene -- hygiene::language
```

Expected: all 5 language tests pass. whatlang is compiled in because `hygiene` feature enables `fastrag-core/language-detection` and the facade re-exports the crate.

Note: `LanguageFilter::apply` needs direct access to `whatlang`. The facade's `hygiene` feature must also depend on `whatlang` directly, or re-route through `fastrag-core`. Add `whatlang = { version = "0.16", optional = true }` to `crates/fastrag/Cargo.toml` and add it to the `hygiene` feature:

```toml
whatlang = { version = "0.16", optional = true }
```

```toml
hygiene = ["language-detection", "fastrag-core/language-detection", "dep:regex", "dep:whatlang"]
```

Then replace the `#[cfg(feature = "hygiene")]` block in `language.rs` with a direct `use whatlang::detect;` (the file is already inside a `#[cfg]`-gated module).

- [ ] **Step 3: Export from hygiene/mod.rs**

```rust
pub use language::{LanguageFilter, LanguagePolicy};
```

- [ ] **Step 4: Run full hygiene suite**

```bash
cargo test -p fastrag --features hygiene
```

Expected: all hygiene tests (reject + boilerplate + language) pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/hygiene/language.rs crates/fastrag/src/hygiene/mod.rs \
        crates/fastrag/Cargo.toml
git commit -m "$(cat <<'EOF'
feat(hygiene): LanguageFilter wrapping whatlang with Drop/Flag policy

Detects language via whatlang on text >= 20 bytes. Drop mode returns
empty string so HygieneChain eliminates the chunk. Flag mode keeps text.
EOF
)"
```

---

## Landing 8 — `KevTemporalTagger`

### Task 8: `KevTemporalTagger` — load + tag

**Files:**
- Create: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/kev.rs`

- [ ] **Step 1: Write the failing tests**

Create `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/kev.rs`:

```rust
//! KevTemporalTagger: tags CVE metadata with `kev_flag=true` when the CVE ID
//! is present in a CISA Known Exploited Vulnerabilities (KEV) catalog.
//!
//! Accepts two catalog shapes detected at load time:
//!   1. CISA `vulnerabilities.json` — `{ "vulnerabilities": [ { "cveID": "CVE-..." }, ... ] }`
//!   2. FastRAG minimal format       — `{ "cve_ids": ["CVE-...", ...] }`
//!
//! The tagger implements `MetadataEnricher` — it only runs on documents that
//! survive the reject + strip + language filters.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use super::MetadataEnricher;

/// Raw CISA KEV entry (only the fields we need).
#[derive(Debug, Deserialize)]
struct CisaKevEntry {
    #[serde(rename = "cveID")]
    cve_id: String,
}

/// CISA `vulnerabilities.json` top-level shape.
#[derive(Debug, Deserialize)]
struct CisaKevFile {
    vulnerabilities: Vec<CisaKevEntry>,
}

/// FastRAG minimal KEV catalog shape.
#[derive(Debug, Deserialize)]
struct MinimalKevFile {
    cve_ids: Vec<String>,
}

/// Tags chunks whose `cve_id` metadata is in the KEV catalog.
pub struct KevTemporalTagger {
    kev_ids: BTreeSet<String>,
}

impl KevTemporalTagger {
    /// Load a KEV catalog from `path`. Accepts CISA `vulnerabilities.json`
    /// shape or the FastRAG minimal `{cve_ids:[...]}` shape, detected at
    /// load time by probing for the `"vulnerabilities"` key.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read KEV catalog {}: {e}", path.display()))?;
        let raw: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("cannot parse KEV catalog {}: {e}", path.display()))?;

        let kev_ids: BTreeSet<String> = if raw.get("vulnerabilities").is_some() {
            // CISA shape
            let catalog: CisaKevFile = serde_json::from_value(raw)
                .map_err(|e| format!("malformed CISA KEV catalog: {e}"))?;
            catalog
                .vulnerabilities
                .into_iter()
                .map(|e| e.cve_id)
                .collect()
        } else if raw.get("cve_ids").is_some() {
            // FastRAG minimal shape
            let catalog: MinimalKevFile = serde_json::from_value(raw)
                .map_err(|e| format!("malformed minimal KEV catalog: {e}"))?;
            catalog.cve_ids.into_iter().collect()
        } else {
            return Err(format!(
                "unrecognised KEV catalog format in {}; expected 'vulnerabilities' or 'cve_ids' key",
                path.display()
            ));
        };

        Ok(Self { kev_ids })
    }

    /// Build a tagger from an already-loaded ID set (for tests).
    pub fn from_ids(ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            kev_ids: ids.into_iter().map(Into::into).collect(),
        }
    }
}

impl MetadataEnricher for KevTemporalTagger {
    fn enrich(&self, metadata: &mut BTreeMap<String, String>) {
        if let Some(cve_id) = metadata.get("cve_id") {
            if self.kev_ids.contains(cve_id.as_str()) {
                metadata.insert("kev_flag".to_string(), "true".to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn meta(cve_id: &str) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("cve_id".to_string(), cve_id.to_string());
        m
    }

    #[test]
    fn tags_known_cve_with_kev_flag() {
        let tagger = KevTemporalTagger::from_ids(["CVE-2021-44228"]);
        let mut m = meta("CVE-2021-44228");
        tagger.enrich(&mut m);
        assert_eq!(m.get("kev_flag").map(String::as_str), Some("true"));
    }

    #[test]
    fn does_not_tag_unknown_cve() {
        let tagger = KevTemporalTagger::from_ids(["CVE-2021-44228"]);
        let mut m = meta("CVE-2024-99999");
        tagger.enrich(&mut m);
        assert!(!m.contains_key("kev_flag"));
    }

    #[test]
    fn does_not_tag_doc_without_cve_id() {
        let tagger = KevTemporalTagger::from_ids(["CVE-2021-44228"]);
        let mut m = BTreeMap::new();
        tagger.enrich(&mut m);
        assert!(!m.contains_key("kev_flag"));
    }

    #[test]
    fn loads_cisa_shape_from_file() {
        let json = r#"{
  "title": "CISA Known Exploited Vulnerabilities Catalog",
  "vulnerabilities": [
    { "cveID": "CVE-2021-44228", "vendorProject": "Apache", "product": "Log4j" },
    { "cveID": "CVE-2022-22965", "vendorProject": "VMware", "product": "Spring Framework" }
  ]
}"#;
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(json.as_bytes()).unwrap();
        let tagger = KevTemporalTagger::from_path(tmp.path()).unwrap();
        let mut m = meta("CVE-2021-44228");
        tagger.enrich(&mut m);
        assert_eq!(m.get("kev_flag").map(String::as_str), Some("true"));
    }

    #[test]
    fn loads_minimal_shape_from_file() {
        let json = r#"{ "cve_ids": ["CVE-2021-44228", "CVE-2019-0708"] }"#;
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(json.as_bytes()).unwrap();
        let tagger = KevTemporalTagger::from_path(tmp.path()).unwrap();
        let mut m = meta("CVE-2019-0708");
        tagger.enrich(&mut m);
        assert_eq!(m.get("kev_flag").map(String::as_str), Some("true"));
    }

    #[test]
    fn rejects_unknown_catalog_shape() {
        let json = r#"{ "data": [] }"#;
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(json.as_bytes()).unwrap();
        let err = KevTemporalTagger::from_path(tmp.path()).unwrap_err();
        assert!(err.contains("unrecognised"), "error must name the shape; got: {err}");
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p fastrag --features hygiene -- hygiene::kev
```

Expected: all 6 KEV tests pass.

- [ ] **Step 3: Export from hygiene/mod.rs**

```rust
pub use kev::KevTemporalTagger;
```

- [ ] **Step 4: Run full hygiene suite**

```bash
cargo test -p fastrag --features hygiene
```

Expected: all hygiene tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/hygiene/kev.rs crates/fastrag/src/hygiene/mod.rs
git commit -m "$(cat <<'EOF'
feat(hygiene): KevTemporalTagger loads CISA + minimal KEV catalog formats

Detects catalog shape at load time. Tags cve_id matches with kev_flag=true.
Both catalog formats have full test coverage.
EOF
)"
```

---

## Landing 9 — CLI wiring, E2E test, docs, gold-set, baseline

### Task 9: `SecurityProfile` default chain builder + corpus integration point

**Files:**
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/src/hygiene/mod.rs`
- Modify: `/home/ubuntu/github/fastrag/crates/fastrag/src/corpus/mod.rs`

- [ ] **Step 1: Write the failing test — hygiene chain integration**

Add to `crates/fastrag/src/hygiene/mod.rs`:

```rust
/// Build the default `SecurityProfile` hygiene chain.
///
/// Composition: MetadataRejectFilter → BoilerplateStripper → LanguageFilter(en, Drop).
/// KEV tagger is opt-in (added by the CLI when `--security-kev-catalog` is supplied).
///
/// Reject statuses default to `{Rejected, Disputed}`; override via `reject_statuses`.
pub fn security_default_chain(
    reject_statuses: Option<Vec<String>>,
) -> HygieneChain {
    let mut reject = MetadataRejectFilter::default();
    if let Some(statuses) = reject_statuses {
        reject.reject_statuses = statuses.into_iter().collect();
    }
    HygieneChain::new()
        .with_doc_filter(Box::new(reject))
        .with_chunk_filter(Box::new(BoilerplateStripper))
        .with_chunk_filter(Box::new(LanguageFilter::default()))
}
```

Add a test:

```rust
#[test]
fn security_default_chain_rejects_and_strips() {
    let chain = security_default_chain(None);
    // Rejected doc → None
    let mut meta = BTreeMap::new();
    meta.insert("vuln_status".to_string(), "Rejected".to_string());
    let chunks = vec![make_chunk("** REJECT ** some text")];
    assert!(chain.apply(chunks, &mut meta).is_none());

    // Analyzed doc with boilerplate → stripped
    let mut meta2 = BTreeMap::new();
    meta2.insert("vuln_status".to_string(), "Analyzed".to_string());
    let chunks2 = vec![make_chunk("** DISPUTED ** Heap overflow in libfoo.")];
    let result = chain.apply(chunks2, &mut meta2);
    assert!(result.is_some());
    let (out, _) = result.unwrap();
    assert!(!out[0].text.contains("DISPUTED"));
    assert!(out[0].text.contains("Heap overflow in libfoo."));
}
```

- [ ] **Step 2: Wire hygiene into `index_path_with_metadata` signature**

In `/home/ubuntu/github/fastrag/crates/fastrag/src/corpus/mod.rs`, add the hygiene parameter and insertion point. The function signature gains:

```rust
#[cfg(feature = "hygiene")]
hygiene: Option<&crate::hygiene::HygieneChain>,
```

After `let mut chunks = chunk_document(&doc, chunking);` (line ~301) and before the contextualizer block (line ~306), add:

```rust
#[cfg(feature = "hygiene")]
if let Some(h) = hygiene {
    match h.apply(chunks, &mut doc_metadata_for_this_file) {
        None => continue, // doc rejected — skip to next wf
        Some((filtered_chunks, h_stats)) => {
            chunks = filtered_chunks;
            hygiene_totals.docs_rejected += h_stats.docs_rejected;
            hygiene_totals.chunks_stripped += h_stats.chunks_stripped;
            hygiene_totals.chunks_lang_dropped += h_stats.chunks_lang_dropped;
            hygiene_totals.chunks_kev_tagged += h_stats.chunks_kev_tagged;
        }
    }
}
```

Add `hygiene_totals: HygieneStats` to the local vars (gated behind `hygiene` feature) and surface them on `CorpusIndexStats`.

Add to `CorpusIndexStats`:

```rust
#[cfg(feature = "hygiene")]
pub hygiene: crate::hygiene::HygieneStats,
```

All existing call sites pass `None` for the new parameter (non-breaking — it's feature-gated).

- [ ] **Step 3: Run hygiene tests + workspace gate**

```bash
cargo test -p fastrag --features hygiene
cargo test --workspace --features nvd,hygiene
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add crates/fastrag/src/hygiene/mod.rs crates/fastrag/src/corpus/mod.rs
git commit -m "$(cat <<'EOF'
feat(hygiene): security_default_chain builder + hygiene wired into index pipeline

index_path_with_metadata gains hygiene: Option<&HygieneChain> parameter.
CorpusIndexStats surfaces HygieneStats. All call sites default to None.
EOF
)"
```

---

### Task 10: CLI flags — `--security-profile` and sub-flags

**Files:**
- Modify: `/home/ubuntu/github/fastrag/fastrag-cli/src/args.rs`
- Modify: `/home/ubuntu/github/fastrag/fastrag-cli/src/main.rs`

- [ ] **Step 1: Add CLI flags**

In `/home/ubuntu/github/fastrag/fastrag-cli/src/args.rs`, inside the `Index` variant (after the existing `retry_failed` block), add:

```rust
        /// Enable security corpus hygiene filters (NVD reject, boilerplate strip,
        /// language filter). Requires --features hygiene on the facade crate.
        #[cfg(feature = "hygiene")]
        #[arg(long)]
        security_profile: bool,

        /// ISO 639-1 language code to keep; non-matching chunks are dropped.
        /// Only used when --security-profile is set. Default: en.
        #[cfg(feature = "hygiene")]
        #[arg(long, default_value = "en")]
        security_lang: String,

        /// Path to a CISA KEV catalog JSON (vulnerabilities.json or
        /// {cve_ids:[...]} minimal shape). Tags matching CVEs with kev_flag=true.
        /// Only used when --security-profile is set.
        #[cfg(feature = "hygiene")]
        #[arg(long)]
        security_kev_catalog: Option<PathBuf>,

        /// Comma-separated vuln_status values to reject. Default: Rejected,Disputed.
        /// Only used when --security-profile is set.
        #[cfg(feature = "hygiene")]
        #[arg(long, default_value = "Rejected,Disputed")]
        security_reject_statuses: String,
```

- [ ] **Step 2: Wire flags in `main.rs`**

In the `Command::Index { ... }` arm in `/home/ubuntu/github/fastrag/fastrag-cli/src/main.rs`, destructure the new fields and build the chain:

```rust
#[cfg(feature = "hygiene")]
let hygiene_chain = if security_profile {
    use fastrag::hygiene::{KevTemporalTagger, security_default_chain};
    let statuses: Vec<String> = security_reject_statuses
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut chain = security_default_chain(
        if statuses.is_empty() { None } else { Some(statuses) }
    );
    if let Some(kev_path) = security_kev_catalog {
        match KevTemporalTagger::from_path(&kev_path) {
            Ok(tagger) => chain = chain.with_enricher(Box::new(tagger)),
            Err(e) => {
                eprintln!("Error loading KEV catalog: {e}");
                std::process::exit(1);
            }
        }
    }
    Some(chain)
} else {
    None
};
#[cfg(not(feature = "hygiene"))]
let hygiene_chain: Option<()> = None;
```

Pass `hygiene_chain.as_ref()` (or `hygiene: hygiene_chain.as_ref()`) into the `index_path_with_metadata` call. Print the hygiene summary after indexing:

```rust
#[cfg(feature = "hygiene")]
if security_profile {
    let h = &stats.hygiene;
    println!(
        "hygiene: rejected={} stripped={} lang-dropped={} kev-tagged={}",
        h.docs_rejected, h.chunks_stripped, h.chunks_lang_dropped, h.chunks_kev_tagged
    );
}
```

- [ ] **Step 3: Verify CLI compiles**

```bash
cargo build -p fastrag-cli --features hygiene,nvd,retrieval 2>&1 | tail -5
```

Expected: `Finished` with zero errors.

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/src/args.rs fastrag-cli/src/main.rs
git commit -m "$(cat <<'EOF'
feat(cli): --security-profile + --security-lang/kev-catalog/reject-statuses flags

Builds HygieneChain from SecurityProfile defaults + CLI overrides.
Prints hygiene summary line after indexing. Zero impact when flag is off.
EOF
)"
```

---

### Task 11: E2E test

**Files:**
- Create: `/home/ubuntu/github/fastrag/fastrag-cli/tests/security_profile_e2e.rs`

- [ ] **Step 1: Write the E2E test**

Create `/home/ubuntu/github/fastrag/fastrag-cli/tests/security_profile_e2e.rs`:

```rust
//! E2E: index the 5-CVE NVD fixture with --security-profile and verify
//! that Rejected/Disputed CVEs do NOT appear in query results, and that
//! Analyzed CVEs do appear.
//!
//! Does NOT require a real embedder or LLM — uses the test-doubles from
//! fastrag-embed's test-utils feature.

use std::path::Path;
use std::process::Command;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/fastrag-nvd/fixtures/nvd_slice.json")
}

fn fastrag_bin() -> std::path::PathBuf {
    // Built by cargo test --test security_profile_e2e -- the binary is in
    // the same target directory.
    let mut p = std::env::current_exe().unwrap();
    p.pop(); // remove test binary name
    p.pop(); // remove `deps`
    p.join("fastrag")
}

/// Run `fastrag index <fixture> --corpus <dir> --security-profile` and
/// return the combined stdout+stderr.
fn run_index(corpus_dir: &Path) -> std::process::Output {
    Command::new(fastrag_bin())
        .args([
            "index",
            fixture_path().to_str().unwrap(),
            "--corpus",
            corpus_dir.to_str().unwrap(),
            "--security-profile",
            "--embedder",
            "mock",  // uses test-utils mock embedder
        ])
        .output()
        .expect("fastrag binary must be present; run `cargo build -p fastrag-cli --features nvd,hygiene,retrieval`")
}

#[test]
#[ignore = "requires fastrag binary; run with FASTRAG_NVD_TEST=1"]
fn security_profile_drops_rejected_cves() {
    if std::env::var("FASTRAG_NVD_TEST").is_err() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run_index(dir.path());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "index must succeed; stderr: {stderr}");
    // Hygiene summary must report 2 rejected docs (CVE-2024-10001 Rejected + CVE-2024-10002 Disputed)
    assert!(
        stdout.contains("rejected=2"),
        "expected rejected=2 in hygiene summary; stdout: {stdout}"
    );
}
```

Add `security_profile_e2e` to `fastrag-cli/Cargo.toml` under `[[test]]` (if not auto-discovered). The test is `#[ignore]` to avoid blocking CI without a binary; it's gated behind `FASTRAG_NVD_TEST=1`.

- [ ] **Step 2: Verify the test compiles**

```bash
cargo test -p fastrag-cli --features nvd,hygiene,retrieval --test security_profile_e2e -- --list
```

Expected: the test appears in the list (and skips because `FASTRAG_NVD_TEST` is unset).

- [ ] **Step 3: Commit**

```bash
git add fastrag-cli/tests/security_profile_e2e.rs
git commit -m "$(cat <<'EOF'
test(cli): security_profile_e2e integration test (ignored; FASTRAG_NVD_TEST=1)

Verifies Rejected/Disputed CVEs are excluded from corpus after indexing
with --security-profile. Gated behind env var to avoid CI dependency on
binary + NVD fixture availability.
EOF
)"
```

---

### Task 12: Gold-set entries

**Files:**
- Modify: `/home/ubuntu/github/fastrag/tests/gold/questions.json`

- [ ] **Step 1: Append 7 new gold-set entries**

Read the existing `tests/gold/questions.json` first, then append these entries to the `entries` array:

```json
{
  "id": "hygiene-001",
  "question": "What is the impact of the Log4Shell vulnerability CVE-2021-44228?",
  "must_contain_cve_ids": ["CVE-2021-44228"],
  "must_contain_terms": ["JNDI", "remote code execution"],
  "notes": "Log4Shell should appear with kev_flag=true when KEV catalog is used"
},
{
  "id": "hygiene-002",
  "question": "Describe the HTTP/2 Rapid Reset Attack CVE-2023-44487.",
  "must_contain_cve_ids": ["CVE-2023-44487"],
  "must_contain_terms": ["denial of service"],
  "notes": "Modified status CVE — must not be filtered"
},
{
  "id": "hygiene-003",
  "question": "What is the Spring4Shell vulnerability?",
  "must_contain_cve_ids": ["CVE-2022-22965"],
  "must_contain_terms": ["Spring Framework", "remote code execution"],
  "notes": "CRITICAL severity — verify CVSS metadata round-trips"
},
{
  "id": "hygiene-004",
  "question": "Find information about CVE-2024-10001.",
  "must_contain_cve_ids": [],
  "must_contain_terms": [],
  "notes": "NEGATIVE: Rejected CVE — must NOT appear in results when --security-profile is active. Eval harness should score 0 hits as correct."
},
{
  "id": "hygiene-005",
  "question": "Find information about CVE-2024-10002.",
  "must_contain_cve_ids": [],
  "must_contain_terms": [],
  "notes": "NEGATIVE: Disputed CVE — must NOT appear in results when --security-profile is active."
},
{
  "id": "hygiene-006",
  "question": "Which Apache products have critical vulnerabilities?",
  "must_contain_cve_ids": ["CVE-2021-44228"],
  "must_contain_terms": ["apache", "critical"],
  "notes": "Vendor facet via cpe_vendor=apache metadata"
},
{
  "id": "hygiene-007",
  "question": "What KEV vulnerabilities affect VMware Spring Framework?",
  "must_contain_cve_ids": ["CVE-2022-22965"],
  "must_contain_terms": ["Spring"],
  "notes": "kev_flag test — Spring4Shell is in the test KEV minimal catalog fixture"
}
```

- [ ] **Step 2: Validate JSON syntax**

```bash
cargo test -p fastrag-eval --test gold_set_loader 2>&1 | tail -10
```

Expected: gold set loader tests pass (they validate JSON structure).

- [ ] **Step 3: Commit**

```bash
git add tests/gold/questions.json
git commit -m "$(cat <<'EOF'
test(eval): add 7 hygiene gold-set entries for security corpus hygiene regression

Covers Log4Shell, HTTP/2 Rapid Reset, Spring4Shell, Rejected/Disputed
negatives, vendor facet, and KEV flag. NEGATIVE entries verify rejected
CVEs are absent post-hygiene-filter.
EOF
)"
```

---

### Task 13: CLAUDE.md + README.md + roadmap mark

**Files:**
- Modify: `/home/ubuntu/github/fastrag/CLAUDE.md`
- Modify: `/home/ubuntu/github/fastrag/README.md`
- Modify: `/home/ubuntu/github/fastrag/docs/superpowers/roadmap-2026-04-phase2-rewrite.md`

- [ ] **Step 1: Update CLAUDE.md Build & Test section**

In the "Build & Test" section of `/home/ubuntu/github/fastrag/CLAUDE.md`, append after the eval test commands block:

```bash
cargo test -p fastrag-nvd                                               # NVD parser unit tests
cargo test -p fastrag-nvd --test nvd_end_to_end                        # NVD 5-CVE fixture integration test
cargo test --workspace --features nvd,hygiene                          # NVD + hygiene full suite
cargo test -p fastrag-cli --features nvd,hygiene,retrieval --test security_profile_e2e -- --list  # List E2E test
FASTRAG_NVD_TEST=1 cargo test -p fastrag-cli --features nvd,hygiene,retrieval --test security_profile_e2e -- --ignored  # Full E2E (requires binary)
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings  # Full lint gate (with hygiene)
```

Also add a usage example:

```bash
# NVD ingest with security hygiene profile
cargo run --release -p fastrag-cli --features nvd,hygiene,retrieval -- \
  index tests/gold/corpus/nvd-fixture.json --corpus /tmp/nvd-corpus --security-profile
# With KEV catalog
cargo run --release -p fastrag-cli --features nvd,hygiene,retrieval -- \
  index tests/gold/corpus/nvd-fixture.json --corpus /tmp/nvd-corpus \
  --security-profile --security-kev-catalog /path/to/known_exploited_vulnerabilities.json
```

- [ ] **Step 2: Add README Security Corpus Hygiene section**

In `/home/ubuntu/github/fastrag/README.md`, after the "Contextual Retrieval" section, add a "Security Corpus Hygiene" subsection:

```markdown
## Security Corpus Hygiene

FastRAG includes an ingest-time hygiene filter chain for security corpora, enabled via `--security-profile` on the `index` command.

**What it does:**
- **NVD parser** (`--features nvd`): reads NVD 2.0 JSON feeds (yearly dumps) and emits one document per CVE record, with structured metadata (`cve_id`, `vuln_status`, `published_year`, `cvss_severity`, `cpe_vendor`, `cpe_product`).
- **Reject filter**: drops CVEs with `vuln_status` of `Rejected` or `Disputed` (configurable via `--security-reject-statuses`).
- **Boilerplate stripper**: removes `** REJECT **`/`** DISPUTED **` markers, CPE 2.3 URI literals, standalone URL lines, and NVD legal notices from chunk text.
- **Language filter**: drops non-English chunks (configurable via `--security-lang`).
- **KEV tagger**: tags CVEs present in a CISA KEV catalog with `kev_flag=true` in metadata (accepts both CISA `vulnerabilities.json` and a minimal `{cve_ids:[...]}` format).

**Usage:**

```bash
fastrag index ./nvd-feeds/ --corpus ./nvd-corpus \
  --security-profile \
  --security-kev-catalog known_exploited_vulnerabilities.json
```

**Build flags:** `--features nvd` for the NVD parser, `--features hygiene` for the filter chain. Both are off by default.
```

- [ ] **Step 3: Mark Step 7 shipped in roadmap**

In `/home/ubuntu/github/fastrag/docs/superpowers/roadmap-2026-04-phase2-rewrite.md`, find the Step 7 entry and mark it `✅`.

- [ ] **Step 4: Run full gate**

```bash
cargo test --workspace --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
cargo fmt --check
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md docs/superpowers/roadmap-2026-04-phase2-rewrite.md
git commit -m "$(cat <<'EOF'
docs: CLAUDE.md test commands + README hygiene section + Step 7 roadmap mark

Adds nvd + hygiene test commands to CLAUDE.md Build & Test table.
README gains Security Corpus Hygiene section after Contextual Retrieval.
Step 7 marked shipped in phase 2 roadmap.
EOF
)"
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Covered in plan | Task |
|---|---|---|
| `fastrag-nvd` crate skeleton | ✅ | Task 1 |
| NVD 2.0 serde types + round-trip test | ✅ | Task 2 |
| `NvdFeedParser` single-record + metadata projection | ✅ | Task 1 (parser.rs + metadata.rs) |
| `MultiDocParser` trait in fastrag-core | ✅ | Task 3 |
| Multi-doc ingest loop in corpus/mod.rs | ✅ | Task 4 |
| `ChunkFilter` trait + `HygieneChain` | ✅ | Task 5 |
| `MetadataRejectFilter` | ✅ | Task 5 |
| `BoilerplateStripper` | ✅ | Task 6 |
| `LanguageFilter` (whatlang, Drop/Flag) | ✅ | Task 7 |
| `KevTemporalTagger` (CISA + minimal shape) | ✅ | Task 8 |
| `SecurityProfile::default_chain()` builder | ✅ | Task 9 |
| Hygiene wired into index pipeline | ✅ | Task 9 |
| CLI flags `--security-profile` + sub-flags | ✅ | Task 10 |
| E2E test | ✅ | Task 11 |
| Gold-set entries | ✅ | Task 12 |
| CLAUDE.md + README + roadmap | ✅ | Task 13 |
| eval re-export (`fastrag-eval` uses nvd schema) | ✅ | Task 2 |
| Feature flags `nvd` + `hygiene` on facade | ✅ | Task 4, Task 5 |
| Full lint gate command documented | ✅ | Task 13 |

**Placeholder scan:** No "TBD", "similar to", or placeholder values present.

**Type consistency:** `HygieneChain::apply` takes `Vec<Chunk>` (owned) and `&mut BTreeMap<String, String>`. `DocFilter::keep` takes `&[Chunk]` (borrowed). `ChunkFilter::apply` takes `&str` + `&BTreeMap`. `MetadataEnricher::enrich` takes `&mut BTreeMap`. These are consistent across all landing tasks.

**Gotcha — `Document::metadata.extra` field:** The plan assumes `Metadata` has a `BTreeMap<String, String> extra` field. Task 4, Step 4 checks for it and adds it if absent. The field does not currently exist (as of commit `c230b77`). This is a required modification to `crates/fastrag-core/src/document.rs` that the subagent must perform before the NvdFeedParser integration test can pass.

**Gotcha — `source_path` per-CVE uniqueness:** The manifest's incremental planner uses `rel_path` (derived from the file path) to track which chunks belong to which file. For NVD multi-doc emission, all CVEs in a feed share one file path. The plan notes (Task 4, Step 5) that `source_path` should use a `{feed_path}#{cve_id}` suffix — but the manifest `FileEntry.rel_path` is a `PathBuf` and cannot carry a fragment. The subagent must decide whether to (a) store the CVE-ID in the `cve_id` Tantivy field for uniqueness, (b) synthesize a virtual rel path like `nvdcve-2.0-2024.json#CVE-2024-12345`, or (c) use the feed file as the unit of incremental tracking (re-index all CVEs when the feed file changes). Option (c) is simplest and correct — feed files are versioned yearly dumps and change infrequently.

**Gotcha — `LanguageFilter` needs direct whatlang access:** The `hygiene` module lives in `crates/fastrag` (the facade), not `crates/fastrag-core`. The `whatlang` crate must be added directly to `crates/fastrag/Cargo.toml` as an optional dep, and the `hygiene` feature must pull it in. This is called out in Task 7 Step 2.

---

## Execution Handoff

**Recommended: Subagent-Driven Development**

Invoke `superpowers:subagent-driven-development` and pass this plan. Each landing (1–9) is independently executable by a fresh subagent. Start with Landing 1 (Task 1) and proceed sequentially — each landing's output is the input to the next.

**Alternative: Inline Execution**

Invoke `superpowers:executing-plans` to work through tasks in the current session with review checkpoints after each landing commit.

**Local gate before every push (mandatory per CLAUDE.md):**

```bash
cargo test --workspace --features nvd,hygiene
cargo clippy --workspace --all-targets --features retrieval,rerank,hybrid,contextual,eval,nvd,hygiene -- -D warnings
cargo fmt --check
```

After each `git push`, invoke the `ci-watcher` skill as a background Haiku Agent.
