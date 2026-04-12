//! NvdFeedParser: detects NVD 2.0 JSON feeds and emits one Document per CVE.

use std::path::Path;

use fastrag_core::{
    Document, Element, ElementKind, FastRagError, FileFormat, Metadata, MultiDocParser,
};

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
/// Implements `MultiDocParser` — one call on a yearly dump emits one `Document`
/// per CVE record. Each document's `metadata.extra` map carries the Step 7
/// metadata contract keys (cve_id, vuln_status, published_year, etc.).
pub struct NvdFeedParser;

impl MultiDocParser for NvdFeedParser {
    fn parse_all(&self, path: &Path) -> Result<Vec<Document>, FastRagError> {
        let bytes = std::fs::read(path)?;
        let feed: NvdFeed = serde_json::from_slice(&bytes).map_err(|e| FastRagError::Parse {
            format: FileFormat::NvdFeed,
            message: format!("NVD JSON parse error: {e}"),
        })?;

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

            let metadata_map = cve_to_metadata(&cve);

            let mut meta = Metadata::new(FileFormat::NvdFeed);
            meta.title = Some(id.clone());
            meta.source_file = Some(path.to_string_lossy().to_string());
            // Store the flat metadata map in `extra` — the canonical ingest
            // metadata field for structured parsers.
            meta.extra = metadata_map;

            let doc = Document {
                metadata: meta,
                elements: vec![Element::new(ElementKind::Paragraph, text)],
            };
            docs.push(doc);
        }

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::MultiDocParser;
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
        let extra = &docs[0].metadata.extra;
        assert_eq!(
            extra.get("cve_id").map(String::as_str),
            Some("CVE-2024-99001")
        );
        assert_eq!(
            extra.get("vuln_status").map(String::as_str),
            Some("Analyzed")
        );
        assert_eq!(
            extra.get("published_year").map(String::as_str),
            Some("2024")
        );
        assert_eq!(extra.get("source").map(String::as_str), Some("nvd"));
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
        assert!(is_nvd_feed(
            b"{\"format\":\"NVD_CVE\",\"vulnerabilities\":[]}"
        ));
        assert!(is_nvd_feed(b"{\"vulnerabilities\":[]}"));
        assert!(!is_nvd_feed(b"{\"hello\":\"world\"}"));
    }
}
