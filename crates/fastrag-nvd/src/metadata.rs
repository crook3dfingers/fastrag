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
        assert_eq!(
            map.get("cve_id").map(String::as_str),
            Some("CVE-2024-12345")
        );
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
        assert_eq!(
            map.get("cve_id").map(String::as_str),
            Some("CVE-2024-00001")
        );
    }
}
