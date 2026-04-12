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
    #[serde(rename = "vulnStatus")]
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
