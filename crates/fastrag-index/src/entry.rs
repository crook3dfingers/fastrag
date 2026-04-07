use crate::ElementKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    pub id: u64,
    pub vector: Vec<f32>,
    pub chunk_text: String,
    pub source_path: PathBuf,
    pub chunk_index: usize,
    pub section: Option<String>,
    pub element_kinds: Vec<ElementKind>,
    pub pages: Vec<usize>,
    pub language: Option<String>,
    /// User-supplied metadata (customer, severity, year, project, ...).
    /// Populated via `.meta.json` sidecar files or `--metadata k=v` at index time.
    /// Empty on older indexes — `#[serde(default)]` keeps them loadable.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl IndexEntry {
    /// Check whether every `filter` key/value pair is present in the entry's metadata.
    /// An empty filter always matches.
    pub fn matches_filter(&self, filter: &BTreeMap<String, String>) -> bool {
        filter
            .iter()
            .all(|(k, v)| self.metadata.get(k).map(|m| m == v).unwrap_or(false))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchHit {
    pub entry: IndexEntry,
    pub score: f32, // cosine similarity
}
