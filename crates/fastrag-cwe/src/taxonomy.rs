use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TaxonomyError {
    #[error("malformed taxonomy JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Taxonomy {
    version: String,
    view: String,
    /// Map from CWE id → `[self, descendants...]` (self is first, rest sorted ascending).
    closure: HashMap<u32, Vec<u32>>,
}

impl Taxonomy {
    pub fn from_json(bytes: &[u8]) -> Result<Self, TaxonomyError> {
        let parsed: Taxonomy = serde_json::from_slice(bytes)?;
        Ok(parsed)
    }

    /// Internal constructor used by the compile tool. Not part of the public
    /// runtime API.
    pub fn from_components(version: String, view: String, closure: HashMap<u32, Vec<u32>>) -> Self {
        Self {
            version,
            view,
            closure,
        }
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn view(&self) -> &str {
        &self.view
    }

    /// Return the closure for `cwe`. Falls back to a single-element slice
    /// holding `cwe` when the id is not present in the taxonomy.
    pub fn expand(&self, cwe: u32) -> Vec<u32> {
        match self.closure.get(&cwe) {
            Some(ids) => ids.clone(),
            None => vec![cwe],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> &'static str {
        r#"{
            "version": "4.16-test",
            "view": "1000",
            "closure": {
                "89": [89, 564, 943],
                "79": [79, 80, 81]
            }
        }"#
    }

    #[test]
    fn parses_version_and_view() {
        let tx = Taxonomy::from_json(fixture_json().as_bytes()).unwrap();
        assert_eq!(tx.version(), "4.16-test");
        assert_eq!(tx.view(), "1000");
    }

    #[test]
    fn expand_known_id_returns_closure() {
        let tx = Taxonomy::from_json(fixture_json().as_bytes()).unwrap();
        let got = tx.expand(89);
        assert!(got.contains(&89), "expand(89) missing self: {got:?}");
        assert!(got.contains(&564), "expand(89) missing child 564: {got:?}");
        assert!(got.contains(&943), "expand(89) missing child 943: {got:?}");
    }

    #[test]
    fn expand_unknown_id_returns_singleton() {
        let tx = Taxonomy::from_json(fixture_json().as_bytes()).unwrap();
        assert_eq!(tx.expand(9999), vec![9999]);
    }

    #[test]
    fn expand_is_idempotent_on_repeat_calls() {
        let tx = Taxonomy::from_json(fixture_json().as_bytes()).unwrap();
        let first = tx.expand(79);
        let second = tx.expand(79);
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_json_errors() {
        let result = Taxonomy::from_json(b"not json");
        assert!(matches!(result, Err(TaxonomyError::Parse(_))));
    }
}
