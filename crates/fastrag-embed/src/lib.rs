mod bge;
mod error;

#[cfg(feature = "http-embedders")]
pub mod http;

#[cfg(feature = "test-utils")]
pub mod test_utils;

pub use crate::bge::BgeSmallEmbedder;
pub use crate::error::EmbedError;

use serde::{Deserialize, Serialize};

/// A query-side input. Distinct from `PassageText` at the type level so prefix
/// conventions cannot be confused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryText(String);

impl QueryText {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A passage-side input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageText(String);

impl PassageText {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Prefix pair used by asymmetric retrievers (E5, nomic, arctic, …). Empty
/// strings mean "no prefix" (BGE-small, OpenAI, mock).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixScheme {
    pub query: &'static str,
    pub passage: &'static str,
}

impl PrefixScheme {
    pub const NONE: PrefixScheme = PrefixScheme {
        query: "",
        passage: "",
    };

    pub const fn new(query: &'static str, passage: &'static str) -> Self {
        Self { query, passage }
    }

    /// FNV-1a 64-bit hash of `"{query}\0{passage}"`. Deterministic across runs.
    pub const fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let qb = self.query.as_bytes();
        let mut i = 0;
        while i < qb.len() {
            h ^= qb[i] as u64;
            h = h.wrapping_mul(0x100000001b3);
            i += 1;
        }
        h ^= 0;
        h = h.wrapping_mul(0x100000001b3);
        let pb = self.passage.as_bytes();
        let mut j = 0;
        while j < pb.len() {
            h ^= pb[j] as u64;
            h = h.wrapping_mul(0x100000001b3);
            j += 1;
        }
        h
    }
}

/// Identity of the embedder that produced a corpus. Persisted in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderIdentity {
    pub model_id: String,
    pub dim: usize,
    pub prefix_scheme_hash: u64,
}

/// Fixed canary text, embedded once at corpus creation and re-embedded on load
/// to detect silent drift.
pub const CANARY_TEXT: &str = "fastrag canary v1: the quick brown fox jumps over the lazy dog";

pub const CANARY_COSINE_TOLERANCE: f32 = 0.999;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Canary {
    pub text_version: u32,
    pub vector: Vec<f32>,
}

/// An embedder produces fixed-size vectors for input texts.
pub trait Embedder: Send + Sync {
    /// An identifier for the embedding model implementation used.
    ///
    /// This is written into corpus manifests to enforce compatibility at load time.
    ///
    /// Returns an owned String so HTTP-backed embedders can encode runtime-chosen model
    /// names (e.g. "openai:text-embedding-3-small").
    fn model_id(&self) -> String {
        "unknown".to_string()
    }

    fn dim(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Default safe batch size for `embed_batched`. Implementations that materialize a
    /// `(batch, seq_len, hidden)` tensor (BGE, E5, …) should override this to a value that
    /// keeps peak RSS bounded on real corpora. The runner relies on this so it does not
    /// have to know transformer-specific memory characteristics.
    fn default_batch_size(&self) -> usize {
        64
    }

    /// Embed `texts` in fixed-size batches. Default implementation loops `embed`; override
    /// only if the underlying model can do something smarter (true async batching, GPU
    /// streams, etc.). Callers should prefer this over `embed` for any input that is not
    /// already known to be small.
    fn embed_batched(
        &self,
        texts: &[&str],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch = batch_size.max(1);
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch) {
            out.extend(self.embed(chunk)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod core_type_tests {
    use super::*;

    #[test]
    fn prefix_scheme_hash_is_stable_for_same_prefixes() {
        let a = PrefixScheme::new("query: ", "passage: ");
        let b = PrefixScheme::new("query: ", "passage: ");
        assert_eq!(a.hash(), b.hash());
    }

    #[test]
    fn prefix_scheme_hash_differs_when_prefixes_differ() {
        let a = PrefixScheme::new("query: ", "passage: ");
        let b = PrefixScheme::new("search_query: ", "search_document: ");
        assert_ne!(a.hash(), b.hash());
    }

    #[test]
    fn query_and_passage_text_are_distinct_types() {
        let q = QueryText::new("hi");
        let p = PassageText::new("hi");
        assert_eq!(q.as_str(), "hi");
        assert_eq!(p.as_str(), "hi");
    }

    #[test]
    fn embedder_identity_equality_is_field_wise() {
        let a = EmbedderIdentity {
            model_id: "fastrag/bge-small-en-v1.5".into(),
            dim: 384,
            prefix_scheme_hash: 0xDEADBEEF,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    struct CountingEmbedder {
        calls: std::sync::Mutex<Vec<usize>>,
    }

    impl Embedder for CountingEmbedder {
        fn dim(&self) -> usize {
            3
        }
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            self.calls.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|_| vec![0.0; 3]).collect())
        }
    }

    #[test]
    fn embed_batched_respects_batch_size() {
        let e = CountingEmbedder {
            calls: Default::default(),
        };
        let texts: Vec<&str> = (0..10).map(|_| "x").collect();
        let out = e.embed_batched(&texts, 4).unwrap();
        assert_eq!(out.len(), 10);
        assert_eq!(*e.calls.lock().unwrap(), vec![4, 4, 2]);
    }

    #[test]
    fn embed_batched_empty_input_is_zero_calls() {
        let e = CountingEmbedder {
            calls: Default::default(),
        };
        let out = e.embed_batched(&[], 4).unwrap();
        assert!(out.is_empty());
        assert!(e.calls.lock().unwrap().is_empty());
    }
}
