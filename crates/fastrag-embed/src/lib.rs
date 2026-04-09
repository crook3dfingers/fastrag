#[cfg(feature = "legacy-candle")]
mod bge;
mod error;

#[cfg(feature = "http-embedders")]
pub mod http;

#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;

#[cfg(feature = "test-utils")]
pub mod test_utils;

#[cfg(feature = "legacy-candle")]
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

/// Static embedder trait. Every implementation must burn its dim, model id,
/// and prefix scheme into associated consts so the compiler can enforce
/// compatibility wherever the concrete type is known.
pub trait Embedder: Send + Sync + 'static {
    const DIM: usize;
    const MODEL_ID: &'static str;
    const PREFIX_SCHEME: PrefixScheme;

    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError>;

    fn default_batch_size(&self) -> usize {
        64
    }
}

/// Dyn-safe mirror of `Embedder`. Corpora store `Arc<dyn DynEmbedderTrait>`
/// and get runtime access to the same invariants that the static trait
/// guarantees at construction time.
pub trait DynEmbedderTrait: Send + Sync + 'static {
    fn model_id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn prefix_scheme(&self) -> PrefixScheme;
    fn prefix_scheme_hash(&self) -> u64;
    fn identity(&self) -> EmbedderIdentity {
        EmbedderIdentity {
            model_id: self.model_id().to_string(),
            dim: self.dim(),
            prefix_scheme_hash: self.prefix_scheme_hash(),
        }
    }
    fn default_batch_size(&self) -> usize;
    fn embed_query_dyn(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn embed_passage_dyn(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError>;
}

impl<E: Embedder> DynEmbedderTrait for E {
    fn model_id(&self) -> &'static str {
        <E as Embedder>::MODEL_ID
    }
    fn dim(&self) -> usize {
        <E as Embedder>::DIM
    }
    fn prefix_scheme(&self) -> PrefixScheme {
        <E as Embedder>::PREFIX_SCHEME
    }
    fn prefix_scheme_hash(&self) -> u64 {
        <E as Embedder>::PREFIX_SCHEME.hash()
    }
    fn default_batch_size(&self) -> usize {
        <E as Embedder>::default_batch_size(self)
    }
    fn embed_query_dyn(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        <E as Embedder>::embed_query(self, texts)
    }
    fn embed_passage_dyn(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        <E as Embedder>::embed_passage(self, texts)
    }
}

/// Convenience alias for the erased form.
pub type DynEmbedder = std::sync::Arc<dyn DynEmbedderTrait>;

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

    #[test]
    fn dyn_embedder_forwards_to_static_impl() {
        use std::sync::Arc;

        struct Toy;
        impl Embedder for Toy {
            const DIM: usize = 2;
            const MODEL_ID: &'static str = "toy/v1";
            const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

            fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }

            fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(texts.iter().map(|_| vec![0.0, 1.0]).collect())
            }
        }

        let erased: Arc<dyn DynEmbedderTrait> = Arc::new(Toy);
        assert_eq!(erased.dim(), 2);
        assert_eq!(erased.model_id(), "toy/v1");
        assert_eq!(erased.prefix_scheme_hash(), PrefixScheme::NONE.hash());

        let qv = erased.embed_query_dyn(&[QueryText::new("q")]).unwrap();
        assert_eq!(qv, vec![vec![1.0, 0.0]]);
        let pv = erased.embed_passage_dyn(&[PassageText::new("p")]).unwrap();
        assert_eq!(pv, vec![vec![0.0, 1.0]]);
    }
}
