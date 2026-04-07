mod bge;
mod error;

#[cfg(feature = "test-utils")]
pub mod test_utils;

pub use crate::bge::BgeSmallEmbedder;
pub use crate::error::EmbedError;

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
