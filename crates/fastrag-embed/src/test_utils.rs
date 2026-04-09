use crate::{Embedder, EmbedError, PassageText, PrefixScheme, QueryText};

/// Deterministic hash-based embedder for tests.
///
/// - 16 dimensions
/// - L2-normalized output
/// - Stable across platforms (does not use Rust's randomized Hash)
#[derive(Debug, Default, Clone)]
pub struct MockEmbedder;

impl MockEmbedder {
    fn fingerprint(text: &str) -> Vec<f32> {
        const DIM: usize = 16;
        let lower = text.to_lowercase();
        let bytes = lower.as_bytes();
        let mut v = vec![0.0f32; DIM];

        if bytes.is_empty() {
            return v;
        }

        // Feature-hash byte trigrams into a small fixed vector.
        if bytes.len() < 3 {
            for (i, &b) in bytes.iter().enumerate() {
                let idx = (b as usize).wrapping_add(i) % DIM;
                v[idx] += 1.0;
            }
        } else {
            for w in bytes.windows(3) {
                let h = fnv1a64(w);
                let idx = (h as usize) % DIM;
                let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
                v[idx] += sign;
            }
        }

        l2_normalize(&mut v);
        v
    }
}

impl Embedder for MockEmbedder {
    const DIM: usize = 16;
    const MODEL_ID: &'static str = "fastrag/mock-embedder-16d-v1";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::fingerprint(t.as_str())).collect())
    }

    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| Self::fingerprint(t.as_str())).collect())
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }
    for x in v {
        *x /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    #[test]
    fn dim_is_16() {
        assert_eq!(MockEmbedder::DIM, 16);
    }

    #[test]
    fn model_id_is_stable() {
        assert_eq!(MockEmbedder::MODEL_ID, "fastrag/mock-embedder-16d-v1");
    }

    #[test]
    fn deterministic_for_same_input() {
        let e = MockEmbedder;
        let a = e.embed_query(&[QueryText::new("hello world")]).unwrap();
        let b = e.embed_query(&[QueryText::new("hello world")]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn batch_preserves_order() {
        let e = MockEmbedder;
        let batch = e
            .embed_query(&[
                QueryText::new("one"),
                QueryText::new("two"),
                QueryText::new("three"),
            ])
            .unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0], e.embed_query(&[QueryText::new("one")]).unwrap()[0]);
        assert_eq!(batch[1], e.embed_query(&[QueryText::new("two")]).unwrap()[0]);
        assert_eq!(
            batch[2],
            e.embed_query(&[QueryText::new("three")]).unwrap()[0]
        );
    }

    #[test]
    fn normalized_when_non_empty() {
        let e = MockEmbedder;
        let v = e
            .embed_query(&[QueryText::new("some text")])
            .unwrap()
            .pop()
            .unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similar_texts_have_higher_cosine_than_unrelated() {
        let e = MockEmbedder;
        let a = e
            .embed_query(&[QueryText::new("cat sits on mat")])
            .unwrap()
            .pop()
            .unwrap();
        let b = e
            .embed_query(&[QueryText::new("a cat on the mat")])
            .unwrap()
            .pop()
            .unwrap();
        let c = e
            .embed_query(&[QueryText::new("Rust async runtime")])
            .unwrap()
            .pop()
            .unwrap();
        assert!(cosine(&a, &b) > cosine(&a, &c));
    }
}

#[cfg(test)]
mod mock_invariant_tests {
    use super::*;
    use crate::{Embedder, PassageText, QueryText};

    #[test]
    fn mock_consts_are_pinned() {
        assert_eq!(MockEmbedder::DIM, 16);
        assert_eq!(MockEmbedder::MODEL_ID, "fastrag/mock-embedder-16d-v1");
    }

    #[test]
    fn mock_embed_query_returns_16d_vectors() {
        let m = MockEmbedder;
        let v = m.embed_query(&[QueryText::new("hello")]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 16);
    }

    #[test]
    fn mock_query_and_passage_match_for_same_input() {
        let m = MockEmbedder;
        let q = m.embed_query(&[QueryText::new("same")]).unwrap();
        let p = m.embed_passage(&[PassageText::new("same")]).unwrap();
        assert_eq!(q, p);
    }
}
