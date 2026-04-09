use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::{EmbedError, Embedder, PassageText, PrefixScheme, QueryText};

const MODEL_REPO_ID: &str = "BAAI/bge-small-en-v1.5";
const MODEL_CACHE_SUBDIR: &str = "fastrag/models/bge-small-en-v1.5";
const EXPECTED_DIM: usize = 384;

/// CPU embedder for BGE-small-en-v1.5.
///
/// Notes:
/// - Uses mean pooling over token embeddings with attention masking.
/// - Applies L2 normalization (cosine-ready).
pub struct BgeSmallEmbedder {
    device: Device,
    tokenizer: Tokenizer,
    model: candle_transformers::models::bert::BertModel,
    dim: usize,
}

impl BgeSmallEmbedder {
    /// Load the model from a local directory containing at least:
    /// - `tokenizer.json`
    /// - `config.json`
    /// - `model.safetensors`
    pub fn from_local(model_dir: &Path) -> Result<Self, EmbedError> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");

        ensure_exists(&tokenizer_path)?;
        ensure_exists(&config_path)?;
        ensure_exists(&weights_path)?;

        let device = Device::Cpu;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: 512,
            ..Default::default()
        }))?;

        // candle-transformers uses serde for config types; parse from json.
        let config_json = fs::read_to_string(&config_path)?;
        let config: candle_transformers::models::bert::Config =
            serde_json::from_str(&config_json).map_err(|e| EmbedError::Candle(e.to_string()))?;

        if config.hidden_size != EXPECTED_DIM {
            return Err(EmbedError::UnexpectedDim {
                expected: EXPECTED_DIM,
                got: config.hidden_size,
            });
        }

        let vb = unsafe {
            // SAFETY: we map a file owned by the caller/HF cache as read-only model weights.
            // candle's API requires this to be marked unsafe.
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = candle_transformers::models::bert::BertModel::load(vb, &config)?;

        Ok(Self {
            device,
            tokenizer,
            model,
            dim: config.hidden_size,
        })
    }

    /// Download the model from HF Hub (if needed) and cache it under:
    /// `~/.cache/fastrag/models/bge-small-en-v1.5`.
    pub fn from_hf_hub() -> Result<Self, EmbedError> {
        let base = dirs::cache_dir().ok_or(EmbedError::NoCacheDir)?;
        let model_dir = base.join(MODEL_CACHE_SUBDIR);
        fs::create_dir_all(&model_dir)?;

        // Keep hf-hub's internal cache out of the model directory, but still under ~/.cache/fastrag/.
        let hf_cache_dir = base.join("fastrag/hf-hub");
        fs::create_dir_all(&hf_cache_dir)?;

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(hf_cache_dir)
            .build()?;
        let repo = api.model(MODEL_REPO_ID.to_string());

        download_into(&repo, "tokenizer.json", &model_dir)?;
        download_into(&repo, "config.json", &model_dir)?;
        download_into(&repo, "model.safetensors", &model_dir)?;

        Self::from_local(&model_dir)
    }

    fn embed_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Err(EmbedError::EmptyInput);
        }

        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true)?;
        let batch = encodings.len();
        let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        // tokenizers is configured for batch padding, so all encodings have identical length.
        let mut input_ids: Vec<u32> = Vec::with_capacity(batch * seq_len);
        let mut attention_mask: Vec<u32> = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            input_ids.extend_from_slice(enc.get_ids());
            attention_mask.extend_from_slice(enc.get_attention_mask());
        }

        let input_ids =
            Tensor::from_vec(input_ids, (batch, seq_len), &self.device)?.to_dtype(DType::I64)?;
        let attention_mask = Tensor::from_vec(attention_mask, (batch, seq_len), &self.device)?
            .to_dtype(DType::F32)?;

        // Some bert configs need token type ids; using zeros is standard for single-sentence inputs.
        let token_type_ids = Tensor::zeros((batch, seq_len), DType::I64, &self.device)?;

        // Forward pass: output shape is expected to be (batch, seq_len, hidden).
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            .to_dtype(DType::F32)?;

        let pooled = mean_pool(&hidden, &attention_mask)?;
        let normalized = l2_normalize_rows(&pooled)?;

        // Convert to Vec<Vec<f32>>
        let vecs = normalized.to_vec2::<f32>()?;
        for v in &vecs {
            if v.len() != self.dim {
                return Err(EmbedError::UnexpectedDim {
                    expected: self.dim,
                    got: v.len(),
                });
            }
        }
        Ok(vecs)
    }
}

fn download_into(
    repo: &hf_hub::api::sync::ApiRepo,
    filename: &str,
    model_dir: &Path,
) -> Result<(), EmbedError> {
    let dst = model_dir.join(filename);
    if dst.exists() {
        return Ok(());
    }
    let src = repo.get(filename)?;
    fs::copy(src, &dst)?;
    Ok(())
}

impl Embedder for BgeSmallEmbedder {
    const DIM: usize = 384;
    const MODEL_ID: &'static str = "fastrag/bge-small-en-v1.5";
    const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;

    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(QueryText::as_str).collect();
        self.embed_raw(&refs)
    }

    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let refs: Vec<&str> = texts.iter().map(PassageText::as_str).collect();
        self.embed_raw(&refs)
    }

    fn default_batch_size(&self) -> usize {
        16
    }
}

fn ensure_exists(path: &Path) -> Result<(), EmbedError> {
    if path.exists() {
        Ok(())
    } else {
        Err(EmbedError::MissingModelFile {
            path: path.to_path_buf(),
        })
    }
}

fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EmbedError> {
    // hidden: (batch, seq_len, hidden)
    // attention_mask: (batch, seq_len)
    let mask = attention_mask.unsqueeze(2)?; // (batch, seq_len, 1)
    let masked = hidden.broadcast_mul(&mask)?;
    let summed = masked.sum(1)?; // (batch, hidden)
    let denom = mask.sum(1)?.clamp(1e-9f32, f32::MAX)?; // (batch, 1)
    Ok(summed.broadcast_div(&denom)?)
}

fn l2_normalize_rows(x: &Tensor) -> Result<Tensor, EmbedError> {
    // x: (batch, dim)
    let sq = x.sqr()?;
    let sum = sq.sum(1)?; // (batch,)
    let norm = sum.sqrt()?.unsqueeze(1)?; // (batch, 1)
    let norm = norm.clamp(1e-12f32, f32::MAX)?;
    Ok(x.broadcast_div(&norm)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    #[ignore]
    fn real_embed_returns_correct_dim() {
        if std::env::var("FASTRAG_EMBED_TEST").ok().as_deref() != Some("1") {
            return;
        }
        let embedder = BgeSmallEmbedder::from_hf_hub().unwrap();
        let out = embedder
            .embed_query(&[QueryText::new("hello world")])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), EXPECTED_DIM);
    }

    #[test]
    #[ignore]
    fn real_embed_is_normalized() {
        if std::env::var("FASTRAG_EMBED_TEST").ok().as_deref() != Some("1") {
            return;
        }
        let embedder = BgeSmallEmbedder::from_hf_hub().unwrap();
        let v = embedder
            .embed_query(&[QueryText::new("hello world")])
            .unwrap()
            .pop()
            .unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    #[ignore]
    fn real_similar_texts_closer_than_unrelated() {
        if std::env::var("FASTRAG_EMBED_TEST").ok().as_deref() != Some("1") {
            return;
        }
        let embedder = BgeSmallEmbedder::from_hf_hub().unwrap();
        let a = embedder
            .embed_query(&[QueryText::new("cat sits on mat")])
            .unwrap()
            .pop()
            .unwrap();
        let b = embedder
            .embed_query(&[QueryText::new("a cat on the mat")])
            .unwrap()
            .pop()
            .unwrap();
        let c = embedder
            .embed_query(&[QueryText::new("Rust async runtime")])
            .unwrap()
            .pop()
            .unwrap();
        assert!(cosine(&a, &b) > cosine(&a, &c));
    }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;
    use crate::{Embedder, PrefixScheme};

    #[test]
    fn bge_model_id_matches_fastrag_namespace() {
        assert_eq!(BgeSmallEmbedder::MODEL_ID, "fastrag/bge-small-en-v1.5");
    }

    #[test]
    fn bge_dim_is_384() {
        assert_eq!(BgeSmallEmbedder::DIM, 384);
    }

    #[test]
    fn bge_prefix_scheme_is_none() {
        assert_eq!(
            BgeSmallEmbedder::PREFIX_SCHEME.hash(),
            PrefixScheme::NONE.hash()
        );
    }
}
