use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastrag::DynEmbedder;
use fastrag_embed::{
    BgeSmallEmbedder, DynEmbedderTrait, Embedder,
    http::{
        ollama::OllamaEmbedder,
        openai::{OpenAiLarge, OpenAiSmall},
    },
};
use thiserror::Error;

use crate::args::EmbedderKindArg;

#[derive(Debug, Error)]
pub enum EmbedLoaderError {
    #[error("embedding model error: {0}")]
    Embed(String),
    #[error("unsupported model path: {0}")]
    UnsupportedModelPath(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse corpus manifest: {0}")]
    Manifest(String),
    #[error(
        "embedder identity mismatch: corpus built with `{existing}`, --embedder specifies `{requested}`"
    )]
    KindMismatch { existing: String, requested: String },
}

impl From<fastrag_embed::EmbedError> for EmbedLoaderError {
    fn from(e: fastrag_embed::EmbedError) -> Self {
        EmbedLoaderError::Embed(e.to_string())
    }
}

#[derive(Clone)]
pub struct EmbedderOptions {
    pub kind: Option<EmbedderKindArg>,
    pub model_path: Option<PathBuf>,
    pub openai_model: String,
    pub openai_base_url: String,
    pub ollama_model: String,
    pub ollama_url: String,
}

pub fn load_for_write(opts: &EmbedderOptions) -> Result<DynEmbedder, EmbedLoaderError> {
    let kind = opts.kind.unwrap_or(EmbedderKindArg::Bge);
    build(kind, opts)
}

pub fn load_for_read(
    corpus_dir: &Path,
    opts: &EmbedderOptions,
) -> Result<DynEmbedder, EmbedLoaderError> {
    let manifest_path = corpus_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| EmbedLoaderError::Manifest(e.to_string()))?;
    let existing = value
        .get("identity")
        .and_then(|i| i.get("model_id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| EmbedLoaderError::Manifest("missing identity.model_id".into()))?
        .to_string();

    let detected_kind = detect_from_model_id(&existing)?;
    let kind = opts.kind.unwrap_or(detected_kind);
    if kind != detected_kind {
        return Err(EmbedLoaderError::KindMismatch {
            existing,
            requested: kind_name(kind).to_string(),
        });
    }

    let mut effective = opts.clone();
    if let Some(rest) = existing.strip_prefix("openai:") {
        effective.openai_model = rest.to_string();
    } else if let Some(rest) = existing.strip_prefix("ollama:") {
        effective.ollama_model = rest.to_string();
    }

    build(kind, &effective)
}

fn detect_from_model_id(existing: &str) -> Result<EmbedderKindArg, EmbedLoaderError> {
    if existing.starts_with("openai:") {
        Ok(EmbedderKindArg::Openai)
    } else if existing.starts_with("ollama:") {
        Ok(EmbedderKindArg::Ollama)
    } else if existing == BgeSmallEmbedder::MODEL_ID {
        Ok(EmbedderKindArg::Bge)
    } else {
        Err(EmbedLoaderError::Manifest(format!(
            "unrecognized identity.model_id `{existing}`; pass --embedder explicitly"
        )))
    }
}

fn kind_name(kind: EmbedderKindArg) -> &'static str {
    match kind {
        EmbedderKindArg::Bge => "bge",
        EmbedderKindArg::Openai => "openai",
        EmbedderKindArg::Ollama => "ollama",
    }
}

fn build(kind: EmbedderKindArg, opts: &EmbedderOptions) -> Result<DynEmbedder, EmbedLoaderError> {
    match kind {
        EmbedderKindArg::Bge => {
            let e = match &opts.model_path {
                Some(path) => BgeSmallEmbedder::from_local(path)?,
                None => BgeSmallEmbedder::from_hf_hub()?,
            };
            let arc: Arc<dyn DynEmbedderTrait> = Arc::new(e);
            Ok(arc)
        }
        EmbedderKindArg::Openai => match opts.openai_model.as_str() {
            "text-embedding-3-small" => {
                let e = OpenAiSmall::new()?.with_base_url(opts.openai_base_url.clone());
                let arc: Arc<dyn DynEmbedderTrait> = Arc::new(e);
                Ok(arc)
            }
            "text-embedding-3-large" => {
                let e = OpenAiLarge::new()?.with_base_url(opts.openai_base_url.clone());
                let arc: Arc<dyn DynEmbedderTrait> = Arc::new(e);
                Ok(arc)
            }
            other => Err(EmbedLoaderError::Manifest(format!(
                "unknown OpenAI model `{other}` — supported: text-embedding-3-small, text-embedding-3-large"
            ))),
        },
        EmbedderKindArg::Ollama => {
            unsafe { std::env::set_var("OLLAMA_HOST", &opts.ollama_url) };
            let e = OllamaEmbedder::new(opts.ollama_model.clone())?;
            let arc: Arc<dyn DynEmbedderTrait> = Arc::new(e);
            Ok(arc)
        }
    }
}
