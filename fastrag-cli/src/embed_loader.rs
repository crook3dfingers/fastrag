use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastrag::{BgeSmallEmbedder, Embedder};
use thiserror::Error;

use crate::args::EmbedderKindArg;

#[derive(Debug, Error)]
pub enum EmbedLoaderError {
    #[error("embedding model error: {0}")]
    Embed(#[from] fastrag::EmbedderError),
    #[error("unsupported model path: {0}")]
    UnsupportedModelPath(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse corpus manifest: {0}")]
    Manifest(String),
    #[error(
        "embedder mismatch: corpus built with `{existing}`, --embedder specifies `{requested}`"
    )]
    Mismatch { existing: String, requested: String },
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

pub fn load_for_write(opts: &EmbedderOptions) -> Result<Arc<dyn Embedder>, EmbedLoaderError> {
    let kind = opts.kind.unwrap_or(EmbedderKindArg::Bge);
    build(kind, opts)
}

pub fn load_for_read(
    corpus_dir: &Path,
    opts: &EmbedderOptions,
) -> Result<Arc<dyn Embedder>, EmbedLoaderError> {
    let manifest_path = corpus_dir.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| EmbedLoaderError::Manifest(e.to_string()))?;
    let existing = value
        .get("embedding_model_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| EmbedLoaderError::Manifest("missing embedding_model_id".into()))?
        .to_string();

    let (detected_kind, model_override) = detect_from_manifest(&existing)?;
    let kind = opts.kind.unwrap_or(detected_kind);

    if kind != detected_kind {
        return Err(EmbedLoaderError::Mismatch {
            existing,
            requested: kind_name(kind).to_string(),
        });
    }

    let mut effective = opts.clone();
    if let Some(m) = model_override {
        match kind {
            EmbedderKindArg::Openai => effective.openai_model = m,
            EmbedderKindArg::Ollama => effective.ollama_model = m,
            EmbedderKindArg::Bge => {}
        }
    }

    let emb = build(kind, &effective)?;
    let requested = emb.model_id();
    if requested != existing {
        return Err(EmbedLoaderError::Mismatch {
            existing,
            requested,
        });
    }
    Ok(emb)
}

fn detect_from_manifest(
    existing: &str,
) -> Result<(EmbedderKindArg, Option<String>), EmbedLoaderError> {
    if let Some(rest) = existing.strip_prefix("openai:") {
        Ok((EmbedderKindArg::Openai, Some(rest.to_string())))
    } else if let Some(rest) = existing.strip_prefix("ollama:") {
        Ok((EmbedderKindArg::Ollama, Some(rest.to_string())))
    } else if existing.starts_with("fastrag/bge") {
        Ok((EmbedderKindArg::Bge, None))
    } else {
        Err(EmbedLoaderError::Manifest(format!(
            "unrecognized embedding_model_id `{existing}`; pass --embedder explicitly"
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

fn build(
    kind: EmbedderKindArg,
    opts: &EmbedderOptions,
) -> Result<Arc<dyn Embedder>, EmbedLoaderError> {
    match kind {
        EmbedderKindArg::Bge => {
            let e = match &opts.model_path {
                Some(path) => BgeSmallEmbedder::from_local(path)?,
                None => BgeSmallEmbedder::from_hf_hub()?,
            };
            Ok(Arc::new(e))
        }
        EmbedderKindArg::Openai => {
            use fastrag_embed::http::openai::OpenAIEmbedder;
            let e = OpenAIEmbedder::new(opts.openai_model.clone())?
                .with_base_url(opts.openai_base_url.clone());
            Ok(Arc::new(e))
        }
        EmbedderKindArg::Ollama => {
            use fastrag_embed::http::ollama::OllamaEmbedder;
            unsafe { std::env::set_var("OLLAMA_HOST", &opts.ollama_url) };
            let e = OllamaEmbedder::new(opts.ollama_model.clone())?;
            Ok(Arc::new(e))
        }
    }
}
