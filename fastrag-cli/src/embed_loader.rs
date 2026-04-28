use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastrag::DynEmbedder;
use fastrag_embed::{
    DynEmbedderTrait, EmbedError, EmbedderIdentity, PrefixScheme,
    http::{
        ollama::OllamaEmbedder,
        openai::{OpenAiLarge, OpenAiSmall},
    },
    llama_cpp::{GenericLlamaCppEmbedder, LlamaServerConfig},
};
use serde::Deserialize;
use thiserror::Error;

use crate::embed_profile::{EmbedBackend, PrefixConfig, ResolvedEmbedderProfile};

const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";

#[derive(Debug, Error)]
pub enum EmbedLoaderError {
    #[error("embedding model error: {0}")]
    Embed(String),
    #[error("unsupported model path: {0}")]
    UnsupportedModelPath(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<EmbedError> for EmbedLoaderError {
    fn from(e: EmbedError) -> Self {
        EmbedLoaderError::Embed(e.to_string())
    }
}

#[derive(Deserialize)]
struct ManifestIdentity {
    identity: EmbedderIdentity,
}

pub fn runtime_identity_for_profile(
    profile: &ResolvedEmbedderProfile,
) -> Result<EmbedderIdentity, EmbedLoaderError> {
    Ok(EmbedderIdentity {
        model_id: format!("{}:{}", backend_name(profile.backend), profile.model),
        dim: profile.dim_override.unwrap_or(0),
        prefix_scheme_hash: prefix_scheme_hash(&profile.prefix),
    })
}

pub fn load_from_profile(
    profile: &ResolvedEmbedderProfile,
) -> Result<DynEmbedder, EmbedLoaderError> {
    validate_profile(profile)?;
    match profile.backend {
        EmbedBackend::Openai => load_openai(profile),
        EmbedBackend::Ollama => load_ollama(profile),
        EmbedBackend::LlamaCpp => load_llama_cpp(profile),
    }
}

fn validate_profile(profile: &ResolvedEmbedderProfile) -> Result<(), EmbedLoaderError> {
    match profile.backend {
        EmbedBackend::Openai => {
            if has_prefix_override(&profile.prefix) {
                return Err(EmbedLoaderError::Embed(format!(
                    "embedder profile `{}` on backend `{}` has an unsupported prefix override",
                    profile.name,
                    backend_name(profile.backend)
                )));
            }
            if profile.dim_override.is_some() {
                return Err(EmbedLoaderError::Embed(format!(
                    "embedder profile `{}` on backend `{}` has an unsupported dim override",
                    profile.name,
                    backend_name(profile.backend)
                )));
            }
            Ok(())
        }
        EmbedBackend::Ollama | EmbedBackend::LlamaCpp => Ok(()),
    }
}

fn has_prefix_override(prefix: &PrefixConfig) -> bool {
    !prefix.query.is_empty() || !prefix.passage.is_empty()
}

pub fn load_from_manifest(corpus_dir: &Path) -> Result<DynEmbedder, EmbedLoaderError> {
    let manifest_path = corpus_dir.join("manifest.json");
    let manifest: ManifestIdentity = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .map_err(|e| EmbedLoaderError::Embed(format!("failed to parse corpus manifest: {e}")))?;

    if manifest.identity.prefix_scheme_hash != PrefixScheme::NONE.hash() {
        return Err(EmbedLoaderError::Embed(
            "cannot auto-load a prefix-aware embedder from corpus manifest; use a resolved profile instead"
                .into(),
        ));
    }

    let profile = profile_from_manifest_identity(manifest.identity)?;
    load_from_profile(&profile)
}

fn load_openai(profile: &ResolvedEmbedderProfile) -> Result<DynEmbedder, EmbedLoaderError> {
    match profile.model.as_str() {
        "text-embedding-3-small" => {
            let mut embedder = OpenAiSmall::new()?;
            if let Some(base_url) = &profile.base_url {
                embedder = embedder.with_base_url(base_url.clone());
            }
            Ok(Arc::new(embedder) as Arc<dyn DynEmbedderTrait>)
        }
        "text-embedding-3-large" => {
            let mut embedder = OpenAiLarge::new()?;
            if let Some(base_url) = &profile.base_url {
                embedder = embedder.with_base_url(base_url.clone());
            }
            Ok(Arc::new(embedder) as Arc<dyn DynEmbedderTrait>)
        }
        other => Err(EmbedLoaderError::Embed(format!(
            "unknown OpenAI model `{other}`; supported: text-embedding-3-small, text-embedding-3-large"
        ))),
    }
}

fn load_ollama(profile: &ResolvedEmbedderProfile) -> Result<DynEmbedder, EmbedLoaderError> {
    let embedder = OllamaEmbedder::new_with_prefix(
        profile.model.clone(),
        resolve_ollama_base_url(
            profile.base_url.as_deref(),
            std::env::var("OLLAMA_HOST").ok().as_deref(),
        ),
        prefix_scheme_for_config(&profile.prefix),
    )?;
    Ok(Arc::new(embedder) as Arc<dyn DynEmbedderTrait>)
}

pub fn resolve_ollama_base_url(profile_base_url: Option<&str>, env_host: Option<&str>) -> String {
    profile_base_url
        .or(env_host)
        .unwrap_or(DEFAULT_OLLAMA_BASE_URL)
        .to_string()
}

fn load_llama_cpp(profile: &ResolvedEmbedderProfile) -> Result<DynEmbedder, EmbedLoaderError> {
    let model_path = PathBuf::from(&profile.model);
    if !model_path.exists() {
        return Err(EmbedLoaderError::UnsupportedModelPath(model_path));
    }

    // Bump physical + logical batch from llama.cpp's 512 default. Embedder
    // chunks can exceed 512 tokens (e.g. VIPER playbook sections at 1k chars
    // tokenize to 540+ tokens with Nomic), and llama-server returns HTTP 500
    // "input is too large to process. increase the physical batch size" if
    // any single input exceeds --ubatch. 4096 fits Nomic v1.5 (8192 ctx) and
    // is harmless for smaller-context embedders (batch caps compute size,
    // not max input length — the model arch still gates that).
    //
    // GPU offload: opt-in via FASTRAG_LLAMA_NGL (default unset = CPU). Set
    // to 999 for full offload. CPU-only stays the default so deployments
    // without a GPU don't fail on a missing CUDA/Vulkan backend.
    let mut extra_args = vec![
        "--model".to_string(),
        profile.model.clone(),
        "--embedding".to_string(),
        "--pooling".to_string(),
        "mean".to_string(),
        "-ub".to_string(),
        "4096".to_string(),
        "-b".to_string(),
        "4096".to_string(),
    ];
    if let Ok(ngl) = std::env::var("FASTRAG_LLAMA_NGL") {
        if !ngl.is_empty() {
            extra_args.push("-ngl".to_string());
            extra_args.push(ngl);
        }
    }
    let cfg = LlamaServerConfig {
        binary_path: find_llama_server()?,
        port: free_port()?,
        health_timeout: std::time::Duration::from_secs(120),
        extra_args,
        skip_version_check: false,
    };

    let embedder = GenericLlamaCppEmbedder::load(
        cfg,
        profile.model.clone(),
        prefix_scheme_for_config(&profile.prefix),
    )?;
    Ok(Arc::new(embedder) as Arc<dyn DynEmbedderTrait>)
}

fn backend_name(backend: EmbedBackend) -> &'static str {
    match backend {
        EmbedBackend::Openai => "openai",
        EmbedBackend::Ollama => "ollama",
        EmbedBackend::LlamaCpp => "llama-cpp",
    }
}

fn profile_from_manifest_identity(
    identity: EmbedderIdentity,
) -> Result<ResolvedEmbedderProfile, EmbedLoaderError> {
    let (backend, model) = if let Some(model) = identity.model_id.strip_prefix("openai:") {
        (EmbedBackend::Openai, model.to_string())
    } else if let Some(model) = identity.model_id.strip_prefix("ollama:") {
        (EmbedBackend::Ollama, model.to_string())
    } else if let Some(model) = identity.model_id.strip_prefix("llama-cpp:") {
        (EmbedBackend::LlamaCpp, model.to_string())
    } else {
        return Err(EmbedLoaderError::Embed(format!(
            "unsupported corpus manifest embedder `{}`",
            identity.model_id
        )));
    };

    Ok(ResolvedEmbedderProfile {
        name: "manifest".into(),
        backend,
        model,
        base_url: None,
        prefix: PrefixConfig::default(),
        dim_override: manifest_dim_override(backend, identity.dim),
    })
}

fn manifest_dim_override(backend: EmbedBackend, dim: usize) -> Option<usize> {
    match backend {
        EmbedBackend::Openai => None,
        EmbedBackend::Ollama | EmbedBackend::LlamaCpp => Some(dim),
    }
}

fn prefix_scheme_for_config(prefix: &PrefixConfig) -> PrefixScheme {
    PrefixScheme::new(
        intern_prefix(prefix.query.clone()),
        intern_prefix(prefix.passage.clone()),
    )
}

fn intern_prefix(prefix: String) -> &'static str {
    if prefix.is_empty() {
        ""
    } else {
        Box::leak(prefix.into_boxed_str())
    }
}

fn prefix_scheme_hash(prefix: &PrefixConfig) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in prefix.query.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0;
    hash = hash.wrapping_mul(0x100000001b3);
    for &byte in prefix.passage.as_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn find_llama_server() -> Result<PathBuf, EmbedLoaderError> {
    if let Ok(path) = std::env::var("LLAMA_SERVER_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
    }

    which_llama_server().ok_or_else(|| {
        EmbedLoaderError::Embed(
            "llama-server not found in $PATH; install from https://github.com/ggml-org/llama.cpp/releases or set LLAMA_SERVER_PATH"
                .into(),
        )
    })
}

fn which_llama_server() -> Option<PathBuf> {
    let output = std::process::Command::new("which")
        .arg("llama-server")
        .output()
        .ok()?;
    if output.status.success() {
        let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn free_port() -> Result<u16, EmbedLoaderError> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| EmbedLoaderError::Embed(format!("bind ephemeral port: {e}")))?;
    Ok(listener
        .local_addr()
        .map_err(|e| EmbedLoaderError::Embed(format!("local_addr: {e}")))?
        .port())
}
