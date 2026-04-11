//! Helpers for bringing up the contextualization stack at CLI startup.
//!
//! Compiled only when `--features contextual` is active. The
//! [`ContextState`] struct owns the `llama-server` subprocess (via
//! [`LlamaServerHandle`]), the chat client, the backend contextualizer, and
//! the SQLite cache. Dropping it tears the subprocess down in RAII order.

#![cfg(feature = "contextual")]

use std::path::Path;

use thiserror::Error;

use fastrag_context::Contextualizer;
#[cfg(feature = "contextual-llama")]
use fastrag_context::LlamaCppContextualizer;
use fastrag_context::{ContextCache, ContextError};
#[cfg(feature = "contextual-llama")]
use fastrag_embed::llama_cpp::{
    DefaultCompletionPreset, HfHubDownloader, LlamaCppChatClient, LlamaServerConfig,
    LlamaServerHandle, resolve_model_path_default,
};
#[cfg(feature = "contextual-llama")]
use std::net::TcpListener;
#[cfg(feature = "contextual-llama")]
use std::path::PathBuf;

/// Error surface for `context_loader::load_context_state`.
#[derive(Debug, Error)]
pub enum ContextLoaderError {
    #[error("contextualization is only supported with the `contextual-llama` feature enabled")]
    BackendNotCompiled,
    #[cfg(feature = "contextual-llama")]
    #[error(
        "llama-server not found in $PATH; install from https://github.com/ggml-org/llama.cpp/releases or set LLAMA_SERVER_PATH"
    )]
    LlamaServerNotFound,
    #[cfg(feature = "contextual-llama")]
    #[error("failed to resolve completion GGUF: {0}")]
    ResolveGguf(String),
    #[cfg(feature = "contextual-llama")]
    #[error("failed to spawn llama-server: {0}")]
    Spawn(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("cache error: {0}")]
    Cache(#[from] ContextError),
}

/// Live contextualization state. Drop order:
/// 1. `cache` (SQLite connection)
/// 2. `contextualizer` (blocking HTTP client — no teardown work)
/// 3. `_server` (kills the `llama-server` subprocess via [`LlamaServerHandle::Drop`])
///
/// Keep `_server` last in the struct so it outlives the contextualizer that
/// depends on its HTTP endpoint. The `contextualizer` field is held as a
/// trait object so the struct definition is feature-flag–independent; in
/// builds without `contextual-llama` no construction path exists.
pub struct ContextState {
    pub cache: ContextCache,
    pub contextualizer: Box<dyn Contextualizer>,
    /// Private — dropped last to tear down the llama-server subprocess.
    #[cfg(feature = "contextual-llama")]
    _server: LlamaServerHandle,
}

/// Bring up the completion server, build a [`LlamaCppContextualizer`], and
/// open the SQLite cache at `<corpus>/contextualization.sqlite`.
///
/// The corpus directory is created if missing so first-time ingest can open
/// the cache before the index is persisted.
#[cfg(feature = "contextual-llama")]
pub fn load_context_state(corpus: &Path) -> Result<ContextState, ContextLoaderError> {
    std::fs::create_dir_all(corpus)?;

    let binary_path = find_llama_server().ok_or(ContextLoaderError::LlamaServerNotFound)?;
    let model_path =
        resolve_model_path_default(&DefaultCompletionPreset::model_source(), &HfHubDownloader)
            .map_err(|e| ContextLoaderError::ResolveGguf(e.to_string()))?;
    let port = free_port()?;
    let cfg = LlamaServerConfig {
        binary_path,
        port,
        health_timeout: std::time::Duration::from_secs(120),
        extra_args: vec![
            "--model".to_string(),
            model_path.display().to_string(),
            "--ctx-size".to_string(),
            DefaultCompletionPreset::CONTEXT_WINDOW.to_string(),
        ],
        skip_version_check: false,
    };
    let server =
        LlamaServerHandle::spawn(cfg).map_err(|e| ContextLoaderError::Spawn(e.to_string()))?;
    let client = LlamaCppChatClient::new(
        server.base_url().to_string(),
        DefaultCompletionPreset::MODEL_ID,
    );
    let contextualizer = LlamaCppContextualizer::new(client, DefaultCompletionPreset::MODEL_ID);

    let cache_path = corpus.join("contextualization.sqlite");
    let cache = ContextCache::open(&cache_path)?;

    Ok(ContextState {
        cache,
        contextualizer: Box::new(contextualizer),
        _server: server,
    })
}

/// Fallback for builds that enable `contextual` without `contextual-llama`.
/// Contextualization has no working backend in that configuration, so the
/// loader fails fast with a clear error.
#[cfg(not(feature = "contextual-llama"))]
pub fn load_context_state(_corpus: &Path) -> Result<ContextState, ContextLoaderError> {
    Err(ContextLoaderError::BackendNotCompiled)
}

#[cfg(feature = "contextual-llama")]
fn find_llama_server() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LLAMA_SERVER_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let output = std::process::Command::new("which")
        .arg("llama-server")
        .output()
        .ok()?;
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout);
        let path = PathBuf::from(s.trim());
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(feature = "contextual-llama")]
fn free_port() -> Result<u16, ContextLoaderError> {
    let l = TcpListener::bind("127.0.0.1:0")?;
    Ok(l.local_addr()?.port())
}
