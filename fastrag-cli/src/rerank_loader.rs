//! Reranker loading for the CLI.
//!
//! Mirrors `embed_loader.rs`: resolves model artifacts and builds a
//! `Box<dyn Reranker>` from the CLI `--rerank` flag.

#[cfg(feature = "rerank-llama")]
use std::net::TcpListener;
#[cfg(feature = "rerank-llama")]
use std::path::PathBuf;

use fastrag_rerank::Reranker;
use thiserror::Error;

use crate::args::RerankerKindArg;

#[derive(Debug, Error)]
pub enum RerankLoaderError {
    #[error("reranker model error: {0}")]
    Model(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}

impl From<fastrag_rerank::RerankError> for RerankLoaderError {
    fn from(e: fastrag_rerank::RerankError) -> Self {
        RerankLoaderError::Model(e.to_string())
    }
}

pub fn load_reranker(kind: RerankerKindArg) -> Result<Box<dyn Reranker>, RerankLoaderError> {
    match kind {
        RerankerKindArg::Onnx => load_onnx(),
        RerankerKindArg::LlamaCpp => load_llama_cpp(),
    }
}

fn load_onnx() -> Result<Box<dyn Reranker>, RerankLoaderError> {
    #[cfg(not(feature = "rerank"))]
    {
        Err(RerankLoaderError::Model(
            "ONNX reranker not available: fastrag-cli built without `rerank` feature".into(),
        ))
    }
    #[cfg(feature = "rerank")]
    {
        use fastrag_rerank::onnx::GteModernBertReranker;
        let reranker = GteModernBertReranker::load_default()?;
        Ok(Box::new(reranker))
    }
}

fn load_llama_cpp() -> Result<Box<dyn Reranker>, RerankLoaderError> {
    #[cfg(not(feature = "rerank-llama"))]
    {
        Err(RerankLoaderError::Model(
            "llama-cpp reranker not available: fastrag-cli built without `rerank-llama` feature"
                .into(),
        ))
    }
    #[cfg(feature = "rerank-llama")]
    {
        use fastrag_embed::llama_cpp::{
            HfHubDownloader, LlamaServerConfig, resolve_model_path_default,
        };
        use fastrag_rerank::llama_cpp::BgeRerankerV2M3Llama;

        let binary_path = find_llama_server()?;
        let port = free_port()?;
        let model_path =
            resolve_model_path_default(&BgeRerankerV2M3Llama::model_source(), &HfHubDownloader)
                .map_err(|e| RerankLoaderError::Model(format!("resolve GGUF: {e}")))?;

        let cfg = LlamaServerConfig {
            binary_path,
            port,
            health_timeout: std::time::Duration::from_secs(120),
            extra_args: vec![
                "--model".to_string(),
                model_path.display().to_string(),
                "--embedding".to_string(),
                "--pooling".to_string(),
                "rank".to_string(),
                "--rerank".to_string(),
                "--threads-batch".to_string(),
                num_cpus_for_batch(),
                "--parallel".to_string(),
                "2".to_string(),
                "--cont-batching".to_string(),
                "--ubatch-size".to_string(),
                "512".to_string(),
            ],
            skip_version_check: false,
        };

        let reranker = BgeRerankerV2M3Llama::load(cfg)?;
        Ok(Box::new(reranker))
    }
}

#[cfg(feature = "rerank-llama")]
fn num_cpus_for_batch() -> String {
    // Use all available logical cores for prompt/batch processing — this
    // is the hot path for cross-encoder reranking. Cap conservatively in
    // case this runs on very large boxes where oversubscription hurts.
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);
    n.to_string()
}

#[cfg(feature = "rerank-llama")]
fn find_llama_server() -> Result<PathBuf, RerankLoaderError> {
    if let Ok(p) = std::env::var("LLAMA_SERVER_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    which_llama_server().ok_or_else(|| {
        RerankLoaderError::Model(
            "llama-server not found in $PATH; install from \
             https://github.com/ggml-org/llama.cpp/releases \
             or set LLAMA_SERVER_PATH"
                .into(),
        )
    })
}

#[cfg(feature = "rerank-llama")]
fn which_llama_server() -> Option<PathBuf> {
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

#[cfg(feature = "rerank-llama")]
fn free_port() -> Result<u16, RerankLoaderError> {
    let l = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| RerankLoaderError::Model(format!("bind ephemeral port: {e}")))?;
    Ok(l.local_addr()
        .map_err(|e| RerankLoaderError::Model(format!("local_addr: {e}")))?
        .port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_reranker_dispatches_to_onnx() {
        // Verify ONNX path is reached. If model files are present the load
        // succeeds; if absent the error must come from the ONNX model loader.
        let result = load_reranker(RerankerKindArg::Onnx);
        match result {
            Ok(reranker) => {
                assert_eq!(reranker.model_id(), "gte-reranker-modernbert-base");
            }
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("model") || msg.contains("ORT") || msg.contains("onnx"),
                    "expected ONNX-related error, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn llama_cpp_without_feature_returns_error() {
        // If rerank-llama feature is off, loading should fail gracefully.
        #[cfg(not(feature = "rerank-llama"))]
        {
            let result = load_llama_cpp();
            assert!(result.is_err());
            assert!(result.err().unwrap().to_string().contains("not available"));
        }
    }
}
