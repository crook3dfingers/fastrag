//! 3-tier model directory resolution for ONNX reranker files.
//!
//! Unlike GGUF models (single file), ONNX rerankers require a directory
//! containing at least `model.onnx` and `tokenizer.json`.
//!
//! Resolution order:
//! 1. `$FASTRAG_MODEL_DIR/<model_name>/` (if env var set and both files exist)
//! 2. `<cache_base>/fastrag/models/<model_name>/` (platform cache dir)
//! 3. Download both files via `OnnxModelDownloader` trait

use std::path::{Path, PathBuf};

use crate::RerankError;

/// Files required in the model directory.
pub const MODEL_FILE: &str = "model.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Coordinates for an ONNX model on HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct OnnxModelSource {
    /// HuggingFace repo ID (e.g. `"Alibaba-NLP/gte-reranker-modernbert-base"`)
    pub repo: &'static str,
    /// Subdirectory name used in the local cache (e.g. `"gte-reranker-modernbert-base"`)
    pub dir_name: &'static str,
    /// Path to the ONNX file within the repo (e.g. `"onnx/model.onnx"` or `"model.onnx"`)
    pub model_path_in_repo: &'static str,
    /// Path to the tokenizer file within the repo
    pub tokenizer_path_in_repo: &'static str,
}

/// Abstraction over model downloading so tests can mock it.
pub trait OnnxModelDownloader {
    fn download_file(&self, repo: &str, file_in_repo: &str, dest: &Path)
    -> Result<(), RerankError>;
}

/// Check if a model directory has both required files.
fn dir_is_complete(dir: &Path) -> bool {
    dir.join(MODEL_FILE).exists() && dir.join(TOKENIZER_FILE).exists()
}

/// Resolve a model directory with explicit overrides for testability.
pub fn resolve_model_dir(
    source: &OnnxModelSource,
    env_model_dir: Option<&Path>,
    cache_base: Option<&Path>,
    downloader: &dyn OnnxModelDownloader,
) -> Result<PathBuf, RerankError> {
    // 1. Env override
    if let Some(dir) = env_model_dir {
        let candidate = dir.join(source.dir_name);
        if dir_is_complete(&candidate) {
            return Ok(candidate);
        }
    }
    // 2. Platform cache
    if let Some(base) = cache_base {
        let candidate = base.join("fastrag").join("models").join(source.dir_name);
        if dir_is_complete(&candidate) {
            return Ok(candidate);
        }
    }
    // 3. Download
    let dest_base = cache_base.ok_or_else(|| {
        RerankError::Model("no cache directory available for model download".into())
    })?;
    let model_dir = dest_base
        .join("fastrag")
        .join("models")
        .join(source.dir_name);
    std::fs::create_dir_all(&model_dir)?;

    let model_dest = model_dir.join(MODEL_FILE);
    if !model_dest.exists() {
        downloader.download_file(source.repo, source.model_path_in_repo, &model_dest)?;
    }

    let tokenizer_dest = model_dir.join(TOKENIZER_FILE);
    if !tokenizer_dest.exists() {
        downloader.download_file(source.repo, source.tokenizer_path_in_repo, &tokenizer_dest)?;
    }

    Ok(model_dir)
}

/// Convenience wrapper that reads env vars and cache dir for production use.
pub fn resolve_model_dir_default(
    source: &OnnxModelSource,
    downloader: &dyn OnnxModelDownloader,
) -> Result<PathBuf, RerankError> {
    let env_dir = std::env::var("FASTRAG_MODEL_DIR").ok().map(PathBuf::from);
    let cache_base = dirs::cache_dir();
    resolve_model_dir(
        source,
        env_dir.as_deref(),
        cache_base.as_deref(),
        downloader,
    )
}

/// Default downloader that fetches from HuggingFace Hub via the sync API.
pub struct HfHubOnnxDownloader;

impl OnnxModelDownloader for HfHubOnnxDownloader {
    fn download_file(
        &self,
        repo: &str,
        file_in_repo: &str,
        dest: &Path,
    ) -> Result<(), RerankError> {
        let cache_dir = dest
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .unwrap_or(dest)
            .join("hf-hub");
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| RerankError::Model(format!("create hf-hub cache dir: {e}")))?;

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .map_err(|e| RerankError::Model(format!("hf-hub API: {e}")))?;

        let src = api
            .model(repo.to_string())
            .get(file_in_repo)
            .map_err(|e| RerankError::Model(format!("hf-hub download {file_in_repo}: {e}")))?;

        std::fs::copy(&src, dest)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::fs;

    struct MockDownloader {
        calls: RefCell<Vec<(String, String)>>,
    }

    impl MockDownloader {
        fn new() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl OnnxModelDownloader for MockDownloader {
        fn download_file(
            &self,
            repo: &str,
            file_in_repo: &str,
            dest: &Path,
        ) -> Result<(), RerankError> {
            self.calls
                .borrow_mut()
                .push((repo.to_string(), file_in_repo.to_string()));
            // Simulate download by creating the file
            fs::write(dest, b"fake-model-data")?;
            Ok(())
        }
    }

    fn source() -> OnnxModelSource {
        OnnxModelSource {
            repo: "Alibaba-NLP/gte-reranker-modernbert-base",
            dir_name: "gte-reranker-modernbert-base",
            model_path_in_repo: "onnx/model.onnx",
            tokenizer_path_in_repo: "tokenizer.json",
        }
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("create tmpdir")
    }

    #[test]
    fn env_dir_takes_priority() {
        let env_dir = tmpdir();
        let model_dir = env_dir.path().join("gte-reranker-modernbert-base");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join(MODEL_FILE), b"fake").unwrap();
        fs::write(model_dir.join(TOKENIZER_FILE), b"fake").unwrap();

        let mock = MockDownloader::new();
        let result = resolve_model_dir(&source(), Some(env_dir.path()), None, &mock).unwrap();
        assert_eq!(result, model_dir);
        assert!(
            mock.calls.borrow().is_empty(),
            "downloader must not be called"
        );
    }

    #[test]
    fn env_dir_incomplete_falls_through() {
        let env_dir = tmpdir();
        let model_dir = env_dir.path().join("gte-reranker-modernbert-base");
        fs::create_dir_all(&model_dir).unwrap();
        // Only model.onnx, missing tokenizer.json
        fs::write(model_dir.join(MODEL_FILE), b"fake").unwrap();

        let cache_base = tmpdir();
        let mock = MockDownloader::new();
        let result = resolve_model_dir(
            &source(),
            Some(env_dir.path()),
            Some(cache_base.path()),
            &mock,
        )
        .unwrap();
        // Should download to cache dir
        assert!(result.starts_with(cache_base.path()));
        assert_eq!(mock.calls.borrow().len(), 2);
    }

    #[test]
    fn cache_dir_found() {
        let cache_base = tmpdir();
        let model_dir = cache_base
            .path()
            .join("fastrag")
            .join("models")
            .join("gte-reranker-modernbert-base");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join(MODEL_FILE), b"fake").unwrap();
        fs::write(model_dir.join(TOKENIZER_FILE), b"fake").unwrap();

        let mock = MockDownloader::new();
        let result = resolve_model_dir(&source(), None, Some(cache_base.path()), &mock).unwrap();
        assert_eq!(result, model_dir);
        assert!(mock.calls.borrow().is_empty());
    }

    #[test]
    fn downloads_when_missing() {
        let cache_base = tmpdir();
        let mock = MockDownloader::new();

        let result = resolve_model_dir(&source(), None, Some(cache_base.path()), &mock).unwrap();

        let expected = cache_base
            .path()
            .join("fastrag")
            .join("models")
            .join("gte-reranker-modernbert-base");
        assert_eq!(result, expected);

        let calls = mock.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].1, "onnx/model.onnx");
        assert_eq!(calls[1].1, "tokenizer.json");
    }

    #[test]
    fn skips_already_downloaded_files() {
        let cache_base = tmpdir();
        let model_dir = cache_base
            .path()
            .join("fastrag")
            .join("models")
            .join("gte-reranker-modernbert-base");
        fs::create_dir_all(&model_dir).unwrap();
        // model.onnx exists but tokenizer.json doesn't
        fs::write(model_dir.join(MODEL_FILE), b"existing").unwrap();

        let mock = MockDownloader::new();
        let _result = resolve_model_dir(&source(), None, Some(cache_base.path()), &mock).unwrap();

        let calls = mock.calls.borrow();
        assert_eq!(calls.len(), 1, "should only download tokenizer");
        assert_eq!(calls[0].1, "tokenizer.json");
    }

    #[test]
    fn no_cache_dir_errors() {
        let mock = MockDownloader::new();
        let err = resolve_model_dir(&source(), None, None, &mock).unwrap_err();
        assert!(
            matches!(err, RerankError::Model(_)),
            "expected Model error, got {err:?}"
        );
    }
}
