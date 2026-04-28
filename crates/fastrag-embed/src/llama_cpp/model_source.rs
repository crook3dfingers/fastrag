//! GGUF model path resolution with HuggingFace Hub auto-download fallback.
//!
//! Resolution order for `ModelSource::HfHub`:
//! 1. `$FASTRAG_MODEL_DIR/<file>` (if env var set and file exists)
//! 2. `<cache_base>/fastrag/models/<file>` (platform cache dir)
//! 3. Download via `ModelDownloader` trait

use std::path::{Path, PathBuf};

use crate::error::EmbedError;

/// Where to find the GGUF model file.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// An explicit local path — no network fallback.
    Local(PathBuf),
    /// HuggingFace Hub coordinates — resolved through the 3-tier fallback.
    HfHub {
        repo: &'static str,
        file: &'static str,
    },
}

/// Abstraction over model downloading so tests can mock it.
pub trait ModelDownloader {
    fn download(&self, repo: &str, file: &str, dest_dir: &Path) -> Result<PathBuf, EmbedError>;
}

/// Resolve a GGUF model path with explicit overrides for testability.
///
/// - `env_model_dir`: the caller reads `$FASTRAG_MODEL_DIR` and passes it in.
/// - `cache_base`: the caller reads `dirs::cache_dir()` and passes it in.
pub fn resolve_model_path(
    source: &ModelSource,
    env_model_dir: Option<&Path>,
    cache_base: Option<&Path>,
    downloader: &dyn ModelDownloader,
) -> Result<PathBuf, EmbedError> {
    match source {
        ModelSource::Local(p) => {
            if p.exists() {
                Ok(p.clone())
            } else {
                Err(EmbedError::MissingModelFile { path: p.clone() })
            }
        }
        ModelSource::HfHub { repo, file } => {
            // 1. Env override
            if let Some(dir) = env_model_dir {
                let candidate = dir.join(file);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
            // 2. Platform cache
            if let Some(base) = cache_base {
                let candidate = base.join("fastrag").join("models").join(file);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
            // 3. Download
            let dest_dir = cache_base.ok_or(EmbedError::NoCacheDir)?;
            let model_dir = dest_dir.join("fastrag").join("models");
            std::fs::create_dir_all(&model_dir)?;
            downloader.download(repo, file, &model_dir)
        }
    }
}

/// Convenience wrapper that reads `$FASTRAG_MODEL_DIR` and
/// `dirs::cache_dir()` for production use.
pub fn resolve_model_path_default(
    source: &ModelSource,
    downloader: &dyn ModelDownloader,
) -> Result<PathBuf, EmbedError> {
    let env_dir = std::env::var("FASTRAG_MODEL_DIR").ok().map(PathBuf::from);
    let cache_base = dirs::cache_dir();
    resolve_model_path(
        source,
        env_dir.as_deref(),
        cache_base.as_deref(),
        downloader,
    )
}

/// Default downloader that fetches from HuggingFace Hub via the sync API.
pub struct HfHubDownloader;

impl ModelDownloader for HfHubDownloader {
    fn download(&self, repo: &str, file: &str, dest_dir: &Path) -> Result<PathBuf, EmbedError> {
        let hf_cache = dest_dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(dest_dir)
            .join("fastrag")
            .join("hf-hub");
        std::fs::create_dir_all(&hf_cache)?;

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(hf_cache)
            .with_token(std::env::var("HF_TOKEN").ok())
            .build()
            .map_err(|e| EmbedError::HfHub(e.to_string()))?;
        let src = api.model(repo.to_string()).get(file)?;

        let dst = dest_dir.join(file);
        if !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::fs;

    struct MockDownloader {
        called: Cell<bool>,
        captured_repo: std::cell::RefCell<String>,
        captured_file: std::cell::RefCell<String>,
        return_path: PathBuf,
    }

    impl MockDownloader {
        fn new(return_path: PathBuf) -> Self {
            Self {
                called: Cell::new(false),
                captured_repo: std::cell::RefCell::new(String::new()),
                captured_file: std::cell::RefCell::new(String::new()),
                return_path,
            }
        }
    }

    impl ModelDownloader for MockDownloader {
        fn download(
            &self,
            repo: &str,
            file: &str,
            _dest_dir: &Path,
        ) -> Result<PathBuf, EmbedError> {
            self.called.set(true);
            *self.captured_repo.borrow_mut() = repo.to_string();
            *self.captured_file.borrow_mut() = file.to_string();
            Ok(self.return_path.clone())
        }
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("create tmpdir")
    }

    #[test]
    fn local_path_found() {
        let dir = tmpdir();
        let file = dir.path().join("model.gguf");
        fs::write(&file, b"fake").unwrap();
        let mock = MockDownloader::new(PathBuf::new());
        let result =
            resolve_model_path(&ModelSource::Local(file.clone()), None, None, &mock).unwrap();
        assert_eq!(result, file);
        assert!(
            !mock.called.get(),
            "downloader must not be called for Local"
        );
    }

    #[test]
    fn local_path_missing_errors() {
        let mock = MockDownloader::new(PathBuf::new());
        let bad = PathBuf::from("/nonexistent/model.gguf");
        let err =
            resolve_model_path(&ModelSource::Local(bad.clone()), None, None, &mock).unwrap_err();
        match err {
            EmbedError::MissingModelFile { path } => assert_eq!(path, bad),
            other => panic!("expected MissingModelFile, got {other:?}"),
        }
    }

    #[test]
    fn env_dir_takes_priority() {
        let env_dir = tmpdir();
        let file = env_dir.path().join("arctic-test.gguf");
        fs::write(&file, b"fake").unwrap();

        let mock = MockDownloader::new(PathBuf::new());
        let result = resolve_model_path(
            &ModelSource::HfHub {
                repo: "tarmotech/snowflake-arctic-embed-l-gguf-private",
                file: "arctic-test.gguf",
            },
            Some(env_dir.path()),
            None,
            &mock,
        )
        .unwrap();
        assert_eq!(result, file);
        assert!(
            !mock.called.get(),
            "downloader must not be called when env dir has file"
        );
    }

    #[test]
    fn cache_dir_fallback() {
        let cache_base = tmpdir();
        let model_dir = cache_base.path().join("fastrag").join("models");
        fs::create_dir_all(&model_dir).unwrap();
        let file = model_dir.join("arctic-test.gguf");
        fs::write(&file, b"fake").unwrap();

        let mock = MockDownloader::new(PathBuf::new());
        let result = resolve_model_path(
            &ModelSource::HfHub {
                repo: "tarmotech/snowflake-arctic-embed-l-gguf-private",
                file: "arctic-test.gguf",
            },
            None,
            Some(cache_base.path()),
            &mock,
        )
        .unwrap();
        assert_eq!(result, file);
        assert!(
            !mock.called.get(),
            "downloader must not be called when cache has file"
        );
    }

    #[test]
    fn downloader_invoked_when_missing() {
        let cache_base = tmpdir();
        let expected_path = cache_base.path().join("fastrag/models/arctic-test.gguf");
        let mock = MockDownloader::new(expected_path.clone());

        let result = resolve_model_path(
            &ModelSource::HfHub {
                repo: "tarmotech/snowflake-arctic-embed-l-gguf-private",
                file: "arctic-test.gguf",
            },
            None,
            Some(cache_base.path()),
            &mock,
        )
        .unwrap();
        assert!(
            mock.called.get(),
            "downloader must be called when file missing"
        );
        assert_eq!(
            *mock.captured_repo.borrow(),
            "tarmotech/snowflake-arctic-embed-l-gguf-private"
        );
        assert_eq!(*mock.captured_file.borrow(), "arctic-test.gguf");
        assert_eq!(result, expected_path);
    }
}
