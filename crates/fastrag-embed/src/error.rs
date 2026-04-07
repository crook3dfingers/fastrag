use std::path::PathBuf;

/// Errors that can occur during embedding.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to locate a cache directory for model downloads")]
    NoCacheDir,

    #[error("missing required model file: {path}")]
    MissingModelFile { path: PathBuf },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("candle error: {0}")]
    Candle(String),

    #[error("hf-hub error: {0}")]
    HfHub(String),

    #[error("unexpected embedding dimension: expected {expected}, got {got}")]
    UnexpectedDim { expected: usize, got: usize },

    #[error("missing required environment variable: {0}")]
    MissingEnv(&'static str),

    #[error("http transport error: {0}")]
    Http(String),

    #[error("api error: status {status}: {message}")]
    Api { status: u16, message: String },

    #[error("dimension probe failed: {0}")]
    DimensionProbeFailed(String),

    #[error("unknown model for backend {backend}: {model}")]
    UnknownModel {
        backend: &'static str,
        model: String,
    },

    #[error("empty input")]
    EmptyInput,
}

impl From<candle_core::Error> for EmbedError {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value.to_string())
    }
}

impl From<tokenizers::Error> for EmbedError {
    fn from(value: tokenizers::Error) -> Self {
        Self::Tokenizer(value.to_string())
    }
}

impl From<hf_hub::api::sync::ApiError> for EmbedError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        Self::HfHub(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_variants_format_cleanly() {
        let e = EmbedError::MissingEnv("OPENAI_API_KEY");
        assert_eq!(
            e.to_string(),
            "missing required environment variable: OPENAI_API_KEY"
        );

        let e = EmbedError::Http("connection reset".into());
        assert_eq!(e.to_string(), "http transport error: connection reset");

        let e = EmbedError::Api {
            status: 401,
            message: "bad key".into(),
        };
        assert_eq!(e.to_string(), "api error: status 401: bad key");

        let e = EmbedError::DimensionProbeFailed("refused".into());
        assert_eq!(e.to_string(), "dimension probe failed: refused");

        let e = EmbedError::UnknownModel {
            backend: "openai",
            model: "text-embedding-9001".into(),
        };
        assert_eq!(
            e.to_string(),
            "unknown model for backend openai: text-embedding-9001"
        );
    }
}
