use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("corpus is empty")]
    EmptyCorpus,

    #[error("corpus file missing: {path}")]
    MissingCorpusFile { path: PathBuf },

    #[error("corpus is corrupt: {message}")]
    CorruptCorpus { message: String },

    #[error(
        "embedder identity mismatch: corpus was built with `{existing}` (dim {existing_dim}), caller provided `{requested}` (dim {requested_dim})"
    )]
    IdentityMismatch {
        existing: String,
        existing_dim: usize,
        requested: String,
        requested_dim: usize,
    },

    #[error(
        "canary vector mismatch: live cosine {cosine:.6} below tolerance {tolerance:.6} — embedder weights or tokenizer have drifted since this corpus was built"
    )]
    CanaryMismatch { cosine: f32, tolerance: f32 },

    #[error("unsupported corpus schema: got v{got}, expected v3")]
    UnsupportedSchema { got: u32 },

    #[error("embedder error during canary verification: {0}")]
    CanaryEmbed(String),
}

pub type IndexResult<T> = Result<T, IndexError>;
