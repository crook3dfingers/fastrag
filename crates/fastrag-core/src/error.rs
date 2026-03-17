use crate::format::FileFormat;

/// Errors that can occur during document parsing.
#[derive(Debug, thiserror::Error)]
pub enum FastRagError {
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error ({format}): {message}")]
    Parse { format: FileFormat, message: String },

    #[error("encoding error: {0}")]
    Encoding(String),
}
