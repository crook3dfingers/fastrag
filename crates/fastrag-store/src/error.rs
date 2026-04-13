use crate::schema::TypedKind;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("index error: {0}")]
    Index(#[from] fastrag_index::IndexError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("schema conflict on field `{field}`: existing={existing:?}, incoming={incoming:?}")]
    SchemaConflict {
        field: String,
        existing: TypedKind,
        incoming: TypedKind,
    },

    #[error("missing field: {0}")]
    MissingField(String),

    #[error("corrupt store: {0}")]
    Corrupt(String),
}

pub type StoreResult<T> = Result<T, StoreError>;
