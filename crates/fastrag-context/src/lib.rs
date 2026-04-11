//! Contextual Retrieval for fastrag (Step 5).
//!
//! Adds an opt-in ingest-time stage that asks a small instruct LLM to produce
//! a 50–100 token context prefix for each chunk, which is then prepended to
//! the chunk text before dense embedding and BM25 indexing. The raw chunk is
//! preserved for display and for CVE/CWE exact-match lookups.
//!
//! See `docs/superpowers/specs/2026-04-10-contextual-retrieval-design.md` for
//! the full design.

mod cache;
mod contextualizer;
#[cfg(feature = "llama-cpp")]
mod llama;
mod prompt;
mod stage;

#[cfg(any(feature = "test-utils", test))]
pub mod test_utils;

pub use cache::{CacheKey, CacheStatus, CachedContext, ContextCache};
pub use contextualizer::{Contextualizer, ContextualizerMeta, NoContextualizer};
#[cfg(feature = "llama-cpp")]
pub use llama::LlamaCppContextualizer;
pub use prompt::{PROMPT, PROMPT_VERSION, format_prompt};
pub use stage::run_contextualize_stage;

/// Crate-level schema version for the SQLite cache. Bump when the table
/// definition, primary key shape, or canonical content of existing rows
/// changes in a way that invalidates prior caches.
pub const CTX_VERSION: u32 = 1;

/// Error surface for the fastrag-context crate.
#[derive(thiserror::Error, Debug)]
pub enum ContextError {
    #[error("llama-server HTTP error: {0}")]
    Http(String),
    #[error("llama-server returned non-200: status={status}, body={body}")]
    BadStatus { status: u16, body: String },
    #[error("llama-server request timed out after {0:?}")]
    Timeout(std::time::Duration),
    #[error("llama-server returned empty completion")]
    EmptyCompletion,
    #[error("cache error: {0}")]
    Cache(#[from] rusqlite::Error),
    #[error("prompt template error: {0}")]
    Template(String),
    #[error("response parse error: {0}")]
    Parse(String),
}
