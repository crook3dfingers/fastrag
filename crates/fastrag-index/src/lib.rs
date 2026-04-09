mod entry;
mod error;
pub mod hash;
mod hnsw;
mod manifest;

pub use entry::{IndexEntry, SearchHit};
pub use error::{IndexError, IndexResult};
pub use hnsw::HnswIndex;
pub use manifest::{CorpusManifest, FileEntry, ManifestChunkingStrategy, RootEntry};

pub use fastrag_core::ElementKind;

use std::path::Path;

/// A persistent vector index for approximate nearest-neighbor search.
pub trait VectorIndex {
    fn add(&mut self, entries: Vec<IndexEntry>) -> IndexResult<()>;
    fn query(&self, vector: &[f32], top_k: usize) -> IndexResult<Vec<SearchHit>>;
    fn save(&self, dir: &Path) -> IndexResult<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
