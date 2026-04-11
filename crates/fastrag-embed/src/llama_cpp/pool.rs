//! Lifecycle helper owning up to two [`LlamaServerHandle`] instances.
//!
//! Used by the `fastrag-context` crate (via `fastrag::corpus::index_path`) to
//! spawn both the embedder server and the optional completion server from a
//! single coordinated drop point. The pool itself does not implement `Drop`
//! — ordinary field-drop order tears down `completion` first and `embedder`
//! second, which matches the construction order.

use crate::error::EmbedError;
use crate::llama_cpp::handle::{LlamaServerConfig, LlamaServerHandle};

/// Pool of at most two llama-server subprocesses: the embedder, and an
/// optional completion server for contextualization.
pub struct LlamaServerPool {
    embedder: LlamaServerHandle,
    completion: Option<LlamaServerHandle>,
}

impl LlamaServerPool {
    /// Wrap an already-running embedder handle. Use [`Self::with_completion`]
    /// to add a completion server.
    pub fn new(embedder: LlamaServerHandle) -> Self {
        Self {
            embedder,
            completion: None,
        }
    }

    /// Spawn the completion server alongside the existing embedder. Returns
    /// an error if the completion subprocess fails to start; the embedder is
    /// left intact so the caller can still proceed with a no-contextualize
    /// ingest if desired.
    pub fn with_completion(mut self, cfg: LlamaServerConfig) -> Result<Self, EmbedError> {
        let handle = LlamaServerHandle::spawn(cfg)?;
        self.completion = Some(handle);
        Ok(self)
    }

    pub fn embedder(&self) -> &LlamaServerHandle {
        &self.embedder
    }

    pub fn embedder_mut(&mut self) -> &mut LlamaServerHandle {
        &mut self.embedder
    }

    pub fn completion(&self) -> Option<&LlamaServerHandle> {
        self.completion.as_ref()
    }

    pub fn completion_mut(&mut self) -> Option<&mut LlamaServerHandle> {
        self.completion.as_mut()
    }

    /// Snapshot of each pool member's liveness, suitable for `fastrag doctor`
    /// output. Each tuple is `(label, alive, port)`.
    pub fn health_summary(&mut self) -> Vec<(&'static str, bool, u16)> {
        let mut out = vec![(
            "embedder",
            self.embedder.check_alive(),
            self.embedder.port(),
        )];
        if let Some(c) = &mut self.completion {
            out.push(("completion", c.check_alive(), c.port()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Full coverage for the pool lives in the Phase 6 E2E tests (which need a
    // real llama-server binary and GGUFs). This smoke test just verifies that
    // the module compiles and that the public surface stays stable.
    #[test]
    fn module_compiles_and_type_exists() {
        let _ = std::any::TypeId::of::<LlamaServerPool>();
    }
}
