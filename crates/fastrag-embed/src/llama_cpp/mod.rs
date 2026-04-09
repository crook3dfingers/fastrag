//! llama.cpp HTTP embedding backend.
//!
//! Talks to a `llama-server` process (llama.cpp ≥ b5000) running the
//! `/v1/embeddings` OpenAI-compatible endpoint.  The server process lifecycle
//! (spawn, health-check, shutdown) lives in `LlamaServerHandle` (Task 3).
//! The blocking HTTP client helpers (`build_client`, `send_with_retry`,
//! `ensure_success`) are shared with the `http-embedders` path via
//! `crate::http`.
//!
//! Nothing is implemented yet — Tasks 3 and 4 fill this module.

#[cfg(test)]
mod tests {
    /// Verifies that the `llama-cpp` feature correctly implies `http-embedders`
    /// so the shared `crate::http` helpers are accessible from this module.
    /// This will fail to compile if the feature-flag implication is broken.
    #[test]
    fn shared_http_client_builds_under_llama_cpp_feature() {
        let client = crate::http::build_client().expect("reqwest client must build");
        // A freshly-built blocking client has no base URL; confirm it is usable
        // by inspecting its debug representation — not a no-op because it
        // exercises the reqwest builder path gated on the feature.
        let repr = format!("{client:?}");
        assert!(
            repr.contains("Client"),
            "expected debug repr to contain 'Client', got: {repr}"
        );
    }
}
