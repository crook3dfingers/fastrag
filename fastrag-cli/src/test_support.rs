//! Helpers for integration tests — build an in-memory `axum::Router` against
//! a synthetic bundle on disk so tests don't spawn a full OS process.
//!
//! The caller supplies the embedder: that keeps this module free of
//! `fastrag-embed/test-utils` symbols, which are only guaranteed under
//! dev-dependencies of the integration test binaries themselves.

use std::path::Path;

use fastrag::DynEmbedder;

/// Construct a router fronting a minimally valid bundle on disk. The
/// returned `TempDir` must stay alive for the duration of the test.
pub async fn build_router_with_bundle(
    admin_token: Option<String>,
    embedder: DynEmbedder,
) -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    write_minimal_bundle(tmp.path());
    let state = crate::http::TestAppState::from_bundle(tmp.path(), admin_token, embedder).unwrap();
    let router = crate::http::build_router_for_test(state);
    (router, tmp)
}

/// Same as [`build_router_with_bundle`] but also wires a read-path auth token
/// so tests can prove that specific routes (e.g. `/ready`) stay reachable
/// without the token.
pub async fn build_router_with_bundle_and_token(
    read_token: Option<String>,
    admin_token: Option<String>,
    embedder: DynEmbedder,
) -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    write_minimal_bundle(tmp.path());
    let state = crate::http::TestAppState::from_bundle(tmp.path(), admin_token, embedder).unwrap();
    let router = crate::http::build_router_for_test_with_token(state, read_token);
    (router, tmp)
}

/// Build a router whose `AppState.bundle` is `None` — the mode the server
/// runs in when started with only `--corpus`. Bundle-only routes (`/cve`,
/// `/cwe`, `/cwe/relation`, `/ready`, `/admin/reload`) must surface
/// `bundle_not_loaded` / 503 in this mode.
pub fn build_router_no_bundle(embedder: DynEmbedder) -> axum::Router {
    let state = crate::http::TestAppState::without_bundle(embedder);
    crate::http::build_router_for_test(state)
}

/// Same as [`build_router_with_bundle`] but seeds the taxonomy with the DAG
/// `89 -> [707, 943] -> [20]` used by the `/cwe/relation` tests.
pub async fn build_router_with_bundle_dag(
    embedder: DynEmbedder,
) -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    write_minimal_bundle(tmp.path());
    // CWE taxonomy semantics: `parents[x]` is the list of more-abstract CWEs
    // directly above `x`; `closure[x]` enumerates `x` plus all transitively
    // reachable descendants (more specific CWEs below). DAG shape used here:
    //   20 -> {707, 943} -> {89}   (20 is most abstract, 89 most specific)
    std::fs::write(
        tmp.path().join("taxonomy/cwe-taxonomy.json"),
        r#"{
            "schema_version": 2,
            "version": "4.15",
            "view": "1000",
            "closure": {
                "20":  [20, 707, 943, 89],
                "707": [707, 89],
                "943": [943, 89],
                "89":  [89]
            },
            "parents": {
                "89":  [707, 943],
                "707": [20],
                "943": [20],
                "20":  []
            }
        }"#,
    )
    .unwrap();
    let state = crate::http::TestAppState::from_bundle(tmp.path(), None, embedder).unwrap();
    let router = crate::http::build_router_for_test(state);
    (router, tmp)
}

fn write_minimal_bundle(root: &Path) {
    std::fs::write(
        root.join("bundle.json"),
        r#"{
            "schema_version": 1,
            "bundle_id": "test",
            "built_at": "2026-04-16T00:00:00Z",
            "corpora": ["cve","cwe","kev"],
            "taxonomy": "cwe-taxonomy.json"
        }"#,
    )
    .unwrap();
    std::fs::create_dir_all(root.join("taxonomy")).unwrap();
    std::fs::write(
        root.join("taxonomy/cwe-taxonomy.json"),
        r#"{"schema_version":2,"version":"4.15","view":"1000",
             "closure":{"89":[89]},"parents":{"89":[]}}"#,
    )
    .unwrap();
    for c in ["cve", "cwe", "kev"] {
        let d = root.join("corpora").join(c);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("manifest.json"), "{}").unwrap();
        std::fs::write(d.join("index.bin"), b"").unwrap();
        std::fs::write(d.join("entries.bin"), b"").unwrap();
    }
}
