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
    write_bundle_with_id(root, "test");
}

/// Build a minimally valid bundle rooted at `root` with the given
/// `bundle_id`. Used by the multi-bundle helpers below so reload tests can
/// verify `bundle_id` / `previous_bundle_id` flip on swap.
fn write_bundle_with_id(root: &Path, bundle_id: &str) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("bundle.json"),
        format!(
            r#"{{
            "schema_version": 1,
            "bundle_id": "{bundle_id}",
            "built_at": "2026-04-16T00:00:00Z",
            "corpora": ["cve","cwe","kev"],
            "taxonomy": "cwe-taxonomy.json"
        }}"#
        ),
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

/// Build a router against a `bundles-dir` containing `fastrag-first/` and
/// `fastrag-second/`, with `fastrag-first` loaded. The server's
/// `bundles_dir` is set to the parent `TempDir` so `/admin/reload` can
/// resolve either sibling directory name.
pub async fn build_router_with_two_bundles(
    admin_token: Option<String>,
    embedder: DynEmbedder,
) -> (axum::Router, tempfile::TempDir) {
    build_router_with_two_bundles_inner(admin_token, embedder, None).await
}

/// Same as [`build_router_with_two_bundles`] but injects a 500ms sleep into
/// the `/admin/reload` handler while it holds the reload mutex, so tests
/// can deterministically observe the 409 `reload_in_progress` branch when
/// two requests race.
pub async fn build_router_with_two_bundles_slow(
    admin_token: Option<String>,
    embedder: DynEmbedder,
) -> (axum::Router, tempfile::TempDir) {
    build_router_with_two_bundles_inner(
        admin_token,
        embedder,
        Some(std::time::Duration::from_millis(500)),
    )
    .await
}

async fn build_router_with_two_bundles_inner(
    admin_token: Option<String>,
    embedder: DynEmbedder,
    reload_delay: Option<std::time::Duration>,
) -> (axum::Router, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let first = tmp.path().join("fastrag-first");
    let second = tmp.path().join("fastrag-second");
    write_bundle_with_id(&first, "fastrag-first");
    write_bundle_with_id(&second, "fastrag-second");
    let state = crate::http::TestAppState::from_bundle_with_bundles_dir(
        &first,
        tmp.path().to_path_buf(),
        admin_token,
        embedder,
        reload_delay,
    )
    .unwrap();
    let router = crate::http::build_router_for_test(state);
    (router, tmp)
}
