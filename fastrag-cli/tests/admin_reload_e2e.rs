//! `POST /admin/reload` loads a new bundle from `--bundles-dir` and performs
//! an atomic `ArcSwap` swap. The response reports the new and previous
//! bundle ids; in-flight reads are unaffected by the swap.

use axum_test::TestServer;

#[tokio::test]
async fn admin_reload_swaps_bundle_atomically() {
    let (router, tmp) = fastrag_cli::test_support::build_router_with_two_bundles(
        Some("admintok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    // Baseline: first bundle loaded; /ready is 200.
    let resp = server.get("/ready").await;
    assert_eq!(resp.status_code(), 200);

    // Reload to second bundle.
    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "admintok")
        .json(&serde_json::json!({"bundle_path": "fastrag-second"}))
        .await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["reloaded"], true);
    assert_eq!(body["bundle_id"], "fastrag-second");
    assert_eq!(body["previous_bundle_id"], "fastrag-first");

    drop(tmp);
}

#[tokio::test]
async fn admin_reload_rejects_nonexistent_bundle() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_two_bundles(
        Some("admintok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "admintok")
        .json(&serde_json::json!({"bundle_path": "does-not-exist"}))
        .await;
    assert_eq!(resp.status_code(), 400);
    let body: serde_json::Value = resp.json();
    let err = body["error"].as_str().unwrap();
    assert!(
        err == "bundle_missing" || err.contains("manifest"),
        "unexpected error: {err}"
    );
}
