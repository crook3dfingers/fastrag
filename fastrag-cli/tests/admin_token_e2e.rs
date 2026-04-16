//! The `/admin/*` routes require a separate admin token. The general
//! `--token` (used by `/query` etc.) never grants admin access.

use axum_test::TestServer;

#[tokio::test]
async fn admin_reload_requires_admin_token() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        /* admin_token */ Some("sekret".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .json(&serde_json::json!({"bundle_path": "x"}))
        .await;
    assert_eq!(resp.status_code(), 401);
}

#[tokio::test]
async fn admin_reload_accepts_matching_admin_token() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        /* admin_token */ Some("sekret".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "sekret")
        .json(&serde_json::json!({"bundle_path": "x"}))
        .await;
    // Task 6 adds real behaviour; for now we only assert the auth gate passes
    // (i.e. the response is NOT 401).
    assert_ne!(resp.status_code(), 401, "expected auth to pass, got 401");
}

#[tokio::test]
async fn admin_routes_disabled_without_admin_token() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        /* admin_token */ None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "anything")
        .json(&serde_json::json!({"bundle_path": "x"}))
        .await;
    assert_eq!(resp.status_code(), 401);
}
