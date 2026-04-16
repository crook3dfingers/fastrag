//! `POST /admin/reload` must reject any `bundle_path` that tries to escape
//! the configured `--bundles-dir` via parent traversal or absolute paths.
//! Both get 400 `path_escape` before any disk I/O is attempted.

use axum_test::TestServer;

#[tokio::test]
async fn admin_reload_rejects_parent_traversal() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_two_bundles(
        Some("tok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "tok")
        .json(&serde_json::json!({"bundle_path": "../../etc"}))
        .await;
    assert_eq!(resp.status_code(), 400);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "path_escape");
}

#[tokio::test]
async fn admin_reload_rejects_absolute_path() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_two_bundles(
        Some("tok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .post("/admin/reload")
        .add_header("x-fastrag-admin-token", "tok")
        .json(&serde_json::json!({"bundle_path": "/etc/passwd"}))
        .await;
    assert_eq!(resp.status_code(), 400);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "path_escape");
}
