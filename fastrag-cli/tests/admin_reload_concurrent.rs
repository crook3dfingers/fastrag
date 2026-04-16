//! `POST /admin/reload` serialises through a reload mutex. A second caller
//! that arrives while the handler is mid-load gets `409 reload_in_progress`,
//! never a partial swap or a panic. The first caller still succeeds.

use axum_test::TestServer;

#[tokio::test]
async fn concurrent_reload_returns_409() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_two_bundles_slow(
        Some("tok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let first_fut = async {
        server
            .post("/admin/reload")
            .add_header("x-fastrag-admin-token", "tok")
            .json(&serde_json::json!({"bundle_path": "fastrag-second"}))
            .await
    };
    let second_fut = async {
        // Delay so the first request wins the reload mutex before we fire.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        server
            .post("/admin/reload")
            .add_header("x-fastrag-admin-token", "tok")
            .json(&serde_json::json!({"bundle_path": "fastrag-second"}))
            .await
    };

    let (first, second) = tokio::join!(first_fut, second_fut);

    assert_eq!(second.status_code(), 409, "expected 409 from second caller");
    let body: serde_json::Value = second.json();
    assert_eq!(body["error"], "reload_in_progress");

    assert_eq!(first.status_code(), 200, "expected 200 from first caller");
    let body: serde_json::Value = first.json();
    assert_eq!(body["bundle_id"], "fastrag-second");
}
