//! Rolling back to a previous bundle is just another reload. A→B→A must
//! succeed without state leaking between swaps — the `previous_bundle_id`
//! response field reflects the bundle loaded immediately before.

use axum_test::TestServer;

#[tokio::test]
async fn rollback_sequence_succeeds() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_two_bundles(
        Some("tok".to_string()),
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    for (target, prev) in [
        ("fastrag-second", "fastrag-first"),
        ("fastrag-first", "fastrag-second"),
    ] {
        let resp = server
            .post("/admin/reload")
            .add_header("x-fastrag-admin-token", "tok")
            .json(&serde_json::json!({"bundle_path": target}))
            .await;
        assert_eq!(resp.status_code(), 200, "reload to {target} failed");
        let body: serde_json::Value = resp.json();
        assert_eq!(body["reloaded"], true);
        assert_eq!(body["bundle_id"], target);
        assert_eq!(body["previous_bundle_id"], prev);
    }
}
