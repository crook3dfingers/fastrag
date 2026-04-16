//! `GET /ready` distinguishes from `/health`: it returns 503 until the bundle
//! is loaded and every required corpus is present. External probes use it
//! without auth to gate traffic on a freshly-started container.

use axum_test::TestServer;

fn mock_embedder() -> fastrag::DynEmbedder {
    std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder)
}

#[tokio::test]
async fn ready_200_when_bundle_loaded() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle(None, mock_embedder()).await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/ready").await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["ready"], true);
}

#[tokio::test]
async fn ready_503_when_no_bundle() {
    let router = fastrag_cli::test_support::build_router_no_bundle(mock_embedder());
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/ready").await;
    assert_eq!(resp.status_code(), 503);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["ready"], false);
    let reasons: Vec<String> = body["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(reasons.iter().any(|r| r == "bundle_not_loaded"));
}

#[tokio::test]
async fn ready_is_unauthenticated() {
    // /ready must not require the read token — external probes don't have it.
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle_and_token(
        Some("secret".to_string()),
        None,
        mock_embedder(),
    )
    .await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/ready").await;
    assert_eq!(resp.status_code(), 200);
}
