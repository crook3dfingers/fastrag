//! `GET /cwe/{id}` accepts both bare numeric ids ("89") and the common
//! prefixed form ("CWE-89"). Non-numeric ids are rejected at the handler
//! before the corpus is touched.

use axum_test::TestServer;

#[tokio::test]
async fn get_cwe_accepts_numeric_id() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/89").await;
    // Test bundle has no CWE records, so 404 is acceptable; 200 if bundle seeded.
    assert!(
        resp.status_code() == 200 || resp.status_code() == 404,
        "unexpected status: {}",
        resp.status_code()
    );
}

#[tokio::test]
async fn get_cwe_accepts_prefixed_id() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/CWE-89").await;
    assert!(
        resp.status_code() == 200 || resp.status_code() == 404,
        "unexpected status: {}",
        resp.status_code()
    );
}

#[tokio::test]
async fn get_cwe_rejects_non_numeric() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/not-a-number").await;
    assert_eq!(resp.status_code(), 400);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "invalid_cwe_id");
}

#[tokio::test]
async fn get_cwe_404_for_unknown_id() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/99999").await;
    assert_eq!(resp.status_code(), 404);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "cwe_not_found");
    assert_eq!(body["id"], 99999);
}
