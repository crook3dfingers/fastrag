//! `GET /cve/{id}` is a direct-lookup endpoint backed by the bundle's `cve`
//! corpus. It matches on the `cve_id` user field and returns 404 when no
//! document matches.

use axum_test::TestServer;

#[tokio::test]
async fn get_cve_404_for_unknown_id() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server.get("/cve/CVE-9999-0000").await;
    assert_eq!(resp.status_code(), 404);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "cve_not_found");
    assert_eq!(body["id"], "CVE-9999-0000");
}

#[tokio::test]
async fn get_cve_rejects_query_params() {
    let (router, _tmp) = fastrag_cli::test_support::build_router_with_bundle(
        None,
        std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder),
    )
    .await;
    let server = TestServer::new(router).unwrap();

    let resp = server.get("/cve/CVE-2021-44228?q=anything").await;
    assert_eq!(resp.status_code(), 400);
}
