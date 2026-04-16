//! `GET /cwe/relation` walks the in-memory `Taxonomy` (no corpus hit needed)
//! and returns ancestors and/or descendants of the given CWE id.
//!
//! The seeded DAG from `build_router_with_bundle_dag` is:
//!   89 -> [707, 943] -> [20]

use axum_test::TestServer;

fn mock_embedder() -> fastrag::DynEmbedder {
    std::sync::Arc::new(fastrag_embed::test_utils::MockEmbedder)
}

#[tokio::test]
async fn cwe_relation_returns_ancestors_and_descendants() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle_dag(mock_embedder()).await;
    let server = TestServer::new(router).unwrap();

    let resp = server.get("/cwe/relation?cwe_id=89").await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["cwe_id"], 89);
    let ancestors: Vec<u64> = body["ancestors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert!(ancestors.contains(&707), "missing 707: {ancestors:?}");
    assert!(ancestors.contains(&943), "missing 943: {ancestors:?}");
    assert!(ancestors.contains(&20), "missing 20: {ancestors:?}");
    assert!(body["descendants"].is_array());
}

#[tokio::test]
async fn cwe_relation_respects_direction_ancestors() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle_dag(mock_embedder()).await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .get("/cwe/relation?cwe_id=89&direction=ancestors")
        .await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    let ancestors = body["ancestors"].as_array().unwrap();
    assert!(!ancestors.is_empty());
    let descendants = body["descendants"].as_array().unwrap();
    assert!(descendants.is_empty(), "got: {descendants:?}");
}

#[tokio::test]
async fn cwe_relation_respects_direction_descendants() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle_dag(mock_embedder()).await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .get("/cwe/relation?cwe_id=20&direction=descendants")
        .await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    let ancestors = body["ancestors"].as_array().unwrap();
    assert!(ancestors.is_empty());
    let descendants: Vec<u64> = body["descendants"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert!(
        descendants.contains(&89),
        "expected 89 under 20: {descendants:?}"
    );
    assert!(!descendants.contains(&20), "self should be stripped");
}

#[tokio::test]
async fn cwe_relation_respects_max_depth() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle_dag(mock_embedder()).await;
    let server = TestServer::new(router).unwrap();

    let resp = server
        .get("/cwe/relation?cwe_id=89&direction=ancestors&max_depth=1")
        .await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    let ancestors: Vec<u64> = body["ancestors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert!(ancestors.contains(&707));
    assert!(ancestors.contains(&943));
    assert!(
        !ancestors.contains(&20),
        "max_depth=1 must exclude grandparents: {ancestors:?}"
    );
}

#[tokio::test]
async fn cwe_relation_bad_id_is_400() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle(None, mock_embedder()).await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/relation?cwe_id=not-a-number").await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn cwe_relation_missing_id_is_400() {
    let (router, _tmp) =
        fastrag_cli::test_support::build_router_with_bundle(None, mock_embedder()).await;
    let server = TestServer::new(router).unwrap();
    let resp = server.get("/cwe/relation").await;
    assert_eq!(resp.status_code(), 400);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["error"], "missing_cwe_id");
}
