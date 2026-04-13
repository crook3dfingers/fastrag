use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Extension, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastrag::corpus::{CorpusError, SearchHitDto};
use fastrag::default_separators;
use fastrag::{DynEmbedder, DynEmbedderTrait, ops};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::Deserialize;
use serde_json::json;
use subtle::ConstantTimeEq;
use thiserror::Error;
use tracing::{info, info_span, warn};

/// Per-corpus read-write locks. Ingest/delete acquire write; queries acquire read.
type IngestLocks =
    Arc<std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::RwLock<()>>>>>;

fn get_or_create_lock(locks: &IngestLocks, corpus: &str) -> Arc<tokio::sync::RwLock<()>> {
    let mut map = locks.lock().expect("IngestLocks poisoned");
    map.entry(corpus.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(())))
        .clone()
}

#[derive(Clone)]
struct AppState {
    registry: fastrag::corpus::CorpusRegistry,
    embedder: DynEmbedder,
    metrics: PrometheusHandle,
    dense_only: bool,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_locks: IngestLocks,
    ingest_max_body: usize,
    #[cfg(feature = "rerank")]
    reranker: Option<std::sync::Arc<dyn fastrag_rerank::Reranker>>,
    #[cfg(feature = "rerank")]
    rerank_over_fetch: usize,
}

/// Shared-secret auth state. `None` = auth disabled (the server logs a warning
/// at startup and accepts every request — matches pre-#26 behaviour for trusted
/// localhost). `Some(token)` = every protected route must present a matching
/// `X-Fastrag-Token` or `Authorization: Bearer` header.
#[derive(Clone)]
struct AuthState {
    token: Option<Arc<String>>,
}

impl AuthState {
    fn check(&self, provided: Option<&str>) -> bool {
        let expected = match &self.token {
            Some(t) => t,
            None => return true,
        };
        let got = match provided {
            Some(g) => g,
            None => return false,
        };
        // ConstantTimeEq short-circuits on length mismatch via ct_eq on bytes —
        // but we also must not leak via Option branching, so the Option check
        // above runs unconditionally before we reach here.
        expected.as_bytes().ct_eq(got.as_bytes()).into()
    }
}

async fn auth_middleware(
    State(state): State<AuthState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let headers = req.headers();
    let provided = headers
        .get("x-fastrag-token")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| {
                    s.strip_prefix("Bearer ")
                        .or_else(|| s.strip_prefix("bearer "))
                })
        });
    if state.check(provided) {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Tenant identity injected by middleware into request extensions when
/// `tenant_field` is configured on the server. The `field` names the metadata
/// key used to scope searches; `value` comes from the `X-Fastrag-Tenant` header.
#[derive(Clone)]
pub struct TenantFilter {
    pub field: String,
    pub value: String,
}

async fn tenant_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(ref field) = state.tenant_field else {
        return Ok(next.run(req).await);
    };

    let tenant_value = req
        .headers()
        .get("x-fastrag-tenant")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let Some(tenant) = tenant_value else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    req.extensions_mut().insert(TenantFilter {
        field: field.clone(),
        value: tenant,
    });

    Ok(next.run(req).await)
}

#[derive(Debug, Deserialize)]
struct QueryParams {
    q: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Comma-separated equality filters: `customer=acme,severity=high`.
    #[serde(default)]
    filter: Option<String>,
    /// Search mode (currently ignored — hybrid removed; dense-only is the only path).
    #[serde(default)]
    #[allow(dead_code)]
    mode: Option<String>,
    /// Set to `off` to skip reranking for this request.
    #[serde(default)]
    rerank: Option<String>,
    /// Override the rerank over-fetch multiplier for this request.
    #[serde(default)]
    over_fetch: Option<usize>,
    /// Named corpus to query. Defaults to `"default"`.
    #[serde(default)]
    corpus: Option<String>,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, serde::Deserialize)]
struct BatchQueryRequest {
    queries: Vec<BatchQueryItem>,
}

#[derive(Debug, serde::Deserialize)]
struct BatchQueryItem {
    q: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Accepts either string syntax ("severity = HIGH") or JSON AST (FilterExpr).
    #[serde(default)]
    filter: Option<serde_json::Value>,
    /// Named corpus to query. Defaults to `"default"`.
    #[serde(default)]
    corpus: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct BatchQueryResponse {
    results: Vec<BatchResultItem>,
}

#[derive(Debug, serde::Serialize)]
struct BatchResultItem {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    hits: Option<Vec<fastrag::corpus::SearchHitDto>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Error)]
pub enum HttpError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corpus error: {0}")]
    Corpus(#[from] fastrag::corpus::CorpusError),
    #[error("embed loader error: {0}")]
    EmbedLoader(#[from] crate::embed_loader::EmbedLoaderError),
    #[error("server error: {0}")]
    Server(String),
    #[error("metrics setup: {0}")]
    Metrics(String),
}

/// Optional reranker configuration for the HTTP server.
#[derive(Clone)]
pub struct HttpRerankerConfig {
    #[cfg(feature = "rerank")]
    pub reranker: Option<std::sync::Arc<dyn fastrag_rerank::Reranker>>,
    #[cfg(feature = "rerank")]
    pub over_fetch: usize,
}

impl Default for HttpRerankerConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "rerank")]
            reranker: None,
            #[cfg(feature = "rerank")]
            over_fetch: 10,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http(
    corpus_dir: PathBuf,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    ingest_max_body: usize,
) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    serve_http_with_embedder(
        corpus_dir,
        listener,
        embedder,
        token,
        dense_only,
        rerank_cfg,
        batch_max_queries,
        ingest_max_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_embedder(
    corpus_dir: PathBuf,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    ingest_max_body: usize,
) -> Result<(), HttpError> {
    let registry = fastrag::corpus::CorpusRegistry::new();
    registry.register("default", corpus_dir);
    serve_http_with_registry(
        registry,
        listener,
        embedder,
        token,
        dense_only,
        rerank_cfg,
        batch_max_queries,
        None,
        ingest_max_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry_port(
    registry: fastrag::corpus::CorpusRegistry,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    serve_http_with_registry(
        registry,
        listener,
        embedder,
        token,
        dense_only,
        rerank_cfg,
        batch_max_queries,
        tenant_field,
        ingest_max_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry(
    registry: fastrag::corpus::CorpusRegistry,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
) -> Result<(), HttpError> {
    // The `metrics` crate allows exactly one global recorder per process, but in
    // test binaries multiple serve_http_with_embedder calls share one process.
    // Install exactly once, then reuse the handle for every subsequent server.
    static METRICS_HANDLE: std::sync::OnceLock<PrometheusHandle> = std::sync::OnceLock::new();
    let metrics = METRICS_HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("install prometheus recorder")
        })
        .clone();

    metrics::describe_counter!("fastrag_query_total", "Total /query requests served");
    metrics::describe_histogram!(
        "fastrag_query_duration_seconds",
        "Latency of /query requests in seconds"
    );
    metrics::describe_gauge!(
        "fastrag_index_entries",
        "Number of entries in the loaded corpus index"
    );

    // Update index entry gauge for every registered corpus.
    for (_name, path, _loaded) in registry.list() {
        if let Ok(info) =
            fastrag::corpus::corpus_info(&path, embedder.as_ref() as &dyn DynEmbedderTrait)
        {
            metrics::gauge!("fastrag_index_entries").set(info.entry_count as f64);
        }
    }

    let auth_state = AuthState {
        token: token.map(Arc::new),
    };
    if auth_state.token.is_none() {
        warn!(
            "serve-http started without --token / FASTRAG_TOKEN — all requests are accepted. \
             Set a token before exposing on a shared network."
        );
    }

    let app_state = AppState {
        registry,
        embedder,
        metrics,
        dense_only,
        batch_max_queries,
        tenant_field,
        ingest_locks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        ingest_max_body,
        #[cfg(feature = "rerank")]
        reranker: rerank_cfg.reranker,
        #[cfg(feature = "rerank")]
        rerank_over_fetch: rerank_cfg.over_fetch,
    };

    let max_body = app_state.ingest_max_body;

    // /health stays unauthenticated for liveness probes.
    let protected = Router::new()
        .route("/query", get(query))
        .route(
            "/batch-query",
            axum::routing::post(batch_query_handler).layer(DefaultBodyLimit::max(10 * 1024 * 1024)), // 10 MB
        )
        .route(
            "/ingest",
            axum::routing::post(ingest_handler).layer(DefaultBodyLimit::max(max_body)),
        )
        .route("/ingest/:id", axum::routing::delete(delete_handler))
        .route("/metrics", get(metrics_handler))
        // /corpora is inside the protected router intentionally: when tenant_field is set,
        // listing corpora requires the X-Fastrag-Tenant header. This prevents corpus
        // enumeration without tenant credentials.
        .route("/corpora", get(list_corpora))
        .route("/stats", get(stats_handler))
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            tenant_middleware,
        ))
        .route_layer(middleware::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ));

    let app = Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(app_state);

    axum::serve(listener, app)
        .await
        .map_err(|e| HttpError::Server(e.to_string()))?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct IngestQueryParams {
    /// Named corpus to ingest into. Defaults to "default".
    #[serde(default = "default_corpus")]
    corpus: String,
    /// Field to use as the external record identifier.
    id_field: String,
    /// Comma-separated text fields whose content forms the record body.
    text_fields: String,
    /// Comma-separated fields to extract as metadata.
    #[serde(default)]
    metadata_fields: Option<String>,
    /// Comma-separated type overrides (e.g. "cvss=numeric,published=date").
    #[serde(default)]
    metadata_types: Option<String>,
    /// Comma-separated array fields.
    #[serde(default)]
    array_fields: Option<String>,
    /// Chunking strategy: "recursive" (default), "basic".
    #[serde(default = "default_chunk_strategy")]
    chunk_strategy: String,
    /// Max characters per chunk.
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,
    /// Overlap characters between chunks.
    #[serde(default = "default_chunk_overlap")]
    chunk_overlap: usize,
}

fn default_corpus() -> String {
    "default".to_string()
}
fn default_chunk_strategy() -> String {
    "recursive".to_string()
}
fn default_chunk_size() -> usize {
    1000
}
fn default_chunk_overlap() -> usize {
    200
}

#[derive(Debug, Deserialize)]
struct DeleteQueryParams {
    #[serde(default = "default_corpus")]
    corpus: String,
}

#[derive(Debug, Deserialize)]
struct StatsQueryParams {
    #[serde(default = "default_corpus")]
    corpus: String,
}

#[derive(Debug, serde::Serialize)]
struct DeleteResponse {
    corpus: String,
    id: String,
    deleted: bool,
}

#[derive(Debug, serde::Serialize)]
struct IngestResponse {
    corpus: String,
    records_new: usize,
    records_updated: usize,
    records_unchanged: usize,
    chunks_added: usize,
}

async fn ingest_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantFilter>>,
    Query(params): Query<IngestQueryParams>,
    body: axum::body::Bytes,
) -> Result<Json<IngestResponse>, Response> {
    // 1. Validate body size (belt-and-suspenders; the layer also enforces this).
    if body.len() > state.ingest_max_body {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "body exceeds ingest-max-body",
        )
            .into_response());
    }

    // 2. Resolve corpus directory.
    let corpus_dir = state.registry.corpus_path(&params.corpus).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("corpus {:?} not found", params.corpus),
        )
            .into_response()
    })?;

    // 3. Parse config from query params.
    let text_fields: Vec<String> = params
        .text_fields
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let metadata_fields: Vec<String> = params
        .metadata_fields
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let metadata_types: std::collections::BTreeMap<String, fastrag_store::schema::TypedKind> =
        params
            .metadata_types
            .as_deref()
            .unwrap_or("")
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| {
                let (k, v) = s.split_once('=')?;
                let kind = match v.trim() {
                    "numeric" => fastrag_store::schema::TypedKind::Numeric,
                    "date" => fastrag_store::schema::TypedKind::Date,
                    "bool" => fastrag_store::schema::TypedKind::Bool,
                    "array" => fastrag_store::schema::TypedKind::Array,
                    _ => return None,
                };
                Some((k.trim().to_string(), kind))
            })
            .collect();
    let array_fields: Vec<String> = params
        .array_fields
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    let config = fastrag::ingest::jsonl::JsonlIngestConfig {
        text_fields,
        id_field: params.id_field.clone(),
        metadata_fields,
        metadata_types,
        array_fields,
    };

    // 4. Build ChunkingStrategy from params.
    let chunking = match params.chunk_strategy.as_str() {
        "basic" => fastrag::ChunkingStrategy::Basic {
            max_characters: params.chunk_size,
            overlap: params.chunk_overlap,
        },
        _ => fastrag::ChunkingStrategy::RecursiveCharacter {
            max_characters: params.chunk_size,
            overlap: params.chunk_overlap,
            separators: default_separators(),
        },
    };

    // 5. Inject tenant metadata if active.
    let mut body_bytes = body.to_vec();
    if let Some(Extension(ref tf)) = tenant_ext {
        let mut output = Vec::new();
        for line in body_bytes.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let mut record: serde_json::Value = serde_json::from_slice(line).map_err(|e| {
                (StatusCode::BAD_REQUEST, format!("malformed JSON: {e}")).into_response()
            })?;
            if let Some(obj) = record.as_object_mut() {
                obj.insert(
                    tf.field.clone(),
                    serde_json::Value::String(tf.value.clone()),
                );
            }
            serde_json::to_writer(&mut output, &record).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("JSON write error: {e}"),
                )
                    .into_response()
            })?;
            output.push(b'\n');
        }
        body_bytes = output;
    }

    // 6. Write to temp file.
    let tmp = tempfile::NamedTempFile::new().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("tempfile: {e}")).into_response()
    })?;
    std::fs::write(tmp.path(), &body_bytes).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("write tempfile: {e}"),
        )
            .into_response()
    })?;

    // 7. Acquire write lock for this corpus.
    let lock = get_or_create_lock(&state.ingest_locks, &params.corpus);
    let _write_guard = lock.write().await;

    // 8. Run ingest in blocking thread.
    let tmp_path = tmp.path().to_path_buf();
    let corpus_dir_clone = corpus_dir.clone();
    let embedder = state.embedder.clone();
    let stats = tokio::task::spawn_blocking(move || {
        fastrag::ingest::engine::index_jsonl(
            &tmp_path,
            &corpus_dir_clone,
            &chunking,
            embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
            &config,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response())?
    .map_err(|e| {
        let status = if e.to_string().contains("embed") {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, format!("ingest error: {e}")).into_response()
    })?;

    // 9. Return stats.
    Ok(Json(IngestResponse {
        corpus: params.corpus,
        records_new: stats.records_new,
        records_updated: stats.records_upserted,
        records_unchanged: stats.records_skipped,
        chunks_added: stats.chunks_created,
    }))
}

async fn delete_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(params): Query<DeleteQueryParams>,
) -> Result<Json<DeleteResponse>, Response> {
    let corpus_dir = state.registry.corpus_path(&params.corpus).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("corpus {:?} not found", params.corpus),
        )
            .into_response()
    })?;

    let lock = get_or_create_lock(&state.ingest_locks, &params.corpus);
    let _write_guard = lock.write().await;

    let id_clone = id.clone();
    let corpus_dir_clone = corpus_dir.clone();
    let deleted_ids = tokio::task::spawn_blocking(move || {
        let mut store = fastrag_store::Store::open_no_embedder(&corpus_dir_clone)
            .map_err(|e| format!("open store: {e}"))?;
        let ids = store
            .delete_by_external_id(&id_clone)
            .map_err(|e| format!("delete: {e}"))?;
        store.save().map_err(|e| format!("save: {e}"))?;
        Ok::<Vec<u64>, String>(ids)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response())?
    .map_err(|e: String| (StatusCode::INTERNAL_SERVER_ERROR, e).into_response())?;

    Ok(Json(DeleteResponse {
        corpus: params.corpus,
        id,
        deleted: !deleted_ids.is_empty(),
    }))
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.render()
}

async fn list_corpora(State(state): State<AppState>) -> impl IntoResponse {
    let entries: Vec<serde_json::Value> = state
        .registry
        .list()
        .into_iter()
        .map(|(name, path, loaded)| {
            serde_json::json!({
                "name": name,
                "path": path,
                "status": if loaded { "loaded" } else { "unloaded" }
            })
        })
        .collect();
    Json(serde_json::json!({ "corpora": entries }))
}

async fn stats_handler(
    State(state): State<AppState>,
    Query(params): Query<StatsQueryParams>,
) -> Result<Json<serde_json::Value>, Response> {
    let corpus_dir = state.registry.corpus_path(&params.corpus).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("corpus not found: {}", params.corpus) })),
        )
            .into_response()
    })?;

    let lock = get_or_create_lock(&state.ingest_locks, &params.corpus);
    let _read_guard = lock.read().await;

    let corpus_name = params.corpus.clone();
    let stats = tokio::task::spawn_blocking(move || {
        fastrag::corpus::corpus_stats(&corpus_dir, &corpus_name)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response())?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("stats: {e}")).into_response())?;

    Ok(Json(serde_json::to_value(stats).unwrap()))
}

fn run_query(
    state: &AppState,
    params: &QueryParams,
    filter: Option<&fastrag::filter::FilterExpr>,
    corpus_name: &str,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    let corpus_dir = state
        .registry
        .corpus_path(corpus_name)
        .ok_or_else(|| CorpusError::NotFound(corpus_name.to_string()))?;

    let _ = state.dense_only; // hybrid removed; dense-only is the only path

    #[cfg(feature = "rerank")]
    {
        let skip = params.rerank.as_deref() == Some("off");
        if !skip && let Some(ref reranker) = state.reranker {
            let over_fetch = params.over_fetch.unwrap_or(state.rerank_over_fetch);
            return ops::query_corpus_reranked(
                &corpus_dir,
                &params.q,
                params.top_k,
                over_fetch,
                state.embedder.as_ref() as &dyn DynEmbedderTrait,
                reranker.as_ref(),
                filter,
                &mut fastrag::corpus::LatencyBreakdown::default(),
            );
        }
    }

    ops::query_corpus_with_filter(
        &corpus_dir,
        &params.q,
        params.top_k,
        state.embedder.as_ref() as &dyn DynEmbedderTrait,
        filter,
        &mut fastrag::corpus::LatencyBreakdown::default(),
    )
}

async fn batch_query_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantFilter>>,
    Json(req): Json<BatchQueryRequest>,
) -> Result<Json<BatchQueryResponse>, Response> {
    use fastrag_embed::QueryText;

    if req.queries.len() > state.batch_max_queries {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "batch size {} exceeds limit {}",
                req.queries.len(),
                state.batch_max_queries
            ),
        )
            .into_response());
    }

    // Validate all queries target the same corpus (per-corpus batch routing is out of scope).
    let corpus_name: &str = {
        let mut specified: Option<&str> = None;
        for (i, item) in req.queries.iter().enumerate() {
            if let Some(c) = item.corpus.as_deref() {
                match specified {
                    None => specified = Some(c),
                    Some(prev) if prev != c => {
                        return Err((
                            StatusCode::BAD_REQUEST,
                            format!(
                                "all queries in a batch must target the same corpus; \
                                 query 0 targets {:?} but query {} targets {:?}",
                                prev, i, c
                            ),
                        )
                            .into_response());
                    }
                    Some(_) => {} // same corpus, ok
                }
            }
        }
        specified.unwrap_or("default")
    };
    let corpus_dir = match state.registry.corpus_path(corpus_name) {
        Some(p) => p,
        None => {
            return Err((StatusCode::NOT_FOUND, "corpus not found").into_response());
        }
    };
    let lock = get_or_create_lock(&state.ingest_locks, corpus_name);
    let _read_guard = lock.read().await;

    // Parse all filters up front. Track per-query filter parse errors.
    let mut filter_errors: Vec<Option<String>> = vec![None; req.queries.len()];
    let mut filter_exprs: Vec<Option<fastrag::filter::FilterExpr>> =
        Vec::with_capacity(req.queries.len());

    for (i, item) in req.queries.iter().enumerate() {
        match &item.filter {
            None => filter_exprs.push(None),
            Some(serde_json::Value::String(s)) => match fastrag::filter::parse(s) {
                Ok(f) => filter_exprs.push(Some(f)),
                Err(e) => {
                    filter_exprs.push(None);
                    filter_errors[i] = Some(format!("bad filter: {e}"));
                }
            },
            Some(json_val) => match serde_json::from_value(json_val.clone()) {
                Ok(f) => filter_exprs.push(Some(f)),
                Err(e) => {
                    filter_exprs.push(None);
                    filter_errors[i] = Some(format!("bad filter: {e}"));
                }
            },
        }
    }

    // Embed all query texts in a single call.
    let texts: Vec<QueryText> = req
        .queries
        .iter()
        .map(|item| QueryText::new(&item.q))
        .collect();

    let embeddings =
        match (state.embedder.as_ref() as &dyn fastrag::DynEmbedderTrait).embed_query_dyn(&texts) {
            Ok(vecs) => vecs,
            Err(e) => {
                return Err(
                    (StatusCode::SERVICE_UNAVAILABLE, format!("embed error: {e}")).into_response(),
                );
            }
        };

    // Build per-query params, merging tenant filter if present.
    let tenant_filter: Option<TenantFilter> = tenant_ext.map(|Extension(tf)| tf);
    let params: Vec<fastrag::corpus::BatchQueryParams> = req
        .queries
        .iter()
        .zip(filter_exprs.into_iter())
        .map(|(item, f)| {
            let filter = if let Some(ref tf) = tenant_filter {
                let tenant_cond = fastrag::filter::FilterExpr::Eq {
                    field: tf.field.clone(),
                    value: fastrag_store::schema::TypedValue::String(tf.value.clone()),
                };
                Some(match f {
                    Some(existing) => fastrag::filter::FilterExpr::And(vec![tenant_cond, existing]),
                    None => tenant_cond,
                })
            } else {
                f
            };
            fastrag::corpus::BatchQueryParams {
                text: item.q.clone(),
                top_k: item.top_k,
                filter,
            }
        })
        .collect();

    // Run batch retrieval.
    #[cfg(feature = "rerank")]
    let raw_results =
        fastrag::corpus::batch_query(&corpus_dir, &embeddings, &params, state.reranker.as_deref());
    #[cfg(not(feature = "rerank"))]
    let raw_results = fastrag::corpus::batch_query(&corpus_dir, &embeddings, &params);

    // Merge filter parse errors with retrieval results.
    let results: Vec<BatchResultItem> = raw_results
        .into_iter()
        .enumerate()
        .map(|(i, result)| {
            if let Some(ref err) = filter_errors[i] {
                return BatchResultItem {
                    index: i,
                    hits: None,
                    error: Some(err.clone()),
                };
            }
            match result {
                Ok(hits) => BatchResultItem {
                    index: i,
                    hits: Some(hits),
                    error: None,
                },
                Err(e) => BatchResultItem {
                    index: i,
                    hits: None,
                    error: Some(e.to_string()),
                },
            }
        })
        .collect();

    Ok(Json(BatchQueryResponse { results }))
}

async fn query(
    State(state): State<AppState>,
    Query(params): Query<QueryParams>,
    tenant_ext: Option<Extension<TenantFilter>>,
) -> Result<Json<Vec<SearchHitDto>>, Response> {
    let span = info_span!(
        "query",
        q = %params.q,
        top_k = params.top_k,
        hit_count = tracing::field::Empty,
        latency_ms = tracing::field::Empty,
    );
    let _enter = span.enter();
    let start = Instant::now();

    let base_filter: Option<fastrag::filter::FilterExpr> = match params.filter.as_deref() {
        Some(s) => match fastrag::filter::parse(s) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
        None => None,
    };

    let filter_expr: Option<fastrag::filter::FilterExpr> = if let Some(Extension(tf)) = tenant_ext {
        let tenant_cond = fastrag::filter::FilterExpr::Eq {
            field: tf.field.clone(),
            value: fastrag_store::schema::TypedValue::String(tf.value.clone()),
        };
        Some(match base_filter {
            Some(existing) => fastrag::filter::FilterExpr::And(vec![tenant_cond, existing]),
            None => tenant_cond,
        })
    } else {
        base_filter
    };

    let corpus_name = params.corpus.as_deref().unwrap_or("default");
    let lock = get_or_create_lock(&state.ingest_locks, corpus_name);
    let _read_guard = lock.read().await;
    let result = run_query(&state, &params, filter_expr.as_ref(), corpus_name);

    let elapsed = start.elapsed();
    metrics::counter!("fastrag_query_total").increment(1);
    metrics::histogram!("fastrag_query_duration_seconds").record(elapsed.as_secs_f64());
    span.record("latency_ms", elapsed.as_millis() as u64);

    match result {
        Ok(hits) => {
            span.record("hit_count", hits.len());
            info!("query served");
            Ok(Json(hits))
        }
        Err(CorpusError::NotFound(_)) => {
            warn!(corpus = corpus_name, "corpus not found");
            Err((StatusCode::NOT_FOUND, "corpus not found").into_response())
        }
        Err(err) => {
            warn!(error = %err, "query failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response())
        }
    }
}
