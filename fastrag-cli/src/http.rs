use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use axum::extract::{DefaultBodyLimit, Extension, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastrag::bundle::BundleState;
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
    cwe_expand_default: bool,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_locks: IngestLocks,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
    #[cfg(feature = "rerank")]
    reranker: Option<std::sync::Arc<dyn fastrag_rerank::Reranker>>,
    #[cfg(feature = "rerank")]
    rerank_over_fetch: usize,

    /// Bundle-mode state. `None` means the server was started with only
    /// `--corpus`; bundle-only routes (`/cve`, `/cwe`, `/cwe/relation`,
    /// `/ready`, `/admin/reload`) return 503 in that mode.
    #[allow(dead_code)] // consumed by Task 4+ handlers
    bundle: Option<Arc<ArcSwap<BundleState>>>,
    /// Root dir for bundles. Names passed to `/admin/reload` resolve here.
    #[allow(dead_code)] // consumed by Task 6 (/admin/reload)
    bundles_dir: Option<PathBuf>,
    /// Separate admin credential. `/admin/*` require this; read-path
    /// `--token` never grants admin.
    admin_token: Option<Arc<String>>,
    /// Serializes concurrent `/admin/reload` calls.
    #[allow(dead_code)] // consumed by Task 6 (/admin/reload)
    reload_lock: Arc<tokio::sync::Mutex<()>>,
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
    /// Max characters for snippet. 0 disables. Default 150.
    #[serde(default = "default_snippet_len")]
    snippet_len: usize,
    /// Comma-separated field projection (e.g. "score,snippet" or "-chunk_text").
    #[serde(default)]
    fields: Option<String>,
    /// Override server default `--cwe-expand`. None = use server default.
    #[serde(default)]
    cwe_expand: Option<bool>,
    /// Enable BM25 + dense hybrid retrieval via RRF.
    #[serde(default)]
    hybrid: Option<bool>,
    /// RRF k. Default 60.
    #[serde(default)]
    rrf_k: Option<u32>,
    /// Per-retriever overfetch multiplier.
    #[serde(default)]
    rrf_overfetch: Option<usize>,
    /// Name of the Date metadata field for temporal decay. Implies hybrid.
    #[serde(default)]
    time_decay_field: Option<String>,
    /// Humantime halflife string (e.g. "30d").
    #[serde(default)]
    time_decay_halflife: Option<String>,
    /// Alpha floor (0..=1).
    #[serde(default)]
    time_decay_weight: Option<f32>,
    /// Dateless prior (0..=1).
    #[serde(default)]
    time_decay_dateless_prior: Option<f32>,
    /// Blend: "multiplicative" or "additive".
    #[serde(default)]
    time_decay_blend: Option<String>,
}

fn default_top_k() -> usize {
    5
}

fn default_snippet_len() -> usize {
    150
}

/// Request body for POST /query.
///
/// Mirrors the GET /query query-string params but adds first-class `temporal_policy`
/// and `date_fields` fields. Precedence rule for temporal settings:
///
/// 1. If `temporal_policy` is explicitly set in the body (`"off"` or `{favor_recent: …}`),
///    use it verbatim and take `date_fields` from the body.
/// 2. Else if legacy `time_decay_field` is set, bridge it to `FavorRecent(Medium)` as
///    Task 9 did (backward-compat path).
/// 3. Else default to `Auto` with no date fields (abstaining detector path).
///
/// Note: if the body sends `"auto"` and legacy `time_decay_field` is also set, the
/// explicit `Auto` is indistinguishable from the default, so rule 2 takes effect.
/// This is acceptable — `"auto"` + a decay field is a degenerate case.
#[derive(Debug, Deserialize)]
struct PostQueryBody {
    q: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Accepts either string syntax ("severity = HIGH") or JSON AST (FilterExpr).
    #[serde(default)]
    filter: Option<serde_json::Value>,
    /// Named corpus to query. Defaults to `"default"`.
    #[serde(default)]
    corpus: Option<String>,
    /// Max characters for snippet. 0 disables. Default 150.
    #[serde(default = "default_snippet_len")]
    snippet_len: usize,
    /// Comma-separated field projection (e.g. "score,snippet" or "-chunk_text").
    #[serde(default)]
    fields: Option<String>,
    /// Override server default `--cwe-expand`.
    #[serde(default)]
    cwe_expand: Option<bool>,
    /// Enable BM25 + dense hybrid retrieval via RRF.
    #[serde(default)]
    hybrid: Option<bool>,
    /// RRF k. Default 60.
    #[serde(default)]
    rrf_k: Option<u32>,
    /// Per-retriever overfetch multiplier.
    #[serde(default)]
    rrf_overfetch: Option<usize>,
    /// Per-query temporal policy. Default `auto` (abstaining detector).
    #[serde(default)]
    temporal_policy: fastrag::corpus::temporal::TemporalPolicy,
    /// Ordered list of date metadata field names for decay. Empty disables decay.
    #[serde(default)]
    date_fields: Vec<String>,
    // Legacy time_decay_* fields — accepted for backward-compat, bridged to
    // `FavorRecent(Medium)` when `temporal_policy` is default `Auto`.
    #[serde(default)]
    time_decay_field: Option<String>,
    #[serde(default)]
    time_decay_halflife: Option<String>,
    #[serde(default)]
    time_decay_weight: Option<f32>,
    #[serde(default)]
    time_decay_dateless_prior: Option<f32>,
    #[serde(default)]
    time_decay_blend: Option<String>,
    /// Set to `off` to skip reranking for this request.
    #[serde(default)]
    rerank: Option<String>,
    /// Override the rerank over-fetch multiplier for this request.
    #[serde(default)]
    over_fetch: Option<usize>,
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
    #[serde(default = "default_snippet_len")]
    snippet_len: usize,
    #[serde(default)]
    fields: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct BatchQueryResponse {
    results: Vec<BatchResultItem>,
}

#[derive(Debug, serde::Serialize)]
struct BatchResultItem {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    hits: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct SimilarRequest {
    text: String,
    threshold: f32,
    max_results: usize,
    #[serde(default)]
    corpus: Option<String>,
    #[serde(default)]
    corpora: Option<Vec<String>>,
    /// Accepts either string syntax ("severity = HIGH") or JSON AST.
    #[serde(default)]
    filter: Option<serde_json::Value>,
    #[serde(default)]
    fields: Option<String>,
    // Catch-all for rejected params. Any of these set -> 400.
    #[serde(default)]
    hybrid: Option<serde_json::Value>,
    #[serde(default)]
    rrf_k: Option<serde_json::Value>,
    #[serde(default)]
    rrf_overfetch: Option<serde_json::Value>,
    #[serde(default)]
    time_decay_field: Option<serde_json::Value>,
    #[serde(default)]
    time_decay_halflife: Option<serde_json::Value>,
    #[serde(default)]
    time_decay_weight: Option<serde_json::Value>,
    #[serde(default)]
    time_decay_dateless_prior: Option<serde_json::Value>,
    #[serde(default)]
    time_decay_blend: Option<serde_json::Value>,
    #[serde(default)]
    rerank: Option<serde_json::Value>,
    #[serde(default)]
    cwe_expand: Option<serde_json::Value>,
    /// Optional post-ANN verification. Shape: `{ method: "minhash", threshold: 0.7 }`.
    /// Parsed as raw JSON to produce precise 400 messages for bad shapes.
    #[serde(default)]
    verify: Option<serde_json::Value>,
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

#[derive(Debug, Clone)]
enum FieldSelection {
    All,
    Include(Vec<String>),
    Exclude(Vec<String>),
}

/// Parse a `verify` block from raw JSON, producing precise 400 messages.
fn parse_verify_block(v: &serde_json::Value) -> Result<fastrag::corpus::VerifyConfig, String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "verify must be an object".to_string())?;
    for k in obj.keys() {
        if k != "method" && k != "threshold" {
            return Err(format!("verify has unknown field `{k}`"));
        }
    }
    let method_val = obj
        .get("method")
        .ok_or_else(|| "verify.method is required".to_string())?;
    let method_str = method_val
        .as_str()
        .ok_or_else(|| "verify.method must be a string".to_string())?;
    let method = match method_str {
        "minhash" => fastrag::corpus::VerifyMethod::MinHash,
        other => {
            return Err(format!(
                "verify.method `{other}` is not supported; expected \"minhash\""
            ));
        }
    };
    let threshold_val = obj
        .get("threshold")
        .ok_or_else(|| "verify.threshold is required".to_string())?;
    let threshold_f64 = threshold_val
        .as_f64()
        .ok_or_else(|| "verify.threshold must be a number".to_string())?;
    if !(0.0..=1.0).contains(&threshold_f64) {
        return Err("verify.threshold must be in [0.0, 1.0]".to_string());
    }
    Ok(fastrag::corpus::VerifyConfig {
        method,
        threshold: threshold_f64 as f32,
    })
}

fn parse_field_selection(fields: Option<&str>) -> Result<FieldSelection, String> {
    let raw = match fields {
        None | Some("") => return Ok(FieldSelection::All),
        Some(s) => s,
    };
    let parts: Vec<&str> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Ok(FieldSelection::All);
    }
    let has_exclude = parts.iter().any(|p| p.starts_with('-'));
    let has_include = parts.iter().any(|p| !p.starts_with('-'));
    if has_exclude && has_include {
        return Err("cannot mix include and exclude field selectors".to_string());
    }
    if has_exclude {
        Ok(FieldSelection::Exclude(
            parts
                .iter()
                .map(|p| p.trim_start_matches('-').to_string())
                .collect(),
        ))
    } else {
        Ok(FieldSelection::Include(
            parts.iter().map(|p| p.to_string()).collect(),
        ))
    }
}

fn apply_field_selection(hits: &mut [serde_json::Value], selection: &FieldSelection) {
    match selection {
        FieldSelection::All => {}
        FieldSelection::Include(fields) => {
            let mut top_level: Vec<&str> = Vec::new();
            let mut source_sub: Vec<&str> = Vec::new();
            for f in fields {
                if let Some(sub) = f.strip_prefix("source.") {
                    source_sub.push(sub);
                } else {
                    top_level.push(f);
                }
            }
            if !source_sub.is_empty() && !top_level.contains(&"source") {
                top_level.push("source");
            }
            for hit in hits.iter_mut() {
                if let Some(obj) = hit.as_object_mut() {
                    let keys: Vec<String> = obj.keys().cloned().collect();
                    for key in keys {
                        if !top_level.contains(&key.as_str()) {
                            obj.remove(&key);
                        }
                    }
                    if !source_sub.is_empty()
                        && let Some(source) = obj.get_mut("source").and_then(|v| v.as_object_mut())
                    {
                        let src_keys: Vec<String> = source.keys().cloned().collect();
                        for key in src_keys {
                            if !source_sub.contains(&key.as_str()) {
                                source.remove(&key);
                            }
                        }
                    }
                }
            }
        }
        FieldSelection::Exclude(fields) => {
            for hit in hits.iter_mut() {
                if let Some(obj) = hit.as_object_mut() {
                    for f in fields {
                        if let Some(sub) = f.strip_prefix("source.") {
                            if let Some(source) =
                                obj.get_mut("source").and_then(|v| v.as_object_mut())
                            {
                                source.remove(sub);
                            }
                        } else {
                            obj.remove(f.as_str());
                        }
                    }
                }
            }
        }
    }
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

/// Optional bundle-mode configuration for the HTTP server. When `state` is
/// `Some`, `/cve`, `/cwe`, `/cwe/relation`, `/ready`, and `/admin/reload`
/// become live. Leaving it `None` preserves the legacy `--corpus`-only
/// surface.
#[derive(Clone, Default)]
pub struct HttpBundleConfig {
    pub state: Option<Arc<ArcSwap<BundleState>>>,
    pub bundles_dir: Option<PathBuf>,
    pub admin_token: Option<String>,
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
        false,
        rerank_cfg,
        batch_max_queries,
        None,
        ingest_max_body,
        10_000,
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
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
) -> Result<(), HttpError> {
    serve_http_with_registry_port_bundle(
        registry,
        port,
        embedder,
        token,
        dense_only,
        cwe_expand_default,
        rerank_cfg,
        batch_max_queries,
        tenant_field,
        ingest_max_body,
        similar_overfetch_cap,
        HttpBundleConfig::default(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry_port_bundle(
    registry: fastrag::corpus::CorpusRegistry,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
    bundle_cfg: HttpBundleConfig,
) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    serve_http_with_registry_and_bundle(
        registry,
        listener,
        embedder,
        token,
        dense_only,
        cwe_expand_default,
        rerank_cfg,
        batch_max_queries,
        tenant_field,
        ingest_max_body,
        similar_overfetch_cap,
        bundle_cfg,
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
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
) -> Result<(), HttpError> {
    serve_http_with_registry_and_bundle(
        registry,
        listener,
        embedder,
        token,
        dense_only,
        cwe_expand_default,
        rerank_cfg,
        batch_max_queries,
        tenant_field,
        ingest_max_body,
        similar_overfetch_cap,
        HttpBundleConfig::default(),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn serve_http_with_registry_and_bundle(
    registry: fastrag::corpus::CorpusRegistry,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    cwe_expand_default: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    ingest_max_body: usize,
    similar_overfetch_cap: usize,
    bundle_cfg: HttpBundleConfig,
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
    metrics::describe_counter!("fastrag_similar_total", "Total /similar requests served");
    metrics::describe_histogram!(
        "fastrag_similar_duration_seconds",
        "Latency of /similar requests in seconds"
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
        cwe_expand_default,
        batch_max_queries,
        tenant_field,
        ingest_locks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        ingest_max_body,
        similar_overfetch_cap,
        #[cfg(feature = "rerank")]
        reranker: rerank_cfg.reranker,
        #[cfg(feature = "rerank")]
        rerank_over_fetch: rerank_cfg.over_fetch,
        bundle: bundle_cfg.state,
        bundles_dir: bundle_cfg.bundles_dir,
        admin_token: bundle_cfg.admin_token.map(Arc::new),
        reload_lock: Arc::new(tokio::sync::Mutex::new(())),
    };

    let app = build_router(app_state, auth_state);

    axum::serve(listener, app)
        .await
        .map_err(|e| HttpError::Server(e.to_string()))?;
    Ok(())
}

fn build_router(app_state: AppState, auth_state: AuthState) -> Router {
    let max_body = app_state.ingest_max_body;

    // /health stays unauthenticated for liveness probes.
    let protected = Router::new()
        .route("/query", get(query).post(post_query))
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
        .route(
            "/similar",
            axum::routing::post(similar_handler).layer(DefaultBodyLimit::max(1024 * 1024)),
        )
        .route("/cve/:id", get(get_cve_handler))
        .route("/cwe/relation", get(cwe_relation_handler))
        .route("/cwe/:id", get(get_cwe_handler))
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            tenant_middleware,
        ))
        .route_layer(middleware::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ));

    // /admin/* is gated by a SEPARATE token (never by --token). If no admin
    // token is configured, every call returns 401. Real handlers land in
    // Task 6; the stub returns 501 so a live admin_token proves the auth
    // gate passed.
    let admin = Router::new()
        .route(
            "/admin/reload",
            axum::routing::post(admin_reload_stub_handler),
        )
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            admin_auth_middleware,
        ));

    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .merge(admin)
        .with_state(app_state)
}

/// Test-only builder: assemble an `AppState` + router from a loaded bundle
/// and a caller-supplied embedder. Integration tests pass a `MockEmbedder`
/// via their own dev-dependency so the lib crate doesn't need
/// `fastrag-embed/test-utils` enabled in its regular feature set.
pub struct TestAppState(AppState);

impl TestAppState {
    pub fn from_bundle(
        bundle_path: &std::path::Path,
        admin_token: Option<String>,
        embedder: DynEmbedder,
    ) -> Result<Self, fastrag::bundle::BundleError> {
        let state = fastrag::bundle::BundleState::load(bundle_path)?;
        let bundle = Some(Arc::new(ArcSwap::from_pointee(state)));
        let bundles_dir = bundle_path.parent().map(|p| p.to_path_buf());
        let registry = fastrag::corpus::CorpusRegistry::new();

        static TEST_METRICS: std::sync::OnceLock<PrometheusHandle> = std::sync::OnceLock::new();
        let metrics = TEST_METRICS
            .get_or_init(|| {
                PrometheusBuilder::new()
                    .install_recorder()
                    .unwrap_or_else(|_| PrometheusBuilder::new().build_recorder().handle())
            })
            .clone();

        Ok(Self(AppState {
            registry,
            embedder,
            metrics,
            dense_only: false,
            cwe_expand_default: false,
            batch_max_queries: 100,
            tenant_field: None,
            ingest_locks: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            ingest_max_body: 52_428_800,
            similar_overfetch_cap: 10_000,
            #[cfg(feature = "rerank")]
            reranker: None,
            #[cfg(feature = "rerank")]
            rerank_over_fetch: 10,
            bundle,
            bundles_dir,
            admin_token: admin_token.map(Arc::new),
            reload_lock: Arc::new(tokio::sync::Mutex::new(())),
        }))
    }
}

pub fn build_router_for_test(app_state_internal: TestAppState) -> Router {
    let auth_state = AuthState { token: None };
    build_router(app_state_internal.0, auth_state)
}

async fn admin_auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(expected) = state.admin_token.as_deref() else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    let provided = req
        .headers()
        .get("x-fastrag-admin-token")
        .and_then(|v| v.to_str().ok());
    let provided = match provided {
        Some(p) => p,
        None => return Err(StatusCode::UNAUTHORIZED),
    };
    if expected.as_bytes().ct_eq(provided.as_bytes()).into() {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn admin_reload_stub_handler() -> (StatusCode, Json<serde_json::Value>) {
    // Real implementation lands in Task 6.
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({"error": "not_implemented"})),
    )
}

async fn get_cve_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    if !params.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "unexpected_query_params",
                "message": "/cve/{id} is a direct lookup; query params not allowed"
            })),
        )
            .into_response();
    }
    let Some(bundle) = state.bundle.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "bundle_not_loaded"})),
        )
            .into_response();
    };
    let guard = bundle.load_full();
    let Some(corpus) = guard.corpora.get("cve") else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "corpus_cve_missing"})),
        )
            .into_response();
    };

    match fastrag::corpus::lookup_by_field(corpus.dir(), "cve_id", &id) {
        Ok(hits) if !hits.is_empty() => {
            (StatusCode::OK, Json(json!({"hits": hits}))).into_response()
        }
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "cve_not_found", "id": id})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "lookup_failed", "message": e.to_string()})),
        )
            .into_response(),
    }
}

async fn get_cwe_handler(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let numeric = id.strip_prefix("CWE-").unwrap_or(&id);
    let cwe_id: u32 = match numeric.parse() {
        Ok(n) => n,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "invalid_cwe_id",
                    "message": "cwe id must be integer or CWE-<integer>"
                })),
            )
                .into_response();
        }
    };
    let Some(bundle) = state.bundle.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "bundle_not_loaded"})),
        )
            .into_response();
    };
    let guard = bundle.load_full();
    let Some(corpus) = guard.corpora.get("cwe") else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "corpus_cwe_missing"})),
        )
            .into_response();
    };

    match fastrag::corpus::lookup_by_field(corpus.dir(), "cwe_id", &cwe_id.to_string()) {
        Ok(hits) if !hits.is_empty() => {
            (StatusCode::OK, Json(json!({"hits": hits}))).into_response()
        }
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "cwe_not_found", "id": cwe_id})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "lookup_failed", "message": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct CweRelationParams {
    cwe_id: Option<String>,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default)]
    max_depth: Option<usize>,
}

async fn cwe_relation_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<CweRelationParams>,
) -> Response {
    let Some(raw) = params.cwe_id else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing_cwe_id"})),
        )
            .into_response();
    };
    let raw_trim = raw.strip_prefix("CWE-").unwrap_or(&raw);
    let cwe: u32 = match raw_trim.parse() {
        Ok(n) => n,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "invalid_cwe_id"})),
            )
                .into_response();
        }
    };
    let dir = params.direction.as_deref().unwrap_or("both");
    if !matches!(dir, "ancestors" | "descendants" | "both") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid_direction"})),
        )
            .into_response();
    }
    let Some(bundle) = state.bundle.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "bundle_not_loaded"})),
        )
            .into_response();
    };
    let guard = bundle.load_full();
    let tax = guard.taxonomy.as_ref();

    let ancestors: Vec<u32> = if matches!(dir, "ancestors" | "both") {
        match params.max_depth {
            Some(d) => tax.ancestors_bounded(cwe, d),
            None => tax.ancestors(cwe),
        }
    } else {
        Vec::new()
    };
    let descendants: Vec<u32> = if matches!(dir, "descendants" | "both") {
        let mut d = tax.expand(cwe);
        d.retain(|&x| x != cwe);
        d
    } else {
        Vec::new()
    };

    (
        StatusCode::OK,
        Json(json!({
            "cwe_id": cwe,
            "ancestors": ancestors,
            "descendants": descendants,
        })),
    )
        .into_response()
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
        cwe_field: None,
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

/// Resolved parameters for the core query dispatcher.
struct QueryCoreArgs<'a> {
    q: &'a str,
    top_k: usize,
    snippet_len: usize,
    cwe_expand: bool,
    rerank_skip: bool,
    over_fetch_override: Option<usize>,
    filter: Option<&'a fastrag::filter::FilterExpr>,
    corpus_name: &'a str,
    hybrid: fastrag::corpus::hybrid::HybridOpts,
    temporal_policy: fastrag::corpus::temporal::TemporalPolicy,
    date_fields: Vec<String>,
}

/// Core query dispatcher shared by the GET and POST /query handlers.
///
/// Callers must resolve `temporal_policy` and `date_fields` before calling this;
/// the GET handler applies the legacy bridge (time_decay_field → FavorRecent(Medium)),
/// the POST handler uses the body fields directly.
fn run_query_core(
    state: &AppState,
    args: QueryCoreArgs<'_>,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    let corpus_dir = state
        .registry
        .corpus_path(args.corpus_name)
        .ok_or_else(|| CorpusError::NotFound(args.corpus_name.to_string()))?;

    // dense_only is a server-level flag; per-request hybrid opts take precedence.
    let _ = state.dense_only;

    let query_opts = ops::QueryOpts {
        cwe_expand: args.cwe_expand,
        hybrid: args.hybrid,
        temporal_policy: args.temporal_policy,
        date_fields: args.date_fields,
    };

    #[cfg(feature = "rerank")]
    if !args.rerank_skip
        && let Some(ref reranker) = state.reranker
    {
        let over_fetch = args.over_fetch_override.unwrap_or(state.rerank_over_fetch);
        return ops::query_corpus_reranked_opts(
            &corpus_dir,
            args.q,
            args.top_k,
            over_fetch,
            state.embedder.as_ref() as &dyn DynEmbedderTrait,
            reranker.as_ref(),
            args.filter,
            &query_opts,
            &mut fastrag::corpus::LatencyBreakdown::default(),
            args.snippet_len,
        );
    }

    #[cfg(not(feature = "rerank"))]
    let _ = (args.rerank_skip, args.over_fetch_override);

    ops::query_corpus_with_filter_opts(
        &corpus_dir,
        args.q,
        args.top_k,
        state.embedder.as_ref() as &dyn DynEmbedderTrait,
        args.filter,
        &query_opts,
        &mut fastrag::corpus::LatencyBreakdown::default(),
        args.snippet_len,
    )
}

fn run_query(
    state: &AppState,
    params: &QueryParams,
    filter: Option<&fastrag::filter::FilterExpr>,
    corpus_name: &str,
    hybrid: fastrag::corpus::hybrid::HybridOpts,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    // If the caller supplied a time_decay_field, bridge it into the new
    // late-stage temporal policy path.  Default to FavorRecent(Medium); the
    // legacy halflife/weight/prior/blend params are accepted for compatibility
    // but the strength ladder governs the new path.
    let (temporal_policy, date_fields) = match &params.time_decay_field {
        Some(field) => (
            fastrag::corpus::temporal::TemporalPolicy::FavorRecent(
                fastrag::corpus::temporal::Strength::Medium,
            ),
            vec![field.clone()],
        ),
        None => (Default::default(), vec![]),
    };

    run_query_core(
        state,
        QueryCoreArgs {
            q: &params.q,
            top_k: params.top_k,
            snippet_len: params.snippet_len,
            cwe_expand: params.cwe_expand.unwrap_or(state.cwe_expand_default),
            rerank_skip: params.rerank.as_deref() == Some("off"),
            over_fetch_override: params.over_fetch,
            filter,
            corpus_name,
            hybrid,
            temporal_policy,
            date_fields,
        },
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
                snippet_len: item.snippet_len,
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
                Ok(dtos) => {
                    let field_sel = parse_field_selection(req.queries[i].fields.as_deref())
                        .unwrap_or(FieldSelection::All);
                    let mut json_hits: Vec<serde_json::Value> = dtos
                        .iter()
                        .map(|h| serde_json::to_value(h).unwrap())
                        .collect();
                    apply_field_selection(&mut json_hits, &field_sel);
                    BatchResultItem {
                        index: i,
                        hits: Some(json_hits),
                        error: None,
                    }
                }
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
) -> Result<Json<Vec<serde_json::Value>>, Response> {
    let field_sel = match parse_field_selection(params.fields.as_deref()) {
        Ok(sel) => sel,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e).into_response()),
    };

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

    // Validate + build hybrid opts. Return 400 on bad inputs.
    let hybrid_opts = fastrag::corpus::hybrid::build_hybrid_opts_from_parts(
        params.hybrid.unwrap_or(false),
        params.rrf_k.unwrap_or(60),
        params.rrf_overfetch.unwrap_or(4),
        params.time_decay_field.clone(),
        params.time_decay_halflife.as_deref().unwrap_or("30d"),
        params.time_decay_weight.unwrap_or(0.3),
        params.time_decay_dateless_prior.unwrap_or(0.5),
        params
            .time_decay_blend
            .as_deref()
            .unwrap_or("multiplicative"),
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e).into_response())?;

    let corpus_name = params.corpus.as_deref().unwrap_or("default");
    let lock = get_or_create_lock(&state.ingest_locks, corpus_name);
    let _read_guard = lock.read().await;
    let result = run_query(
        &state,
        &params,
        filter_expr.as_ref(),
        corpus_name,
        hybrid_opts,
    );

    let elapsed = start.elapsed();
    metrics::counter!("fastrag_query_total").increment(1);
    metrics::histogram!("fastrag_query_duration_seconds").record(elapsed.as_secs_f64());
    span.record("latency_ms", elapsed.as_millis() as u64);

    match result {
        Ok(hits) => {
            span.record("hit_count", hits.len());
            info!("query served");
            let mut json_hits: Vec<serde_json::Value> = hits
                .iter()
                .map(|h| serde_json::to_value(h).unwrap())
                .collect();
            apply_field_selection(&mut json_hits, &field_sel);
            Ok(Json(json_hits))
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

async fn post_query(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantFilter>>,
    Json(body): Json<PostQueryBody>,
) -> Result<Json<Vec<serde_json::Value>>, Response> {
    let field_sel = match parse_field_selection(body.fields.as_deref()) {
        Ok(sel) => sel,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e).into_response()),
    };

    let span = info_span!(
        "post_query",
        q = %body.q,
        top_k = body.top_k,
        hit_count = tracing::field::Empty,
        latency_ms = tracing::field::Empty,
    );
    let _enter = span.enter();
    let start = Instant::now();

    let base_filter: Option<fastrag::filter::FilterExpr> = match &body.filter {
        None => None,
        Some(serde_json::Value::String(s)) => match fastrag::filter::parse(s) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
        Some(json_val) => match serde_json::from_value(json_val.clone()) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
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

    // Validate + build hybrid opts.
    let hybrid_opts = fastrag::corpus::hybrid::build_hybrid_opts_from_parts(
        body.hybrid.unwrap_or(false),
        body.rrf_k.unwrap_or(60),
        body.rrf_overfetch.unwrap_or(4),
        // Pass legacy time_decay_field through so HybridOpts is populated when
        // only the legacy params are used; the temporal_policy bridge below handles
        // which policy actually fires.
        body.time_decay_field.clone(),
        body.time_decay_halflife.as_deref().unwrap_or("30d"),
        body.time_decay_weight.unwrap_or(0.3),
        body.time_decay_dateless_prior.unwrap_or(0.5),
        body.time_decay_blend.as_deref().unwrap_or("multiplicative"),
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e).into_response())?;

    // Precedence for temporal settings (see PostQueryBody doc comment):
    // 1. explicit temporal_policy in body (non-default) + body.date_fields
    // 2. legacy time_decay_field → FavorRecent(Medium)
    // 3. default Auto, no date fields
    let (temporal_policy, date_fields) =
        if body.temporal_policy != fastrag::corpus::temporal::TemporalPolicy::Auto {
            // Explicit non-default policy supplied — use it verbatim.
            (body.temporal_policy, body.date_fields)
        } else if let Some(ref field) = body.time_decay_field {
            // Legacy bridge: time_decay_field set, temporal_policy is default Auto.
            (
                fastrag::corpus::temporal::TemporalPolicy::FavorRecent(
                    fastrag::corpus::temporal::Strength::Medium,
                ),
                vec![field.clone()],
            )
        } else if !body.date_fields.is_empty() {
            // date_fields provided without an explicit policy — default to Auto
            // (abstaining detector will decide per query).
            (
                fastrag::corpus::temporal::TemporalPolicy::Auto,
                body.date_fields,
            )
        } else {
            (Default::default(), vec![])
        };

    let corpus_name = body.corpus.as_deref().unwrap_or("default");
    let lock = get_or_create_lock(&state.ingest_locks, corpus_name);
    let _read_guard = lock.read().await;
    let result = run_query_core(
        &state,
        QueryCoreArgs {
            q: &body.q,
            top_k: body.top_k,
            snippet_len: body.snippet_len,
            cwe_expand: body.cwe_expand.unwrap_or(state.cwe_expand_default),
            rerank_skip: body.rerank.as_deref() == Some("off"),
            over_fetch_override: body.over_fetch,
            filter: filter_expr.as_ref(),
            corpus_name,
            hybrid: hybrid_opts,
            temporal_policy,
            date_fields,
        },
    );

    let elapsed = start.elapsed();
    metrics::counter!("fastrag_query_total").increment(1);
    metrics::histogram!("fastrag_query_duration_seconds").record(elapsed.as_secs_f64());
    span.record("latency_ms", elapsed.as_millis() as u64);

    match result {
        Ok(hits) => {
            span.record("hit_count", hits.len());
            info!("post_query served");
            let mut json_hits: Vec<serde_json::Value> = hits
                .iter()
                .map(|h| serde_json::to_value(h).unwrap())
                .collect();
            apply_field_selection(&mut json_hits, &field_sel);
            Ok(Json(json_hits))
        }
        Err(CorpusError::NotFound(_)) => {
            warn!(corpus = corpus_name, "corpus not found");
            Err((StatusCode::NOT_FOUND, "corpus not found").into_response())
        }
        Err(err) => {
            warn!(error = %err, "post_query failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response())
        }
    }
}

async fn similar_handler(
    State(state): State<AppState>,
    tenant_ext: Option<Extension<TenantFilter>>,
    Json(req): Json<SimilarRequest>,
) -> Result<Json<serde_json::Value>, Response> {
    // Reject unsupported params up front.
    if req.hybrid.is_some()
        || req.rrf_k.is_some()
        || req.rrf_overfetch.is_some()
        || req.time_decay_field.is_some()
        || req.time_decay_halflife.is_some()
        || req.time_decay_weight.is_some()
        || req.time_decay_dateless_prior.is_some()
        || req.time_decay_blend.is_some()
        || req.cwe_expand.is_some()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "/similar does not support hybrid or temporal decay; see /query",
        )
            .into_response());
    }
    if req.rerank.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "/similar does not support reranking",
        )
            .into_response());
    }

    // Validate scalars.
    if req.text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text must be non-empty").into_response());
    }
    if !(0.0..=1.0).contains(&req.threshold) {
        return Err((StatusCode::BAD_REQUEST, "threshold must be in [0.0, 1.0]").into_response());
    }
    if req.max_results == 0 || req.max_results > 1000 {
        return Err((StatusCode::BAD_REQUEST, "max_results must be in [1, 1000]").into_response());
    }

    // Resolve target corpora.
    if req.corpus.is_some() && req.corpora.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "exactly one of `corpus` or `corpora` may be set",
        )
            .into_response());
    }
    let names: Vec<String> = match (&req.corpus, &req.corpora) {
        (Some(n), None) => vec![n.clone()],
        (None, Some(v)) => {
            if v.is_empty() {
                return Err(
                    (StatusCode::BAD_REQUEST, "`corpora` must be non-empty").into_response()
                );
            }
            v.clone()
        }
        (None, None) => vec!["default".into()],
        (Some(_), Some(_)) => unreachable!(), // handled above
    };
    let mut targets: Vec<(String, std::path::PathBuf)> = Vec::with_capacity(names.len());
    for name in &names {
        let Some(path) = state.registry.corpus_path(name) else {
            return Err(
                (StatusCode::NOT_FOUND, format!("corpus not found: {name}")).into_response()
            );
        };
        targets.push((name.clone(), path));
    }

    // Parse filter (string or JSON AST) + AND-in tenant filter.
    let base_filter: Option<fastrag::filter::FilterExpr> = match &req.filter {
        None => None,
        Some(serde_json::Value::String(s)) => match fastrag::filter::parse(s) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
        Some(v) => match serde_json::from_value::<fastrag::filter::FilterExpr>(v.clone()) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
    };
    let filter = if let Some(Extension(tf)) = tenant_ext {
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

    // Acquire per-corpus read locks (same pattern as /query).
    let mut guards = Vec::with_capacity(targets.len());
    for (name, _) in &targets {
        let lock = get_or_create_lock(&state.ingest_locks, name);
        guards.push(lock);
    }
    let _read_guards: Vec<_> = {
        let mut out = Vec::with_capacity(guards.len());
        for lock in &guards {
            out.push(lock.read().await);
        }
        out
    };

    // Parse optional verify block.
    let verify_cfg = match &req.verify {
        None => None,
        Some(v) => match parse_verify_block(v) {
            Ok(cfg) => Some(cfg),
            Err(msg) => return Err((StatusCode::BAD_REQUEST, msg).into_response()),
        },
    };

    let snippet_len = 150;
    let request = fastrag::corpus::SimilarityRequest {
        text: req.text.clone(),
        threshold: req.threshold,
        max_results: req.max_results,
        targets,
        filter,
        snippet_len,
        overfetch_cap: state.similar_overfetch_cap,
        verify: verify_cfg,
    };
    let embedder = state.embedder.clone();

    let field_sel = match parse_field_selection(req.fields.as_deref()) {
        Ok(sel) => sel,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e).into_response()),
    };

    let span = info_span!(
        "similar",
        text = %request.text,
        threshold = request.threshold,
        max_results = request.max_results,
        corpora_count = request.targets.len(),
    );
    let _enter = span.enter();
    let start = Instant::now();

    let result = tokio::task::spawn_blocking(move || {
        fastrag::corpus::similarity_search(
            embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
            &request,
        )
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")).into_response())?;

    let elapsed = start.elapsed();
    metrics::counter!("fastrag_similar_total").increment(1);
    metrics::histogram!("fastrag_similar_duration_seconds").record(elapsed.as_secs_f64());

    match result {
        Ok(resp) => {
            let mut value = serde_json::to_value(&resp).unwrap();
            // Apply field projection to each hit's source via existing helper.
            if let Some(hits) = value.get_mut("hits").and_then(|v| v.as_array_mut()) {
                apply_field_selection(hits, &field_sel);
            }
            info!(hit_count = resp.hits.len(), "similar served");
            Ok(Json(value))
        }
        Err(CorpusError::NotFound(name)) => {
            Err((StatusCode::NOT_FOUND, format!("corpus not found: {name}")).into_response())
        }
        Err(e) => {
            warn!(error = %e, "similar failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response())
        }
    }
}
