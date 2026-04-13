use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastrag::corpus::{CorpusError, SearchHitDto};
use fastrag::{DynEmbedder, DynEmbedderTrait, ops};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::Deserialize;
use serde_json::json;
use subtle::ConstantTimeEq;
use thiserror::Error;
use tracing::{info, info_span, warn};

#[derive(Clone)]
struct AppState {
    corpus_dir: PathBuf,
    embedder: DynEmbedder,
    metrics: PrometheusHandle,
    dense_only: bool,
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
}

fn default_top_k() -> usize {
    5
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

pub async fn serve_http(
    corpus_dir: PathBuf,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
) -> Result<(), HttpError> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    serve_http_with_embedder(
        corpus_dir, listener, embedder, token, dense_only, rerank_cfg,
    )
    .await
}

pub async fn serve_http_with_embedder(
    corpus_dir: PathBuf,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
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

    if let Ok(info) =
        fastrag::corpus::corpus_info(&corpus_dir, embedder.as_ref() as &dyn DynEmbedderTrait)
    {
        metrics::gauge!("fastrag_index_entries").set(info.entry_count as f64);
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
        corpus_dir,
        embedder,
        metrics,
        dense_only,
        #[cfg(feature = "rerank")]
        reranker: rerank_cfg.reranker,
        #[cfg(feature = "rerank")]
        rerank_over_fetch: rerank_cfg.over_fetch,
    };

    // /health stays unauthenticated for liveness probes.
    let protected = Router::new()
        .route("/query", get(query))
        .route("/metrics", get(metrics_handler))
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

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.render()
}

fn run_query(
    state: &AppState,
    params: &QueryParams,
    filter: Option<&fastrag::filter::FilterExpr>,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    let _ = state.dense_only; // hybrid removed; dense-only is the only path

    #[cfg(feature = "rerank")]
    {
        let skip = params.rerank.as_deref() == Some("off");
        if !skip && let Some(ref reranker) = state.reranker {
            let over_fetch = params.over_fetch.unwrap_or(state.rerank_over_fetch);
            return ops::query_corpus_reranked(
                &state.corpus_dir,
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
        &state.corpus_dir,
        &params.q,
        params.top_k,
        state.embedder.as_ref() as &dyn DynEmbedderTrait,
        filter,
        &mut fastrag::corpus::LatencyBreakdown::default(),
    )
}

async fn query(
    State(state): State<AppState>,
    Query(params): Query<QueryParams>,
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

    let filter_expr: Option<fastrag::filter::FilterExpr> = match params.filter.as_deref() {
        Some(s) => match fastrag::filter::parse(s) {
            Ok(f) => Some(f),
            Err(e) => {
                return Err((StatusCode::BAD_REQUEST, format!("bad filter: {e}")).into_response());
            }
        },
        None => None,
    };

    let result = run_query(&state, &params, filter_expr.as_ref());

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
        Err(err) => {
            warn!(error = %err, "query failed");
            Err((StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response())
        }
    }
}
