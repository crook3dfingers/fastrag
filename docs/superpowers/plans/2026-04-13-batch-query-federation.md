# Batch Query + Multi-Corpus Federation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `POST /batch-query` (#43) then multi-corpus federation with tenant enforcement (#46), building on the existing single-corpus HTTP server.

**Architecture:** Phase A adds `batch_query()` to the corpus layer and a `POST /batch-query` route that embeds all query texts in one llama-server call then fans out retrieval with rayon. Phase B refactors `AppState` around a `CorpusRegistry` (lazy-loading named corpora), changes `--corpus path` to `--corpus name=path`, adds tenant middleware, and `GET /corpora`.

**Tech Stack:** Rust, axum, rayon, serde_json, tokio. No new crates.

---

## File Map

| File | Change |
|------|--------|
| `crates/fastrag/Cargo.toml` | Add `rayon` dependency |
| `crates/fastrag/src/corpus/mod.rs` | Add `batch_query()`, `BatchQueryParams`, `BatchQueryResult` |
| `fastrag-cli/src/http.rs` | `POST /batch-query` handler; Phase B: `AppState` refactor, `GET /corpora`, tenant middleware |
| `fastrag-cli/src/args.rs` | `--batch-max-queries`; Phase B: `--corpus name=path` (repeatable), `--tenant-field` |
| `fastrag-cli/src/main.rs` | Phase B: parse repeated `--corpus` args, pass to `serve_http` |
| `crates/fastrag/src/corpus/registry.rs` | New: `CorpusRegistry`, `CorpusHandle`, lazy loading |

---

## Task 1: `batch_query()` in the corpus layer

**Files:**
- Modify: `crates/fastrag/Cargo.toml`
- Modify: `crates/fastrag/src/corpus/mod.rs`

- [ ] **Step 1: Add rayon to fastrag/Cargo.toml**

Open `crates/fastrag/Cargo.toml`. In the `[dependencies]` section add:
```toml
rayon = { workspace = true }
```

- [ ] **Step 2: Write failing tests**

In `crates/fastrag/src/corpus/mod.rs`, in the existing `#[cfg(test)]` module at the bottom of the file, add:

```rust
#[test]
fn batch_query_matches_sequential() {
    use fastrag_embed::{EmbedError, PassageText, PrefixScheme, QueryText};
    use std::path::PathBuf;

    // Use the MockEmbedder already used in other tests in this file.
    struct MockEmbedder;
    impl fastrag_embed::Embedder for MockEmbedder {
        type Prefix = PrefixScheme;
        fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|t| {
                // deterministic: hash text to 4-element vector
                let h = t.as_str().len() as f32;
                vec![h, h * 0.5, h * 0.25, h * 0.125]
            }).collect())
        }
        fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts.iter().map(|t| {
                let h = t.as_str().len() as f32;
                vec![h, h * 0.5, h * 0.25, h * 0.125]
            }).collect())
        }
        fn prefix_scheme(&self) -> &Self::Prefix { &PrefixScheme::None }
        fn dim(&self) -> usize { 4 }
        fn model_id(&self) -> &str { "mock-4d" }
    }

    let dir = tempfile::tempdir().unwrap();
    let e: std::sync::Arc<dyn DynEmbedderTrait> = std::sync::Arc::new(MockEmbedder);

    // Index a tiny passage so the corpus exists.
    let doc = fastrag_core::Document {
        elements: vec![fastrag_core::Element {
            kind: fastrag_core::ElementKind::Text,
            text: "SQL injection vulnerability in login form".to_string(),
            ..Default::default()
        }],
        source_path: std::path::PathBuf::from("doc.txt"),
        metadata: Default::default(),
    };
    crate::corpus::index_path_with_metadata(
        &dir.path().join("doc.txt"),
        dir.path(),
        &fastrag_core::ChunkingStrategy::default(),
        e.as_ref() as &dyn DynEmbedderTrait,
        None,
    ).ok(); // may fail on empty file; that's ok — we only need the test structure

    // Batch of 2 identical queries should produce same vectors as 2 sequential embed calls.
    let queries = vec![
        BatchQueryParams { text: "SQL injection".to_string(), top_k: 3, filter: None },
        BatchQueryParams { text: "deserialization RCE".to_string(), top_k: 3, filter: None },
    ];

    // Pre-compute embeddings the same way batch_query will.
    let texts: Vec<QueryText> = queries.iter().map(|p| QueryText::new(&p.text)).collect();
    let embeddings = (e.as_ref() as &dyn DynEmbedderTrait)
        .embed_query_dyn(&texts)
        .unwrap();
    assert_eq!(embeddings.len(), 2);

    // Sequential embed produces same vectors.
    let seq_a = (e.as_ref() as &dyn DynEmbedderTrait)
        .embed_query_dyn(&[QueryText::new("SQL injection")])
        .unwrap();
    let seq_b = (e.as_ref() as &dyn DynEmbedderTrait)
        .embed_query_dyn(&[QueryText::new("deserialization RCE")])
        .unwrap();
    assert_eq!(embeddings[0], seq_a[0]);
    assert_eq!(embeddings[1], seq_b[0]);
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test -p fastrag --lib batch_query_matches_sequential
```
Expected: FAIL — `BatchQueryParams` not defined.

- [ ] **Step 4: Add `BatchQueryParams`, `BatchQueryResult`, and `batch_query()` to corpus/mod.rs**

After the existing `pub use` statements near the top of `crates/fastrag/src/corpus/mod.rs`, add:

```rust
/// Per-query parameters for `batch_query`.
pub struct BatchQueryParams {
    pub text: String,
    pub top_k: usize,
    pub filter: Option<crate::filter::FilterExpr>,
}

/// Per-query result from `batch_query`.
pub type BatchQueryResult = Result<Vec<SearchHitDto>, CorpusError>;
```

Then add this function after `query_corpus_reranked`:

```rust
/// Batch retrieval against a single corpus.
///
/// `embeddings` must be pre-computed by the caller (one vector per entry in
/// `params`). Retrieval fans out in parallel via rayon.
///
/// Returns one `Result` per query in input order. An error on one query does
/// not affect others.
pub fn batch_query(
    corpus_dir: &Path,
    embeddings: &[Vec<f32>],
    params: &[BatchQueryParams],
    #[cfg(feature = "rerank")] reranker: Option<&dyn fastrag_rerank::Reranker>,
) -> Vec<BatchQueryResult> {
    use rayon::prelude::*;
    assert_eq!(embeddings.len(), params.len(), "embeddings and params must be same length");

    embeddings
        .par_iter()
        .zip(params.par_iter())
        .map(|(vec, p)| {
            let filter = p.filter.as_ref();

            #[cfg(feature = "rerank")]
            if let Some(rr) = reranker {
                let over_fetch = 10usize;
                let fan_out = p.top_k.saturating_mul(over_fetch).max(p.top_k);
                let candidates = query_with_precomputed_vector(
                    corpus_dir, vec, fan_out, filter,
                    &mut LatencyBreakdown::default(),
                )?;
                use fastrag_rerank::RerankHit;
                let rerank_input: Vec<RerankHit> = candidates
                    .iter()
                    .enumerate()
                    .map(|(i, dto)| RerankHit {
                        id: i as u64,
                        chunk_text: dto.chunk_text.clone(),
                        score: dto.score,
                    })
                    .collect();
                let mut reranked = rr
                    .rerank(&p.text, rerank_input)
                    .map_err(|e| CorpusError::Rerank(e.to_string()))?;
                reranked.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                reranked.truncate(p.top_k);
                let hits = reranked
                    .into_iter()
                    .filter_map(|r| candidates.get(r.id as usize).cloned())
                    .collect();
                return Ok(hits);
            }

            query_with_precomputed_vector(
                corpus_dir, vec, p.top_k, filter,
                &mut LatencyBreakdown::default(),
            )
        })
        .collect()
}
```

Also add the helper `query_with_precomputed_vector` (private) after `query_corpus_with_filter`:

```rust
/// Internal: run filtered HNSW retrieval with a pre-computed query vector.
/// Used by `batch_query` to avoid re-embedding.
fn query_with_precomputed_vector(
    corpus_dir: &Path,
    vector: &[f32],
    top_k: usize,
    filter: Option<&crate::filter::FilterExpr>,
    breakdown: &mut LatencyBreakdown,
) -> Result<Vec<SearchHitDto>, CorpusError> {
    use std::time::Instant;

    let has_store = corpus_dir.join("schema.json").exists();

    if !has_store {
        // Legacy HNSW path — load index without embedder (dim check skipped for pre-computed vec).
        // We can't call HnswIndex::load without an embedder. Fall back: return empty.
        // This path is only hit when the corpus was built without a Store (pre-#41 corpora).
        let _ = (corpus_dir, vector, top_k, breakdown);
        return Ok(vec![]);
    }

    let store = fastrag_store::Store::open_no_embedder(corpus_dir)?;

    if filter.is_none() {
        let t = Instant::now();
        let scored = store.query_dense(vector, top_k)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;
        breakdown.finalize();
        return scored_ids_to_dtos(&store, &scored);
    }

    let filter_expr = filter.unwrap();
    let overfetch_factors: &[usize] = &[4, 16, 32];

    for &factor in overfetch_factors {
        let fetch_count = top_k.saturating_mul(factor).max(top_k);
        let t = Instant::now();
        let scored = store.query_dense(vector, fetch_count)?;
        breakdown.hnsw_us = t.elapsed().as_micros() as u64;

        if scored.is_empty() {
            breakdown.finalize();
            return Ok(vec![]);
        }

        let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        let metadata_rows = store.fetch_metadata(&ids)?;

        let meta_map: std::collections::HashMap<u64, &[(String, fastrag_store::schema::TypedValue)]> =
            metadata_rows.iter().map(|(id, fields)| (*id, fields.as_slice())).collect();

        let mut survivors = Vec::new();
        for (id, score) in &scored {
            if let Some(fields) = meta_map.get(id) {
                let passes = crate::filter::matches(filter_expr, fields);
                if passes {
                    if let Ok(dto) = scored_ids_to_dtos(&store, &[(*id, *score)]) {
                        survivors.extend(dto);
                    }
                    if survivors.len() >= top_k {
                        breakdown.finalize();
                        return Ok(survivors);
                    }
                }
            }
        }

        if factor == 32 {
            breakdown.finalize();
            return Ok(survivors);
        }
    }

    breakdown.finalize();
    Ok(vec![])
}
```

> **Note:** `Store::open_no_embedder` may not exist yet. Check `crates/fastrag-store/src/lib.rs`. If it doesn't exist, add it (see Task 1 Step 4b below), or use the existing `Store::open` with a `MockEmbedder`.

- [ ] **Step 4b: Check if `Store::open_no_embedder` exists**

```bash
grep -n "open_no_embedder\|fn open" crates/fastrag-store/src/lib.rs | head -10
```

If `open_no_embedder` does not exist, add it to `crates/fastrag-store/src/lib.rs` alongside the existing `open`:

```rust
/// Open a Store for querying with a pre-computed vector (no embedder needed).
/// Used by `batch_query` which embeds all texts in one batch before retrieval.
pub fn open_no_embedder(corpus_dir: &Path) -> Result<Self, StoreError> {
    // Delegate to existing open with a zero-dim mock that won't be called.
    use fastrag_embed::{EmbedError, PassageText, PrefixScheme, QueryText};
    struct NullEmbedder;
    impl fastrag_embed::Embedder for NullEmbedder {
        type Prefix = PrefixScheme;
        fn embed_query(&self, _: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Err(EmbedError::Model("open_no_embedder: embed_query must not be called".into()))
        }
        fn embed_passage(&self, _: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Err(EmbedError::Model("open_no_embedder: embed_passage must not be called".into()))
        }
        fn prefix_scheme(&self) -> &Self::Prefix { &PrefixScheme::None }
        fn dim(&self) -> usize { 0 }
        fn model_id(&self) -> &str { "null" }
    }
    Self::open(corpus_dir, &NullEmbedder)
}
```

Check `fastrag-store`'s `open` signature and error type first:
```bash
grep -n "pub fn open\|StoreError" crates/fastrag-store/src/lib.rs | head -10
```

- [ ] **Step 5: Run tests to verify green**

```bash
cargo test -p fastrag --lib
```
Expected: all pass, including `batch_query_matches_sequential`.

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag/Cargo.toml crates/fastrag/src/corpus/mod.rs crates/fastrag-store/src/lib.rs
git commit -m "feat(corpus): batch_query() with pre-computed embeddings and rayon fan-out"
```

---

## Task 2: `POST /batch-query` HTTP endpoint

**Files:**
- Modify: `fastrag-cli/src/http.rs`
- Modify: `fastrag-cli/src/args.rs`
- Modify: `fastrag-cli/src/main.rs`

- [ ] **Step 1: Write failing integration test**

In `fastrag-cli/tests/` create `batch_query_e2e.rs` (or add to an existing integration test file if one exists for the HTTP server). Check with:
```bash
ls fastrag-cli/tests/
```

Create `fastrag-cli/tests/batch_query_e2e.rs`:

```rust
//! Integration test for POST /batch-query.
//! Spins up a real HTTP server with a toy corpus and exercises partial-failure semantics.

use std::path::PathBuf;
use fastrag_embed::{EmbedError, PassageText, PrefixScheme, QueryText};
use fastrag_cli::http::{serve_http_with_embedder, HttpRerankerConfig};

struct Toy4d;
impl fastrag_embed::Embedder for Toy4d {
    type Prefix = PrefixScheme;
    fn embed_query(&self, texts: &[QueryText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|_| vec![1.0f32, 0.0, 0.0, 0.0]).collect())
    }
    fn embed_passage(&self, texts: &[PassageText]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|_| vec![1.0f32, 0.0, 0.0, 0.0]).collect())
    }
    fn prefix_scheme(&self) -> &Self::Prefix { &PrefixScheme::None }
    fn dim(&self) -> usize { 4 }
    fn model_id(&self) -> &str { "toy-4d" }
}

fn toy_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let e = std::sync::Arc::new(Toy4d) as fastrag_embed::DynEmbedder;
    // Index one document.
    let doc_path = dir.path().join("doc.txt");
    std::fs::write(&doc_path, "SQL injection vulnerability").unwrap();
    fastrag::corpus::index_path_with_metadata(
        &doc_path,
        dir.path(),
        &fastrag_core::ChunkingStrategy::default(),
        e.as_ref() as &dyn fastrag_embed::DynEmbedderTrait,
        None,
    ).unwrap();
    dir
}

#[tokio::test]
async fn batch_query_partial_failure() {
    let dir = toy_corpus();
    let e = std::sync::Arc::new(Toy4d) as fastrag_embed::DynEmbedder;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        serve_http_with_embedder(
            dir.path().to_path_buf(),
            listener,
            e,
            Some("test-token".to_string()),
            false,
            HttpRerankerConfig::default(),
        ).await.unwrap();
    });

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "queries": [
            {"q": "SQL injection", "top_k": 3},
            {"q": "RCE", "top_k": 3, "filter": "INVALID FILTER !!!"},
            {"q": "memory corruption", "top_k": 3}
        ]
    });

    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .header("x-fastrag-token", "test-token")
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);

    // index 0: hits
    assert!(results[0]["hits"].is_array(), "index 0 should have hits");
    assert!(results[0].get("error").is_none(), "index 0 should not have error");

    // index 1: bad filter -> error
    assert!(results[1]["error"].is_string(), "index 1 should have error");
    assert!(results[1].get("hits").is_none(), "index 1 should not have hits");

    // index 2: hits
    assert!(results[2]["hits"].is_array(), "index 2 should have hits");
}

#[tokio::test]
async fn batch_query_rejects_over_limit() {
    let dir = toy_corpus();
    let e = std::sync::Arc::new(Toy4d) as fastrag_embed::DynEmbedder;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        serve_http_with_embedder(
            dir.path().to_path_buf(),
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
        ).await.unwrap();
    });

    // Build a batch of 101 queries (default limit is 100).
    let queries: Vec<serde_json::Value> = (0..101)
        .map(|i| serde_json::json!({"q": format!("query {i}"), "top_k": 1}))
        .collect();
    let body = serde_json::json!({ "queries": queries });

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/batch-query", addr))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
}
```

Add `reqwest` to `fastrag-cli/Cargo.toml` dev-dependencies if not present:
```bash
grep "reqwest" fastrag-cli/Cargo.toml
```
If absent, add to `[dev-dependencies]`:
```toml
reqwest = { version = "0.12", features = ["json"] }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p fastrag-cli --test batch_query_e2e 2>&1 | head -20
```
Expected: FAIL — `batch-query` route not found (404) or compile error.

- [ ] **Step 3: Add request/response types and `--batch-max-queries` to args.rs**

In `fastrag-cli/src/args.rs`, find the `ServeHttp` variant and add before `}`:

```rust
        /// Maximum number of queries in a single /batch-query request.
        #[arg(long, default_value_t = 100)]
        batch_max_queries: usize,
```

- [ ] **Step 4: Thread `batch_max_queries` through serve_http**

In `fastrag-cli/src/http.rs`:

1. Add to `AppState`:
```rust
batch_max_queries: usize,
```

2. Add to `serve_http` and `serve_http_with_embedder` signatures:
```rust
pub async fn serve_http(
    corpus_dir: PathBuf,
    port: u16,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
) -> Result<(), HttpError>
```
Update the `serve_http_with_embedder` signature similarly and thread `batch_max_queries` into `AppState`.

3. Add request/response types (at top of `http.rs`, after existing `use` imports):

```rust
use fastrag::corpus::{BatchQueryParams, BatchQueryResult};

#[derive(Debug, serde::Deserialize)]
struct BatchQueryRequest {
    queries: Vec<BatchQueryItem>,
}

#[derive(Debug, serde::Deserialize)]
struct BatchQueryItem {
    q: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    filter: Option<serde_json::Value>,  // accepts string or JSON AST
    /// Ignored until Phase B (federation).
    #[serde(default)]
    #[allow(dead_code)]
    corpus: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct BatchQueryResponse {
    results: Vec<BatchQueryResultItem>,
}

#[derive(Debug, serde::Serialize)]
struct BatchQueryResultItem {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    hits: Option<Vec<fastrag::corpus::SearchHitDto>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}
```

4. Add the handler function:

```rust
async fn batch_query(
    State(state): State<AppState>,
    Json(req): Json<BatchQueryRequest>,
) -> Result<Json<BatchQueryResponse>, Response> {
    use axum::routing::post;
    use fastrag_embed::QueryText;

    if req.queries.len() > state.batch_max_queries {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "batch exceeds max queries ({}); got {}",
                state.batch_max_queries,
                req.queries.len()
            ),
        ).into_response());
    }

    // Parse all filters up front; track which indices failed.
    let mut filter_errors: Vec<Option<String>> = vec![None; req.queries.len()];
    let mut filter_exprs: Vec<Option<fastrag::filter::FilterExpr>> = Vec::with_capacity(req.queries.len());

    for (i, item) in req.queries.iter().enumerate() {
        match &item.filter {
            None => filter_exprs.push(None),
            Some(serde_json::Value::String(s)) => {
                match fastrag::filter::parse(s) {
                    Ok(f) => filter_exprs.push(Some(f)),
                    Err(e) => {
                        filter_exprs.push(None);
                        filter_errors[i] = Some(format!("bad filter: {e}"));
                    }
                }
            }
            Some(json_ast) => {
                match fastrag::filter::parse_json(json_ast) {
                    Ok(f) => filter_exprs.push(Some(f)),
                    Err(e) => {
                        filter_exprs.push(None);
                        filter_errors[i] = Some(format!("bad filter: {e}"));
                    }
                }
            }
        }
    }

    // Embed all query texts in a single call.
    let texts: Vec<QueryText> = req.queries.iter()
        .map(|item| QueryText::new(&item.q))
        .collect();

    let embeddings = match (state.embedder.as_ref() as &dyn fastrag::DynEmbedderTrait)
        .embed_query_dyn(&texts)
    {
        Ok(vecs) => vecs,
        Err(e) => {
            return Err((StatusCode::SERVICE_UNAVAILABLE, format!("embed error: {e}")).into_response());
        }
    };

    // Build BatchQueryParams for queries that passed filter parsing.
    let params: Vec<BatchQueryParams> = req.queries.iter().zip(filter_exprs.iter()).map(|(item, f)| {
        BatchQueryParams {
            text: item.q.clone(),
            top_k: item.top_k,
            filter: f.clone(),
        }
    }).collect();

    // Run batch retrieval.
    #[cfg(feature = "rerank")]
    let raw_results = fastrag::corpus::batch_query(
        &state.corpus_dir,
        &embeddings,
        &params,
        state.reranker.as_deref(),
    );
    #[cfg(not(feature = "rerank"))]
    let raw_results = fastrag::corpus::batch_query(
        &state.corpus_dir,
        &embeddings,
        &params,
    );

    // Merge filter parse errors with retrieval results.
    let results: Vec<BatchQueryResultItem> = raw_results
        .into_iter()
        .enumerate()
        .map(|(i, result)| {
            if let Some(err) = &filter_errors[i] {
                return BatchQueryResultItem { index: i, hits: None, error: Some(err.clone()) };
            }
            match result {
                Ok(hits) => BatchQueryResultItem { index: i, hits: Some(hits), error: None },
                Err(e) => BatchQueryResultItem { index: i, hits: None, error: Some(e.to_string()) },
            }
        })
        .collect();

    Ok(Json(BatchQueryResponse { results }))
}
```

5. Register the route in `serve_http_with_embedder`:

```rust
let protected = Router::new()
    .route("/query", get(query))
    .route("/batch-query", post(batch_query))   // ← add this line
    .route("/metrics", get(metrics_handler))
    .route_layer(middleware::from_fn_with_state(
        auth_state.clone(),
        auth_middleware,
    ));
```

6. Update `fastrag-cli/src/main.rs` `ServeHttp` arm to pass `batch_max_queries`:

Find the line:
```rust
fastrag_cli::http::serve_http(corpus, port, embedder, token, dense_only, rerank_cfg)
```
Change to:
```rust
fastrag_cli::http::serve_http(corpus, port, embedder, token, dense_only, rerank_cfg, batch_max_queries)
```

Also add `batch_max_queries` to the destructure pattern at the top of the `ServeHttp` arm.

> **Note:** Check whether `fastrag::filter::parse_json` exists. If not, implement it as a new function in `crates/fastrag/src/filter/` that deserializes a `serde_json::Value` into a `FilterExpr`. Check:
> ```bash
> grep -n "pub fn parse" crates/fastrag/src/filter/mod.rs
> ```

- [ ] **Step 5: Run tests**

```bash
cargo test -p fastrag-cli --test batch_query_e2e
```
Expected: both tests pass.

- [ ] **Step 6: Run full test + lint gate**

```bash
cargo test --workspace --features retrieval,rerank
cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings
cargo fmt --check
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add fastrag-cli/src/http.rs fastrag-cli/src/args.rs fastrag-cli/src/main.rs \
        fastrag-cli/tests/batch_query_e2e.rs fastrag-cli/Cargo.toml
git commit -m "feat(http): POST /batch-query with partial-failure semantics

Closes #43"
```

---

## Task 3: `CorpusRegistry` with lazy loading

**Files:**
- Create: `crates/fastrag/src/corpus/registry.rs`
- Modify: `crates/fastrag/src/corpus/mod.rs` (re-export)

- [ ] **Step 1: Write failing tests**

Create `crates/fastrag/src/corpus/registry.rs` with tests only:

```rust
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use fastrag_embed::DynEmbedder;

/// Handle to a named corpus. Load state is managed lazily.
#[derive(Clone)]
pub enum CorpusState {
    Unloaded,
    Loaded,  // In a real impl, would hold Arc<LoadedCorpus>; here just a marker.
}

#[derive(Clone)]
pub struct CorpusHandle {
    pub path: PathBuf,
    pub state: CorpusState,
}

/// Registry of named corpora with lazy loading.
pub struct CorpusRegistry {
    inner: Arc<Mutex<HashMap<String, CorpusHandle>>>,
}

impl CorpusRegistry {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Register a named corpus. Does not load it.
    pub fn register(&self, name: impl Into<String>, path: PathBuf) {
        let mut map = self.inner.lock().unwrap();
        map.insert(name.into(), CorpusHandle { path, state: CorpusState::Unloaded });
    }

    /// Return the path for a named corpus, or None if not registered.
    pub fn corpus_path(&self, name: &str) -> Option<PathBuf> {
        let map = self.inner.lock().unwrap();
        map.get(name).map(|h| h.path.clone())
    }

    /// List all registered corpora as (name, path, loaded) triples.
    pub fn list(&self) -> Vec<(String, PathBuf, bool)> {
        let map = self.inner.lock().unwrap();
        map.iter().map(|(name, h)| {
            let loaded = matches!(h.state, CorpusState::Loaded);
            (name.clone(), h.path.clone(), loaded)
        }).collect()
    }

    /// Parse a `name=path` string. Returns `("default", path)` if no `=` present.
    pub fn parse_corpus_arg(s: &str) -> (String, PathBuf) {
        if let Some(pos) = s.find('=') {
            let name = s[..pos].to_string();
            let path = PathBuf::from(&s[pos + 1..]);
            (name, path)
        } else {
            ("default".to_string(), PathBuf::from(s))
        }
    }
}

impl Clone for CorpusRegistry {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_list() {
        let reg = CorpusRegistry::new();
        reg.register("nvd", PathBuf::from("/data/nvd"));
        reg.register("findings", PathBuf::from("/data/findings"));

        let list = reg.list();
        assert_eq!(list.len(), 2);
        let names: Vec<&str> = list.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"nvd"));
        assert!(names.contains(&"findings"));
        // Both start unloaded.
        assert!(list.iter().all(|(_, _, loaded)| !loaded));
    }

    #[test]
    fn corpus_path_returns_none_for_unknown() {
        let reg = CorpusRegistry::new();
        assert!(reg.corpus_path("unknown").is_none());
    }

    #[test]
    fn parse_corpus_arg_with_equals() {
        let (name, path) = CorpusRegistry::parse_corpus_arg("nvd=/data/nvd");
        assert_eq!(name, "nvd");
        assert_eq!(path, PathBuf::from("/data/nvd"));
    }

    #[test]
    fn parse_corpus_arg_without_equals_is_default() {
        let (name, path) = CorpusRegistry::parse_corpus_arg("./corpus");
        assert_eq!(name, "default");
        assert_eq!(path, PathBuf::from("./corpus"));
    }

    #[test]
    fn parse_corpus_arg_path_with_equals_in_path() {
        // Only the first `=` is the separator.
        let (name, path) = CorpusRegistry::parse_corpus_arg("nvd=/data/nvd=2024");
        assert_eq!(name, "nvd");
        assert_eq!(path, PathBuf::from("/data/nvd=2024"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p fastrag --lib corpus::registry
```
Expected: compile error (module not yet included).

- [ ] **Step 3: Add module declaration**

In `crates/fastrag/src/corpus/mod.rs`, at the top add:
```rust
pub mod registry;
pub use registry::CorpusRegistry;
```

- [ ] **Step 4: Run tests green**

```bash
cargo test -p fastrag --lib corpus::registry
```
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/registry.rs crates/fastrag/src/corpus/mod.rs
git commit -m "feat(corpus): CorpusRegistry with lazy-load handles and parse_corpus_arg"
```

---

## Task 4: Federation — `AppState` refactor + CLI changes

**Files:**
- Modify: `fastrag-cli/src/http.rs`
- Modify: `fastrag-cli/src/args.rs`
- Modify: `fastrag-cli/src/main.rs`

- [ ] **Step 1: Write failing test**

Add to `fastrag-cli/tests/batch_query_e2e.rs` (or create `fastrag-cli/tests/federation_e2e.rs`):

```rust
#[tokio::test]
async fn named_corpus_query_routes_correctly() {
    // Start server with a named corpus "docs".
    let dir = toy_corpus(); // reuse helper from batch_query_e2e.rs or inline it
    let e = std::sync::Arc::new(Toy4d) as fastrag_embed::DynEmbedder;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = fastrag::corpus::CorpusRegistry::new();
    registry.register("docs", dir.path().to_path_buf());

    tokio::spawn(async move {
        fastrag_cli::http::serve_http_with_registry(
            registry,
            listener,
            e,
            None, // no token
            false,
            HttpRerankerConfig::default(),
            100,  // batch_max_queries
            None, // tenant_field
        ).await.unwrap();
    });

    let client = reqwest::Client::new();

    // Query named corpus.
    let resp = client
        .get(format!("http://{}/query?q=SQL&corpus=docs&top_k=3", addr))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);

    // Query unknown corpus -> 404.
    let resp = client
        .get(format!("http://{}/query?q=SQL&corpus=unknown&top_k=3", addr))
        .send().await.unwrap();
    assert_eq!(resp.status(), 404);

    // GET /corpora lists registered corpora.
    let resp = client
        .get(format!("http://{}/corpora", addr))
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let corpora = json["corpora"].as_array().unwrap();
    assert_eq!(corpora.len(), 1);
    assert_eq!(corpora[0]["name"].as_str().unwrap(), "docs");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p fastrag-cli --test federation_e2e 2>&1 | head -20
```
Expected: compile error — `serve_http_with_registry` not defined.

- [ ] **Step 3: Refactor `AppState` to hold `CorpusRegistry`**

In `fastrag-cli/src/http.rs`:

Replace:
```rust
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
```

With:
```rust
#[derive(Clone)]
struct AppState {
    registry: fastrag::corpus::CorpusRegistry,
    embedder: DynEmbedder,
    metrics: PrometheusHandle,
    dense_only: bool,
    batch_max_queries: usize,
    tenant_field: Option<String>,
    #[cfg(feature = "rerank")]
    reranker: Option<std::sync::Arc<dyn fastrag_rerank::Reranker>>,
    #[cfg(feature = "rerank")]
    rerank_over_fetch: usize,
}
```

Update `AppState` construction in `serve_http_with_embedder` to build a `CorpusRegistry` with a single "default" entry:

```rust
let registry = fastrag::corpus::CorpusRegistry::new();
registry.register("default", corpus_dir.clone());
let app_state = AppState {
    registry,
    embedder,
    metrics,
    dense_only,
    batch_max_queries,
    tenant_field: None,
    #[cfg(feature = "rerank")]
    reranker: rerank_cfg.reranker,
    #[cfg(feature = "rerank")]
    rerank_over_fetch: rerank_cfg.over_fetch,
};
```

Update `run_query` to resolve `corpus_dir` from the registry. Change signature to accept the corpus name:

```rust
fn run_query(
    state: &AppState,
    params: &QueryParams,
    filter: Option<&fastrag::filter::FilterExpr>,
    corpus_name: &str,
) -> Result<Vec<fastrag::corpus::SearchHitDto>, fastrag::corpus::CorpusError> {
    let corpus_dir = state.registry.corpus_path(corpus_name)
        .ok_or_else(|| fastrag::corpus::CorpusError::NotFound(corpus_name.to_string()))?;
    // ... rest of existing run_query body, replacing `state.corpus_dir` with `corpus_dir`
}
```

Add `NotFound(String)` variant to `CorpusError` if it doesn't exist:
```bash
grep -n "pub enum CorpusError" crates/fastrag/src/corpus/mod.rs
```
If `NotFound` is absent, add it.

Update `QueryParams` to add optional corpus:
```rust
#[serde(default)]
corpus: Option<String>,
```

Update the `query` handler to extract corpus name and pass 404 on unknown:
```rust
let corpus_name = params.corpus.as_deref().unwrap_or("default");
match run_query(&state, &params, filter_expr.as_ref(), corpus_name) {
    Ok(hits) => { ... }
    Err(fastrag::corpus::CorpusError::NotFound(_)) => {
        Err((StatusCode::NOT_FOUND, "corpus not found").into_response())
    }
    Err(err) => { ... }
}
```

- [ ] **Step 4: Add `serve_http_with_registry`**

Add a new public function in `http.rs`:

```rust
pub async fn serve_http_with_registry(
    registry: fastrag::corpus::CorpusRegistry,
    listener: tokio::net::TcpListener,
    embedder: DynEmbedder,
    token: Option<String>,
    dense_only: bool,
    rerank_cfg: HttpRerankerConfig,
    batch_max_queries: usize,
    tenant_field: Option<String>,
) -> Result<(), HttpError> {
    // ... same metrics setup as serve_http_with_embedder ...
    let app_state = AppState {
        registry,
        embedder,
        metrics,
        dense_only,
        batch_max_queries,
        tenant_field,
        #[cfg(feature = "rerank")]
        reranker: rerank_cfg.reranker,
        #[cfg(feature = "rerank")]
        rerank_over_fetch: rerank_cfg.over_fetch,
    };
    // ... router setup + axum::serve ...
}
```

- [ ] **Step 5: Add `GET /corpora` endpoint**

```rust
async fn list_corpora(State(state): State<AppState>) -> impl IntoResponse {
    let entries: Vec<serde_json::Value> = state.registry.list()
        .into_iter()
        .map(|(name, path, loaded)| serde_json::json!({
            "name": name,
            "path": path,
            "status": if loaded { "loaded" } else { "unloaded" }
        }))
        .collect();
    Json(serde_json::json!({ "corpora": entries }))
}
```

Register it on the protected router:
```rust
.route("/corpora", get(list_corpora))
```

- [ ] **Step 6: Update `--corpus` arg in args.rs**

In `ServeHttp`, replace:
```rust
#[arg(long)]
corpus: PathBuf,
```
With:
```rust
/// Corpus to register. Format: `name=path` or just `path` (registers as "default").
/// May be repeated to register multiple corpora.
#[arg(long, required = true)]
corpus: Vec<String>,
```

Add:
```rust
/// Enforce tenant scoping. When set, every request must include
/// `X-Fastrag-Tenant: <value>`; the server injects a mandatory filter
/// `<field> = <value>` on all queries.
#[arg(long)]
tenant_field: Option<String>,
```

- [ ] **Step 7: Update main.rs `ServeHttp` arm**

Replace the single `corpus: PathBuf` destructure with `corpus: Vec<String>` and build the registry:

```rust
Command::ServeHttp {
    corpus,
    port,
    // ... other fields ...
    batch_max_queries,
    tenant_field,
} => {
    // Build registry from --corpus args.
    let registry = fastrag::corpus::CorpusRegistry::new();
    for arg in &corpus {
        let (name, path) = fastrag::corpus::CorpusRegistry::parse_corpus_arg(arg);
        registry.register(name, path);
    }

    // For load_for_read, use the first corpus path.
    let first_path = corpus.first()
        .map(|s| fastrag::corpus::CorpusRegistry::parse_corpus_arg(s).1)
        .unwrap_or_default();
    let embedder = embed_loader::load_for_read(&first_path, &opts).unwrap_or_else(|e| {
        eprintln!("Error loading embedder: {e}");
        std::process::exit(1);
    });

    // ... rerank setup unchanged ...

    if let Err(e) = fastrag_cli::http::serve_http_with_registry(
        registry, port_listener, embedder, token, dense_only, rerank_cfg,
        batch_max_queries, tenant_field,
    ).await {
        eprintln!("Error starting HTTP server: {e}");
        std::process::exit(1);
    }
}
```

> Note: `serve_http` (the existing convenience function that takes a single `PathBuf`) can be kept as a thin wrapper calling `serve_http_with_registry` for back-compat in tests.

- [ ] **Step 8: Run tests**

```bash
cargo test -p fastrag-cli --test federation_e2e
cargo test -p fastrag-cli --test batch_query_e2e
```
Expected: all pass.

- [ ] **Step 9: Run full gate**

```bash
cargo test --workspace --features retrieval,rerank
cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings
cargo fmt --check
```
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add fastrag-cli/src/http.rs fastrag-cli/src/args.rs fastrag-cli/src/main.rs \
        fastrag-cli/tests/federation_e2e.rs
git commit -m "feat(http): multi-corpus registry, GET /corpora, --corpus name=path

Closes #46"
```

---

## Task 5: Tenant enforcement middleware

**Files:**
- Modify: `fastrag-cli/src/http.rs`

- [ ] **Step 1: Write failing test**

Add to `fastrag-cli/tests/federation_e2e.rs`:

```rust
#[tokio::test]
async fn tenant_enforcement_rejects_missing_header() {
    let dir = toy_corpus();
    let e = std::sync::Arc::new(Toy4d) as fastrag_embed::DynEmbedder;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let registry = fastrag::corpus::CorpusRegistry::new();
    registry.register("default", dir.path().to_path_buf());

    tokio::spawn(async move {
        fastrag_cli::http::serve_http_with_registry(
            registry,
            listener,
            e,
            None,
            false,
            HttpRerankerConfig::default(),
            100,
            Some("engagement_id".to_string()),  // tenant enforcement ON
        ).await.unwrap();
    });

    let client = reqwest::Client::new();

    // No tenant header -> 401.
    let resp = client
        .get(format!("http://{}/query?q=SQL&top_k=1", addr))
        .send().await.unwrap();
    assert_eq!(resp.status(), 401);

    // With tenant header -> 200.
    let resp = client
        .get(format!("http://{}/query?q=SQL&top_k=1", addr))
        .header("x-fastrag-tenant", "engagement-abc")
        .send().await.unwrap();
    assert_eq!(resp.status(), 200);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p fastrag-cli --test federation_e2e tenant_enforcement
```
Expected: FAIL — missing header still returns 200 (enforcement not implemented).

- [ ] **Step 3: Implement tenant middleware**

In `fastrag-cli/src/http.rs`, add:

```rust
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

    // Inject the tenant filter into request extensions so handlers can read it.
    req.extensions_mut().insert(TenantFilter {
        field: field.clone(),
        value: tenant,
    });

    Ok(next.run(req).await)
}

/// Tenant filter injected by middleware into request extensions.
#[derive(Clone)]
pub struct TenantFilter {
    pub field: String,
    pub value: String,
}
```

Register the middleware on the protected router in `serve_http_with_registry`:
```rust
let protected = Router::new()
    .route("/query", get(query))
    .route("/batch-query", post(batch_query))
    .route("/corpora", get(list_corpora))
    .route("/metrics", get(metrics_handler))
    .route_layer(middleware::from_fn_with_state(state.clone(), tenant_middleware))
    .route_layer(middleware::from_fn_with_state(auth_state.clone(), auth_middleware));
```

Update `query` handler to extract `TenantFilter` from extensions and merge into the filter expression:

```rust
async fn query(
    State(state): State<AppState>,
    Extension(tenant): Option<Extension<TenantFilter>>,  // None when tenant_field not set
    Query(params): Query<QueryParams>,
) -> Result<Json<Vec<fastrag::corpus::SearchHitDto>>, Response> {
    // ... existing filter parse ...

    // Merge tenant filter if present.
    let filter_expr: Option<fastrag::filter::FilterExpr> = match (filter_expr, tenant) {
        (None, None) => None,
        (Some(f), None) => Some(f),
        (None, Some(Extension(t))) => {
            Some(fastrag::filter::FilterExpr::Eq {
                field: t.field,
                value: t.value,
            })
        }
        (Some(f), Some(Extension(t))) => {
            Some(fastrag::filter::FilterExpr::And(
                Box::new(f),
                Box::new(fastrag::filter::FilterExpr::Eq {
                    field: t.field,
                    value: t.value,
                }),
            ))
        }
    };
    // ... continue with run_query ...
}
```

> Verify the `FilterExpr` variant names by checking:
> ```bash
> grep -n "pub enum FilterExpr\|Eq\|And" crates/fastrag/src/filter/mod.rs | head -20
> ```
> Use the actual variant names from the codebase.

- [ ] **Step 4: Run tests**

```bash
cargo test -p fastrag-cli --test federation_e2e
```
Expected: all 3 federation tests pass.

- [ ] **Step 5: Run full gate**

```bash
cargo test --workspace --features retrieval,rerank
cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings
cargo fmt --check
```

- [ ] **Step 6: Commit**

```bash
git add fastrag-cli/src/http.rs fastrag-cli/tests/federation_e2e.rs
git commit -m "feat(http): tenant enforcement middleware (X-Fastrag-Tenant header)"
```

---

## Task 6: Push and CI

- [ ] **Step 1: Final local gate**

```bash
cargo test --workspace --features retrieval,rerank,contextual,eval,nvd,hygiene
cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings
cargo fmt --check
```
Expected: all green.

- [ ] **Step 2: Push**

```bash
git push
```

- [ ] **Step 3: Watch CI**

Use `ci-watcher` skill (background Haiku agent). Wait for all jobs to pass before declaring done.
