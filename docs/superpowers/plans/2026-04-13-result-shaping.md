# Result Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add snippet generation, source JSON exposure, and field selection to query responses (#49), reducing payload for LLM consumers.

**Architecture:** Three layers: (1) `TantivyStore::generate_snippets()` wraps Tantivy's SnippetGenerator for BM25 highlighting, (2) `SearchHitDto` gains `snippet` + `source` fields threaded from Store, (3) HTTP handler applies field selection as a post-filter on serialized JSON. All additions are backward compatible — new fields are optional, field selection is opt-in.

**Tech Stack:** Rust, tantivy 0.22 (`tantivy::snippet::SnippetGenerator`), axum, serde_json

---

## File Map

| File | Change |
|------|--------|
| `crates/fastrag-store/src/tantivy.rs` | Add `generate_snippets()` method |
| `crates/fastrag-store/src/lib.rs` | Add `Store::generate_snippets()` delegation |
| `crates/fastrag/src/corpus/mod.rs` | Add `snippet` + `source` to `SearchHitDto`; update `scored_ids_to_dtos`; remove `deny_unknown_fields` on SearchHitDto |
| `fastrag-cli/src/http.rs` | Add `snippet_len` + `fields` query params; `FieldSelection` type; `apply_field_selection` filter; wire snippets into query + batch-query handlers |
| `fastrag-cli/tests/snippet_e2e.rs` | New: 5 integration tests |

---

### Task 1: Add `snippet` and `source` to SearchHitDto

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs`

This task adds the two new fields and threads `source` from `SearchHit` into the DTO. No snippet generation yet — that comes in Task 2.

- [ ] **Step 1: Add fields to `SearchHitDto`**

In `crates/fastrag/src/corpus/mod.rs`, modify the `SearchHitDto` struct (line 355-369). Remove `#[serde(deny_unknown_fields)]` (it prevents adding optional fields without breaking deserialization of old data). Add `snippet` and `source`:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHitDto {
    pub score: f32,
    pub chunk_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
    pub source_path: PathBuf,
    pub chunk_index: usize,
    pub section: Option<String>,
    pub pages: Vec<usize>,
    pub element_kinds: Vec<ElementKind>,
    pub language: Option<String>,
    #[cfg(feature = "store")]
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub metadata: std::collections::BTreeMap<String, fastrag_store::schema::TypedValue>,
}
```

- [ ] **Step 2: Update `From<VectorHit>` impl**

Update the `impl From<VectorHit> for SearchHitDto` (line 371-386) to include the new fields:

```rust
impl From<VectorHit> for SearchHitDto {
    fn from(value: VectorHit) -> Self {
        Self {
            score: value.score,
            chunk_text: String::new(),
            snippet: None,
            source: None,
            source_path: PathBuf::new(),
            chunk_index: 0,
            section: None,
            pages: Vec::new(),
            element_kinds: Vec::new(),
            language: None,
            #[cfg(feature = "store")]
            metadata: std::collections::BTreeMap::new(),
        }
    }
}
```

- [ ] **Step 3: Update `scored_ids_to_dtos` to thread `source`**

In `scored_ids_to_dtos` (line 999-1034), thread `hit.source` through to the DTO. The `SearchHit` struct (in `crates/fastrag-store/src/lib.rs:53-59`) already has `pub source: Option<serde_json::Value>`. Currently the loop ignores it:

```rust
fn scored_ids_to_dtos(
    store: &fastrag_store::Store,
    scored: &[(u64, f32)],
) -> Result<Vec<SearchHitDto>, CorpusError> {
    if scored.is_empty() {
        return Ok(vec![]);
    }

    let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
    let metadata_rows = store.fetch_metadata(&ids)?;
    let meta_map: std::collections::HashMap<u64, Vec<(String, fastrag_store::schema::TypedValue)>> =
        metadata_rows.into_iter().collect();

    let hits = store.fetch_hits(scored)?;
    let mut dtos = Vec::with_capacity(scored.len());
    for hit in &hits {
        for chunk in &hit.chunks {
            let metadata_fields = meta_map.get(&chunk.id).cloned().unwrap_or_default();
            dtos.push(SearchHitDto {
                score: chunk.score,
                chunk_text: chunk.chunk_text.clone(),
                snippet: None,
                source: hit.source.clone(),
                source_path: PathBuf::from(&hit.external_id),
                chunk_index: chunk.chunk_index,
                section: None,
                pages: vec![],
                element_kinds: vec![],
                language: None,
                #[cfg(feature = "store")]
                metadata: metadata_fields.into_iter().collect(),
            });
        }
    }
    Ok(dtos)
}
```

The key change: `source: hit.source.clone()` instead of not setting it.

- [ ] **Step 4: Verify compilation and existing tests pass**

Run: `cargo test --workspace --features retrieval,rerank`
Expected: all pass (new fields are `None`/`Option`, backward compatible)

- [ ] **Step 5: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs
git commit -m "feat(corpus): add snippet + source fields to SearchHitDto"
```

---

### Task 2: Snippet generation in TantivyStore

**Files:**
- Modify: `crates/fastrag-store/src/tantivy.rs`
- Modify: `crates/fastrag-store/src/lib.rs`

- [ ] **Step 1: Write failing unit test**

In the `#[cfg(test)] mod tests` block of `crates/fastrag-store/src/tantivy.rs`, add:

```rust
#[test]
fn generate_snippets_highlights_matching_terms() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);
    let core = store.core();

    let mut writer = store.writer().unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_u64(core.id, 1);
    doc.add_text(core.external_id, "doc-1");
    doc.add_text(core.content_hash, "h1");
    doc.add_u64(core.chunk_index, 0);
    doc.add_text(core.source_path, "/t.txt");
    doc.add_text(core.source, "{}");
    doc.add_text(
        core.chunk_text,
        "The SQL injection vulnerability allows remote code execution on the server",
    );
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();
    store.reload().unwrap();

    let snippets = store.generate_snippets("SQL injection", &[1], 150);
    assert_eq!(snippets.len(), 1);
    let snippet = snippets[0].as_ref().expect("snippet should be Some");
    assert!(
        snippet.contains("<b>"),
        "snippet should contain highlight tags: {snippet}"
    );
}

#[test]
fn generate_snippets_returns_none_for_missing_doc() {
    let dir = TempDir::new().unwrap();
    let store = make_store(&dir);
    let snippets = store.generate_snippets("test", &[999], 150);
    assert_eq!(snippets.len(), 1);
    assert!(snippets[0].is_none(), "missing doc should produce None");
}
```

- [ ] **Step 2: Run tests — expect compile failure**

Run: `cargo test -p fastrag-store -- generate_snippets`
Expected: compile error — `generate_snippets` not found

- [ ] **Step 3: Implement `TantivyStore::generate_snippets()`**

In `crates/fastrag-store/src/tantivy.rs`, add to `impl TantivyStore` after `field_stats()`:

```rust
/// Generate highlighted snippets for the given internal IDs using Tantivy's
/// SnippetGenerator. Returns one `Option<String>` per ID in input order.
/// `None` if the document is not found or snippet generation fails.
pub fn generate_snippets(
    &self,
    query_text: &str,
    ids: &[u64],
    max_chars: usize,
) -> Vec<Option<String>> {
    use tantivy::query::QueryParser;
    use tantivy::snippet::SnippetGenerator;

    let searcher = self.reader.searcher();
    let parser = QueryParser::for_index(&searcher.index(), vec![self.core.chunk_text]);
    let query = match parser.parse_query(query_text) {
        Ok(q) => q,
        Err(_) => return vec![None; ids.len()],
    };

    let mut generator = match SnippetGenerator::create(&searcher, &*query, self.core.chunk_text)
    {
        Ok(g) => g,
        Err(_) => return vec![None; ids.len()],
    };
    generator.set_max_num_chars(max_chars);

    let mut results = Vec::with_capacity(ids.len());
    for &id_val in ids {
        let term = tantivy::Term::from_field_u64(self.core.id, id_val);
        let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let top = match searcher.search(&query, &tantivy::collector::TopDocs::with_limit(1)) {
            Ok(t) => t,
            Err(_) => {
                results.push(None);
                continue;
            }
        };
        if let Some((_score, addr)) = top.first() {
            match searcher.doc::<TantivyDocument>(*addr) {
                Ok(doc) => {
                    let snippet = generator.snippet_from_doc(&doc);
                    let html = snippet.to_html();
                    if html.is_empty() {
                        results.push(None);
                    } else {
                        results.push(Some(html));
                    }
                }
                Err(_) => results.push(None),
            }
        } else {
            results.push(None);
        }
    }
    results
}
```

- [ ] **Step 4: Add `Store::generate_snippets()` delegation in `lib.rs`**

In `crates/fastrag-store/src/lib.rs`, after `field_stats()`:

```rust
/// Generate highlighted snippets for the given internal IDs.
pub fn generate_snippets(
    &self,
    query_text: &str,
    ids: &[u64],
    max_chars: usize,
) -> Vec<Option<String>> {
    self.tantivy.generate_snippets(query_text, ids, max_chars)
}
```

- [ ] **Step 5: Run tests — expect green**

Run: `cargo test -p fastrag-store -- generate_snippets`
Expected: both tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/fastrag-store/src/tantivy.rs crates/fastrag-store/src/lib.rs
git commit -m "feat(store): add generate_snippets() using Tantivy SnippetGenerator"
```

---

### Task 3: Wire snippets into corpus query path

**Files:**
- Modify: `crates/fastrag/src/corpus/mod.rs`

Update `scored_ids_to_dtos` to accept an optional query text and snippet length, and populate `snippet` on each DTO.

- [ ] **Step 1: Change `scored_ids_to_dtos` signature**

Change the function signature to accept snippet parameters:

```rust
fn scored_ids_to_dtos(
    store: &fastrag_store::Store,
    scored: &[(u64, f32)],
    snippet_query: Option<&str>,
    snippet_len: usize,
) -> Result<Vec<SearchHitDto>, CorpusError> {
```

- [ ] **Step 2: Add snippet generation to the function body**

After the `fetch_hits` call, generate snippets if requested:

```rust
    let snippet_map: std::collections::HashMap<u64, String> = match snippet_query {
        Some(query_text) if snippet_len > 0 => {
            let all_ids: Vec<u64> = hits
                .iter()
                .flat_map(|h| h.chunks.iter().map(|c| c.id))
                .collect();
            let snippets = store.generate_snippets(query_text, &all_ids, snippet_len);
            all_ids
                .into_iter()
                .zip(snippets)
                .filter_map(|(id, s)| s.map(|text| (id, text)))
                .collect()
        }
        _ => std::collections::HashMap::new(),
    };
```

Then in the DTO construction loop, look up the snippet:

```rust
            dtos.push(SearchHitDto {
                score: chunk.score,
                chunk_text: chunk.chunk_text.clone(),
                snippet: snippet_map.get(&chunk.id).cloned(),
                source: hit.source.clone(),
                // ... rest unchanged
            });
```

- [ ] **Step 3: Update all callers of `scored_ids_to_dtos`**

Search for all call sites in `corpus/mod.rs`. Each needs the two new params. For the query functions that have `query_text` available, pass `Some(query_text)`. Use `150` as the default snippet_len for now (the HTTP handler will override this).

Actually, to keep the corpus layer clean, add `snippet_query` and `snippet_len` as parameters that callers thread through. The public query functions (`query_corpus_with_filter`, `batch_query`, `query_corpus_reranked`) gain these params too. This is a wide signature change — update all call sites.

Simpler approach: make `scored_ids_to_dtos` accept the new params, and update its callers in the same file. The public functions like `query_corpus_with_filter` already receive `query_text` as `&str`. Thread it through.

For each call site of `scored_ids_to_dtos` in the file, change from:
```rust
scored_ids_to_dtos(store, &scored)
// or
scored_ids_to_dtos(&store, &scored)
```
to:
```rust
scored_ids_to_dtos(store, &scored, Some(query_text), snippet_len)
// or
scored_ids_to_dtos(&store, &scored, Some(query_text), snippet_len)
```

Where `query_text` is the `q` parameter already available in each function, and `snippet_len` is a new parameter threaded through the public functions.

Add `snippet_len: usize` parameter to:
- `query_corpus_with_filter` (line 895)
- `query_corpus` (line 867) 
- `query_corpus_reranked` (if it exists under rerank feature)
- `batch_query` (line 398)

For `query_corpus` and `query_corpus_with_filter`, add `snippet_len: usize` as the last parameter. Default callers in http.rs and tests will pass a value.

- [ ] **Step 4: Update batch_query similarly**

The `batch_query` function uses `scored_ids_to_dtos` or similar. Thread `snippet_len` through it as well. The `BatchQueryParams` struct gains `snippet_len: usize`.

- [ ] **Step 5: For dense-only queries without BM25 terms, provide truncated preview**

In `scored_ids_to_dtos`, when `snippet_map` is empty for a chunk (no BM25 highlight produced), fall back to a truncated preview:

```rust
snippet: snippet_map.get(&chunk.id).cloned().or_else(|| {
    if snippet_len > 0 && snippet_query.is_some() {
        // Dense-only fallback: truncate chunk_text at word boundary
        let text = &chunk.chunk_text;
        if text.len() <= snippet_len {
            Some(text.clone())
        } else {
            let truncated = &text[..snippet_len];
            // Find last space for word boundary
            let end = truncated.rfind(' ').unwrap_or(snippet_len);
            Some(format!("{}...", &text[..end]))
        }
    } else {
        None
    }
}),
```

- [ ] **Step 6: Fix compilation errors from all callers**

Run: `cargo check --workspace --features retrieval,rerank`

Fix any callers in the workspace that now have the wrong number of arguments. The main callers are in `http.rs` (updated in Task 4) and tests. For now, update callers in `corpus/mod.rs` and `ops.rs` to thread the new param. External callers (http.rs) will be fixed in Task 4.

- [ ] **Step 7: Verify tests pass**

Run: `cargo test --workspace --features retrieval,rerank`

- [ ] **Step 8: Commit**

```bash
git add crates/fastrag/src/corpus/mod.rs
git commit -m "feat(corpus): wire snippet generation into query path"
```

---

### Task 4: HTTP handler — snippet_len, fields, field selection

**Files:**
- Modify: `fastrag-cli/src/http.rs`

- [ ] **Step 1: Add `snippet_len` and `fields` to QueryParams**

In `QueryParams` (line 136-156), add:

```rust
    /// Max characters for snippet. 0 disables. Default 150.
    #[serde(default = "default_snippet_len")]
    snippet_len: usize,
    /// Comma-separated field projection (e.g. "score,snippet" or "-chunk_text").
    #[serde(default)]
    fields: Option<String>,
```

Add the default function:

```rust
fn default_snippet_len() -> usize {
    150
}
```

- [ ] **Step 2: Add `snippet_len` and `fields` to BatchQueryItem**

In `BatchQueryItem` (line 167-178), add:

```rust
    #[serde(default = "default_snippet_len")]
    snippet_len: usize,
    #[serde(default)]
    fields: Option<String>,
```

- [ ] **Step 3: Add `FieldSelection` type and parsing**

After the query param structs, add:

```rust
#[derive(Debug, Clone)]
enum FieldSelection {
    All,
    Include(Vec<String>),
    Exclude(Vec<String>),
}

fn parse_field_selection(fields: Option<&str>) -> Result<FieldSelection, String> {
    let raw = match fields {
        None | Some("") => return Ok(FieldSelection::All),
        Some(s) => s,
    };
    let parts: Vec<&str> = raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
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
            parts.iter().map(|p| p.trim_start_matches('-').to_string()).collect(),
        ))
    } else {
        Ok(FieldSelection::Include(parts.iter().map(|p| p.to_string()).collect()))
    }
}
```

- [ ] **Step 4: Add `apply_field_selection` function**

```rust
fn apply_field_selection(hits: &mut Vec<serde_json::Value>, selection: &FieldSelection) {
    match selection {
        FieldSelection::All => {}
        FieldSelection::Include(fields) => {
            // Separate top-level includes from dotted paths (e.g. "source.severity")
            let mut top_level: Vec<&str> = Vec::new();
            let mut source_sub: Vec<&str> = Vec::new();
            for f in fields {
                if let Some(sub) = f.strip_prefix("source.") {
                    source_sub.push(sub);
                } else {
                    top_level.push(f);
                }
            }
            // If any source.X paths exist, implicitly include "source"
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
                    // Prune source sub-keys if specified
                    if !source_sub.is_empty() {
                        if let Some(source) = obj.get_mut("source").and_then(|v| v.as_object_mut()) {
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
        }
        FieldSelection::Exclude(fields) => {
            for hit in hits.iter_mut() {
                if let Some(obj) = hit.as_object_mut() {
                    for f in fields {
                        if let Some(sub) = f.strip_prefix("source.") {
                            if let Some(source) = obj.get_mut("source").and_then(|v| v.as_object_mut()) {
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
```

- [ ] **Step 5: Update `run_query` to pass `snippet_len`**

Modify `run_query` (line 739) to accept and pass `snippet_len`:

```rust
fn run_query(
    state: &AppState,
    params: &QueryParams,
    filter: Option<&fastrag::filter::FilterExpr>,
    corpus_name: &str,
) -> Result<Vec<SearchHitDto>, CorpusError> {
```

Thread `params.snippet_len` through to `ops::query_corpus_with_filter` and `ops::query_corpus_reranked`. Both functions gained `snippet_len: usize` in Task 3.

- [ ] **Step 6: Update `query` handler to apply field selection**

In the `query` handler (line 938), parse fields and apply selection:

```rust
    // Parse field selection early so we can return 400 before querying
    let field_sel = match parse_field_selection(params.fields.as_deref()) {
        Ok(sel) => sel,
        Err(e) => return Err((StatusCode::BAD_REQUEST, e).into_response()),
    };
```

Then in the `Ok(hits)` arm, serialize to `serde_json::Value`, apply selection, return:

```rust
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
```

Note: the return type changes from `Json<Vec<SearchHitDto>>` to `Json<Vec<serde_json::Value>>` to support dynamic field filtering.

- [ ] **Step 7: Update `batch_query_handler` similarly**

Thread `snippet_len` from each `BatchQueryItem` into the batch query. Apply field selection per-result.

- [ ] **Step 8: Verify compilation**

Run: `cargo check -p fastrag-cli`

- [ ] **Step 9: Commit**

```bash
git add fastrag-cli/src/http.rs
git commit -m "feat(http): add snippet_len + fields query params with field selection"
```

---

### Task 5: Integration tests

**Files:**
- Create: `fastrag-cli/tests/snippet_e2e.rs`

- [ ] **Step 1: Create test file with 5 tests**

```rust
//! Integration tests for snippet generation and field selection.

use std::sync::Arc;

use fastrag::corpus::CorpusRegistry;
use fastrag_cli::http::{HttpRerankerConfig, serve_http_with_registry};
use fastrag_embed::test_utils::MockEmbedder;

async fn spawn_server(registry: CorpusRegistry) -> std::net::SocketAddr {
    let embedder: fastrag::DynEmbedder = Arc::new(MockEmbedder);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve_http_with_registry(
            registry, listener, embedder, None, false,
            HttpRerankerConfig::default(), 100, None, 52_428_800,
        )
        .await
        .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    addr
}

async fn ingest_test_records(addr: std::net::SocketAddr) {
    let client = reqwest::Client::new();
    let body = concat!(
        r#"{"id":"v1","body":"SQL injection vulnerability allows remote code execution","severity":"HIGH","cvss":9.8}"#,
        "\n",
        r#"{"id":"v2","body":"Buffer overflow in kernel network stack","severity":"CRITICAL","cvss":9.1}"#,
        "\n",
    );
    let resp = client
        .post(format!(
            "http://{}/ingest?id_field=id&text_fields=body&metadata_fields=severity,cvss&metadata_types=cvss=numeric",
            addr
        ))
        .header("content-type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn source_exposed_in_response() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/query?q=SQL+injection&top_k=3", addr))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty(), "expected hits");
    // source should contain the original JSON record
    let source = &hits[0]["source"];
    assert!(source.is_object(), "source should be an object: {source}");
    assert!(source["id"].is_string(), "source should have id field");
}

#[tokio::test]
async fn snippet_present_in_response() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&snippet_len=200",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    // snippet should be present (either BM25 highlighted or dense truncated)
    let snippet = &hits[0]["snippet"];
    assert!(
        snippet.is_string(),
        "snippet should be a string: {snippet}"
    );
    assert!(
        !snippet.as_str().unwrap().is_empty(),
        "snippet should not be empty"
    );
}

#[tokio::test]
async fn snippet_disabled_when_zero() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&snippet_len=0",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    // snippet should be absent (None → not serialized)
    assert!(
        hits[0].get("snippet").is_none(),
        "snippet should be absent when snippet_len=0"
    );
}

#[tokio::test]
async fn field_selection_include() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&fields=score,snippet",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    let obj = hits[0].as_object().unwrap();
    assert!(obj.contains_key("score"), "score should be present");
    assert!(
        !obj.contains_key("chunk_text"),
        "chunk_text should be excluded by field selection"
    );
    assert!(
        !obj.contains_key("source_path"),
        "source_path should be excluded"
    );
}

#[tokio::test]
async fn field_selection_exclude() {
    let corpus_dir = tempfile::tempdir().unwrap();
    let registry = CorpusRegistry::new();
    registry.register("default", corpus_dir.path().to_path_buf());
    let addr = spawn_server(registry).await;
    ingest_test_records(addr).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!(
            "http://{}/query?q=SQL+injection&top_k=3&fields=-chunk_text,-source",
            addr
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let hits: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert!(!hits.is_empty());
    let obj = hits[0].as_object().unwrap();
    assert!(
        !obj.contains_key("chunk_text"),
        "chunk_text should be excluded"
    );
    assert!(!obj.contains_key("source"), "source should be excluded");
    assert!(obj.contains_key("score"), "score should remain");
}
```

- [ ] **Step 2: Run integration tests**

Run: `cargo test -p fastrag-cli --test snippet_e2e`
Expected: all 5 pass

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace --features retrieval,rerank`

- [ ] **Step 4: Commit**

```bash
git add fastrag-cli/tests/snippet_e2e.rs
git commit -m "test(http): add snippet and field selection integration tests"
```

---

### Task 6: Local gate, push, CI

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings`

- [ ] **Step 2: Run fmt check**

Run: `cargo fmt --check`

- [ ] **Step 3: Push**

```bash
git push
```

- [ ] **Step 4: Run ci-watcher**

Invoke the ci-watcher skill as a background Haiku agent.

---

## Verification

```bash
# Store-level snippet tests
cargo test -p fastrag-store -- generate_snippets

# Integration tests
cargo test -p fastrag-cli --test snippet_e2e

# Full workspace
cargo test --workspace --features retrieval,rerank

# Lint
cargo clippy --workspace --all-targets --features retrieval,rerank -- -D warnings
cargo fmt --check

# Manual smoke test
echo '{"id":"cve-1","body":"log4j RCE allows remote code execution","severity":"CRITICAL"}' | \
  curl -s -X POST 'http://localhost:8081/ingest?id_field=id&text_fields=body&metadata_fields=severity' \
       -H 'Content-Type: application/x-ndjson' --data-binary @-
curl -s 'http://localhost:8081/query?q=log4j&top_k=3&snippet_len=100' | jq .
curl -s 'http://localhost:8081/query?q=log4j&top_k=3&fields=score,snippet' | jq .
curl -s 'http://localhost:8081/query?q=log4j&top_k=3&fields=-chunk_text' | jq .
```
