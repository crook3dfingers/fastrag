# Rich Metadata Filters — Design Spec

## Context

Issue #42. fastrag's query path accepts `--filter k=v,k2=v2` at all surfaces (CLI, HTTP, MCP), but the filter parameter is ignored (TODO in `corpus/mod.rs:532`). Phase 3 Step 1 (#41) landed `TypedValue`, `DynamicSchema`, and Tantivy field indexing. This step adds the filtering engine.

## Decision: Post-Filter Only (Deferred Pre-Filter)

The issue describes three retrieval modes selected by Tantivy doc-count selectivity estimation. We ship **post-filter with adaptive overfetch only**. Tantivy pre-filter and selectivity switching are deferred — post-filter covers the common case and decouples the filter AST from Tantivy query compilation.

## Decision: Hand-Rolled Parser

The string syntax parser is a hand-rolled recursive descent — zero new dependencies, precise positional error messages, and a small grammar (~15 productions).

## 1. Filter AST

New module: `crates/fastrag/src/filter/`

```rust
// crates/fastrag/src/filter/ast.rs

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterExpr {
    Eq { field: String, value: TypedValue },
    Neq { field: String, value: TypedValue },
    Gt { field: String, value: TypedValue },
    Gte { field: String, value: TypedValue },
    Lt { field: String, value: TypedValue },
    Lte { field: String, value: TypedValue },
    In { field: String, values: Vec<TypedValue> },
    NotIn { field: String, values: Vec<TypedValue> },
    Contains { field: String, value: TypedValue },
    All { field: String, values: Vec<TypedValue> },
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
}
```

JSON AST deserialization uses serde's tagged enum. The string syntax parses into the same `FilterExpr`.

Backward compat: `k=v,k2=v2` is detected by the absence of keywords or operators and parsed as `And(vec![Eq{k,v}, Eq{k2,v2}])`.

## 2. String Syntax Parser

Hand-rolled recursive descent in `crates/fastrag/src/filter/parser.rs`.

```
expr       → or_expr
or_expr    → and_expr ("OR" and_expr)*
and_expr   → not_expr ("AND" not_expr)*
not_expr   → "NOT" not_expr | primary
primary    → "(" expr ")" | comparison
comparison → field operator value
           | field "IN" "(" value_list ")"
           | field "NOT" "IN" "(" value_list ")"
           | field "CONTAINS" value
           | field "ALL" "(" value_list ")"

field      → identifier | identifier "." identifier
operator   → "=" | "!=" | ">" | ">=" | "<" | "<="
value      → string_lit | number_lit | "true" | "false" | date_lit
string_lit → quoted_string | bare_word
value_list → value ("," value)*
```

- Keywords (`AND`, `OR`, `NOT`, `IN`, `CONTAINS`, `ALL`) are case-insensitive.
- Bare words are valid string values when unambiguous: `severity = HIGH` works.
- Quoted strings handle values with spaces or special characters: `title = "SQL Injection"`.
- Numbers are auto-detected: `7.0` → `Numeric(7.0)`.
- Dates follow ISO 8601: `2024-01-15` → `Date(NaiveDate)`.
- Booleans: `true`/`false` → `Bool`.
- Errors include position: `"unexpected token at position 23, expected value after '>=' "`.

Legacy detection: if the input matches `k=v(,k=v)*` with no keywords, the parser treats it as legacy format via a fast prefix check.

## 3. Filter Evaluation Engine

`crates/fastrag/src/filter/eval.rs` evaluates a `FilterExpr` against a record's metadata.

```rust
pub fn matches(expr: &FilterExpr, fields: &[(String, TypedValue)]) -> bool
```

Type coercion rules:
- `Numeric` vs `String`: attempt f64 parse; failure → no match.
- `Date` vs `String`: attempt `NaiveDate` parse; failure → no match.
- No implicit bool coercion — `Bool` compares only to `Bool`.
- `CONTAINS`: true if any element in the array field equals the value.
- `ALL`: true if every specified value appears in the array field.
- `IN`/`NOT IN`: scalar field value membership in the provided set.

Ordering for `Gt`/`Gte`/`Lt`/`Lte`:
- `Numeric`: f64 ordering.
- `Date`: chronological ordering.
- `String`: lexicographic ordering.
- `Bool`, `Array`: ordering operators return false.

Short-circuit: `And` stops on first false, `Or` stops on first true.

The evaluator is pure — no Tantivy dependency, no I/O.

## 4. Post-Filter with Adaptive Overfetch

Integration into `crates/fastrag/src/corpus/mod.rs` and `crates/fastrag-store/src/lib.rs`.

Flow:
1. Parse filter string → `FilterExpr` (or legacy `k=v` detection).
2. Query HNSW for `top_k * overfetch_factor` candidates (initial factor: 4×).
3. Fetch metadata for candidates via `Store::fetch_metadata()`.
4. Evaluate `FilterExpr` against each candidate's metadata.
5. If ≥ `top_k` results pass, return top_k by score.
6. If fewer pass, retry with 16×, then 32×.
7. If still insufficient after 32×, return whatever matched.

Changes:
- `Store` gains `fetch_metadata(ids) -> Vec<(id, Vec<(String, TypedValue)>)>` to retrieve user_fields from Tantivy docs.
- `SearchHit` and `SearchHitDto` gain `metadata: BTreeMap<String, TypedValue>`.
- `query_corpus_with_filter()` replaces the TODO with adaptive overfetch + eval.
- `query_corpus_reranked()` applies the filter before reranking (filter the overfetched candidates, then rerank survivors).
- All surfaces switch from `BTreeMap<String, String>` to `Option<FilterExpr>`. Parsing happens at each surface; corpus functions receive the AST.

## 5. Surface Integration

### CLI (`fastrag-cli/src/args.rs`, `main.rs`)
- `--filter` accepts both legacy `k=v,k2=v2` and the new string syntax.
- `--filter-json` accepts JSON AST (for scripts piping complex filters).
- Providing both is an error.

### HTTP (`fastrag-cli/src/http.rs`)
- `GET /query?filter=severity IN (HIGH,CRITICAL) AND cvss_score >= 7.0` — string syntax in the query param.
- `POST /query` with a JSON body gains a `filter` field accepting JSON AST.
- Backward compat: `GET /query?filter=k=v,k2=v2` still works.

### MCP (`crates/fastrag-mcp/src/lib.rs`)
- `search_corpus` param `filter` changes from `BTreeMap<String, String>` to `serde_json::Value`.
- Accepts a JSON AST object or a plain string (parsed server-side).

### Result shape
All surfaces include metadata in responses:
```json
{
  "score": 0.87,
  "chunk_text": "...",
  "source": { ... },
  "metadata": { "severity": "HIGH", "cvss_score": 9.1 }
}
```

## 6. Module Layout

```
crates/fastrag/src/filter/
├── mod.rs       — pub mod ast, parser, eval; re-exports
├── ast.rs       — FilterExpr enum, Display impl, JSON serde
├── parser.rs    — recursive descent: parse() -> Result<FilterExpr, FilterError>
└── eval.rs      — matches(expr, fields) -> bool
```

No new crate. The filter module lives in the `fastrag` facade crate, alongside the corpus query logic that consumes it. The filter module has no Tantivy dependency.

## 7. Files Modified

- `crates/fastrag-store/src/lib.rs` — `SearchHit` gains metadata; add `fetch_metadata()`
- `crates/fastrag-store/src/tantivy.rs` — `fetch_metadata()` impl; extract user_fields
- `crates/fastrag/src/corpus/mod.rs` — adaptive overfetch + eval; `SearchHitDto` gains metadata
- `crates/fastrag/src/lib.rs` — `pub mod filter;`
- `fastrag-cli/src/args.rs` — `--filter-json` flag; filter type change
- `fastrag-cli/src/main.rs` — parse filter at CLI surface; pass `FilterExpr`
- `fastrag-cli/src/http.rs` — accept filter in GET param + POST body
- `crates/fastrag-mcp/src/lib.rs` — `filter` param type change

## 8. Tests

- `filter/ast.rs` — JSON round-trip serde for every variant.
- `filter/parser.rs` — every operator, precedence, parentheses, legacy compat, positional error messages.
- `filter/eval.rs` — type coercion, array ops, boolean logic, short-circuit, ordering edge cases.
- `crates/fastrag/src/corpus/` — integration test for adaptive overfetch with mock data.
- `tests/` — end-to-end: JSONL ingest → query with filter → verify filtered results.

## 9. Out of Scope

- Tantivy pre-filter / query compilation (follow-up)
- Selectivity estimation (follow-up)
- Filter validation against corpus schema (follow-up)
- Regex or LIKE operators
