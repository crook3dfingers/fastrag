use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

fn parse_kv(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .ok_or_else(|| format!("expected key=value, got `{s}`"))
}

/// Parse a `--filter` comma list like `customer=acme,severity=high` into a BTreeMap.
pub fn parse_filter(s: &str) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut out = std::collections::BTreeMap::new();
    for pair in s.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (k, v) = parse_kv(pair)?;
        out.insert(k, v);
    }
    Ok(out)
}

/// Parse a `--filter` string into a FilterExpr. Supports both legacy `k=v,k2=v2`
/// format and the new filter syntax (e.g. `severity = HIGH AND cvss_score >= 7.0`).
#[cfg(feature = "store")]
pub fn parse_filter_expr(s: &str) -> Result<fastrag::filter::FilterExpr, String> {
    fastrag::filter::parse(s).map_err(|e| e.to_string())
}

#[cfg(feature = "retrieval")]
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum EmbedderKindArg {
    Bge,
    Openai,
    Ollama,
    Qwen3Q8,
}

#[cfg(feature = "rerank")]
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
pub enum RerankerKindArg {
    Onnx,
    LlamaCpp,
}

#[cfg(feature = "store")]
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
pub enum IngestPresetArg {
    TarmoFinding,
}

#[cfg(feature = "retrieval")]
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum TimeDecayBlendArg {
    Multiplicative,
    Additive,
}

#[cfg(feature = "retrieval")]
#[allow(clippy::too_many_arguments)]
pub fn build_hybrid_opts(
    hybrid: bool,
    rrf_k: u32,
    rrf_overfetch: usize,
    time_decay_field: Option<String>,
    time_decay_halflife: &str,
    time_decay_weight: f32,
    time_decay_dateless_prior: f32,
    time_decay_blend: TimeDecayBlendArg,
) -> Result<fastrag::corpus::hybrid::HybridOpts, String> {
    let blend_str = match time_decay_blend {
        TimeDecayBlendArg::Multiplicative => "multiplicative",
        TimeDecayBlendArg::Additive => "additive",
    };
    fastrag::corpus::hybrid::build_hybrid_opts_from_parts(
        hybrid,
        rrf_k,
        rrf_overfetch,
        time_decay_field,
        time_decay_halflife,
        time_decay_weight,
        time_decay_dateless_prior,
        blend_str,
    )
    .map_err(|e| {
        // Remap generic field names to CLI flag names for better UX.
        e.replace("time_decay_halflife", "--time-decay-halflife")
            .replace("time_decay_weight", "--time-decay-weight")
            .replace("time_decay_dateless_prior", "--time-decay-dateless-prior")
            .replace("time_decay_field", "--time-decay-field")
    })
}

#[derive(Parser)]
#[command(name = "fastrag", about = "Fast document parser for AI/RAG pipelines")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// Parse one or more files
    Parse {
        /// File or directory to parse
        path: String,

        /// Output format
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormatArg,

        /// Output directory (for batch mode)
        #[arg(short, long)]
        output: Option<String>,

        /// Number of parallel workers
        #[arg(short = 'j', long, default_value = "4")]
        workers: usize,

        /// Chunking strategy (none, basic, by-title, recursive)
        #[arg(long, default_value = "none")]
        chunk_strategy: ChunkStrategyArg,

        /// Maximum characters per chunk
        #[arg(long, default_value = "1000")]
        chunk_size: usize,

        /// Number of overlapping characters between consecutive chunks
        #[arg(long, default_value = "0")]
        chunk_overlap: usize,

        /// Comma-separated list of separators for recursive strategy (use \n for newline)
        #[arg(long)]
        chunk_separators: Option<String>,

        /// Similarity threshold for semantic chunking (0.0 to 1.0)
        #[arg(long)]
        similarity_threshold: Option<f32>,

        /// Percentile threshold for semantic chunking (0.0 to 100.0)
        #[arg(long)]
        percentile_threshold: Option<f32>,

        /// Context template for chunk context injection (e.g. "{document_title} > {section}\n\n{chunk_text}")
        #[arg(long)]
        context_template: Option<String>,

        /// Stream elements incrementally (JSONL output, skips hierarchy/captions)
        #[arg(long)]
        stream: bool,

        /// Detect document language
        #[arg(long)]
        detect_language: bool,
    },

    /// List supported formats
    Formats,

    /// Check environment for llama-server and report its version
    Doctor,

    /// Start MCP server for AI assistant integration
    #[cfg(feature = "mcp")]
    Serve,

    /// Index a corpus of files for semantic search
    #[cfg(feature = "retrieval")]
    Index {
        /// File or directory to index
        input: PathBuf,

        /// Corpus directory used for persistence
        #[arg(long)]
        corpus: PathBuf,

        /// Chunking strategy (none, basic, by-title, recursive, semantic)
        #[arg(long, default_value = "recursive")]
        chunk_strategy: ChunkStrategyArg,

        /// Maximum characters per chunk
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        /// Number of overlapping characters between consecutive chunks
        #[arg(long, default_value_t = 200)]
        chunk_overlap: usize,

        /// Comma-separated list of separators for recursive strategy
        #[arg(long)]
        chunk_separators: Option<String>,

        /// Similarity threshold for semantic chunking (0.0 to 1.0)
        #[arg(long)]
        similarity_threshold: Option<f32>,

        /// Percentile threshold for semantic chunking (0.0 to 100.0)
        #[arg(long)]
        percentile_threshold: Option<f32>,

        /// Optional local model path
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Embedder backend to use. Defaults to bge for write paths; auto-detected
        /// from the corpus manifest on read paths if omitted.
        #[arg(long, value_enum)]
        embedder: Option<EmbedderKindArg>,

        /// OpenAI model name.
        #[arg(long, default_value = "text-embedding-3-small")]
        openai_model: String,

        /// OpenAI API base URL.
        #[arg(long, default_value = "https://api.openai.com/v1")]
        openai_base_url: String,

        /// Ollama model name.
        #[arg(long, default_value = "nomic-embed-text")]
        ollama_model: String,

        /// Ollama server URL.
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,

        /// Apply metadata key=value to every file in this run (repeatable).
        /// Per-file `.meta.json` sidecars override these on conflict.
        #[arg(long = "metadata", value_parser = parse_kv)]
        metadata: Vec<(String, String)>,

        /// Opt in to Contextual Retrieval at ingest time. Runs a small
        /// instruct LLM over each chunk to generate a 50–100 token prefix
        /// that situates the chunk in its source document; the prefix is
        /// cached per-corpus in `contextualization.sqlite` and reused on
        /// incremental re-index.
        #[cfg(feature = "contextual")]
        #[arg(long)]
        contextualize: bool,

        /// Override the completion model preset used by `--contextualize`.
        /// Accepts a preset name; currently only `default` is supported.
        #[cfg(feature = "contextual")]
        #[arg(long)]
        context_model: Option<String>,

        /// Hard-fail the ingest on any per-chunk contextualization error.
        /// Without this flag, failing chunks fall back to raw text and the
        /// ingest continues.
        #[cfg(feature = "contextual")]
        #[arg(long)]
        context_strict: bool,

        /// Re-run contextualization only for chunks whose cached row is
        /// `failed`. Requires a corpus already ingested with
        /// `--contextualize`. Does not re-parse source documents.
        #[cfg(feature = "contextual")]
        #[arg(long)]
        retry_failed: bool,

        /// Enable security corpus hygiene filters (NVD reject, boilerplate strip,
        /// language filter). Requires --features hygiene on the facade crate.
        #[cfg(feature = "hygiene")]
        #[arg(long)]
        security_profile: bool,

        /// ISO 639-1 language code to keep; non-matching chunks are dropped.
        /// Only used when --security-profile is set. Default: en.
        #[cfg(feature = "hygiene")]
        #[arg(long, default_value = "en")]
        security_lang: String,

        /// Path to a CISA KEV catalog JSON (vulnerabilities.json or
        /// {cve_ids:[...]} minimal shape). Tags matching CVEs with kev_flag=true.
        /// Only used when --security-profile is set.
        #[cfg(feature = "hygiene")]
        #[arg(long)]
        security_kev_catalog: Option<PathBuf>,

        /// Comma-separated vuln_status values to reject. Default: Rejected,Disputed.
        /// Only used when --security-profile is set.
        #[cfg(feature = "hygiene")]
        #[arg(long, default_value = "Rejected,Disputed")]
        security_reject_statuses: String,

        /// Input format (auto-detect from extension, or specify explicitly)
        #[cfg(feature = "store")]
        #[arg(long = "format")]
        ingest_format: Option<String>,

        /// JSONL: fields to concatenate as embeddable text
        #[cfg(feature = "store")]
        #[arg(long, value_delimiter = ',')]
        text_fields: Option<Vec<String>>,

        /// JSONL: field to use as stable external ID
        #[cfg(feature = "store")]
        #[arg(long)]
        id_field: Option<String>,

        /// JSONL: fields to index as typed metadata
        #[cfg(feature = "store")]
        #[arg(long, value_delimiter = ',')]
        metadata_fields: Option<Vec<String>>,

        /// JSONL: explicit type overrides (field=type, e.g. cvss_score=numeric)
        #[cfg(feature = "store")]
        #[arg(long, value_delimiter = ',')]
        metadata_types: Option<Vec<String>>,

        /// JSONL: fields that contain arrays
        #[cfg(feature = "store")]
        #[arg(long, value_delimiter = ',')]
        array_fields: Option<Vec<String>>,

        /// JSONL: name of the field holding the CWE numeric id. Enables
        /// query-time CWE hierarchy expansion for this corpus.
        #[cfg(feature = "store")]
        #[arg(long)]
        cwe_field: Option<String>,

        /// Ingest preset (pre-fills text-fields, metadata-fields, etc.)
        #[cfg(feature = "store")]
        #[arg(long)]
        preset: Option<IngestPresetArg>,
    },

    /// Query an indexed corpus
    #[cfg(feature = "retrieval")]
    Query {
        /// Search query
        query: String,

        /// Corpus directory used for persistence
        #[arg(long)]
        corpus: PathBuf,

        /// Number of results to return
        #[arg(long, default_value_t = 5)]
        top_k: usize,

        /// Output format
        #[arg(short, long, default_value = "json")]
        format: OutputFormatArg,

        /// Optional local model path
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Embedder backend to use. Defaults to bge for write paths; auto-detected
        /// from the corpus manifest on read paths if omitted.
        #[arg(long, value_enum)]
        embedder: Option<EmbedderKindArg>,

        /// OpenAI model name.
        #[arg(long, default_value = "text-embedding-3-small")]
        openai_model: String,

        /// OpenAI API base URL.
        #[arg(long, default_value = "https://api.openai.com/v1")]
        openai_base_url: String,

        /// Ollama model name.
        #[arg(long, default_value = "nomic-embed-text")]
        ollama_model: String,

        /// Ollama server URL.
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,

        /// Filter expression: supports legacy `k=v,k2=v2` comma format and
        /// rich syntax (e.g. `severity = HIGH AND cvss_score >= 7.0`).
        /// Applied as a post-filter over the HNSW fan-out.
        #[arg(long)]
        filter: Option<String>,

        /// JSON AST filter expression (for scripts piping complex filters).
        /// Mutually exclusive with --filter.
        #[arg(long, conflicts_with = "filter")]
        filter_json: Option<String>,

        /// Enable BM25 + dense hybrid retrieval via Reciprocal Rank Fusion.
        /// Disabled by default (dense-only path preserves current behavior).
        #[arg(long)]
        hybrid: bool,

        /// RRF k parameter. Default 60 per the canonical RRF paper.
        #[arg(long, default_value_t = 60)]
        rrf_k: u32,

        /// Per-retriever overfetch multiplier — fetch rrf_overfetch * top_k
        /// from each retriever before fusion.
        #[arg(long, default_value_t = 4)]
        rrf_overfetch: usize,

        /// Name of the `Date` metadata field to use for temporal decay.
        /// Implies --hybrid.
        #[arg(long)]
        time_decay_field: Option<String>,

        /// Decay halflife (humantime format, e.g. "30d", "7d", "1y").
        #[arg(long, default_value = "30d")]
        time_decay_halflife: String,

        /// Alpha floor: minimum decay factor for very old docs. Range 0..=1.
        #[arg(long, default_value_t = 0.3)]
        time_decay_weight: f32,

        /// Neutral prior used for docs missing the date field. Range 0..=1.
        #[arg(long, default_value_t = 0.5)]
        time_decay_dateless_prior: f32,

        /// Blend mode: multiplicative (default) or additive.
        #[arg(long, value_enum, default_value = "multiplicative")]
        time_decay_blend: TimeDecayBlendArg,

        /// Reranker backend (default: onnx). Reranking is on by default.
        #[cfg(feature = "rerank")]
        #[arg(long, value_enum, default_value = "onnx")]
        rerank: RerankerKindArg,

        /// Disable cross-encoder reranking.
        #[cfg(feature = "rerank")]
        #[arg(long, conflicts_with = "rerank")]
        no_rerank: bool,

        /// Over-fetch multiplier for reranking (fetch top_k * N from HNSW).
        #[cfg(feature = "rerank")]
        #[arg(long, default_value_t = 10)]
        rerank_over_fetch: usize,

        /// Enable query-time CWE hierarchy expansion. Requires the corpus
        /// to have been ingested with --cwe-field. Defaults on when the
        /// manifest records a cwe_field.
        #[arg(long, overrides_with = "no_cwe_expand")]
        cwe_expand: bool,

        /// Disable query-time CWE hierarchy expansion.
        #[arg(long = "no-cwe-expand", overrides_with = "cwe_expand")]
        no_cwe_expand: bool,
    },

    /// Show corpus metadata
    #[cfg(feature = "retrieval")]
    CorpusInfo {
        /// Corpus directory used for persistence
        #[arg(long)]
        corpus: PathBuf,

        /// Optional local model path
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Embedder backend to use. Auto-detected from corpus manifest if omitted.
        #[arg(long, value_enum)]
        embedder: Option<EmbedderKindArg>,

        /// OpenAI model name.
        #[arg(long, default_value = "text-embedding-3-small")]
        openai_model: String,

        /// OpenAI API base URL.
        #[arg(long, default_value = "https://api.openai.com/v1")]
        openai_base_url: String,

        /// Ollama model name.
        #[arg(long, default_value = "nomic-embed-text")]
        ollama_model: String,

        /// Ollama server URL.
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,
    },

    /// Evaluate a retrieval setup on a BEIR-compatible dataset
    #[cfg(feature = "eval")]
    Eval {
        /// Dataset JSON file (mutually exclusive with --dataset-name)
        #[arg(
            long,
            conflicts_with = "dataset_name",
            required_unless_present_any = ["dataset_name", "gold_set"]
        )]
        dataset: Option<PathBuf>,

        /// Built-in dataset to load via fastrag-eval (nfcorpus, scifact, nvd, cwe)
        #[arg(long = "dataset-name", value_enum)]
        dataset_name: Option<EvalDatasetNameArg>,

        /// JSON report output path
        #[arg(long)]
        report: PathBuf,

        /// Embedder to use
        #[arg(long, default_value = "mock")]
        embedder: EvalEmbedderArg,

        /// Number of results to retrieve per query
        #[arg(long, default_value_t = 10)]
        top_k: usize,

        /// Chunking strategy
        #[arg(long, default_value = "basic")]
        chunking: EvalChunkingArg,

        /// Maximum characters per chunk
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        /// Chunk overlap in characters
        #[arg(long, default_value_t = 0)]
        chunk_overlap: usize,

        /// Fail the run if peak RSS exceeds this many MB (off by default)
        #[arg(long)]
        max_rss_mb: Option<u64>,

        /// Cap the indexed corpus to the first N documents (sampled baselines)
        #[arg(long)]
        max_docs: Option<usize>,

        /// Cap the eval to the first N queries (sampled baselines)
        #[arg(long)]
        max_queries: Option<usize>,

        /// Path to a gold-set JSON file. Mutually exclusive with --dataset-name/--dataset.
        #[arg(long)]
        gold_set: Option<PathBuf>,

        /// Path to a built corpus (contextualized when --config-matrix is set).
        #[arg(long)]
        corpus: Option<PathBuf>,

        /// Path to a second corpus built without --contextualize.
        /// Required when --config-matrix is set.
        #[arg(long)]
        corpus_no_contextual: Option<PathBuf>,

        /// Run all 4 matrix variants. Requires --gold-set, --corpus, --corpus-no-contextual.
        #[arg(long, default_value_t = false)]
        config_matrix: bool,

        /// Comma-separated variant labels to run (default: all four).
        /// Values: primary, no_rerank, no_contextual, dense_only
        #[arg(long)]
        variants: Option<String>,

        /// Path to a checked-in baseline JSON. Non-zero exit on >2% regression.
        #[arg(long)]
        baseline: Option<PathBuf>,
    },

    /// Start HTTP retrieval server
    #[cfg(feature = "retrieval")]
    ServeHttp {
        /// Corpus to serve. Format: `name=path` or just `path` (registered as "default").
        /// May be specified multiple times for multi-corpus mode.
        #[arg(long, required = true)]
        corpus: Vec<String>,

        /// Port to bind
        #[arg(long, default_value_t = 8081)]
        port: u16,

        /// Optional local model path
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Embedder backend to use. Defaults to bge for write paths; auto-detected
        /// from the corpus manifest on read paths if omitted.
        #[arg(long, value_enum)]
        embedder: Option<EmbedderKindArg>,

        /// OpenAI model name.
        #[arg(long, default_value = "text-embedding-3-small")]
        openai_model: String,

        /// OpenAI API base URL.
        #[arg(long, default_value = "https://api.openai.com/v1")]
        openai_base_url: String,

        /// Ollama model name.
        #[arg(long, default_value = "nomic-embed-text")]
        ollama_model: String,

        /// Ollama server URL.
        #[arg(long, default_value = "http://localhost:11434")]
        ollama_url: String,

        /// Shared-secret auth token. Also read from `FASTRAG_TOKEN` env var; CLI flag wins.
        /// When set, /query and /metrics require `X-Fastrag-Token: <token>` or
        /// `Authorization: Bearer <token>`. /health stays unauthenticated.
        #[arg(long)]
        token: Option<String>,

        /// Skip BM25/Tantivy and use dense vector search only.
        #[arg(long)]
        dense_only: bool,

        /// Reranker backend (default: onnx). Reranking is on by default.
        #[cfg(feature = "rerank")]
        #[arg(long, value_enum, default_value = "onnx")]
        rerank: RerankerKindArg,

        /// Disable cross-encoder reranking.
        #[cfg(feature = "rerank")]
        #[arg(long, conflicts_with = "rerank")]
        no_rerank: bool,

        /// Over-fetch multiplier for reranking (fetch top_k * N from HNSW).
        #[cfg(feature = "rerank")]
        #[arg(long, default_value_t = 10)]
        rerank_over_fetch: usize,

        /// Maximum number of queries allowed in a single /batch-query request.
        #[arg(long, default_value_t = 100)]
        batch_max_queries: usize,

        /// Tenant enforcement field. When set, every request must supply
        /// `X-Fastrag-Tenant: <value>`; the server injects `<field> = <value>`
        /// as a mandatory filter on all queries.
        #[arg(long)]
        tenant_field: Option<String>,

        /// Maximum body size for POST /ingest (bytes). Default: 52428800 (50 MiB).
        #[arg(long, default_value_t = 52_428_800)]
        ingest_max_body: usize,

        /// Hard cap on per-corpus adaptive overfetch for POST /similar.
        /// Raise for large corpora with permissive thresholds. Default: 10000.
        #[arg(long, default_value_t = 10_000)]
        similar_overfetch_cap: usize,

        /// Default for query-time CWE hierarchy expansion. Per-request
        /// override via the `cwe_expand` query parameter.
        #[arg(long)]
        cwe_expand: bool,
    },

    /// Delete a record by external ID from a store-backed corpus
    #[cfg(feature = "store")]
    Delete {
        /// Path to corpus directory
        #[arg(long)]
        corpus: PathBuf,

        /// External ID of the record to delete
        #[arg(long)]
        id: String,
    },

    /// Compact corpus HNSW index (purge tombstones, rebuild graph)
    #[cfg(feature = "store")]
    Compact {
        /// Path to corpus directory
        #[arg(long)]
        corpus: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormatArg {
    Markdown,
    Json,
    Jsonl,
    Text,
    Html,
}

#[derive(Clone, ValueEnum, PartialEq)]
pub enum ChunkStrategyArg {
    None,
    Basic,
    ByTitle,
    Recursive,
    Semantic,
}

#[cfg(feature = "eval")]
#[derive(Clone, ValueEnum)]
pub enum EvalEmbedderArg {
    Mock,
    BgeSmall,
}

#[cfg(feature = "eval")]
#[derive(Clone, ValueEnum)]
pub enum EvalChunkingArg {
    Basic,
    ByTitle,
    Recursive,
}

#[cfg(feature = "eval")]
#[derive(Clone, Copy, ValueEnum)]
pub enum EvalDatasetNameArg {
    Nfcorpus,
    Scifact,
    Nvd,
    Cwe,
}

#[cfg(feature = "eval")]
impl EvalDatasetNameArg {
    pub fn to_eval(self) -> fastrag_eval::DatasetName {
        match self {
            EvalDatasetNameArg::Nfcorpus => fastrag_eval::DatasetName::NfCorpus,
            EvalDatasetNameArg::Scifact => fastrag_eval::DatasetName::SciFact,
            EvalDatasetNameArg::Nvd => fastrag_eval::DatasetName::Nvd,
            EvalDatasetNameArg::Cwe => fastrag_eval::DatasetName::CweTop25,
        }
    }
}
