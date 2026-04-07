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

#[derive(Parser)]
#[command(name = "fastrag", about = "Fast document parser for AI/RAG pipelines")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
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
        #[arg(long, default_value = "basic")]
        chunk_strategy: ChunkStrategyArg,

        /// Maximum characters per chunk
        #[arg(long, default_value_t = 1000)]
        chunk_size: usize,

        /// Number of overlapping characters between consecutive chunks
        #[arg(long, default_value_t = 0)]
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

        /// Apply metadata key=value to every file in this run (repeatable).
        /// Per-file `.meta.json` sidecars override these on conflict.
        #[arg(long = "metadata", value_parser = parse_kv)]
        metadata: Vec<(String, String)>,
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

        /// Comma-separated equality filters (e.g. `customer=acme,severity=high`).
        /// AND-combined; applied as a post-filter over the HNSW fan-out.
        #[arg(long)]
        filter: Option<String>,
    },

    /// Show corpus metadata
    #[cfg(feature = "retrieval")]
    CorpusInfo {
        /// Corpus directory used for persistence
        #[arg(long)]
        corpus: PathBuf,
    },

    /// Evaluate a retrieval setup on a BEIR-compatible dataset
    #[cfg(feature = "eval")]
    Eval {
        /// Dataset JSON file (mutually exclusive with --dataset-name)
        #[arg(
            long,
            conflicts_with = "dataset_name",
            required_unless_present = "dataset_name"
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
    },

    /// Start HTTP retrieval server
    #[cfg(feature = "retrieval")]
    ServeHttp {
        /// Corpus directory used for persistence
        #[arg(long)]
        corpus: PathBuf,

        /// Port to bind
        #[arg(long, default_value_t = 8081)]
        port: u16,

        /// Optional local model path
        #[arg(long)]
        model_path: Option<PathBuf>,
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
