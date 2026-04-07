use std::path::PathBuf;

#[cfg(feature = "mcp-search")]
use std::sync::Arc;

#[cfg(feature = "mcp-search")]
use fastrag::BgeSmallEmbedder;
use fastrag::ops;
use fastrag::{ChunkingStrategy, OutputFormat, default_separators};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_router};
use serde::Deserialize;
#[cfg(feature = "mcp-search")]
use tokio::sync::OnceCell;

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ParseFileParams {
    /// Absolute path to the file to parse
    #[schemars(description = "Absolute path to the file to parse")]
    pub path: String,
    /// Output format: markdown, json, jsonl, text, or html (default: markdown)
    #[schemars(
        description = "Output format: markdown, json, jsonl, text, or html (default: markdown)"
    )]
    pub format: Option<String>,
    /// Whether to detect the document language
    #[schemars(description = "Whether to detect the document language")]
    pub detect_language: Option<bool>,
    /// Whether to use streaming mode (skips hierarchy/captions, returns raw elements)
    #[schemars(
        description = "Use streaming mode for incremental output (skips hierarchy/captions)"
    )]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ParseDirectoryParams {
    /// Absolute path to the directory to parse
    #[schemars(description = "Absolute path to the directory containing files to parse")]
    pub path: String,
    /// Output format: markdown, json, jsonl, text, or html (default: markdown)
    #[schemars(
        description = "Output format: markdown, json, jsonl, text, or html (default: markdown)"
    )]
    pub format: Option<String>,
    /// Whether to detect the document language
    #[schemars(description = "Whether to detect the document language")]
    pub detect_language: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ChunkDocumentParams {
    /// Absolute path to the file to chunk
    #[schemars(description = "Absolute path to the file to parse and chunk")]
    pub path: String,
    /// Chunking strategy: basic, by-title, or recursive (default: basic)
    #[schemars(description = "Chunking strategy: basic, by-title, or recursive (default: basic)")]
    pub strategy: Option<String>,
    /// Maximum characters per chunk (default: 1000)
    #[schemars(description = "Maximum characters per chunk (default: 1000)")]
    pub max_characters: Option<usize>,
    /// Number of overlapping characters between consecutive chunks (default: 0)
    #[schemars(
        description = "Number of overlapping characters between consecutive chunks (default: 0)"
    )]
    pub overlap: Option<usize>,
    /// Custom separators for recursive strategy (default: paragraph, line, sentence, word, char)
    #[schemars(
        description = "Custom separators for recursive strategy, ordered most to least specific"
    )]
    pub separators: Option<Vec<String>>,
    /// Output format: markdown, json, jsonl, text, or html (default: markdown)
    #[schemars(
        description = "Output format: markdown, json, jsonl, text, or html (default: markdown)"
    )]
    pub format: Option<String>,
    /// Context template for injecting document context into chunks
    #[schemars(
        description = "Template for context injection. Placeholders: {document_title}, {section}, {chunk_text}"
    )]
    pub context_template: Option<String>,
    /// Similarity threshold for semantic chunking (0.0 to 1.0)
    #[schemars(
        description = "Similarity threshold for semantic chunking (0.0 to 1.0). Used when strategy is 'semantic'."
    )]
    pub similarity_threshold: Option<f32>,
    /// Percentile threshold for semantic chunking (0.0 to 100.0)
    #[schemars(
        description = "Percentile threshold for semantic chunking (0.0 to 100.0). Used when strategy is 'semantic'."
    )]
    pub percentile_threshold: Option<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchCorpusParams {
    /// Search query
    #[schemars(description = "Search query")]
    pub query: String,
    /// Corpus path
    #[schemars(description = "Corpus path")]
    pub corpus_path: PathBuf,
    /// Number of hits to return
    #[schemars(description = "Number of hits to return")]
    pub top_k: Option<usize>,
    /// Equality filters applied to entry metadata (AND-combined)
    #[schemars(description = "Equality filters applied to entry metadata (AND-combined)")]
    pub filter: Option<std::collections::BTreeMap<String, String>>,
}

pub struct FastRagMcpServer {
    _tool_router: ToolRouter<Self>,
    #[cfg(feature = "mcp-search")]
    embedder: OnceCell<Arc<BgeSmallEmbedder>>,
}

impl FastRagMcpServer {
    pub fn new() -> Self {
        Self {
            _tool_router: Self::tool_router(),
            #[cfg(feature = "mcp-search")]
            embedder: OnceCell::new(),
        }
    }
}

impl Default for FastRagMcpServer {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_output_format(format: Option<&str>) -> OutputFormat {
    match format.map(|s| s.to_lowercase()).as_deref() {
        Some("json") => OutputFormat::Json,
        Some("jsonl") => OutputFormat::Jsonl,
        Some("text") | Some("plain") | Some("plaintext") => OutputFormat::PlainText,
        Some("html") => OutputFormat::Html,
        _ => OutputFormat::Markdown,
    }
}

#[cfg(feature = "mcp-search")]
async fn load_embedder(
    cell: &OnceCell<Arc<BgeSmallEmbedder>>,
) -> Result<Arc<BgeSmallEmbedder>, String> {
    cell.get_or_try_init(|| async {
        BgeSmallEmbedder::from_hf_hub()
            .map(Arc::new)
            .map_err(|e| format!("Failed to load embedder: {e}"))
    })
    .await
    .map(Arc::clone)
}

#[tool_router]
impl FastRagMcpServer {
    #[tool(
        description = "Parse a single document file. Supports PDF, HTML, Markdown, CSV, \
        Text, DOCX, PPTX, XLSX, and XML. Returns structured content in the requested format \
        (markdown by default)."
    )]
    async fn parse_file(
        &self,
        Parameters(params): Parameters<ParseFileParams>,
    ) -> Result<String, String> {
        let path = PathBuf::from(&params.path);
        let output_format = parse_output_format(params.format.as_deref());
        let detect_language = params.detect_language.unwrap_or(false);
        let stream = params.stream.unwrap_or(false);

        tokio::task::spawn_blocking(move || {
            if stream {
                let (_format, elements) = ops::parse_stream(&path)
                    .map_err(|e| format!("Failed to stream {}: {e}", path.display()))?;
                let collected: Vec<_> = elements.filter_map(|r| r.ok()).collect();
                serde_json::to_string_pretty(&collected)
                    .map_err(|e| format!("Failed to serialize result: {e}"))
            } else {
                let result = ops::parse_single(&path, output_format, None, detect_language)
                    .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
                serde_json::to_string_pretty(&result)
                    .map_err(|e| format!("Failed to serialize result: {e}"))
            }
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    #[tool(
        description = "Parse all supported document files in a directory. Returns an array \
        of parse results for each file found."
    )]
    async fn parse_directory(
        &self,
        Parameters(params): Parameters<ParseDirectoryParams>,
    ) -> Result<String, String> {
        let path = PathBuf::from(&params.path);
        let output_format = parse_output_format(params.format.as_deref());
        let detect_language = params.detect_language.unwrap_or(false);

        tokio::task::spawn_blocking(move || {
            let results = ops::parse_directory(&path, output_format, detect_language)
                .map_err(|e| format!("Failed to parse directory {}: {e}", path.display()))?;
            serde_json::to_string_pretty(&results)
                .map_err(|e| format!("Failed to serialize results: {e}"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    #[tool(description = "List all supported file formats with their file extensions.")]
    fn list_formats(&self) -> String {
        let formats = ops::list_formats();
        serde_json::to_string_pretty(&formats).unwrap_or_else(|_| "[]".to_string())
    }

    #[tool(
        description = "Parse and chunk a document file for RAG pipelines. Splits the \
        document into smaller chunks using the specified strategy. Use 'basic' for simple \
        character-limit splitting, 'by-title' for semantic splitting on headings, or \
        'recursive' for recursive character splitting (LangChain-style). All strategies \
        support an 'overlap' parameter for shared context between chunks."
    )]
    async fn chunk_document(
        &self,
        Parameters(params): Parameters<ChunkDocumentParams>,
    ) -> Result<String, String> {
        let path = PathBuf::from(&params.path);
        let output_format = parse_output_format(params.format.as_deref());
        let max_chars = params.max_characters.unwrap_or(1000);
        let overlap = params.overlap.unwrap_or(0);
        let strategy = match params.strategy.as_deref() {
            Some("by-title") | Some("by_title") | Some("bytitle") => ChunkingStrategy::ByTitle {
                max_characters: max_chars,
                overlap,
            },
            Some("recursive") | Some("recursive-character") | Some("recursive_character") => {
                ChunkingStrategy::RecursiveCharacter {
                    max_characters: max_chars,
                    overlap,
                    separators: params.separators.unwrap_or_else(default_separators),
                }
            }
            Some("semantic") => ChunkingStrategy::Semantic {
                max_characters: max_chars,
                similarity_threshold: params.similarity_threshold,
                percentile_threshold: params.percentile_threshold,
            },
            _ => ChunkingStrategy::Basic {
                max_characters: max_chars,
                overlap,
            },
        };

        let context_injection = params
            .context_template
            .map(|t| fastrag::ContextInjection { template: t });

        tokio::task::spawn_blocking(move || {
            let result = ops::chunk_file_with_context(
                &path,
                &strategy,
                output_format,
                context_injection.as_ref(),
            )
            .map_err(|e| format!("Failed to chunk {}: {e}", path.display()))?;
            serde_json::to_string_pretty(&result)
                .map_err(|e| format!("Failed to serialize result: {e}"))
        })
        .await
        .map_err(|e| format!("Task failed: {e}"))?
    }

    #[tool(description = "Search an indexed corpus for relevant chunks.")]
    async fn search_corpus(
        &self,
        Parameters(params): Parameters<SearchCorpusParams>,
    ) -> Result<String, String> {
        #[cfg(feature = "mcp-search")]
        {
            let embedder = load_embedder(&self.embedder).await?;
            let corpus_path = params.corpus_path.clone();
            let query = params.query.clone();
            let top_k = params.top_k.unwrap_or(5);
            let filter = params.filter.clone().unwrap_or_default();

            let hits = tokio::task::spawn_blocking(move || {
                ops::query_corpus_with_filter(
                    &corpus_path,
                    &query,
                    top_k,
                    embedder.as_ref(),
                    &filter,
                )
            })
            .await
            .map_err(|e| format!("Task failed: {e}"))?
            .map_err(|e| {
                format!(
                    "Failed to query corpus {}: {e}",
                    params.corpus_path.display()
                )
            })?;

            let dto = hits
                .into_iter()
                .map(fastrag::corpus::SearchHitDto::from)
                .collect::<Vec<_>>();
            return serde_json::to_string_pretty(&dto)
                .map_err(|e| format!("Failed to serialize result: {e}"));
        }

        #[cfg(not(feature = "mcp-search"))]
        {
            let _ = params;
            Err("search_corpus requires the mcp-search feature".to_string())
        }
    }
}

impl ServerHandler for FastRagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Use this server to parse documents for RAG pipelines. \
                 Use `parse_file` for single documents, `chunk_document` when you need \
                 RAG-ready chunks, `parse_directory` for batch processing, and \
                 `list_formats` to check supported file types. All tools accept an \
                 optional `format` parameter (markdown/json/jsonl/text/html) — default is markdown \
                 which works best for LLM consumption. If retrieval support is enabled, \
                 use `search_corpus` to search an indexed corpus for relevant chunks.",
        )
    }
}

/// Start the MCP server on stdin/stdout.
pub async fn serve_stdio() -> Result<(), Box<dyn std::error::Error>> {
    let server = FastRagMcpServer::new();
    let transport = rmcp::transport::io::stdio();
    let service = server.serve(transport).await.inspect_err(|e| {
        eprintln!("Failed to start MCP server: {e}");
    })?;
    let _ = service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_file_params_deserializes() {
        let json = serde_json::json!({
            "path": "/tmp/test.pdf",
            "format": "json",
            "detect_language": true
        });
        let params: ParseFileParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.path, "/tmp/test.pdf");
        assert_eq!(params.format.as_deref(), Some("json"));
        assert_eq!(params.detect_language, Some(true));
    }

    #[test]
    fn parse_file_params_minimal() {
        let json = serde_json::json!({ "path": "/tmp/test.txt" });
        let params: ParseFileParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.path, "/tmp/test.txt");
        assert!(params.format.is_none());
        assert!(params.detect_language.is_none());
    }

    #[test]
    fn parse_directory_params_deserializes() {
        let json = serde_json::json!({
            "path": "/tmp/docs",
            "format": "markdown"
        });
        let params: ParseDirectoryParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.path, "/tmp/docs");
        assert_eq!(params.format.as_deref(), Some("markdown"));
    }

    #[test]
    fn chunk_document_params_deserializes() {
        let json = serde_json::json!({
            "path": "/tmp/doc.pdf",
            "strategy": "by-title",
            "max_characters": 500,
            "format": "json"
        });
        let params: ChunkDocumentParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.path, "/tmp/doc.pdf");
        assert_eq!(params.strategy.as_deref(), Some("by-title"));
        assert_eq!(params.max_characters, Some(500));
        assert_eq!(params.format.as_deref(), Some("json"));
    }

    #[test]
    fn chunk_document_params_defaults() {
        let json = serde_json::json!({ "path": "/tmp/doc.txt" });
        let params: ChunkDocumentParams = serde_json::from_value(json).unwrap();
        assert!(params.strategy.is_none());
        assert!(params.max_characters.is_none());
        assert!(params.format.is_none());
    }

    #[test]
    fn parse_output_format_variants() {
        assert_eq!(parse_output_format(None), OutputFormat::Markdown);
        assert_eq!(
            parse_output_format(Some("markdown")),
            OutputFormat::Markdown
        );
        assert_eq!(parse_output_format(Some("json")), OutputFormat::Json);
        assert_eq!(parse_output_format(Some("jsonl")), OutputFormat::Jsonl);
        assert_eq!(parse_output_format(Some("text")), OutputFormat::PlainText);
        assert_eq!(parse_output_format(Some("plain")), OutputFormat::PlainText);
        assert_eq!(
            parse_output_format(Some("plaintext")),
            OutputFormat::PlainText
        );
        assert_eq!(parse_output_format(Some("JSON")), OutputFormat::Json);
        assert_eq!(parse_output_format(Some("JSONL")), OutputFormat::Jsonl);
        assert_eq!(parse_output_format(Some("html")), OutputFormat::Html);
    }

    #[test]
    fn list_formats_tool_returns_json() {
        let server = FastRagMcpServer::new();
        let result = server.list_formats();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 12);
        let names: Vec<&str> = arr.iter().map(|v| v["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"PDF"));
        assert!(names.contains(&"HTML"));
        assert!(names.contains(&"Text"));
    }

    #[test]
    fn server_info_has_tools_capability() {
        let server = FastRagMcpServer::new();
        let info = server.get_info();
        assert!(info.capabilities.tools.is_some());
        assert!(info.instructions.is_some());
    }

    #[tokio::test]
    async fn parse_file_tool_with_fixture() {
        let server = FastRagMcpServer::new();
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = format!("{fixtures}/sample.txt");
        let params = ParseFileParams {
            path,
            format: None,
            detect_language: None,
            stream: None,
        };
        let result = server.parse_file(Parameters(params)).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["filename"], "sample.txt");
        assert_eq!(parsed["format"], "Text");
        assert!(parsed["element_count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn chunk_document_tool_with_fixture() {
        let server = FastRagMcpServer::new();
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = format!("{fixtures}/sample.txt");
        let params = ChunkDocumentParams {
            path,
            strategy: Some("basic".into()),
            max_characters: Some(50),
            overlap: None,
            separators: None,
            format: None,
            context_template: None,
            similarity_threshold: None,
            percentile_threshold: None,
        };
        let result = server.chunk_document(Parameters(params)).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["filename"], "sample.txt");
        assert!(parsed["total_chunks"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn parse_file_nonexistent_returns_error() {
        let server = FastRagMcpServer::new();
        let params = ParseFileParams {
            path: "/nonexistent/file.txt".into(),
            format: None,
            detect_language: None,
            stream: None,
        };
        let result = server.parse_file(Parameters(params)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn parse_file_streaming_mode() {
        let server = FastRagMcpServer::new();
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = format!("{fixtures}/sample.txt");
        let params = ParseFileParams {
            path,
            format: None,
            detect_language: None,
            stream: Some(true),
        };
        let result = server.parse_file(Parameters(params)).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let arr = parsed.as_array().unwrap();
        assert!(
            !arr.is_empty(),
            "streaming should return non-empty element array"
        );
        // Each element should have a kind field
        for el in arr {
            assert!(el["kind"].is_string(), "element missing kind: {el}");
        }
    }
}
