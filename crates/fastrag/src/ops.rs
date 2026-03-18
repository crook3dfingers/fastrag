use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::registry::ParserRegistry;
use crate::{ChunkingStrategy, ContextInjection, Document, FastRagError, FileFormat, OutputFormat};

/// Result of parsing a single file.
#[derive(Debug, Serialize)]
pub struct ParseResult {
    pub filename: String,
    pub format: String,
    pub content: String,
    pub element_count: usize,
    pub metadata: DocumentMetadata,
}

/// Subset of document metadata for API responses.
#[derive(Debug, Serialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub page_count: Option<usize>,
}

/// A single chunk in the chunk result.
#[derive(Debug, Serialize)]
pub struct ChunkInfo {
    pub index: usize,
    pub char_count: usize,
    pub section: Option<String>,
    pub text: String,
}

/// Result of chunking a file.
#[derive(Debug, Serialize)]
pub struct ChunkResult {
    pub filename: String,
    pub chunks: Vec<ChunkInfo>,
    pub total_chunks: usize,
}

/// Information about a supported format.
#[derive(Debug, Serialize)]
pub struct FormatInfo {
    pub name: String,
    pub extensions: Vec<String>,
}

/// Parse a single file and return structured result.
pub fn parse_single(
    path: &Path,
    output_format: OutputFormat,
    chunking: Option<&ChunkingStrategy>,
    detect_language: bool,
) -> Result<ParseResult, FastRagError> {
    parse_single_with_context(path, output_format, chunking, detect_language, None)
}

/// Parse a single file with optional context injection.
pub fn parse_single_with_context(
    path: &Path,
    output_format: OutputFormat,
    chunking: Option<&ChunkingStrategy>,
    detect_language: bool,
    context_injection: Option<&ContextInjection>,
) -> Result<ParseResult, FastRagError> {
    let registry = ParserRegistry::default();
    let mut doc = registry.parse_file(path)?;
    doc.build_hierarchy();
    doc.associate_captions();

    #[cfg(feature = "language-detection")]
    if detect_language {
        doc.detect_language();
        doc.detect_element_languages();
    }
    #[cfg(not(feature = "language-detection"))]
    let _ = detect_language;

    let content = if let Some(strategy) = chunking {
        let mut chunks = doc.chunk(strategy);
        if let Some(injection) = context_injection {
            doc.inject_context(&mut chunks, injection);
        }
        render_chunks(&chunks, output_format)
    } else {
        render_document(&doc, output_format)
    };

    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    Ok(ParseResult {
        filename,
        format: doc.metadata.format.to_string(),
        content,
        element_count: doc.elements.len(),
        metadata: DocumentMetadata {
            title: doc.metadata.title.clone(),
            author: doc.metadata.author.clone(),
            page_count: doc.metadata.page_count,
        },
    })
}

/// Parse all supported files in a directory.
pub fn parse_directory(
    dir: &Path,
    output_format: OutputFormat,
    detect_language: bool,
) -> Result<Vec<ParseResult>, FastRagError> {
    let files = collect_files(dir);
    let mut results = Vec::new();
    for file in files {
        match parse_single(&file, output_format, None, detect_language) {
            Ok(result) => results.push(result),
            Err(e) => {
                eprintln!("Error parsing {}: {e}", file.display());
            }
        }
    }
    Ok(results)
}

/// List all supported formats with their extensions.
pub fn list_formats() -> Vec<FormatInfo> {
    let formats = FileFormat::all_known();
    formats
        .iter()
        .map(|f| FormatInfo {
            name: f.to_string(),
            extensions: extensions_for_format(f),
        })
        .collect()
}

/// Parse and chunk a file for RAG.
pub fn chunk_file(
    path: &Path,
    strategy: &ChunkingStrategy,
    output_format: OutputFormat,
) -> Result<ChunkResult, FastRagError> {
    chunk_file_with_context(path, strategy, output_format, None)
}

/// Parse and chunk a file with optional context injection.
pub fn chunk_file_with_context(
    path: &Path,
    strategy: &ChunkingStrategy,
    output_format: OutputFormat,
    context_injection: Option<&ContextInjection>,
) -> Result<ChunkResult, FastRagError> {
    let registry = ParserRegistry::default();
    let doc = registry.parse_file(path)?;

    let mut chunks = doc.chunk(strategy);
    if let Some(injection) = context_injection {
        doc.inject_context(&mut chunks, injection);
    }
    let chunks = chunks;
    let chunk_infos: Vec<ChunkInfo> = chunks
        .iter()
        .map(|c| ChunkInfo {
            index: c.index,
            char_count: c.char_count,
            section: c.section.clone(),
            text: match output_format {
                OutputFormat::Markdown => {
                    let mut out = String::new();
                    if let Some(ref section) = c.section {
                        out.push_str(&format!("## {section}\n\n"));
                    }
                    out.push_str(&c.text);
                    out
                }
                OutputFormat::Json
                | OutputFormat::Jsonl
                | OutputFormat::PlainText
                | OutputFormat::Html => c.text.clone(),
            },
        })
        .collect();

    let total_chunks = chunk_infos.len();
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    Ok(ChunkResult {
        filename,
        chunks: chunk_infos,
        total_chunks,
    })
}

/// Recursively collect parseable files from a directory.
pub fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let format = FileFormat::detect(&path, &[]);
                    if format != FileFormat::Unknown {
                        files.push(path);
                        continue;
                    }
                    if matches!(
                        ext.to_lowercase().as_str(),
                        "pdf"
                            | "html"
                            | "htm"
                            | "md"
                            | "markdown"
                            | "csv"
                            | "txt"
                            | "text"
                            | "log"
                            | "docx"
                            | "pptx"
                            | "xlsx"
                            | "xml"
                            | "epub"
                            | "rtf"
                            | "eml"
                    ) {
                        files.push(path);
                    }
                }
            } else if path.is_dir() {
                files.extend(collect_files(&path));
            }
        }
    }
    files
}

/// Render chunks to a string in the given output format.
pub fn render_chunks(chunks: &[crate::Chunk], format: OutputFormat) -> String {
    let mut out = String::new();
    for chunk in chunks {
        match format {
            OutputFormat::Markdown => {
                if let Some(ref section) = chunk.section {
                    out.push_str(&format!("## {section}\n\n"));
                }
                out.push_str(&chunk.text);
                out.push_str("\n\n---\n\n");
            }
            OutputFormat::Json | OutputFormat::Jsonl => {
                out.push_str(&format!(
                    "{{\"index\":{},\"char_count\":{},\"section\":{},\"text\":{}}}\n",
                    chunk.index,
                    chunk.char_count,
                    chunk
                        .section
                        .as_ref()
                        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                        .unwrap_or_else(|| "null".to_string()),
                    serde_json::to_string(&chunk.text).unwrap_or_default()
                ));
            }
            OutputFormat::PlainText => {
                out.push_str(&chunk.text);
                out.push_str("\n\n");
            }
            OutputFormat::Html => {
                if let Some(ref section) = chunk.section {
                    out.push_str(&format!("<h2>{section}</h2>\n"));
                }
                out.push_str(&format!("<p>{}</p>\n<hr>\n", chunk.text));
            }
        }
    }
    out.trim_end().to_string()
}

/// Render a document to a string in the given output format.
pub fn render_document(doc: &Document, format: OutputFormat) -> String {
    match format {
        OutputFormat::Markdown => doc.to_markdown(),
        OutputFormat::Json => doc
            .to_json()
            .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}")),
        OutputFormat::Jsonl => doc.to_jsonl(),
        OutputFormat::PlainText => doc.to_plain_text(),
        OutputFormat::Html => doc.to_html(),
    }
}

/// Compute output path for a file given an output directory and format.
pub fn output_path(input: &Path, output_dir: &str, format: OutputFormat) -> PathBuf {
    let out_ext = match format {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
        OutputFormat::Jsonl => "jsonl",
        OutputFormat::PlainText => "txt",
        OutputFormat::Html => "html",
    };
    let filename = input.file_name().unwrap_or_default();
    PathBuf::from(output_dir).join(format!("{}.{out_ext}", filename.to_string_lossy()))
}

fn extensions_for_format(format: &FileFormat) -> Vec<String> {
    match format {
        FileFormat::Pdf => vec!["pdf".into()],
        FileFormat::Html => vec!["html".into(), "htm".into(), "xhtml".into()],
        FileFormat::Markdown => vec!["md".into(), "markdown".into(), "mkd".into()],
        FileFormat::Csv => vec!["csv".into(), "tsv".into()],
        FileFormat::Text => vec!["txt".into(), "text".into(), "log".into()],
        FileFormat::Docx => vec!["docx".into()],
        FileFormat::Pptx => vec!["pptx".into()],
        FileFormat::Xlsx => vec!["xlsx".into()],
        FileFormat::Xml => vec!["xml".into()],
        FileFormat::Epub => vec!["epub".into()],
        FileFormat::Rtf => vec!["rtf".into()],
        FileFormat::Email => vec!["eml".into()],
        FileFormat::Unknown => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_txt_fixture() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");
        let result = parse_single(&path, OutputFormat::Markdown, None, false).unwrap();
        assert_eq!(result.filename, "sample.txt");
        assert_eq!(result.format, "Text");
        assert!(result.element_count > 0);
        assert!(!result.content.is_empty());
    }

    #[test]
    fn parse_single_json_format() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");
        let result = parse_single(&path, OutputFormat::Json, None, false).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert!(parsed["elements"].is_array());
    }

    #[test]
    fn parse_single_with_chunking() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");
        let strategy = ChunkingStrategy::Basic {
            max_characters: 50,
            overlap: 0,
        };
        let result = parse_single(&path, OutputFormat::Markdown, Some(&strategy), false).unwrap();
        assert!(!result.content.is_empty());
    }

    #[test]
    fn list_formats_returns_9() {
        let formats = list_formats();
        assert_eq!(formats.len(), 12);
        let names: Vec<&str> = formats.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"PDF"));
        assert!(names.contains(&"HTML"));
        assert!(names.contains(&"Text"));
    }

    #[test]
    fn list_formats_has_extensions() {
        let formats = list_formats();
        let pdf = formats.iter().find(|f| f.name == "PDF").unwrap();
        assert_eq!(pdf.extensions, vec!["pdf"]);
        let html = formats.iter().find(|f| f.name == "HTML").unwrap();
        assert!(html.extensions.contains(&"html".to_string()));
        assert!(html.extensions.contains(&"htm".to_string()));
    }

    #[test]
    fn chunk_file_txt_fixture() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");
        let strategy = ChunkingStrategy::Basic {
            max_characters: 50,
            overlap: 0,
        };
        let result = chunk_file(&path, &strategy, OutputFormat::Markdown).unwrap();
        assert_eq!(result.filename, "sample.txt");
        assert!(result.total_chunks > 0);
        assert_eq!(result.chunks.len(), result.total_chunks);
    }

    #[test]
    fn parse_result_serializes_to_json() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");
        let result = parse_single(&path, OutputFormat::Markdown, None, false).unwrap();
        let json = serde_json::to_string_pretty(&result).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["filename"], "sample.txt");
        assert!(parsed["element_count"].is_number());
    }

    #[test]
    fn render_document_jsonl() {
        use crate::{Document, Element, ElementKind, FileFormat, Metadata};
        let mut m = Metadata::new(FileFormat::Text);
        m.title = Some("T".to_string());
        let doc = Document {
            metadata: m,
            elements: vec![
                Element::new(ElementKind::Title, "T"),
                Element::new(ElementKind::Paragraph, "P"),
            ],
        };
        let jsonl = render_document(&doc, OutputFormat::Jsonl);
        let lines: Vec<&str> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(parsed["kind"].is_string());
        }
    }

    #[test]
    fn output_path_jsonl() {
        let p = output_path(Path::new("data/sample.csv"), "/out", OutputFormat::Jsonl);
        assert_eq!(p, PathBuf::from("/out/sample.csv.jsonl"));
    }

    #[test]
    fn render_chunks_jsonl() {
        let chunks = vec![crate::Chunk {
            index: 0,
            text: "hello".to_string(),
            char_count: 5,
            section: None,
            elements: vec![],
        }];
        let result = render_chunks(&chunks, OutputFormat::Jsonl);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["index"], 0);
        assert_eq!(parsed["text"], "hello");
    }

    #[test]
    fn collect_files_from_fixtures() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let files = collect_files(Path::new(&fixtures));
        assert_eq!(files.len(), 19);
    }

    #[test]
    fn parse_directory_fixtures() {
        let fixtures = format!("{}/../../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let results = parse_directory(Path::new(&fixtures), OutputFormat::Markdown, false).unwrap();
        assert!(!results.is_empty());
        for r in &results {
            assert!(!r.filename.is_empty());
            assert!(!r.content.is_empty());
        }
    }
}
