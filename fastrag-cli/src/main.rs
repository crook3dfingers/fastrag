mod args;
#[cfg(feature = "eval")]
mod eval;

use std::path::Path;
use std::sync::Arc;

use args::{ChunkStrategyArg, Cli, Command, OutputFormatArg};
use clap::Parser;
use fastrag::ops::{self, collect_files, output_path, render_document};
use fastrag::registry::ParserRegistry;
use fastrag::{ChunkingStrategy, OutputFormat};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt;
    let filter = EnvFilter::try_from_env("FASTRAG_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let json = std::env::var("FASTRAG_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    let builder = fmt().with_env_filter(filter);
    let _ = if json {
        builder.json().try_init()
    } else {
        builder.try_init()
    };
}

#[tokio::main]
async fn main() {
    init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Parse {
            path,
            format,
            output,
            workers,
            chunk_strategy,
            chunk_size,
            chunk_overlap,
            chunk_separators,
            similarity_threshold,
            percentile_threshold,
            context_template,
            stream,
            detect_language,
        } => {
            let output_format = match format {
                OutputFormatArg::Markdown => OutputFormat::Markdown,
                OutputFormatArg::Json => OutputFormat::Json,
                OutputFormatArg::Jsonl => OutputFormat::Jsonl,
                OutputFormatArg::Text => OutputFormat::PlainText,
                OutputFormatArg::Html => OutputFormat::Html,
            };

            let chunking = match chunk_strategy {
                ChunkStrategyArg::None => None,
                ChunkStrategyArg::Basic => Some(ChunkingStrategy::Basic {
                    max_characters: chunk_size,
                    overlap: chunk_overlap,
                }),
                ChunkStrategyArg::ByTitle => Some(ChunkingStrategy::ByTitle {
                    max_characters: chunk_size,
                    overlap: chunk_overlap,
                }),
                ChunkStrategyArg::Recursive => Some(ChunkingStrategy::RecursiveCharacter {
                    max_characters: chunk_size,
                    overlap: chunk_overlap,
                    separators: chunk_separators
                        .map(|s| {
                            s.split(',')
                                .map(|sep| sep.replace("\\n", "\n").replace("\\t", "\t"))
                                .collect()
                        })
                        .unwrap_or_else(fastrag::default_separators),
                }),
                ChunkStrategyArg::Semantic => Some(ChunkingStrategy::Semantic {
                    max_characters: chunk_size,
                    similarity_threshold,
                    percentile_threshold,
                }),
            };

            let context_injection = context_template.map(|t| fastrag::ContextInjection {
                template: t.replace("\\n", "\n"),
            });

            let path = Path::new(&path);
            if stream && path.is_file() {
                stream_file(path);
            } else if path.is_file() {
                parse_single_file(
                    path,
                    output_format,
                    output.as_deref(),
                    chunking.as_ref(),
                    context_injection.as_ref(),
                    detect_language,
                );
            } else if path.is_dir() {
                parse_directory(
                    path,
                    output_format,
                    output.as_deref(),
                    workers,
                    chunking,
                    detect_language,
                )
                .await;
            } else {
                eprintln!("Error: '{}' is not a file or directory", path.display());
                std::process::exit(1);
            }
        }
        Command::Formats => {
            let registry = ParserRegistry::default();
            println!("Supported formats:");
            for format in registry.supported_formats() {
                println!("  - {format}");
            }
        }
        #[cfg(feature = "retrieval")]
        Command::Index {
            input,
            corpus,
            chunk_strategy,
            chunk_size,
            chunk_overlap,
            chunk_separators,
            similarity_threshold,
            percentile_threshold,
            model_path,
            embedder: _,
            openai_model: _,
            openai_base_url: _,
            ollama_model: _,
            ollama_url: _,
            metadata,
        } => {
            let chunking = chunking_from_args(
                chunk_strategy,
                chunk_size,
                chunk_overlap,
                chunk_separators,
                similarity_threshold,
                percentile_threshold,
            );
            let embedder =
                fastrag_cli::embed_loader::load_embedder(model_path).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
            let base_metadata: std::collections::BTreeMap<String, String> =
                metadata.into_iter().collect();
            match ops::index_path_with_metadata(
                &input,
                &corpus,
                &chunking,
                embedder.as_ref(),
                &base_metadata,
            ) {
                Ok(stats) => {
                    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
                    println!(
                        "indexed {} files ({} new, {} changed, {} unchanged, {} deleted) — {} chunks added, {} removed",
                        stats.files_indexed,
                        stats.files_new,
                        stats.files_changed,
                        stats.files_unchanged,
                        stats.files_deleted,
                        stats.chunks_added,
                        stats.chunks_removed,
                    );
                }
                Err(e) => {
                    eprintln!("Error indexing {}: {e}", input.display());
                    std::process::exit(1);
                }
            }
        }
        #[cfg(feature = "retrieval")]
        Command::Query {
            query,
            corpus,
            top_k,
            format,
            model_path,
            embedder: _,
            openai_model: _,
            openai_base_url: _,
            ollama_model: _,
            ollama_url: _,
            filter,
        } => {
            let embedder =
                fastrag_cli::embed_loader::load_embedder(model_path).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
            let filter_map = match filter.as_deref() {
                Some(s) => match args::parse_filter(s) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Error parsing --filter: {e}");
                        std::process::exit(2);
                    }
                },
                None => std::collections::BTreeMap::new(),
            };
            match ops::query_corpus_with_filter(
                &corpus,
                &query,
                top_k,
                embedder.as_ref(),
                &filter_map,
            ) {
                Ok(hits) => print_query_results(&hits, format),
                Err(e) => {
                    eprintln!("Error querying corpus {}: {e}", corpus.display());
                    std::process::exit(1);
                }
            }
        }
        #[cfg(feature = "retrieval")]
        Command::CorpusInfo { corpus } => match ops::corpus_info(&corpus) {
            Ok(info) => println!("{}", serde_json::to_string_pretty(&info).unwrap()),
            Err(e) => {
                eprintln!("Error reading corpus {}: {e}", corpus.display());
                std::process::exit(1);
            }
        },
        #[cfg(feature = "eval")]
        Command::Eval {
            dataset,
            dataset_name,
            report,
            embedder,
            top_k,
            chunking,
            chunk_size,
            chunk_overlap,
            max_rss_mb,
            max_docs,
            max_queries,
        } => {
            if let Err(e) = eval::run_eval(
                dataset,
                dataset_name,
                report,
                embedder,
                top_k,
                chunking,
                chunk_size,
                chunk_overlap,
                max_rss_mb,
                max_docs,
                max_queries,
            )
            .await
            {
                eprintln!("Error running eval: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "retrieval")]
        Command::ServeHttp {
            corpus,
            port,
            model_path,
            embedder: _,
            openai_model: _,
            openai_base_url: _,
            ollama_model: _,
            ollama_url: _,
            token,
        } => {
            let token = token.or_else(|| std::env::var("FASTRAG_TOKEN").ok());
            if let Err(e) = fastrag_cli::http::serve_http(corpus, port, model_path, token).await {
                eprintln!("Error starting HTTP server: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "mcp")]
        Command::Serve => {
            fastrag_mcp::serve_stdio().await.unwrap();
        }
    }
}

#[cfg(feature = "retrieval")]
fn chunking_from_args(
    chunk_strategy: ChunkStrategyArg,
    chunk_size: usize,
    chunk_overlap: usize,
    chunk_separators: Option<String>,
    similarity_threshold: Option<f32>,
    percentile_threshold: Option<f32>,
) -> ChunkingStrategy {
    match chunk_strategy {
        ChunkStrategyArg::None | ChunkStrategyArg::Basic => ChunkingStrategy::Basic {
            max_characters: chunk_size,
            overlap: chunk_overlap,
        },
        ChunkStrategyArg::ByTitle => ChunkingStrategy::ByTitle {
            max_characters: chunk_size,
            overlap: chunk_overlap,
        },
        ChunkStrategyArg::Recursive => ChunkingStrategy::RecursiveCharacter {
            max_characters: chunk_size,
            overlap: chunk_overlap,
            separators: chunk_separators
                .map(|s| {
                    s.split(',')
                        .map(|sep| sep.replace("\\n", "\n").replace("\\t", "\t"))
                        .collect()
                })
                .unwrap_or_else(fastrag::default_separators),
        },
        ChunkStrategyArg::Semantic => ChunkingStrategy::Semantic {
            max_characters: chunk_size,
            similarity_threshold,
            percentile_threshold,
        },
    }
}

#[cfg(feature = "retrieval")]
fn print_query_results(hits: &[fastrag::SearchHit], format: OutputFormatArg) {
    let dtos: Vec<fastrag::corpus::SearchHitDto> = hits.iter().cloned().map(Into::into).collect();
    match format {
        OutputFormatArg::Json => println!("{}", serde_json::to_string_pretty(&dtos).unwrap()),
        OutputFormatArg::Jsonl => {
            for dto in dtos {
                println!("{}", serde_json::to_string(&dto).unwrap());
            }
        }
        OutputFormatArg::Text => {
            for dto in dtos {
                println!(
                    "[{:.4}] {}#{}: {}",
                    dto.score,
                    dto.source_path.display(),
                    dto.chunk_index,
                    dto.chunk_text
                );
            }
        }
        OutputFormatArg::Markdown => {
            for dto in dtos {
                println!(
                    "- **{:.4}** `{}` #{}: {}",
                    dto.score,
                    dto.source_path.display(),
                    dto.chunk_index,
                    dto.chunk_text
                );
            }
        }
        OutputFormatArg::Html => {
            println!("<ul>");
            for dto in dtos {
                println!(
                    "<li><strong>{:.4}</strong> {}#{}: {}</li>",
                    dto.score,
                    dto.source_path.display(),
                    dto.chunk_index,
                    dto.chunk_text
                );
            }
            println!("</ul>");
        }
    }
}

fn stream_file(path: &Path) {
    use std::io::Write;

    match ops::parse_stream(path) {
        Ok((_format, elements)) => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            for result in elements {
                match result {
                    Ok(el) => {
                        if let Ok(json) = serde_json::to_string(&el) {
                            let _ = writeln!(handle, "{json}");
                            let _ = handle.flush();
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {e}");
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error streaming {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}

fn parse_single_file(
    path: &Path,
    output_format: OutputFormat,
    output_dir: Option<&str>,
    chunking: Option<&ChunkingStrategy>,
    context_injection: Option<&fastrag::ContextInjection>,
    detect_language: bool,
) {
    match ops::parse_single_with_context(
        path,
        output_format,
        chunking,
        detect_language,
        context_injection,
    ) {
        Ok(result) => {
            if let Some(dir) = output_dir {
                let out_path = output_path(path, dir, output_format);
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                if let Err(e) = std::fs::write(&out_path, &result.content) {
                    eprintln!("Error writing {}: {e}", out_path.display());
                    std::process::exit(1);
                }
                eprintln!("Wrote {}", out_path.display());
            } else {
                print!("{}", result.content);
            }
        }
        Err(e) => {
            eprintln!("Error parsing {}: {e}", path.display());
            std::process::exit(1);
        }
    }
}

async fn parse_directory(
    dir: &Path,
    output_format: OutputFormat,
    output_dir: Option<&str>,
    workers: usize,
    _chunking: Option<ChunkingStrategy>,
    _detect_language: bool,
) {
    let files = collect_files(dir);
    if files.is_empty() {
        eprintln!("No parseable files found in {}", dir.display());
        return;
    }

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("valid template")
            .progress_chars("=> "),
    );

    let semaphore = Arc::new(Semaphore::new(workers));
    let output_dir = output_dir.map(|s| s.to_string());

    let mut handles = Vec::new();
    for file in files {
        let sem = semaphore.clone();
        let output_dir = output_dir.clone();
        let pb = pb.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.ok();
            tokio::task::spawn_blocking(move || {
                let registry = ParserRegistry::default();
                match registry.parse_file(&file) {
                    Ok(doc) => {
                        let rendered = render_document(&doc, output_format);
                        if let Some(ref dir) = output_dir {
                            let out_path = output_path(&file, dir, output_format);
                            if let Some(parent) = out_path.parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            if let Err(e) = std::fs::write(&out_path, &rendered) {
                                eprintln!("Error writing {}: {e}", out_path.display());
                            }
                        } else {
                            println!("--- {} ---", file.display());
                            println!("{rendered}");
                            println!();
                        }
                    }
                    Err(e) => {
                        eprintln!("Error parsing {}: {e}", file.display());
                    }
                }
                pb.inc(1);
            })
            .await
            .ok();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.ok();
    }

    pb.finish_with_message("done");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use fastrag::{Document, Element, ElementKind, FileFormat, Metadata};

    fn sample_doc() -> Document {
        let mut m = Metadata::new(FileFormat::Text);
        m.title = Some("Title".to_string());
        Document {
            metadata: m,
            elements: vec![
                Element::new(ElementKind::Title, "Title"),
                Element::new(ElementKind::Paragraph, "Body text."),
            ],
        }
    }

    #[test]
    fn render_document_markdown() {
        let doc = sample_doc();
        let md = render_document(&doc, OutputFormat::Markdown);
        assert!(md.contains("# Title"));
    }

    #[test]
    fn render_document_json() {
        let doc = sample_doc();
        let json = render_document(&doc, OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["elements"].is_array());
    }

    #[test]
    fn render_document_plain_text() {
        let doc = sample_doc();
        let text = render_document(&doc, OutputFormat::PlainText);
        assert!(text.contains("Title"));
        assert!(!text.contains("# "));
    }

    #[test]
    fn output_path_markdown() {
        let p = output_path(Path::new("data/sample.csv"), "/out", OutputFormat::Markdown);
        assert_eq!(p, PathBuf::from("/out/sample.csv.md"));
    }

    #[test]
    fn output_path_json() {
        let p = output_path(Path::new("data/sample.csv"), "/out", OutputFormat::Json);
        assert_eq!(p, PathBuf::from("/out/sample.csv.json"));
    }

    #[test]
    fn output_path_plain_text() {
        let p = output_path(
            Path::new("data/sample.csv"),
            "/out",
            OutputFormat::PlainText,
        );
        assert_eq!(p, PathBuf::from("/out/sample.csv.txt"));
    }

    #[test]
    fn output_path_html() {
        let p = output_path(Path::new("data/sample.csv"), "/out", OutputFormat::Html);
        assert_eq!(p, PathBuf::from("/out/sample.csv.html"));
    }

    #[test]
    fn collect_files_fixtures() {
        let fixtures = format!("{}/../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let files = collect_files(Path::new(&fixtures));
        assert_eq!(files.len(), 22); // txt, csv, md, html (x2), pdf (x10), xml, xlsx, docx, pptx, epub, rtf, eml
    }

    #[test]
    fn collect_files_empty_dir() {
        let dir = std::env::temp_dir().join("fastrag_test_empty_dir");
        std::fs::create_dir_all(&dir).ok();
        let files = collect_files(&dir);
        assert!(files.is_empty());
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn stream_file_produces_valid_jsonl() {
        let fixtures = format!("{}/../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let path = PathBuf::from(&fixtures).join("sample.txt");

        let (format, elements) = ops::parse_stream(&path).unwrap();
        assert_eq!(format, "Text");

        let collected: Vec<_> = elements.filter_map(|r| r.ok()).collect();
        assert!(!collected.is_empty());

        // Each element should serialize to valid JSON
        for el in &collected {
            let json = serde_json::to_string(el).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(parsed["kind"].is_string());
        }
    }
}
