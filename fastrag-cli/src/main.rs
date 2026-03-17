mod args;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use args::{Cli, Command, OutputFormatArg};
use clap::Parser;
use fastrag::registry::ParserRegistry;
use fastrag::{FileFormat, OutputFormat};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Parse {
            path,
            format,
            output,
            workers,
        } => {
            let output_format = match format {
                OutputFormatArg::Markdown => OutputFormat::Markdown,
                OutputFormatArg::Json => OutputFormat::Json,
                OutputFormatArg::Text => OutputFormat::PlainText,
            };

            let path = Path::new(&path);
            if path.is_file() {
                parse_single_file(path, output_format, output.as_deref());
            } else if path.is_dir() {
                parse_directory(path, output_format, output.as_deref(), workers).await;
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
    }
}

fn parse_single_file(path: &Path, output_format: OutputFormat, output_dir: Option<&str>) {
    let registry = ParserRegistry::default();
    match registry.parse_file(path) {
        Ok(doc) => {
            let rendered = render_document(&doc, output_format);
            if let Some(dir) = output_dir {
                let out_path = output_path(path, dir, output_format);
                if let Some(parent) = out_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                if let Err(e) = std::fs::write(&out_path, &rendered) {
                    eprintln!("Error writing {}: {e}", out_path.display());
                    std::process::exit(1);
                }
                eprintln!("Wrote {}", out_path.display());
            } else {
                print!("{rendered}");
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

fn collect_files(dir: &Path) -> Vec<PathBuf> {
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
                        "pdf" | "html" | "htm" | "md" | "markdown" | "csv" | "txt" | "text" | "log"
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

fn render_document(doc: &fastrag::Document, format: OutputFormat) -> String {
    match format {
        OutputFormat::Markdown => doc.to_markdown(),
        OutputFormat::Json => doc
            .to_json()
            .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}")),
        OutputFormat::PlainText => doc.to_plain_text(),
    }
}

fn output_path(input: &Path, output_dir: &str, format: OutputFormat) -> PathBuf {
    let out_ext = match format {
        OutputFormat::Markdown => "md",
        OutputFormat::Json => "json",
        OutputFormat::PlainText => "txt",
    };
    let filename = input.file_name().unwrap_or_default();
    PathBuf::from(output_dir).join(format!("{}.{out_ext}", filename.to_string_lossy()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag::{Document, Element, ElementKind, Metadata};

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
    fn collect_files_fixtures() {
        let fixtures = format!("{}/../tests/fixtures", env!("CARGO_MANIFEST_DIR"));
        let files = collect_files(Path::new(&fixtures));
        assert_eq!(files.len(), 5); // txt, csv, md, html, pdf
    }

    #[test]
    fn collect_files_empty_dir() {
        let dir = std::env::temp_dir().join("fastrag_test_empty_dir");
        std::fs::create_dir_all(&dir).ok();
        let files = collect_files(&dir);
        assert!(files.is_empty());
        std::fs::remove_dir(&dir).ok();
    }
}
