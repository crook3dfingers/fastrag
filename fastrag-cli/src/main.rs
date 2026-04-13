mod doctor;
#[cfg(feature = "eval")]
mod eval;
#[cfg(feature = "eval")]
mod eval_matrix;

use std::path::Path;
use std::sync::Arc;

use clap::Parser;
use fastrag::ops::{self, collect_files, output_path, render_document};
use fastrag::registry::ParserRegistry;
use fastrag::{ChunkingStrategy, DynEmbedderTrait, OutputFormat};
use fastrag_cli::args::{self, ChunkStrategyArg, Cli, Command, OutputFormatArg};
#[cfg(feature = "contextual")]
use fastrag_cli::context_loader::load_context_state;
use fastrag_cli::embed_loader;
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
        Command::Doctor => {
            doctor::run();
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
            embedder,
            openai_model,
            openai_base_url,
            ollama_model,
            ollama_url,
            metadata,
            #[cfg(feature = "contextual")]
            contextualize,
            #[cfg(feature = "contextual")]
            context_model,
            #[cfg(feature = "contextual")]
            context_strict,
            #[cfg(feature = "contextual")]
            retry_failed,
            #[cfg(feature = "hygiene")]
            security_profile,
            #[cfg(feature = "hygiene")]
            security_lang,
            #[cfg(feature = "hygiene")]
            security_kev_catalog,
            #[cfg(feature = "hygiene")]
            security_reject_statuses,
            #[cfg(feature = "store")]
            ingest_format,
            #[cfg(feature = "store")]
            text_fields,
            #[cfg(feature = "store")]
            id_field,
            #[cfg(feature = "store")]
            metadata_fields,
            #[cfg(feature = "store")]
            metadata_types,
            #[cfg(feature = "store")]
            array_fields,
            #[cfg(feature = "store")]
            preset,
        } => {
            #[cfg(feature = "contextual")]
            {
                if let Some(preset) = context_model.as_deref()
                    && preset != "default"
                {
                    eprintln!(
                        "Error: --context-model currently only supports `default`, got `{preset}`"
                    );
                    std::process::exit(2);
                }
                if retry_failed && !contextualize {
                    eprintln!("Error: --retry-failed requires --contextualize");
                    std::process::exit(2);
                }
            }
            tokio::task::block_in_place(|| {
                #[cfg(feature = "store")]
                {
                    let is_jsonl = ingest_format.as_deref() == Some("jsonl")
                        || input.extension().map(|e| e == "jsonl").unwrap_or(false);

                    if is_jsonl {
                        let base = preset.map(|p| match p {
                            args::IngestPresetArg::TarmoFinding => {
                                fastrag::ingest::presets::tarmo_finding_preset()
                            }
                        });
                        let config = fastrag::ingest::jsonl::JsonlIngestConfig {
                            text_fields: text_fields.unwrap_or_else(|| {
                                base.as_ref()
                                    .map(|c| c.text_fields.clone())
                                    .unwrap_or_default()
                            }),
                            id_field: id_field.unwrap_or_else(|| {
                                base.as_ref()
                                    .map(|c| c.id_field.clone())
                                    .unwrap_or_else(|| "id".into())
                            }),
                            metadata_fields: metadata_fields.unwrap_or_else(|| {
                                base.as_ref()
                                    .map(|c| c.metadata_fields.clone())
                                    .unwrap_or_default()
                            }),
                            metadata_types: if metadata_types
                                .as_ref()
                                .map(|v| !v.is_empty())
                                .unwrap_or(false)
                            {
                                parse_metadata_types(&metadata_types.unwrap_or_default())
                            } else {
                                base.as_ref()
                                    .map(|c| c.metadata_types.clone())
                                    .unwrap_or_default()
                            },
                            array_fields: array_fields.unwrap_or_else(|| {
                                base.as_ref()
                                    .map(|c| c.array_fields.clone())
                                    .unwrap_or_default()
                            }),
                        };
                        let chunking = chunking_from_args(
                            chunk_strategy,
                            chunk_size,
                            chunk_overlap,
                            chunk_separators,
                            similarity_threshold,
                            percentile_threshold,
                        );
                        let opts = embed_loader::EmbedderOptions {
                            kind: embedder,
                            model_path,
                            openai_model,
                            openai_base_url,
                            ollama_model,
                            ollama_url,
                        };
                        let embedder = embed_loader::load_for_write(&opts).unwrap_or_else(|e| {
                            eprintln!("Error loading embedder: {e}");
                            std::process::exit(1);
                        });
                        match fastrag::ingest::engine::index_jsonl(
                            &input,
                            &corpus,
                            &chunking,
                            embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
                            &config,
                        ) {
                            Ok(stats) => {
                                println!(
                                    "Indexed {} records ({} new, {} upserted, {} skipped), {} chunks",
                                    stats.records_total,
                                    stats.records_new,
                                    stats.records_upserted,
                                    stats.records_skipped,
                                    stats.chunks_created,
                                );
                            }
                            Err(e) => {
                                eprintln!("Error indexing JSONL {}: {e}", input.display());
                                std::process::exit(1);
                            }
                        }
                        return;
                    }
                }

                let chunking = chunking_from_args(
                    chunk_strategy,
                    chunk_size,
                    chunk_overlap,
                    chunk_separators,
                    similarity_threshold,
                    percentile_threshold,
                );
                let opts = embed_loader::EmbedderOptions {
                    kind: embedder,
                    model_path,
                    openai_model,
                    openai_base_url,
                    ollama_model,
                    ollama_url,
                };
                let embedder = embed_loader::load_for_write(&opts).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
                let base_metadata: std::collections::BTreeMap<String, String> =
                    metadata.into_iter().collect();

                #[cfg(feature = "contextual")]
                let mut context_state = if contextualize {
                    match load_context_state(&corpus) {
                        Ok(s) => Some(s),
                        Err(e) => {
                            eprintln!("Error starting contextualizer: {e}");
                            std::process::exit(1);
                        }
                    }
                } else {
                    None
                };

                #[cfg(feature = "contextual")]
                if retry_failed {
                    let state = context_state
                        .as_mut()
                        .expect("retry_failed requires contextualize");
                    let opts_for_retry = fastrag::corpus::ContextualizeOptions {
                        contextualizer: &*state.contextualizer,
                        cache: &mut state.cache,
                        strict: context_strict,
                    };
                    match fastrag::corpus::retry_failed_contextualizations(
                        &corpus,
                        opts_for_retry,
                        embedder.as_ref() as &dyn DynEmbedderTrait,
                    ) {
                        Ok(report) => {
                            println!(
                                "Repaired {}/{} failed chunks (rebuilt dense index: {})",
                                report.repaired, report.total_failed, report.rebuilt_dense,
                            );
                            return;
                        }
                        Err(e) => {
                            eprintln!("Error retrying failed contextualizations: {e}");
                            std::process::exit(1);
                        }
                    }
                }

                #[cfg(feature = "contextual")]
                let contextualize_opts =
                    context_state
                        .as_mut()
                        .map(|s| fastrag::corpus::ContextualizeOptions {
                            contextualizer: &*s.contextualizer,
                            cache: &mut s.cache,
                            strict: context_strict,
                        });

                #[cfg(feature = "hygiene")]
                let hygiene_chain = if security_profile {
                    use fastrag::hygiene::{KevTemporalTagger, security_default_chain};
                    let statuses: Vec<String> = security_reject_statuses
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    let mut chain = security_default_chain(if statuses.is_empty() {
                        None
                    } else {
                        Some(statuses)
                    });
                    if let Some(kev_path) = security_kev_catalog {
                        match KevTemporalTagger::from_path(&kev_path) {
                            Ok(tagger) => chain = chain.with_enricher(Box::new(tagger)),
                            Err(e) => {
                                eprintln!("Error loading KEV catalog: {e}");
                                std::process::exit(1);
                            }
                        }
                    }
                    // Honour --security-lang if it differs from the default "en".
                    if security_lang != "en" {
                        use fastrag::hygiene::{LanguageFilter, LanguagePolicy};
                        // Replace the default LanguageFilter by appending a custom one.
                        // (The default "en" filter is already in the chain; append a
                        //  second pass for the custom lang so the stricter one wins.)
                        chain = chain.with_chunk_filter(Box::new(LanguageFilter::new(
                            security_lang,
                            LanguagePolicy::Drop,
                        )));
                    }
                    Some(chain)
                } else {
                    None
                };

                match ops::index_path_with_metadata(
                    &input,
                    &corpus,
                    &chunking,
                    embedder.as_ref() as &dyn DynEmbedderTrait,
                    &base_metadata,
                    #[cfg(feature = "contextual")]
                    contextualize_opts,
                    #[cfg(feature = "hygiene")]
                    hygiene_chain.as_ref(),
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
                        #[cfg(feature = "hygiene")]
                        if security_profile {
                            let h = &stats.hygiene;
                            println!(
                                "hygiene: rejected={} stripped={} lang-dropped={} kev-tagged={}",
                                h.docs_rejected,
                                h.chunks_stripped,
                                h.chunks_lang_dropped,
                                h.chunks_kev_tagged,
                            );
                        }
                        #[cfg(feature = "contextual")]
                        if contextualize {
                            println!(
                                "Contextualized: {} ok / {} fallback",
                                stats.chunks_contextualized, stats.chunks_contextualize_fallback,
                            );
                        } else {
                            eprintln!();
                            eprintln!(
                                "Hint: re-run with --contextualize for better retrieval on technical"
                            );
                            eprintln!("      queries (one-time per corpus, cached thereafter).");
                        }
                    }
                    Err(e) => {
                        eprintln!("Error indexing {}: {e}", input.display());
                        std::process::exit(1);
                    }
                }
            });
        }
        #[cfg(feature = "retrieval")]
        Command::Query {
            query,
            corpus,
            top_k,
            format,
            model_path,
            embedder,
            openai_model,
            openai_base_url,
            ollama_model,
            ollama_url,
            filter,
            filter_json,
            dense_only,
            #[cfg(feature = "rerank")]
            rerank,
            #[cfg(feature = "rerank")]
            no_rerank,
            #[cfg(feature = "rerank")]
            rerank_over_fetch,
        } => {
            tokio::task::block_in_place(|| {
                let opts = embed_loader::EmbedderOptions {
                    kind: embedder,
                    model_path,
                    openai_model,
                    openai_base_url,
                    ollama_model,
                    ollama_url,
                };
                let embedder = embed_loader::load_for_read(&corpus, &opts).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
                let filter_expr: Option<fastrag::filter::FilterExpr> =
                    match (filter.as_deref(), filter_json.as_deref()) {
                        (Some(s), None) => Some(args::parse_filter_expr(s).unwrap_or_else(|e| {
                            eprintln!("Error parsing --filter: {e}");
                            std::process::exit(2);
                        })),
                        (None, Some(j)) => Some(serde_json::from_str(j).unwrap_or_else(|e| {
                            eprintln!("Error parsing --filter-json: {e}");
                            std::process::exit(2);
                        })),
                        _ => None,
                    };

                #[cfg(feature = "rerank")]
                let use_rerank = !no_rerank;
                #[cfg(not(feature = "rerank"))]
                let use_rerank = false;

                let _ = dense_only; // hybrid removed; dense-only is the only path

                let result = if use_rerank {
                    #[cfg(feature = "rerank")]
                    {
                        let reranker = fastrag_cli::rerank_loader::load_reranker(rerank)
                            .unwrap_or_else(|e| {
                                eprintln!("Error loading reranker: {e}");
                                std::process::exit(1);
                            });
                        ops::query_corpus_reranked(
                            &corpus,
                            &query,
                            top_k,
                            rerank_over_fetch,
                            embedder.as_ref() as &dyn DynEmbedderTrait,
                            reranker.as_ref(),
                            filter_expr.as_ref(),
                            &mut fastrag::corpus::LatencyBreakdown::default(),
                        )
                    }
                    #[cfg(not(feature = "rerank"))]
                    {
                        unreachable!()
                    }
                } else {
                    ops::query_corpus_with_filter(
                        &corpus,
                        &query,
                        top_k,
                        embedder.as_ref() as &dyn DynEmbedderTrait,
                        filter_expr.as_ref(),
                        &mut fastrag::corpus::LatencyBreakdown::default(),
                    )
                };

                match result {
                    Ok(hits) => print_query_results(&hits, format),
                    Err(e) => {
                        eprintln!("Error querying corpus {}: {e}", corpus.display());
                        std::process::exit(1);
                    }
                }
            });
        }
        #[cfg(feature = "retrieval")]
        Command::CorpusInfo {
            corpus,
            model_path,
            embedder,
            openai_model,
            openai_base_url,
            ollama_model,
            ollama_url,
        } => {
            tokio::task::block_in_place(|| {
                let opts = embed_loader::EmbedderOptions {
                    kind: embedder,
                    model_path,
                    openai_model,
                    openai_base_url,
                    ollama_model,
                    ollama_url,
                };
                let emb = embed_loader::load_for_read(&corpus, &opts).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
                match ops::corpus_info(&corpus, emb.as_ref() as &dyn DynEmbedderTrait) {
                    Ok(info) => {
                        println!("{}", serde_json::to_string_pretty(&info).unwrap());
                        #[cfg(feature = "contextual")]
                        print_contextualizer_info(&corpus, &info);
                    }
                    Err(e) => {
                        eprintln!("Error reading corpus {}: {e}", corpus.display());
                        std::process::exit(1);
                    }
                }
            });
        }
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
            gold_set,
            corpus,
            corpus_no_contextual,
            config_matrix,
            variants,
            baseline,
        } => {
            if config_matrix {
                if let Err(e) = eval_matrix::run_config_matrix(
                    gold_set,
                    corpus,
                    corpus_no_contextual,
                    report,
                    top_k,
                    baseline,
                    variants,
                ) {
                    eprintln!("Error running eval matrix: {e}");
                    std::process::exit(1);
                }
            } else if let Err(e) = eval::run_eval(
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
            embedder,
            openai_model,
            openai_base_url,
            ollama_model,
            ollama_url,
            token,
            dense_only,
            #[cfg(feature = "rerank")]
            rerank,
            #[cfg(feature = "rerank")]
            no_rerank,
            #[cfg(feature = "rerank")]
            rerank_over_fetch,
        } => {
            let token = token.or_else(|| std::env::var("FASTRAG_TOKEN").ok());
            let opts = embed_loader::EmbedderOptions {
                kind: embedder,
                model_path,
                openai_model,
                openai_base_url,
                ollama_model,
                ollama_url,
            };
            let embedder = embed_loader::load_for_read(&corpus, &opts).unwrap_or_else(|e| {
                eprintln!("Error loading embedder: {e}");
                std::process::exit(1);
            });

            let mut rerank_cfg = fastrag_cli::http::HttpRerankerConfig::default();

            #[cfg(feature = "rerank")]
            {
                if !no_rerank {
                    let reranker = fastrag_cli::rerank_loader::load_reranker(rerank)
                        .unwrap_or_else(|e| {
                            eprintln!("Error loading reranker: {e}");
                            std::process::exit(1);
                        });
                    rerank_cfg.reranker = Some(std::sync::Arc::from(reranker));
                }
                rerank_cfg.over_fetch = rerank_over_fetch;
            }

            if let Err(e) =
                fastrag_cli::http::serve_http(corpus, port, embedder, token, dense_only, rerank_cfg)
                    .await
            {
                eprintln!("Error starting HTTP server: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "store")]
        Command::Delete { corpus, id } => {
            tokio::task::block_in_place(|| {
                let opts = embed_loader::EmbedderOptions {
                    kind: None,
                    model_path: None,
                    openai_model: String::new(),
                    openai_base_url: String::new(),
                    ollama_model: String::new(),
                    ollama_url: String::new(),
                };
                let embedder = embed_loader::load_for_read(&corpus, &opts).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
                let mut store = fastrag_store::Store::open(
                    &corpus,
                    embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening corpus: {e}");
                    std::process::exit(1);
                });
                let deleted = store.delete_by_external_id(&id).unwrap_or_else(|e| {
                    eprintln!("Error deleting: {e}");
                    std::process::exit(1);
                });
                store.save().unwrap_or_else(|e| {
                    eprintln!("Error saving: {e}");
                    std::process::exit(1);
                });
                println!("Deleted {} chunks for external ID '{}'", deleted.len(), id);
            });
        }
        #[cfg(feature = "store")]
        Command::Compact { corpus } => {
            tokio::task::block_in_place(|| {
                let opts = embed_loader::EmbedderOptions {
                    kind: None,
                    model_path: None,
                    openai_model: String::new(),
                    openai_base_url: String::new(),
                    ollama_model: String::new(),
                    ollama_url: String::new(),
                };
                let embedder = embed_loader::load_for_read(&corpus, &opts).unwrap_or_else(|e| {
                    eprintln!("Error loading embedder: {e}");
                    std::process::exit(1);
                });
                let mut store = fastrag_store::Store::open(
                    &corpus,
                    embedder.as_ref() as &dyn fastrag::DynEmbedderTrait,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening corpus: {e}");
                    std::process::exit(1);
                });
                let before = store.tombstone_count();
                store.compact();
                store.save().unwrap_or_else(|e| {
                    eprintln!("Error saving: {e}");
                    std::process::exit(1);
                });
                println!(
                    "Compacted: purged {} tombstones, {} live entries",
                    before,
                    store.live_count()
                );
            });
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

#[cfg(feature = "store")]
fn parse_metadata_types(
    types: &[String],
) -> std::collections::BTreeMap<String, fastrag_store::schema::TypedKind> {
    let mut map = std::collections::BTreeMap::new();
    for entry in types {
        if let Some((k, v)) = entry.split_once('=') {
            let kind = match v {
                "string" => fastrag_store::schema::TypedKind::String,
                "numeric" => fastrag_store::schema::TypedKind::Numeric,
                "bool" => fastrag_store::schema::TypedKind::Bool,
                "date" => fastrag_store::schema::TypedKind::Date,
                _ => continue,
            };
            map.insert(k.to_string(), kind);
        }
    }
    map
}

#[cfg(feature = "contextual")]
fn print_contextualizer_info(corpus: &Path, info: &fastrag::corpus::CorpusInfo) {
    println!();
    match info.manifest.contextualizer.as_ref() {
        Some(c) => {
            println!("contextualized: true");
            println!("  model_id:       {}", c.model_id);
            println!("  prompt_version: {}", c.prompt_version);
            println!("  prompt_hash:    {}", c.prompt_hash);
            let cache_path = corpus.join("contextualization.sqlite");
            if cache_path.exists() {
                match fastrag_context::ContextCache::open(&cache_path).and_then(|c| c.row_count()) {
                    Ok((ok, failed)) => {
                        println!("  ok:    {ok}");
                        println!("  failed: {failed}");
                    }
                    Err(e) => {
                        println!("  cached:         ERROR — {e}");
                    }
                }
            }
        }
        None => {
            println!("contextualized: false");
        }
    }
}

#[cfg(feature = "retrieval")]
fn print_query_results(dtos: &[fastrag::corpus::SearchHitDto], format: OutputFormatArg) {
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
