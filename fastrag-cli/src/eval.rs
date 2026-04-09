use std::path::PathBuf;

use fastrag_eval::{EvalDataset, Runner, load_by_name};

use fastrag_cli::args::{EvalChunkingArg, EvalDatasetNameArg, EvalEmbedderArg};
use fastrag_embed::DynEmbedderTrait;

#[allow(clippy::too_many_arguments)]
pub async fn run_eval(
    dataset_path: Option<PathBuf>,
    dataset_name: Option<EvalDatasetNameArg>,
    report: PathBuf,
    embedder: EvalEmbedderArg,
    top_k: usize,
    chunking: EvalChunkingArg,
    chunk_size: usize,
    chunk_overlap: usize,
    max_rss_mb: Option<u64>,
    max_docs: Option<usize>,
    max_queries: Option<usize>,
) -> Result<(), fastrag_eval::EvalError> {
    let mut dataset = match (dataset_path, dataset_name) {
        (Some(path), None) => EvalDataset::load(&path)?,
        (None, Some(name)) => load_by_name(name.to_eval())?,
        _ => {
            return Err(fastrag_eval::EvalError::MalformedDataset(
                "exactly one of --dataset or --dataset-name must be provided".to_string(),
            ));
        }
    };

    // Optional sampling. Used for v0-phase1 baselines so the run completes in minutes
    // on a CPU embedder; full-dataset baselines are tracked separately.
    if let Some(n) = max_docs {
        dataset.documents.truncate(n);
        let kept: std::collections::HashSet<&str> =
            dataset.documents.iter().map(|d| d.id.as_str()).collect();
        dataset.qrels.retain(|q| kept.contains(q.doc_id.as_str()));
    }
    if let Some(n) = max_queries {
        dataset.queries.truncate(n);
        let kept: std::collections::HashSet<&str> =
            dataset.queries.iter().map(|q| q.id.as_str()).collect();
        dataset.qrels.retain(|q| kept.contains(q.query_id.as_str()));
    }
    let embedder: Box<dyn DynEmbedderTrait> = match embedder {
        EvalEmbedderArg::Mock => Box::new(fastrag_embed::test_utils::MockEmbedder),
        EvalEmbedderArg::BgeSmall => Box::new(fastrag_embed::BgeSmallEmbedder::from_hf_hub()?),
    };
    let chunking = match chunking {
        EvalChunkingArg::Basic => fastrag::ChunkingStrategy::Basic {
            max_characters: chunk_size,
            overlap: chunk_overlap,
        },
        EvalChunkingArg::ByTitle => fastrag::ChunkingStrategy::ByTitle {
            max_characters: chunk_size,
            overlap: chunk_overlap,
        },
        EvalChunkingArg::Recursive => fastrag::ChunkingStrategy::RecursiveCharacter {
            max_characters: chunk_size,
            overlap: chunk_overlap,
            separators: fastrag::default_separators(),
        },
    };

    let report_value = Runner::new(embedder.as_ref(), chunking, &dataset, top_k).run()?;
    print_report(&report_value);
    report_value.write_json(&report)?;
    println!("Wrote report JSON to {}", report.display());

    if let Some(limit_mb) = max_rss_mb {
        let actual_mb = report_value.memory.peak_rss_bytes / (1024 * 1024);
        if actual_mb > limit_mb {
            return Err(fastrag_eval::EvalError::MalformedDataset(format!(
                "peak RSS {actual_mb} MB exceeded --max-rss-mb {limit_mb}"
            )));
        }
    }
    Ok(())
}

fn print_report(report: &fastrag_eval::EvalReport) {
    println!();
    println!("| Field | Value |");
    println!("| --- | ---: |");
    println!("| dataset | {} |", report.dataset);
    println!("| embedder | {} |", report.embedder);
    println!("| chunking | {} |", report.chunking);
    println!("| build_time_ms | {} |", report.build_time_ms);
    println!("| run_at_unix | {} |", report.run_at_unix);
    println!("| peak_rss_bytes | {} |", report.memory.peak_rss_bytes);
    println!(
        "| current_rss_bytes | {} |",
        report.memory.current_rss_bytes
    );
    println!("| p50_ms | {:.6} |", report.latency.p50_ms);
    println!("| p95_ms | {:.6} |", report.latency.p95_ms);
    println!("| p99_ms | {:.6} |", report.latency.p99_ms);
    println!("| mean_ms | {:.6} |", report.latency.mean_ms);
    println!("| count | {} |", report.latency.count);

    let mut metrics = report.metrics.iter().collect::<Vec<_>>();
    metrics.sort_by(|a, b| a.0.cmp(b.0));
    for (name, value) in metrics {
        println!("| {} | {:.6} |", name, value);
    }
}
