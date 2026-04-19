//! `--config-matrix` dispatch: wire CLI args → RealCorpusDriver → run_matrix.

use std::path::Path;
use std::path::PathBuf;

use fastrag_cli::args::RerankerKindArg;
use fastrag_cli::config::{self, load_app_config};
use fastrag_cli::embed_loader;
use fastrag_cli::rerank_loader;
use fastrag_eval::{
    EvalError,
    baseline::{diff, enforce_temporal_gates, load_baseline},
    gold_set,
    matrix::{ConfigVariant, run_matrix},
    matrix_real::RealCorpusDriver,
    write_matrix_report,
};

#[allow(clippy::too_many_arguments)]
pub fn run_config_matrix(
    gold_set_path: Option<PathBuf>,
    corpus: Option<PathBuf>,
    corpus_no_contextual: Option<PathBuf>,
    report_path: PathBuf,
    top_k: usize,
    baseline_path: Option<PathBuf>,
    variants: Option<String>,
    max_queries: Option<usize>,
) -> Result<(), EvalError> {
    let gs_path = gold_set_path.ok_or(EvalError::MatrixRequiresGoldSet)?;
    let ctx_corpus = corpus
        .ok_or_else(|| EvalError::GoldSetInvalid("--config-matrix requires --corpus".into()))?;
    let raw_corpus = corpus_no_contextual.ok_or(EvalError::MatrixMissingRawCorpus)?;

    let mut gs = gold_set::load(&gs_path)?;
    if let Some(n) = max_queries {
        gs.entries.truncate(n);
    }

    let embedder = load_embedder_for_config_matrix(&ctx_corpus)?;

    // Load reranker via llama-cpp (external process, ~1G RSS) instead of ONNX
    // (in-process, 6G+ RSS due to ORT arena — OOMs on 7G CI runners).
    let reranker = rerank_loader::load_reranker(RerankerKindArg::LlamaCpp)
        .map_err(|e| EvalError::Runner(format!("loading reranker: {e}")))?;

    let driver = RealCorpusDriver::load(
        &ctx_corpus,
        &raw_corpus,
        embedder.as_ref(),
        reranker.as_ref(),
    )
    .map_err(|e| EvalError::Runner(format!("loading corpus driver: {e}")))?;

    let variant_filter: Option<Vec<ConfigVariant>> = variants.map(|s| {
        s.split(',')
            .map(|label| {
                ConfigVariant::from_label(label.trim()).unwrap_or_else(|| {
                    panic!(
                        "unknown variant '{}'; valid: primary, no_rerank, no_contextual, dense_only, temporal_auto, temporal_oracle",
                        label.trim()
                    )
                })
            })
            .collect()
    });

    let matrix_report = run_matrix(&driver, &gs, top_k, variant_filter.as_deref())?;

    write_matrix_report(&matrix_report, &report_path)?;

    println!("Wrote matrix report to {}", report_path.display());

    if let Some(bpath) = baseline_path {
        let baseline = load_baseline(&bpath)?;
        let bdiff = diff(&matrix_report, &baseline)?;
        eprintln!("{}", bdiff.render_report());
        if bdiff.has_regressions() {
            std::process::exit(1);
        }
        enforce_temporal_gates(&matrix_report)?;
    }

    Ok(())
}

fn load_embedder_for_config_matrix(ctx_corpus: &Path) -> Result<fastrag::DynEmbedder, EvalError> {
    match load_app_config(None) {
        Ok(cfg) => {
            let resolved = cfg
                .resolve_embedder_profile(None, &[])
                .map_err(|e| EvalError::Runner(format!("resolving default embedder profile: {e}")))?;
            embed_loader::load_from_profile(&resolved)
                .map_err(|e| EvalError::Runner(format!("loading embedder from default profile: {e}")))
        }
        Err(config::ConfigError::NotFound) => {
            embed_loader::load_from_manifest(ctx_corpus).map_err(|e| {
                let msg = e.to_string();
                if msg.contains("prefix-aware embedder") {
                    EvalError::Runner(
                        "eval --config-matrix needs a usable fastrag.toml default profile for prefix-aware corpora"
                            .into(),
                    )
                } else {
                    EvalError::Runner(format!("loading embedder: {msg}"))
                }
            })
        }
        Err(e) => Err(EvalError::Runner(format!("loading fastrag.toml: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_embed::PrefixScheme;
    use std::fs;
    use std::path::Path;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn write_prefix_aware_manifest(corpus_dir: &Path) {
        fs::create_dir_all(corpus_dir).expect("create corpus dir");
        fs::write(
            corpus_dir.join("manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": 5,
                "identity": {
                    "model_id": "ollama:mixedbread-ai/mxbai-embed-large-v1",
                    "dim": 1024,
                    "prefix_scheme_hash": PrefixScheme::new("query: ", "passage: ").hash(),
                },
                "canary": {
                    "text_version": 1,
                    "vector": vec![0.0_f32; 1024],
                },
                "created_at_unix_seconds": 1,
                "chunk_count": 0,
                "chunking_strategy": {
                    "kind": "basic",
                    "max_characters": 1000,
                    "overlap": 0,
                },
                "roots": [],
                "files": [],
            }))
            .expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    #[test]
    fn config_matrix_uses_default_profile_when_config_available() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        let temp = tempfile::tempdir().expect("tempdir");
        let corpus = temp.path().join("ctx");
        write_prefix_aware_manifest(&corpus);
        fs::write(
            temp.path().join("fastrag.toml"),
            r#"
[embedder]
default_profile = "default"

[embedder.profiles.default]
backend = "openai"
model = "text-embedding-3-small"
"#,
        )
        .expect("write config");

        let prev_dir = std::env::current_dir().expect("cwd");
        let prev_openai = std::env::var("OPENAI_API_KEY").ok();
        std::env::set_current_dir(temp.path()).expect("set cwd");
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }

        let result = load_embedder_for_config_matrix(&corpus);

        std::env::set_current_dir(prev_dir).expect("restore cwd");
        match prev_openai {
            Some(value) => unsafe { std::env::set_var("OPENAI_API_KEY", value) },
            None => unsafe { std::env::remove_var("OPENAI_API_KEY") },
        }

        let err = match result {
            Ok(_) => panic!("missing OPENAI_API_KEY should fail"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("OPENAI_API_KEY"),
            "expected config-first openai failure, got: {msg}"
        );
        assert!(
            !msg.contains("prefix-aware embedder"),
            "unexpected manifest fallback error: {msg}"
        );
    }

    #[test]
    fn config_matrix_requires_default_profile_for_prefix_aware_manifest_without_config() {
        let _guard = TEST_LOCK.lock().expect("test lock");
        let temp = tempfile::tempdir().expect("tempdir");
        let corpus = temp.path().join("ctx");
        write_prefix_aware_manifest(&corpus);

        let prev_dir = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(temp.path()).expect("set cwd");

        let err = match load_embedder_for_config_matrix(&corpus) {
            Ok(_) => panic!("prefix-aware manifest without config should fail"),
            Err(err) => err,
        };

        std::env::set_current_dir(prev_dir).expect("restore cwd");

        let msg = err.to_string();
        assert!(
            msg.contains("fastrag.toml"),
            "expected actionable config guidance, got: {msg}"
        );
        assert!(
            msg.contains("default profile"),
            "expected default profile guidance, got: {msg}"
        );
    }
}
