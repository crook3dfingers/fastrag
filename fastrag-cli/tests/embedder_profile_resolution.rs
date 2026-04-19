use std::fs;
use std::path::PathBuf;

use fastrag_cli::embed_profile::{EmbedBackend, PrefixConfig, ResolvedEmbedderProfile};

fn write_config(dir: &tempfile::TempDir, contents: &str) -> PathBuf {
    let path = dir.path().join("fastrag.toml");
    fs::write(&path, contents).expect("write fastrag.toml");
    path
}

#[test]
fn resolves_default_profile_from_fastrag_toml() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "vams"

[embedder.profiles.vams]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
use_catalog_defaults = true
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let resolved = cfg
        .resolve_embedder_profile(None, &[])
        .expect("resolve profile");

    assert_eq!(resolved.name, "vams");
    assert_eq!(resolved.backend, EmbedBackend::Ollama);
    assert_eq!(resolved.model, "mixedbread-ai/mxbai-embed-large-v1");
    assert_eq!(
        resolved.prefix,
        PrefixConfig {
            query: "Represent this sentence for searching relevant passages: ".into(),
            passage: String::new(),
        }
    );
}

#[test]
fn partial_prefix_overrides_layer_on_catalog_defaults() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "vams"

[embedder.profiles.vams]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
use_catalog_defaults = true

[embedder.profiles.vams.prefix]
passage = "Represent this passage for retrieval: "
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let resolved = cfg
        .resolve_embedder_profile(None, &[])
        .expect("resolve profile");

    assert_eq!(
        resolved.prefix,
        PrefixConfig {
            query: "Represent this sentence for searching relevant passages: ".into(),
            passage: "Represent this passage for retrieval: ".into(),
        }
    );
}

#[test]
fn cli_overrides_base_url_above_config() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "vams"

[embedder.profiles.vams]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
use_catalog_defaults = true
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let resolved = cfg
        .resolve_embedder_profile(
            Some("vams"),
            &[("ollama_url", "http://ollama.internal:11434")],
        )
        .expect("resolve profile");

    assert_eq!(
        resolved.base_url.as_deref(),
        Some("http://ollama.internal:11434")
    );
}

#[test]
fn errors_when_selected_profile_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "vams"

[embedder.profiles.vams]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
use_catalog_defaults = true
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let err = cfg
        .resolve_embedder_profile(Some("missing"), &[])
        .expect_err("missing profile should fail");

    assert!(
        err.to_string().contains("unknown embedder profile"),
        "unexpected error: {err}"
    );
}

#[test]
fn errors_when_default_profile_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]

[embedder.profiles.vams]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
use_catalog_defaults = true
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let err = cfg
        .resolve_embedder_profile(None, &[])
        .expect_err("missing default profile should fail");

    assert!(
        err.to_string().contains("missing default embedder profile"),
        "unexpected error: {err}"
    );
}

#[test]
fn resolved_ollama_profile_produces_runtime_identity_shape() {
    let profile = ResolvedEmbedderProfile {
        name: "vams".into(),
        backend: EmbedBackend::Ollama,
        model: "mixedbread-ai/mxbai-embed-large-v1".into(),
        base_url: Some("http://localhost:11434".into()),
        prefix: PrefixConfig {
            query: "Represent this sentence for searching relevant passages: ".into(),
            passage: "Represent this passage for retrieval: ".into(),
        },
        dim_override: Some(1024),
    };

    let identity = fastrag_cli::embed_loader::runtime_identity_for_profile(&profile)
        .expect("runtime identity");

    assert_eq!(
        identity.model_id,
        "ollama:mixedbread-ai/mxbai-embed-large-v1"
    );
    assert_eq!(identity.dim, 1024);
    assert_ne!(identity.prefix_scheme_hash, 0);
}

#[test]
fn resolved_llama_cpp_profile_produces_runtime_identity_shape() {
    let profile = ResolvedEmbedderProfile {
        name: "local".into(),
        backend: EmbedBackend::LlamaCpp,
        model: "/models/Qwen3-Embedding-0.6B-Q8_0.gguf".into(),
        base_url: None,
        prefix: PrefixConfig {
            query: "query: ".into(),
            passage: "passage: ".into(),
        },
        dim_override: Some(1024),
    };

    let identity = fastrag_cli::embed_loader::runtime_identity_for_profile(&profile)
        .expect("runtime identity");

    assert_eq!(
        identity.model_id,
        "llama-cpp:/models/Qwen3-Embedding-0.6B-Q8_0.gguf"
    );
    assert_eq!(identity.dim, 1024);
    assert_ne!(identity.prefix_scheme_hash, 0);
}

#[test]
fn openai_profile_rejects_unsupported_prefix_override_before_loader_io() {
    let profile = ResolvedEmbedderProfile {
        name: "openai".into(),
        backend: EmbedBackend::Openai,
        model: "text-embedding-3-small".into(),
        base_url: None,
        prefix: PrefixConfig {
            query: "query: ".into(),
            passage: String::new(),
        },
        dim_override: None,
    };

    let err = match fastrag_cli::embed_loader::load_from_profile(&profile) {
        Ok(_) => panic!("openai prefix override should fail before loader startup"),
        Err(err) => err,
    };

    assert!(
        err.to_string().contains("unsupported prefix override"),
        "unexpected error: {err}"
    );
}

#[test]
fn bge_profile_rejects_unsupported_dim_override_before_loader_io() {
    let profile = ResolvedEmbedderProfile {
        name: "bge".into(),
        backend: EmbedBackend::Bge,
        model: "fastrag/bge-small-en-v1.5".into(),
        base_url: None,
        prefix: PrefixConfig::default(),
        dim_override: Some(768),
    };

    let err = match fastrag_cli::embed_loader::load_from_profile(&profile) {
        Ok(_) => panic!("bge dim override should fail before loader startup"),
        Err(err) => err,
    };

    assert!(
        err.to_string().contains("unsupported dim override"),
        "unexpected error: {err}"
    );
}

#[test]
fn openai_profile_rejects_ollama_url_override() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "openai-default"

[embedder.profiles.openai-default]
backend = "openai"
model = "text-embedding-3-small"
base_url = "https://api.openai.com/v1"
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let err = cfg
        .resolve_embedder_profile(
            Some("openai-default"),
            &[("ollama_url", "http://localhost:11434")],
        )
        .expect_err("mismatched ollama_url override should fail");

    assert!(
        err.to_string().contains("ollama_url"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("openai"),
        "unexpected error: {err}"
    );
}

#[test]
fn ollama_profile_rejects_openai_base_url_override() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config_path = write_config(
        &dir,
        r#"
[embedder]
default_profile = "ollama-default"

[embedder.profiles.ollama-default]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://localhost:11434"
"#,
    );

    let cfg = fastrag_cli::config::load_app_config(Some(config_path)).expect("load config");
    let err = cfg
        .resolve_embedder_profile(
            Some("ollama-default"),
            &[("openai_base_url", "https://api.openai.com/v1")],
        )
        .expect_err("mismatched openai_base_url override should fail");

    assert!(
        err.to_string().contains("openai_base_url"),
        "unexpected error: {err}"
    );
    assert!(
        err.to_string().contains("ollama"),
        "unexpected error: {err}"
    );
}

#[test]
fn ollama_base_url_prefers_env_host_when_profile_base_url_missing() {
    let resolved = fastrag_cli::embed_loader::resolve_ollama_base_url(
        None,
        Some("http://ollama.internal:11434"),
    );

    assert_eq!(resolved, "http://ollama.internal:11434");
}
