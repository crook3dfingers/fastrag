use std::fs;
use std::path::PathBuf;

use fastrag_cli::embed_profile::{EmbedBackend, PrefixConfig};

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
