use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use thiserror::Error;

use crate::embed_profile::{EmbedBackend, PrefixConfig, ResolvedEmbedderProfile};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found")]
    NotFound,
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config file: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("unknown embedder profile `{profile}`")]
    UnknownProfile { profile: String },
    #[error("missing default embedder profile")]
    MissingDefaultProfile,
    #[error("missing model for embedder profile `{profile}` on backend `{backend:?}`")]
    MissingModel {
        profile: String,
        backend: EmbedBackend,
    },
    #[error(
        "embedder profile `{profile}` on backend `{backend:?}` does not accept CLI override `{key}`"
    )]
    IncompatibleOverride {
        profile: String,
        backend: EmbedBackend,
        key: String,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub embedder: EmbedderSection,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EmbedderSection {
    #[serde(default)]
    pub default_profile: String,
    #[serde(default)]
    pub profiles: BTreeMap<String, EmbedderProfileConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedderProfileConfig {
    pub backend: EmbedBackend,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub use_catalog_defaults: bool,
    #[serde(default)]
    pub dim_override: Option<usize>,
    #[serde(default)]
    pub prefix: Option<PrefixSection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PrefixSection {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub passage: Option<String>,
}

pub fn default_config_path() -> Option<PathBuf> {
    let local = PathBuf::from("fastrag.toml");
    if local.exists() {
        return Some(local);
    }

    dirs::config_dir().and_then(|dir| {
        let candidate = dir.join("fastrag").join("fastrag.toml");
        candidate.exists().then_some(candidate)
    })
}

pub fn load_app_config(explicit: Option<PathBuf>) -> Result<AppConfig, ConfigError> {
    let path = match explicit {
        Some(path) => path,
        None => default_config_path().ok_or(ConfigError::NotFound)?,
    };

    let contents = fs::read_to_string(&path)?;
    Ok(toml::from_str(&contents)?)
}

impl AppConfig {
    pub fn resolve_embedder_profile(
        &self,
        selected: Option<&str>,
        cli_overrides: &[(&str, &str)],
    ) -> Result<ResolvedEmbedderProfile, ConfigError> {
        let profile_name = match selected.filter(|name| !name.is_empty()) {
            Some(name) => name,
            None => {
                let default_profile = self.embedder.default_profile.trim();
                if default_profile.is_empty() {
                    return Err(ConfigError::MissingDefaultProfile);
                }
                default_profile
            }
        };

        let profile = self.embedder.profiles.get(profile_name).ok_or_else(|| {
            ConfigError::UnknownProfile {
                profile: profile_name.to_string(),
            }
        })?;

        let model = profile
            .model
            .clone()
            .ok_or_else(|| ConfigError::MissingModel {
                profile: profile_name.to_string(),
                backend: profile.backend,
            })?;

        let mut base_url = profile.base_url.clone();
        for (key, value) in cli_overrides {
            if *key == "ollama_url" && profile.backend == EmbedBackend::Ollama {
                base_url = Some((*value).to_string());
            } else if *key == "openai_base_url" && profile.backend == EmbedBackend::Openai {
                base_url = Some((*value).to_string());
            } else if *key == "ollama_url" || *key == "openai_base_url" {
                return Err(ConfigError::IncompatibleOverride {
                    profile: profile_name.to_string(),
                    backend: profile.backend,
                    key: (*key).to_string(),
                });
            }
        }

        let mut prefix = catalog_prefix_defaults(&model, profile.use_catalog_defaults);
        if let Some(profile_prefix) = &profile.prefix {
            if let Some(query) = &profile_prefix.query {
                prefix.query = query.clone();
            }
            if let Some(passage) = &profile_prefix.passage {
                prefix.passage = passage.clone();
            }
        }

        Ok(ResolvedEmbedderProfile {
            name: profile_name.to_string(),
            backend: profile.backend,
            model,
            base_url,
            prefix,
            dim_override: profile.dim_override,
        })
    }
}

pub fn catalog_prefix_defaults(model: &str, enabled: bool) -> PrefixConfig {
    if !enabled {
        return PrefixConfig::default();
    }
    match model {
        "mixedbread-ai/mxbai-embed-large-v1" => PrefixConfig {
            query: "Represent this sentence for searching relevant passages: ".to_string(),
            passage: String::new(),
        },
        "nomic-ai/nomic-embed-text-v1.5" | "nomic-embed-text" => PrefixConfig {
            query: "search_query: ".to_string(),
            passage: "search_document: ".to_string(),
        },
        _ => PrefixConfig::default(),
    }
}

#[cfg(test)]
mod catalog_defaults_tests {
    use super::*;

    #[test]
    fn mxbai_unchanged_when_enabled() {
        let cfg = catalog_prefix_defaults("mixedbread-ai/mxbai-embed-large-v1", true);
        assert_eq!(
            cfg.query,
            "Represent this sentence for searching relevant passages: "
        );
        assert_eq!(cfg.passage, "");
    }

    #[test]
    fn nomic_v15_full_name_gets_asymmetric_prefixes() {
        let cfg = catalog_prefix_defaults("nomic-ai/nomic-embed-text-v1.5", true);
        assert_eq!(cfg.query, "search_query: ");
        assert_eq!(cfg.passage, "search_document: ");
    }

    #[test]
    fn nomic_short_alias_gets_asymmetric_prefixes() {
        let cfg = catalog_prefix_defaults("nomic-embed-text", true);
        assert_eq!(cfg.query, "search_query: ");
        assert_eq!(cfg.passage, "search_document: ");
    }

    #[test]
    fn nomic_returns_default_when_catalog_disabled() {
        let cfg = catalog_prefix_defaults("nomic-ai/nomic-embed-text-v1.5", false);
        assert_eq!(cfg, PrefixConfig::default());
    }

    #[test]
    fn unknown_model_returns_default() {
        let cfg = catalog_prefix_defaults("totally-unknown/embedder", true);
        assert_eq!(cfg, PrefixConfig::default());
    }
}
