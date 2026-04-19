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
    if enabled && model == "mixedbread-ai/mxbai-embed-large-v1" {
        PrefixConfig {
            query: "Represent this sentence for searching relevant passages: ".to_string(),
            passage: String::new(),
        }
    } else {
        PrefixConfig::default()
    }
}
