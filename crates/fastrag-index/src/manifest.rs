use fastrag_embed::{Canary, EmbedderIdentity};
use serde::{Deserialize, Serialize};

/// Persisted corpus metadata stored in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    pub version: u32,
    pub identity: EmbedderIdentity,
    pub canary: Canary,
    pub created_at_unix_seconds: u64,
    pub chunk_count: usize,
    pub chunking_strategy: ManifestChunkingStrategy,
    #[serde(default)]
    pub roots: Vec<RootEntry>,
    #[serde(default)]
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootEntry {
    pub id: u32,
    pub path: std::path::PathBuf,
    pub last_indexed_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub root_id: u32,
    pub rel_path: std::path::PathBuf,
    pub size: u64,
    pub mtime_ns: i128,
    pub content_hash: Option<String>,
    pub chunk_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ManifestChunkingStrategy {
    Basic {
        max_characters: usize,
        overlap: usize,
    },
    ByTitle {
        max_characters: usize,
        overlap: usize,
    },
    RecursiveCharacter {
        max_characters: usize,
        overlap: usize,
        separators: Vec<String>,
    },
    Semantic {
        max_characters: usize,
        similarity_threshold: Option<f32>,
        percentile_threshold: Option<f32>,
    },
}

impl CorpusManifest {
    pub fn new(
        identity: EmbedderIdentity,
        canary: Canary,
        created_at_unix_seconds: u64,
        chunking_strategy: ManifestChunkingStrategy,
    ) -> Self {
        Self {
            version: 3,
            identity,
            canary,
            created_at_unix_seconds,
            chunk_count: 0,
            chunking_strategy,
            roots: Vec::new(),
            files: Vec::new(),
        }
    }
}

#[cfg(test)]
mod v3_tests {
    use super::*;
    use fastrag_embed::{Canary, EmbedderIdentity, PrefixScheme};

    fn sample_identity() -> EmbedderIdentity {
        EmbedderIdentity {
            model_id: "fastrag/mock-embedder-16d-v1".into(),
            dim: 16,
            prefix_scheme_hash: PrefixScheme::NONE.hash(),
        }
    }

    fn sample_canary() -> Canary {
        Canary {
            text_version: 1,
            vector: vec![0.0; 16],
        }
    }

    #[test]
    fn v3_roundtrip() {
        let m = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        assert_eq!(m.version, 3);
        assert_eq!(m.identity.dim, 16);
        assert_eq!(m.canary.vector.len(), 16);
        let s = serde_json::to_string(&m).unwrap();
        let back: CorpusManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn v1_manifest_is_rejected_as_unsupported() {
        let v1 = r#"{
            "version": 1,
            "embedding_model_id": "mock",
            "dim": 3,
            "created_at_unix_seconds": 1,
            "chunk_count": 0,
            "chunking_strategy": {"kind":"basic","max_characters":100,"overlap":0}
        }"#;
        let err = serde_json::from_str::<CorpusManifest>(v1);
        assert!(err.is_err(), "v1 manifests must not deserialize as v3");
    }
}
