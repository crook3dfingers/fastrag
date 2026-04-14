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
    /// Present when the corpus was ingested with contextualization enabled.
    /// Absent on corpora ingested without `--contextualize`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contextualizer: Option<ContextualizerManifest>,
    /// Name of the record field that carries the CWE numeric id. Set at
    /// ingest time via `--cwe-field`. When present, query-time CWE
    /// expansion defaults on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwe_field: Option<String>,
    /// Version string of the CWE taxonomy used when this corpus was built.
    /// Written by the ingest path when `cwe_field` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwe_taxonomy_version: Option<String>,
}

/// Metadata about the contextualizer used at ingest time. Written once per
/// corpus and never re-read at query time — contextualization is an ingest
/// stage only. The fields here exist so `corpus-info` and `doctor` can report
/// on provenance and so a mismatched model on `--retry-failed` surfaces clearly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextualizerManifest {
    pub model_id: String,
    pub prompt_version: u32,
    /// blake3 hex digest of the exact prompt template in use at ingest time.
    pub prompt_hash: String,
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
            version: 5,
            identity,
            canary,
            created_at_unix_seconds,
            chunk_count: 0,
            chunking_strategy,
            roots: Vec::new(),
            files: Vec::new(),
            contextualizer: None,
            cwe_field: None,
            cwe_taxonomy_version: None,
        }
    }
}

#[cfg(test)]
mod v5_tests {
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
    fn v5_roundtrip() {
        let m = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        assert_eq!(m.version, 5);
        assert_eq!(m.identity.dim, 16);
        assert_eq!(m.canary.vector.len(), 16);
        assert!(m.contextualizer.is_none());
        let s = serde_json::to_string(&m).unwrap();
        let back: CorpusManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn v5_with_contextualizer_roundtrip() {
        let mut m = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        m.contextualizer = Some(ContextualizerManifest {
            model_id: "qwen3-4b-instruct-2507-q4-km".to_string(),
            prompt_version: 1,
            prompt_hash: "abc123".to_string(),
        });
        let s = serde_json::to_string(&m).unwrap();
        let back: CorpusManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.contextualizer.as_ref().unwrap().prompt_version, 1);
    }

    #[test]
    fn v5_without_contextualizer_field_deserializes() {
        // An older v4-writer that didn't emit `contextualizer` at all must
        // still load — the field is optional with serde default.
        let m_ref = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        let mut value = serde_json::to_value(&m_ref).unwrap();
        // Strip the contextualizer field so we're loading what an older
        // writer would have produced.
        value.as_object_mut().unwrap().remove("contextualizer");
        let json = serde_json::to_string(&value).unwrap();
        let m: CorpusManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(m.version, 5);
        assert!(m.contextualizer.is_none());
    }

    #[test]
    fn v5_with_cwe_field_roundtrip() {
        let mut m = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        m.cwe_field = Some("cwe_id".to_string());
        m.cwe_taxonomy_version = Some("4.19.1".to_string());
        let s = serde_json::to_string(&m).unwrap();
        let back: CorpusManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.cwe_field.as_deref(), Some("cwe_id"));
        assert_eq!(back.cwe_taxonomy_version.as_deref(), Some("4.19.1"));
    }

    #[test]
    fn v5_without_cwe_fields_deserializes() {
        // Older writer without the new fields.
        let m_ref = CorpusManifest::new(
            sample_identity(),
            sample_canary(),
            1,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        );
        let mut value = serde_json::to_value(&m_ref).unwrap();
        value.as_object_mut().unwrap().remove("cwe_field");
        value
            .as_object_mut()
            .unwrap()
            .remove("cwe_taxonomy_version");
        let json = serde_json::to_string(&value).unwrap();
        let m: CorpusManifest = serde_json::from_str(&json).unwrap();
        assert!(m.cwe_field.is_none());
        assert!(m.cwe_taxonomy_version.is_none());
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
        assert!(err.is_err(), "v1 manifests must not deserialize as v5");
    }
}
