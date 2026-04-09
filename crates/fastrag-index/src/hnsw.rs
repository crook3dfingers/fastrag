use instant_distance::{Builder, HnswMap, Point, Search};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use crate::entry::{IndexEntry, SearchHit};
use crate::error::{IndexError, IndexResult};
use fastrag_embed::DynEmbedderTrait;

#[derive(Clone, Serialize, Deserialize)]
struct VectorPoint {
    vector: Vec<f32>,
}

impl Point for VectorPoint {
    fn distance(&self, other: &Self) -> f32 {
        euclidean_distance(&self.vector, &other.vector)
    }
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    dim: usize,
    manifest: crate::manifest::CorpusManifest,
    entries: Vec<IndexEntry>,
    graph: HnswMap<VectorPoint, usize>,
}

impl HnswIndex {
    pub fn new(manifest: crate::manifest::CorpusManifest) -> Self {
        let dim = manifest.identity.dim;
        let graph = Builder::default().build(Vec::<VectorPoint>::new(), Vec::<usize>::new());
        Self {
            dim,
            manifest,
            entries: Vec::new(),
            graph,
        }
    }

    fn rebuild_graph(&mut self) {
        let points = self
            .entries
            .iter()
            .map(|entry| VectorPoint {
                vector: normalize(entry.vector.clone()),
            })
            .collect::<Vec<_>>();
        let values = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        self.graph = Builder::default().build(points, values);
    }

    fn validate_vector(&self, vector: &[f32]) -> IndexResult<()> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    fn entry_by_index(&self, idx: usize) -> Option<&IndexEntry> {
        self.entries.get(idx)
    }

    fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("manifest.json")
    }

    fn index_path(dir: &Path) -> PathBuf {
        dir.join("index.bin")
    }

    fn entries_path(dir: &Path) -> PathBuf {
        dir.join("entries.bin")
    }

    pub fn manifest(&self) -> &crate::manifest::CorpusManifest {
        &self.manifest
    }

    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }

    pub fn replace_manifest(&mut self, manifest: crate::manifest::CorpusManifest) {
        self.manifest = manifest;
    }

    /// Remove all entries whose `id` is in `ids` and rebuild the graph.
    /// O(n) in total entries; cheap since add() already rebuilds per call.
    pub fn remove_by_chunk_ids(&mut self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let set: std::collections::HashSet<u64> = ids.iter().copied().collect();
        self.entries.retain(|e| !set.contains(&e.id));
        self.manifest.chunk_count = self.entries.len();
        self.rebuild_graph();
    }

    pub fn load(dir: &Path, embedder: &dyn DynEmbedderTrait) -> IndexResult<Self> {
        use fastrag_embed::{CANARY_COSINE_TOLERANCE, CANARY_TEXT, PassageText};

        let manifest_bytes =
            std::fs::read(Self::manifest_path(dir)).map_err(|_| IndexError::MissingCorpusFile {
                path: Self::manifest_path(dir),
            })?;

        let manifest: crate::manifest::CorpusManifest = serde_json::from_slice(&manifest_bytes)?;

        if manifest.version != 3 {
            return Err(IndexError::UnsupportedSchema {
                got: manifest.version,
            });
        }

        let live = embedder.identity();
        if live != manifest.identity {
            return Err(IndexError::IdentityMismatch {
                existing: manifest.identity.model_id.clone(),
                existing_dim: manifest.identity.dim,
                requested: live.model_id,
                requested_dim: live.dim,
            });
        }

        let reembedded = embedder
            .embed_passage_dyn(&[PassageText::new(CANARY_TEXT)])
            .map_err(|e| IndexError::CanaryEmbed(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| IndexError::CanaryEmbed("empty output".into()))?;

        let cosine = cosine_similarity(&reembedded, &manifest.canary.vector);
        if cosine < CANARY_COSINE_TOLERANCE {
            return Err(IndexError::CanaryMismatch {
                cosine,
                tolerance: CANARY_COSINE_TOLERANCE,
            });
        }

        let graph_bytes =
            std::fs::read(Self::index_path(dir)).map_err(|_| IndexError::MissingCorpusFile {
                path: Self::index_path(dir),
            })?;
        let entries_bytes =
            std::fs::read(Self::entries_path(dir)).map_err(|_| IndexError::MissingCorpusFile {
                path: Self::entries_path(dir),
            })?;

        let graph: HnswMap<VectorPoint, usize> = bincode::deserialize(&graph_bytes)?;
        let entries: Vec<IndexEntry> = bincode::deserialize(&entries_bytes)?;

        if manifest.identity.dim == 0 {
            return Err(IndexError::CorruptCorpus {
                message: "manifest dim cannot be 0".to_string(),
            });
        }
        if entries.len() != manifest.chunk_count {
            return Err(IndexError::CorruptCorpus {
                message: format!(
                    "manifest chunk count {} does not match entries {}",
                    manifest.chunk_count,
                    entries.len()
                ),
            });
        }

        let dim = manifest.identity.dim;
        Ok(Self {
            dim,
            manifest,
            entries,
            graph,
        })
    }
}

impl crate::VectorIndex for HnswIndex {
    fn add(&mut self, entries: Vec<IndexEntry>) -> IndexResult<()> {
        for entry in &entries {
            self.validate_vector(&entry.vector)?;
        }
        self.entries.extend(entries.into_iter().map(|mut entry| {
            entry.vector = normalize(entry.vector);
            entry
        }));
        self.manifest.chunk_count = self.entries.len();
        self.rebuild_graph();
        Ok(())
    }

    fn query(&self, vector: &[f32], top_k: usize) -> IndexResult<Vec<SearchHit>> {
        self.validate_vector(vector)?;
        if top_k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_query = normalize(vector.to_vec());
        let query = VectorPoint {
            vector: normalized_query.clone(),
        };
        let mut search = Search::default();
        let mut hits = Vec::new();

        for result in self.graph.search(&query, &mut search).take(top_k) {
            let entry = self
                .entry_by_index(*result.value)
                .ok_or_else(|| IndexError::CorruptCorpus {
                    message: format!("missing entry at index {}", result.value),
                })?
                .clone();
            let score = cosine_similarity(&normalized_query, &result.point.vector);
            hits.push(SearchHit { entry, score });
        }

        hits.sort_by(
            |a, b| match b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal) {
                Ordering::Equal => a.entry.id.cmp(&b.entry.id),
                ord => ord,
            },
        );
        Ok(hits)
    }

    fn save(&self, dir: &Path) -> IndexResult<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            Self::manifest_path(dir),
            serde_json::to_vec_pretty(&self.manifest)?,
        )?;
        std::fs::write(Self::index_path(dir), bincode::serialize(&self.graph)?)?;
        std::fs::write(Self::entries_path(dir), bincode::serialize(&self.entries)?)?;
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IndexEntry, VectorIndex};
    use fastrag_core::ElementKind;
    use fastrag_embed::{
        CANARY_TEXT, Canary, DynEmbedderTrait, Embedder, EmbedderIdentity, PassageText,
        PrefixScheme, test_utils::MockEmbedder,
    };
    use tempfile::tempdir;

    fn mock_identity() -> EmbedderIdentity {
        EmbedderIdentity {
            model_id: MockEmbedder::MODEL_ID.into(),
            dim: MockEmbedder::DIM,
            prefix_scheme_hash: PrefixScheme::NONE.hash(),
        }
    }

    fn mock_canary(e: &MockEmbedder) -> Canary {
        let v = e
            .embed_passage(&[PassageText::new(CANARY_TEXT)])
            .unwrap()
            .remove(0);
        Canary {
            text_version: 1,
            vector: v,
        }
    }

    fn manifest() -> crate::manifest::CorpusManifest {
        let e = MockEmbedder;
        crate::manifest::CorpusManifest::new(
            mock_identity(),
            mock_canary(&e),
            1_700_000_000,
            crate::manifest::ManifestChunkingStrategy::Basic {
                max_characters: 1000,
                overlap: 0,
            },
        )
    }

    fn entry(id: u64, vector: Vec<f32>, text: &str) -> IndexEntry {
        IndexEntry {
            id,
            vector,
            chunk_text: text.to_string(),
            source_path: PathBuf::from(format!("/tmp/{id}.txt")),
            chunk_index: id as usize,
            section: Some("section".to_string()),
            element_kinds: vec![ElementKind::Paragraph],
            pages: vec![1],
            language: Some("en".to_string()),
            metadata: std::collections::BTreeMap::new(),
        }
    }

    #[test]
    fn add_then_query_returns_exact_match() {
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![
                entry(
                    1,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "a",
                ),
                entry(
                    2,
                    vec![
                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "b",
                ),
                entry(
                    3,
                    vec![
                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "c",
                ),
            ])
            .unwrap();
        let hits = index
            .query(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                1,
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.id, 1);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn remove_by_chunk_ids_drops_matching_entries_and_rebuilds() {
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![
                entry(
                    1,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "a",
                ),
                entry(
                    2,
                    vec![
                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "b",
                ),
                entry(
                    3,
                    vec![
                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "c",
                ),
            ])
            .unwrap();

        index.remove_by_chunk_ids(&[2]);
        assert_eq!(index.len(), 2);
        let ids: Vec<u64> = index.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 3]);

        let hits = index
            .query(
                &[
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                3,
            )
            .unwrap();
        assert!(hits.iter().all(|h| h.entry.id != 2));
    }

    #[test]
    fn query_ranks_by_similarity() {
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![
                entry(
                    1,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "a",
                ),
                entry(
                    2,
                    vec![
                        0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "b",
                ),
                entry(
                    3,
                    vec![
                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "c",
                ),
            ])
            .unwrap();
        let hits = index
            .query(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                3,
            )
            .unwrap();
        assert_eq!(
            hits.iter().map(|h| h.entry.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn save_load_roundtrip() {
        let e = MockEmbedder;
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![
                entry(
                    1,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "a",
                ),
                entry(
                    2,
                    vec![
                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "b",
                ),
            ])
            .unwrap();
        let dir = tempdir().unwrap();
        index.save(dir.path()).unwrap();
        let loaded = HnswIndex::load(dir.path(), &e as &dyn DynEmbedderTrait).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.manifest(), index.manifest());
        assert_eq!(
            loaded
                .query(
                    &[
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0
                    ],
                    1
                )
                .unwrap()[0]
                .entry
                .id,
            1
        );
    }

    #[test]
    fn metadata_preserved() {
        let e = MockEmbedder;
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![entry(
                1,
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                "a",
            )])
            .unwrap();
        let dir = tempdir().unwrap();
        index.save(dir.path()).unwrap();
        let loaded = HnswIndex::load(dir.path(), &e as &dyn DynEmbedderTrait).unwrap();
        let hit = &loaded
            .query(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                1,
            )
            .unwrap()[0];
        assert_eq!(hit.entry.source_path, PathBuf::from("/tmp/1.txt"));
        assert_eq!(hit.entry.chunk_index, 1);
        assert_eq!(hit.entry.pages, vec![1]);
        assert_eq!(hit.entry.element_kinds, vec![ElementKind::Paragraph]);
        assert_eq!(hit.entry.language.as_deref(), Some("en"));
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut index = HnswIndex::new(manifest());
        let err = index
            .add(vec![entry(1, vec![1.0, 0.0], "bad")])
            .unwrap_err();
        match err {
            IndexError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 16);
                assert_eq!(got, 2);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn query_order_is_deterministic_when_scores_tie() {
        let mut index = HnswIndex::new(manifest());
        index
            .add(vec![
                entry(
                    2,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "b",
                ),
                entry(
                    1,
                    vec![
                        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    "a",
                ),
            ])
            .unwrap();

        let hits = index
            .query(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                2,
            )
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].entry.id, 1);
        assert_eq!(hits[1].entry.id, 2);
        assert!((hits[0].score - hits[1].score).abs() < 1e-6);
    }
}

#[cfg(test)]
mod canary_tests {
    use super::*;
    use crate::VectorIndex;
    use crate::manifest::ManifestChunkingStrategy;
    use fastrag_embed::{
        CANARY_COSINE_TOLERANCE, CANARY_TEXT, Canary, DynEmbedderTrait, Embedder, EmbedderIdentity,
        PassageText, PrefixScheme, test_utils::MockEmbedder,
    };
    use tempfile::tempdir;

    fn mock_identity() -> EmbedderIdentity {
        EmbedderIdentity {
            model_id: MockEmbedder::MODEL_ID.into(),
            dim: MockEmbedder::DIM,
            prefix_scheme_hash: PrefixScheme::NONE.hash(),
        }
    }

    fn mock_canary(e: &MockEmbedder) -> Canary {
        let v = e
            .embed_passage(&[PassageText::new(CANARY_TEXT)])
            .unwrap()
            .remove(0);
        Canary {
            text_version: 1,
            vector: v,
        }
    }

    #[test]
    fn load_rejects_mismatched_identity() {
        let dir = tempdir().unwrap();
        let e = MockEmbedder;
        let manifest = crate::manifest::CorpusManifest::new(
            mock_identity(),
            mock_canary(&e),
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 10,
                overlap: 0,
            },
        );
        let idx = HnswIndex::new(manifest);
        idx.save(dir.path()).unwrap();

        struct Bogus;
        impl Embedder for Bogus {
            const DIM: usize = 16;
            const MODEL_ID: &'static str = "fastrag/bogus-v1";
            const PREFIX_SCHEME: PrefixScheme = PrefixScheme::NONE;
            fn embed_query(
                &self,
                texts: &[fastrag_embed::QueryText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                Ok(texts.iter().map(|_| vec![0.0; 16]).collect())
            }
            fn embed_passage(
                &self,
                texts: &[PassageText],
            ) -> Result<Vec<Vec<f32>>, fastrag_embed::EmbedError> {
                Ok(texts.iter().map(|_| vec![0.0; 16]).collect())
            }
        }

        let result = HnswIndex::load(dir.path(), &Bogus as &dyn DynEmbedderTrait);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected identity mismatch error"),
        };
        match err {
            IndexError::IdentityMismatch { .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn load_rejects_canary_drift() {
        let dir = tempdir().unwrap();
        let e = MockEmbedder;
        let mut wrong_canary = mock_canary(&e);
        for v in wrong_canary.vector.iter_mut() {
            *v = 0.0;
        }
        let manifest = crate::manifest::CorpusManifest::new(
            mock_identity(),
            wrong_canary,
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 10,
                overlap: 0,
            },
        );
        let idx = HnswIndex::new(manifest);
        idx.save(dir.path()).unwrap();

        let result = HnswIndex::load(dir.path(), &e as &dyn DynEmbedderTrait);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected canary mismatch error"),
        };
        assert!(matches!(err, IndexError::CanaryMismatch { .. }));
        let _ = CANARY_COSINE_TOLERANCE;
    }

    #[test]
    fn load_accepts_matching_identity_and_canary() {
        let dir = tempdir().unwrap();
        let e = MockEmbedder;
        let manifest = crate::manifest::CorpusManifest::new(
            mock_identity(),
            mock_canary(&e),
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 10,
                overlap: 0,
            },
        );
        let idx = HnswIndex::new(manifest);
        idx.save(dir.path()).unwrap();

        HnswIndex::load(dir.path(), &e as &dyn DynEmbedderTrait)
            .map(|_| ())
            .unwrap();
    }
}
