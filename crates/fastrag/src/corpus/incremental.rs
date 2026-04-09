use fastrag_index::{CorpusManifest, FileEntry, RootEntry};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct WalkedFile {
    pub rel_path: PathBuf,
    pub abs_path: PathBuf,
    pub size: u64,
    pub mtime_ns: i128,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct IndexPlan {
    pub root_id: u32,
    pub unchanged: Vec<WalkedFile>,
    pub changed: Vec<WalkedFile>,
    pub new: Vec<WalkedFile>,
    pub deleted: Vec<FileEntry>,
    /// Files whose stat changed but hash matched — need mtime/size updated in manifest.
    pub touched: Vec<(FileEntry, WalkedFile)>,
}

/// Classify walked files against the manifest.
pub fn plan_index(
    root_abs: &Path,
    walked: Vec<WalkedFile>,
    manifest: &mut CorpusManifest,
    hash_file: &dyn Fn(&Path) -> std::io::Result<String>,
) -> std::io::Result<IndexPlan> {
    let root_id = resolve_root(manifest, root_abs);
    let existing: std::collections::HashMap<PathBuf, FileEntry> = manifest
        .files
        .iter()
        .filter(|f| f.root_id == root_id)
        .map(|f| (f.rel_path.clone(), f.clone()))
        .collect();

    let mut plan = IndexPlan {
        root_id,
        ..Default::default()
    };
    let mut seen_rel: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    for wf in walked {
        seen_rel.insert(wf.rel_path.clone());
        match existing.get(&wf.rel_path) {
            None => plan.new.push(wf),
            Some(existing_entry) => {
                if existing_entry.size == wf.size && existing_entry.mtime_ns == wf.mtime_ns {
                    plan.unchanged.push(wf);
                } else {
                    let h = hash_file(&wf.abs_path)?;
                    if existing_entry.content_hash.as_deref() == Some(h.as_str()) {
                        plan.touched.push((existing_entry.clone(), wf));
                    } else {
                        plan.changed.push(wf);
                    }
                }
            }
        }
    }

    for f in &manifest.files {
        if f.root_id == root_id && !seen_rel.contains(&f.rel_path) {
            plan.deleted.push(f.clone());
        }
    }

    Ok(plan)
}

/// Find or append a root entry for the given absolute path.
pub fn resolve_root(manifest: &mut CorpusManifest, abs: &Path) -> u32 {
    if let Some(r) = manifest.roots.iter().find(|r| r.path == abs) {
        return r.id;
    }
    let id = manifest
        .roots
        .iter()
        .map(|r| r.id)
        .max()
        .map_or(0, |m| m + 1);
    manifest.roots.push(RootEntry {
        id,
        path: abs.to_path_buf(),
        last_indexed_unix_seconds: 0,
    });
    id
}

/// Walk `root` via the existing `collect_files` path and produce `WalkedFile`s
/// suitable for `plan_index`. Canonicalizes `root` once.
pub fn walk_for_plan(root: &Path) -> std::io::Result<(PathBuf, Vec<WalkedFile>)> {
    let root_abs = root.canonicalize()?;
    let files = if root_abs.is_file() {
        vec![root_abs.clone()]
    } else {
        crate::ops::collect_files(&root_abs)
    };
    let mut out = Vec::with_capacity(files.len());
    for path in files {
        let rel = path.strip_prefix(&root_abs).unwrap_or(&path).to_path_buf();
        let md = std::fs::metadata(&path)?;
        let mtime_ns = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(0);
        out.push(WalkedFile {
            rel_path: rel,
            abs_path: path,
            size: md.len(),
            mtime_ns,
        });
    }
    Ok((root_abs, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_index::ManifestChunkingStrategy;

    fn empty_manifest() -> CorpusManifest {
        use fastrag_embed::{Canary, EmbedderIdentity, PrefixScheme};
        let identity = EmbedderIdentity {
            model_id: "mock".to_string(),
            dim: 3,
            prefix_scheme_hash: PrefixScheme::NONE.hash(),
        };
        let canary = Canary {
            text_version: 1,
            vector: vec![0.0f32; 3],
        };
        CorpusManifest::new(
            identity,
            canary,
            0,
            ManifestChunkingStrategy::Basic {
                max_characters: 100,
                overlap: 0,
            },
        )
    }

    fn walked(rel: &str, size: u64, mtime: i128) -> WalkedFile {
        WalkedFile {
            rel_path: rel.into(),
            abs_path: format!("/root/{rel}").into(),
            size,
            mtime_ns: mtime,
        }
    }

    fn never_hash(_: &Path) -> std::io::Result<String> {
        panic!("hash_file should not have been called for unchanged files");
    }

    #[test]
    fn all_new_when_manifest_empty() {
        let mut m = empty_manifest();
        let plan = plan_index(
            Path::new("/root"),
            vec![walked("a.txt", 10, 1), walked("b.txt", 20, 2)],
            &mut m,
            &never_hash,
        )
        .unwrap();
        assert_eq!(plan.new.len(), 2);
        assert!(plan.unchanged.is_empty() && plan.changed.is_empty() && plan.deleted.is_empty());
        assert_eq!(m.roots.len(), 1);
        assert_eq!(m.roots[0].path, Path::new("/root"));
    }

    #[test]
    fn unchanged_skips_hash_call() {
        let mut m = empty_manifest();
        m.roots.push(RootEntry {
            id: 0,
            path: "/root".into(),
            last_indexed_unix_seconds: 0,
        });
        m.files.push(FileEntry {
            root_id: 0,
            rel_path: "a.txt".into(),
            size: 10,
            mtime_ns: 1,
            content_hash: Some("blake3:xxx".into()),
            chunk_ids: vec![1],
        });

        let plan = plan_index(
            Path::new("/root"),
            vec![walked("a.txt", 10, 1)],
            &mut m,
            &never_hash,
        )
        .unwrap();
        assert_eq!(plan.unchanged.len(), 1);
        assert!(plan.changed.is_empty() && plan.new.is_empty());
    }

    #[test]
    fn touch_with_same_content_goes_to_touched_not_changed() {
        let mut m = empty_manifest();
        m.roots.push(RootEntry {
            id: 0,
            path: "/root".into(),
            last_indexed_unix_seconds: 0,
        });
        m.files.push(FileEntry {
            root_id: 0,
            rel_path: "a.txt".into(),
            size: 10,
            mtime_ns: 1,
            content_hash: Some("blake3:abc".into()),
            chunk_ids: vec![1],
        });

        let plan = plan_index(
            Path::new("/root"),
            vec![walked("a.txt", 10, 999)],
            &mut m,
            &|_| Ok("blake3:abc".to_string()),
        )
        .unwrap();
        assert_eq!(plan.touched.len(), 1);
        assert!(plan.changed.is_empty());
    }

    #[test]
    fn edit_with_new_content_goes_to_changed() {
        let mut m = empty_manifest();
        m.roots.push(RootEntry {
            id: 0,
            path: "/root".into(),
            last_indexed_unix_seconds: 0,
        });
        m.files.push(FileEntry {
            root_id: 0,
            rel_path: "a.txt".into(),
            size: 10,
            mtime_ns: 1,
            content_hash: Some("blake3:old".into()),
            chunk_ids: vec![1],
        });

        let plan = plan_index(
            Path::new("/root"),
            vec![walked("a.txt", 11, 999)],
            &mut m,
            &|_| Ok("blake3:new".to_string()),
        )
        .unwrap();
        assert_eq!(plan.changed.len(), 1);
    }

    #[test]
    fn missing_file_goes_to_deleted() {
        let mut m = empty_manifest();
        m.roots.push(RootEntry {
            id: 0,
            path: "/root".into(),
            last_indexed_unix_seconds: 0,
        });
        m.files.push(FileEntry {
            root_id: 0,
            rel_path: "gone.txt".into(),
            size: 10,
            mtime_ns: 1,
            content_hash: Some("blake3:x".into()),
            chunk_ids: vec![7, 8],
        });

        let plan = plan_index(Path::new("/root"), vec![], &mut m, &never_hash).unwrap();
        assert_eq!(plan.deleted.len(), 1);
        assert_eq!(plan.deleted[0].chunk_ids, vec![7, 8]);
    }

    #[test]
    fn second_root_is_appended_and_isolated() {
        let mut m = empty_manifest();
        m.roots.push(RootEntry {
            id: 0,
            path: "/a".into(),
            last_indexed_unix_seconds: 0,
        });
        m.files.push(FileEntry {
            root_id: 0,
            rel_path: "doc.txt".into(),
            size: 10,
            mtime_ns: 1,
            content_hash: Some("blake3:x".into()),
            chunk_ids: vec![1],
        });

        let plan = plan_index(
            Path::new("/b"),
            vec![walked("other.txt", 5, 5)],
            &mut m,
            &never_hash,
        )
        .unwrap();
        assert_eq!(plan.root_id, 1);
        assert_eq!(plan.new.len(), 1);
        assert!(plan.deleted.is_empty());
        assert_eq!(m.roots.len(), 2);
    }

    #[test]
    fn walk_for_plan_produces_relative_paths_and_stat() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"world!!").unwrap();
        let (root, files) = walk_for_plan(dir.path()).unwrap();
        assert_eq!(root, dir.path().canonicalize().unwrap());
        let names: Vec<_> = files
            .iter()
            .map(|f| f.rel_path.to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.txt".to_string()));
        assert!(names.contains(&"b.txt".to_string()));
        let a = files
            .iter()
            .find(|f| f.rel_path == std::path::Path::new("a.txt"))
            .unwrap();
        assert_eq!(a.size, 5);
        assert!(a.mtime_ns > 0);
    }
}
