# Incremental Indexing — Design (#28)

## Purpose

Make `fastrag index` re-runnable cheaply: unchanged files must not be re-embedded, edited/new files re-embed themselves, deleted files drop out of the corpus. One corpus must support multiple independent input roots.

## Change detection: hybrid (stat fast path + content hash)

1. Fast path: `(size, mtime_ns)` match → skip (no read).
2. Suspect (size or mtime differs): read file, compute blake3, compare to stored hash.
   - Same hash → update stored mtime/size, skip re-embed (touch-preserving edit, clock skew, rsync).
   - Different hash → re-embed.

Rationale: embeddings are expensive and stale vectors are silent correctness bugs. Pure stat (make/ninja) is too lossy; pure hash (bazel/nix) reads every file. Hybrid is the rsync/turborepo middle ground and is identical to pure-stat in the 99% case.

## Deletion: auto-prune with report

Files present in manifest under the current root but absent from the walk are removed (chunks tombstoned, manifest entry dropped). CLI summary reports `files_deleted` and `chunks_removed`. No `--prune` flag.

## Multi-root corpora

One corpus may contain multiple input roots. Re-indexing with a different root appends a new root rather than erroring; pruning only affects files under the current invocation's root.

## Manifest schema v2

```json
{
  "schema_version": 2,
  "embedder": { "name": "...", "dim": 384 },
  "roots": [
    { "id": 0, "path": "/abs/docs",    "last_indexed": "2026-04-07T..." }
  ],
  "files": [
    {
      "root_id": 0,
      "rel_path": "guide.md",
      "size": 12034,
      "mtime_ns": 1712500000000000000,
      "content_hash": "blake3:...",
      "chunk_ids": [17, 18, 19]
    }
  ],
  "deleted": [42, 43],
  "entry_count": 1234
}
```

- `rel_path` is relative to `roots[root_id].path`.
- `chunk_ids` are stable u32 ids into `entries.bin` / HNSW.
- `deleted` is the tombstone set for HNSW ids that are no longer live.

### v1 → v2 migration (automatic on first load)

- Synthesize one root per distinct `source_path` prefix observed in v1 entries (or one root if all share a common ancestor).
- Populate `size`, `mtime_ns` from `fs::metadata`.
- Leave `content_hash = None` — first incremental pass will hash-verify and fill it.
- Write v2 manifest atomically; keep one `manifest.json.bak`.

## Diff algorithm

Input: CLI root `R` (canonicalized), loaded manifest `M`.

1. Resolve `root_id` for `R` in `M.roots`; append new entry if absent.
2. Walk `R`; collect `(rel_path, size, mtime_ns)` for supported files.
3. Classify each walked file against `M.files` filtered by `root_id`:
   - **Unchanged**: `size` and `mtime_ns` match → skip.
   - **Suspect**: stat differs → read, blake3, compare hash.
     - Hash match → update stat, skip embed.
     - Hash mismatch → **changed**.
   - **New**: not in manifest → **new**.
4. **Deleted**: any manifest file with this `root_id` not seen in walk.
5. Apply:
   - Changed: tombstone old `chunk_ids`, re-chunk, embed, insert new ids, update manifest entry.
   - New: chunk, embed, insert, add manifest entry.
   - Deleted: tombstone chunk_ids, drop manifest entry.
6. Update `roots[root_id].last_indexed`.
7. If `deleted.len() > entries.len() / 4`, rebuild HNSW from live entries and clear tombstones.
8. Atomic rewrite: `manifest.json.tmp` → `rename`; same for `entries.bin` if rebuilt.

## HNSW tombstoning

`instant-distance` does not support deletion. Approach:

- Dead chunk ids are tracked in `manifest.deleted: HashSet<u32>`.
- `corpus::query_corpus*` filters out tombstoned ids in the hit loop.
- Rebuild threshold: >25% tombstoned → full rebuild from live entries.

## Code organization

New module `crates/fastrag/src/corpus/incremental.rs`:

```rust
pub struct IndexPlan {
    pub unchanged: Vec<FileRef>,
    pub changed:   Vec<FileRef>,
    pub new:       Vec<FileRef>,
    pub deleted:   Vec<FileEntry>,
}

pub fn plan_index(root: &Path, manifest: &Manifest) -> io::Result<IndexPlan>;
pub fn apply_plan(plan: IndexPlan, ctx: &mut IndexCtx) -> Result<IndexStats, CorpusError>;
```

`ops::index_path` becomes: `load_or_migrate` → `plan_index` → `apply_plan` → persist.

New `Manifest` methods:

- `load_or_migrate(corpus_dir) -> Manifest`
- `resolve_root(abs_path) -> RootId`
- `tombstone(chunk_ids: &[u32])`
- `needs_rebuild() -> bool`

Query path (`corpus::query_corpus*`) adds one `if manifest.deleted.contains(id) { continue }` per hit.

`IndexStats` gains: `files_unchanged`, `files_changed`, `files_new`, `files_deleted`, `chunks_added`, `chunks_removed`. CLI summary prints them.

## Error handling & atomicity

- **Manifest writes**: write `manifest.json.tmp` then `rename`. Keep one `.bak` generation.
- **entries.bin rewrite** (on HNSW rebuild): same tmp+rename.
- **Partial failure mid-apply**: if embedding a changed file fails, retain the old chunks and old hash for that file; report the failure, continue. Never leave a file in a half-updated state.
- **Hash I/O errors**: warn, leave file untouched in manifest, continue.
- **Root canonicalization failure**: hard error at entry.
- **Concurrent runs**: `corpus/.lock` via `fs2`; second concurrent `index` fails with `CorpusError::Locked`.

## Testing

### Unit (`corpus/incremental.rs`)

- Classifier correctness: unchanged, suspect-same-hash, suspect-changed, new, deleted.
- v1→v2 migration: synthesizes roots, hash = None, stat populated from disk.
- Multi-root isolation: deletion under root A does not touch root B.
- Tombstone rebuild threshold triggers at >25%.

### Integration (`fastrag-cli/tests/incremental_e2e.rs`)

- Re-index unchanged dir → `chunks_added == 0`, `files_unchanged == N`.
- Edit one file → only that file re-embedded; stale chunk ids absent from query results.
- `touch` with identical content → no re-embed (hash fast-reject).
- Delete file → chunks removed, query no longer returns them.
- Two-root corpus: index `./a`, index `./b`; both queryable; delete in `./a` does not affect `./b`.
- Hand-written v1 manifest fixture: loads, auto-migrates, next pass is incremental.

### Regression

Existing `tests/corpus.rs` stays green.

## Out of scope

- Parallel embedding of changed files.
- True incremental HNSW insertion (rebuild-on-threshold suffices).
- `--watch` mode.
- Cross-machine corpus sync.
