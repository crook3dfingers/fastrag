//! SQLite-backed cache for contextualization results.
//!
//! See `docs/superpowers/specs/2026-04-10-contextual-retrieval-design.md` for
//! the full design. The cache is self-contained: every row stores the raw
//! chunk text and the source document title alongside the generated context,
//! so a `--retry-failed` pass can run against the SQLite file alone without
//! opening the Tantivy or HNSW indexes.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};

use crate::ContextError;

/// SQLite-backed cache for contextualization results.
pub struct ContextCache {
    conn: Connection,
}

/// One row from the `context` table. Field order matches the schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedContext {
    pub chunk_hash: [u8; 32],
    pub ctx_version: u32,
    pub model_id: String,
    pub prompt_version: u32,
    pub raw_text: String,
    pub doc_title: String,
    pub context_text: Option<String>,
    pub status: CacheStatus,
    pub error: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    Ok,
    Failed,
}

impl CacheStatus {
    fn parse(s: &str) -> Result<Self, ContextError> {
        match s {
            "ok" => Ok(CacheStatus::Ok),
            "failed" => Ok(CacheStatus::Failed),
            other => Err(ContextError::Template(format!(
                "invalid cache status: {other}"
            ))),
        }
    }
}

/// Composite primary key for a cache row. `model_id` borrows; the lifetime
/// matches the caller's owned string. The rest of the fields are trivially
/// `Copy`, so the whole key is `Copy` when the borrow is static.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey<'a> {
    pub chunk_hash: [u8; 32],
    pub ctx_version: u32,
    pub model_id: &'a str,
    pub prompt_version: u32,
}

/// SELECT column list used by both `get` and `iter_failed` so the row-mapping
/// code can be shared.
const SELECT_COLS: &str = "chunk_hash, ctx_version, model_id, prompt_version, \
     raw_text, doc_title, context_text, status, error, created_at";

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CachedContext> {
    let hash_blob: Vec<u8> = row.get(0)?;
    let mut chunk_hash = [0u8; 32];
    if hash_blob.len() != 32 {
        return Err(rusqlite::Error::InvalidColumnType(
            0,
            "chunk_hash".into(),
            rusqlite::types::Type::Blob,
        ));
    }
    chunk_hash.copy_from_slice(&hash_blob);
    let status_str: String = row.get(7)?;
    let status = CacheStatus::parse(&status_str).map_err(|_| {
        rusqlite::Error::InvalidColumnType(7, "status".into(), rusqlite::types::Type::Text)
    })?;
    Ok(CachedContext {
        chunk_hash,
        ctx_version: row.get(1)?,
        model_id: row.get(2)?,
        prompt_version: row.get(3)?,
        raw_text: row.get(4)?,
        doc_title: row.get(5)?,
        context_text: row.get(6)?,
        status,
        error: row.get(8)?,
        created_at: row.get(9)?,
    })
}

impl ContextCache {
    /// Open the cache at `path`, creating the file and table if needed. WAL
    /// journal mode is enabled so readers never block writers on this single
    /// writer workload.
    pub fn open(path: &Path) -> Result<Self, ContextError> {
        let conn = Connection::open(path)?;
        // PRAGMA journal_mode=WAL returns a row, so it cannot go through
        // execute_batch; set it explicitly first.
        let _: String = conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        conn.execute_batch(
            r#"
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS context (
                chunk_hash      BLOB NOT NULL,
                ctx_version     INTEGER NOT NULL,
                model_id        TEXT NOT NULL,
                prompt_version  INTEGER NOT NULL,
                raw_text        TEXT NOT NULL,
                doc_title       TEXT NOT NULL,
                context_text    TEXT,
                status          TEXT NOT NULL CHECK(status IN ('ok','failed')),
                error           TEXT,
                created_at      INTEGER NOT NULL,
                PRIMARY KEY (chunk_hash, ctx_version, model_id, prompt_version)
            );
            CREATE INDEX IF NOT EXISTS idx_context_status ON context(status);
            "#,
        )?;
        Ok(Self { conn })
    }

    /// Insert or replace a successful contextualization result.
    pub fn put_ok(
        &mut self,
        key: CacheKey<'_>,
        raw_text: &str,
        doc_title: &str,
        context_text: &str,
    ) -> Result<(), ContextError> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO context
              (chunk_hash, ctx_version, model_id, prompt_version,
               raw_text, doc_title, context_text, status, error, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'ok', NULL, ?8)
            "#,
            params![
                &key.chunk_hash[..],
                key.ctx_version,
                key.model_id,
                key.prompt_version,
                raw_text,
                doc_title,
                context_text,
                now_unix(),
            ],
        )?;
        Ok(())
    }

    /// Insert or replace a failure record. The `error` string is truncated
    /// to a bounded length to keep the DB footprint predictable.
    pub fn mark_failed(
        &mut self,
        key: CacheKey<'_>,
        raw_text: &str,
        doc_title: &str,
        error: &str,
    ) -> Result<(), ContextError> {
        const MAX_ERROR_LEN: usize = 500;
        let truncated = if error.len() > MAX_ERROR_LEN {
            // Slice on a char boundary to avoid splitting UTF-8.
            let mut end = MAX_ERROR_LEN;
            while !error.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &error[..end])
        } else {
            error.to_string()
        };
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO context
              (chunk_hash, ctx_version, model_id, prompt_version,
               raw_text, doc_title, context_text, status, error, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, 'failed', ?7, ?8)
            "#,
            params![
                &key.chunk_hash[..],
                key.ctx_version,
                key.model_id,
                key.prompt_version,
                raw_text,
                doc_title,
                truncated,
                now_unix(),
            ],
        )?;
        Ok(())
    }

    /// Look up a single cache row. Returns `Ok(None)` if the key is not
    /// present.
    pub fn get(&self, key: CacheKey<'_>) -> Result<Option<CachedContext>, ContextError> {
        let sql = format!(
            "SELECT {SELECT_COLS} FROM context \
             WHERE chunk_hash = ?1 AND ctx_version = ?2 \
               AND model_id = ?3 AND prompt_version = ?4"
        );
        let row = self
            .conn
            .query_row(
                &sql,
                params![
                    &key.chunk_hash[..],
                    key.ctx_version,
                    key.model_id,
                    key.prompt_version,
                ],
                map_row,
            )
            .optional()?;
        Ok(row)
    }

    /// Materialize every `status='failed'` row into an owned Vec. The failure
    /// count is expected to be <5% of corpus size; materializing is simpler
    /// than lending the connection through a streaming iterator.
    pub fn iter_failed(&self) -> Result<std::vec::IntoIter<CachedContext>, ContextError> {
        let sql = format!("SELECT {SELECT_COLS} FROM context WHERE status = 'failed'");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows: Vec<CachedContext> = stmt
            .query_map([], map_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.into_iter())
    }

    /// Count rows, for diagnostics.
    pub fn row_count(&self) -> Result<(u64, u64), ContextError> {
        let ok: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM context WHERE status = 'ok'",
            [],
            |r| r.get(0),
        )?;
        let failed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM context WHERE status = 'failed'",
            [],
            |r| r.get(0),
        )?;
        Ok((ok as u64, failed as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache() -> (ContextCache, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("context.sqlite");
        let cache = ContextCache::open(&path).expect("open");
        (cache, dir)
    }

    fn sample_key() -> CacheKey<'static> {
        CacheKey {
            chunk_hash: [1u8; 32],
            ctx_version: 1,
            model_id: "test-model",
            prompt_version: 1,
        }
    }

    #[test]
    fn open_creates_schema() {
        let (cache, _dir) = temp_cache();
        let count: i64 = cache
            .conn
            .query_row("SELECT COUNT(*) FROM context", [], |r| r.get(0))
            .expect("select");
        assert_eq!(count, 0);
    }

    #[test]
    fn put_then_get_round_trip() {
        let (mut cache, _dir) = temp_cache();
        let key = sample_key();

        cache
            .put_ok(key, "Raw chunk text.", "Doc Title", "Generated context.")
            .expect("put");

        let row = cache.get(key).expect("get").expect("row present");

        assert_eq!(row.status, CacheStatus::Ok);
        assert_eq!(row.raw_text, "Raw chunk text.");
        assert_eq!(row.doc_title, "Doc Title");
        assert_eq!(row.context_text.as_deref(), Some("Generated context."));
        assert_eq!(row.error, None);
        assert_eq!(row.chunk_hash, [1u8; 32]);
        assert_eq!(row.ctx_version, 1);
        assert_eq!(row.model_id, "test-model");
        assert_eq!(row.prompt_version, 1);
        assert!(row.created_at > 0);
    }

    #[test]
    fn get_missing_returns_none() {
        let (cache, _dir) = temp_cache();
        let missing = cache
            .get(CacheKey {
                chunk_hash: [99u8; 32],
                ctx_version: 1,
                model_id: "x",
                prompt_version: 1,
            })
            .expect("get");
        assert!(missing.is_none());
    }

    #[test]
    fn mark_failed_then_iter_failed_returns_row() {
        let (mut cache, _dir) = temp_cache();
        let key = sample_key();

        cache
            .mark_failed(key, "Raw chunk.", "Doc", "llama-server returned 500")
            .expect("mark_failed");

        let failed: Vec<CachedContext> = cache.iter_failed().expect("iter").collect();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].status, CacheStatus::Failed);
        assert_eq!(failed[0].raw_text, "Raw chunk.");
        assert_eq!(failed[0].doc_title, "Doc");
        assert_eq!(failed[0].context_text, None);
        assert_eq!(
            failed[0].error.as_deref(),
            Some("llama-server returned 500")
        );
    }

    #[test]
    fn put_ok_after_failed_removes_from_iter_failed() {
        let (mut cache, _dir) = temp_cache();
        let key = sample_key();

        cache
            .mark_failed(key, "Raw.", "Doc", "err")
            .expect("mark_failed");
        assert_eq!(cache.iter_failed().expect("iter").count(), 1);

        cache
            .put_ok(key, "Raw.", "Doc", "generated context")
            .expect("put_ok");
        assert_eq!(cache.iter_failed().expect("iter").count(), 0);

        let row = cache.get(key).expect("get").expect("row");
        assert_eq!(row.status, CacheStatus::Ok);
        assert_eq!(row.context_text.as_deref(), Some("generated context"));
        assert_eq!(row.error, None);
    }

    #[test]
    fn distinct_keys_for_same_hash_coexist() {
        let (mut cache, _dir) = temp_cache();
        let hash = [7u8; 32];

        for (ctx_v, model, prompt_v, ctx_text) in [
            (1u32, "model-a", 1u32, "ctx-a1"),
            (1, "model-a", 2, "ctx-a2"),
            (1, "model-b", 1, "ctx-b1"),
            (2, "model-a", 1, "ctx-v2"),
        ] {
            cache
                .put_ok(
                    CacheKey {
                        chunk_hash: hash,
                        ctx_version: ctx_v,
                        model_id: model,
                        prompt_version: prompt_v,
                    },
                    "raw",
                    "",
                    ctx_text,
                )
                .expect("put");
        }

        let count: i64 = cache
            .conn
            .query_row(
                "SELECT COUNT(*) FROM context WHERE chunk_hash = ?1",
                params![&hash[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 4, "each tuple should be a distinct row");

        let row1 = cache
            .get(CacheKey {
                chunk_hash: hash,
                ctx_version: 1,
                model_id: "model-a",
                prompt_version: 1,
            })
            .unwrap()
            .unwrap();
        let row2 = cache
            .get(CacheKey {
                chunk_hash: hash,
                ctx_version: 1,
                model_id: "model-a",
                prompt_version: 2,
            })
            .unwrap()
            .unwrap();
        assert_eq!(row1.context_text.as_deref(), Some("ctx-a1"));
        assert_eq!(row2.context_text.as_deref(), Some("ctx-a2"));
    }

    #[test]
    fn row_count_reports_ok_and_failed_separately() {
        let (mut cache, _dir) = temp_cache();
        for i in 0..3u8 {
            cache
                .put_ok(
                    CacheKey {
                        chunk_hash: [i; 32],
                        ctx_version: 1,
                        model_id: "m",
                        prompt_version: 1,
                    },
                    "raw",
                    "",
                    "ctx",
                )
                .unwrap();
        }
        for i in 10..12u8 {
            cache
                .mark_failed(
                    CacheKey {
                        chunk_hash: [i; 32],
                        ctx_version: 1,
                        model_id: "m",
                        prompt_version: 1,
                    },
                    "raw",
                    "",
                    "boom",
                )
                .unwrap();
        }
        let (ok, failed) = cache.row_count().unwrap();
        assert_eq!(ok, 3);
        assert_eq!(failed, 2);
    }

    #[test]
    fn mark_failed_truncates_long_error() {
        let (mut cache, _dir) = temp_cache();
        let key = sample_key();
        let long: String = "x".repeat(1000);
        cache.mark_failed(key, "raw", "", &long).unwrap();
        let row = cache.get(key).unwrap().unwrap();
        let err = row.error.unwrap();
        // 500 original chars plus the trailing ellipsis; max length is
        // bounded to well under the raw 1000.
        assert!(err.len() < 1000);
        assert!(err.ends_with('…'));
        assert!(err.starts_with("xxxxx"));
    }
}
