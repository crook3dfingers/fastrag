//! Cache resume integration test — verifies rows survive close-and-reopen.
//!
//! The resume guarantee is the foundation of `--retry-failed`: we must be
//! able to crash mid-ingest and come back without losing any prior work.

use fastrag_context::{CacheKey, ContextCache};
use std::path::PathBuf;

fn key(n: u8) -> CacheKey<'static> {
    CacheKey {
        chunk_hash: [n; 32],
        ctx_version: 1,
        model_id: "test",
        prompt_version: 1,
    }
}

#[test]
fn cache_survives_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("ctx.sqlite");

    // Open, write 100 rows, close via drop.
    {
        let mut cache = ContextCache::open(&path).unwrap();
        for i in 0..100u8 {
            cache
                .put_ok(key(i), &format!("raw-{i}"), "Title", &format!("ctx-{i}"))
                .unwrap();
        }
    }

    // Reopen and verify every row.
    let cache = ContextCache::open(&path).unwrap();
    for i in 0..100u8 {
        let row = cache
            .get(key(i))
            .unwrap()
            .unwrap_or_else(|| panic!("missing row {i}"));
        assert_eq!(row.raw_text, format!("raw-{i}"));
        assert_eq!(
            row.context_text.as_deref(),
            Some(format!("ctx-{i}").as_str())
        );
    }
}
