use fastrag::ChunkingStrategy;
use fastrag::corpus::{CorpusIndexStats, LatencyBreakdown, index_path, query_corpus};
use fastrag_embed::test_utils::MockEmbedder;
use std::fs;
use tempfile::tempdir;

fn basic() -> ChunkingStrategy {
    ChunkingStrategy::Basic {
        max_characters: 1000,
        overlap: 0,
    }
}

fn write(dir: &std::path::Path, name: &str, body: &str) {
    fs::write(dir.join(name), body).unwrap();
}

fn reindex(input: &std::path::Path, corpus: &std::path::Path) -> CorpusIndexStats {
    index_path(input, corpus, &basic(), &MockEmbedder).unwrap()
}

#[test]
fn reindex_unchanged_does_no_work() {
    let input = tempdir().unwrap();
    let corpus = tempdir().unwrap();
    write(input.path(), "a.txt", "ALPHA\n\nalpha beta.");
    write(input.path(), "b.txt", "BETA\n\nbeta gamma.");

    let first = reindex(input.path(), corpus.path());
    assert_eq!(first.files_new, 2);
    assert_eq!(first.files_unchanged, 0);

    let second = reindex(input.path(), corpus.path());
    assert_eq!(second.files_new, 0);
    assert_eq!(second.files_changed, 0);
    assert_eq!(second.files_unchanged, 2);
    assert_eq!(second.chunks_added, 0);
    assert_eq!(second.chunks_removed, 0);
}

#[test]
fn edited_file_is_re_embedded_stale_chunks_gone() {
    let input = tempdir().unwrap();
    let corpus = tempdir().unwrap();
    write(input.path(), "a.txt", "alpha original content.");
    reindex(input.path(), corpus.path());

    write(input.path(), "a.txt", "alpha replaced content xyz.");
    let stats = reindex(input.path(), corpus.path());
    assert_eq!(stats.files_changed, 1);
    assert_eq!(stats.files_new, 0);
    assert!(stats.chunks_removed >= 1);
    assert!(stats.chunks_added >= 1);

    let hits = query_corpus(
        corpus.path(),
        "original content",
        5,
        &MockEmbedder,
        &mut LatencyBreakdown::default(),
    )
    .unwrap();
    assert!(
        hits.iter()
            .all(|h| !h.entry.chunk_text.contains("original"))
    );
}

#[test]
fn deleted_file_drops_chunks() {
    let input = tempdir().unwrap();
    let corpus = tempdir().unwrap();
    write(input.path(), "a.txt", "alpha.");
    write(input.path(), "b.txt", "beta.");
    reindex(input.path(), corpus.path());

    fs::remove_file(input.path().join("b.txt")).unwrap();
    let stats = reindex(input.path(), corpus.path());
    assert_eq!(stats.files_deleted, 1);
    assert!(stats.chunks_removed >= 1);

    let hits = query_corpus(
        corpus.path(),
        "beta",
        5,
        &MockEmbedder,
        &mut LatencyBreakdown::default(),
    )
    .unwrap();
    assert!(hits.iter().all(|h| !h.entry.source_path.ends_with("b.txt")));
}

#[test]
fn two_roots_isolated() {
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    let corpus = tempdir().unwrap();
    write(a.path(), "a.txt", "alpha.");
    write(b.path(), "b.txt", "beta.");

    reindex(a.path(), corpus.path());
    reindex(b.path(), corpus.path());

    let hits = query_corpus(
        corpus.path(),
        "alpha",
        5,
        &MockEmbedder,
        &mut LatencyBreakdown::default(),
    )
    .unwrap();
    assert!(hits.iter().any(|h| h.entry.source_path.ends_with("a.txt")));
    let hits = query_corpus(
        corpus.path(),
        "beta",
        5,
        &MockEmbedder,
        &mut LatencyBreakdown::default(),
    )
    .unwrap();
    assert!(hits.iter().any(|h| h.entry.source_path.ends_with("b.txt")));

    fs::remove_file(a.path().join("a.txt")).unwrap();
    let stats = reindex(a.path(), corpus.path());
    assert_eq!(stats.files_deleted, 1);

    let hits = query_corpus(
        corpus.path(),
        "beta",
        5,
        &MockEmbedder,
        &mut LatencyBreakdown::default(),
    )
    .unwrap();
    assert!(hits.iter().any(|h| h.entry.source_path.ends_with("b.txt")));
}
