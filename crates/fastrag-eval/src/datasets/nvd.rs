use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use fastrag_nvd::schema::{NvdDescription, NvdFeed};

use crate::{EvalDataset, EvalDocument, EvalError, EvalQuery, EvalResult, Qrel};

use super::common::{cache_root, download_to_path, file_name_from_url, read_gz_json, sha256_file};

const NVD_2023_URL: &str = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2023.json.gz";
const NVD_2024_URL: &str = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-2024.json.gz";

const BUNDLED_SECURITY_QUERIES: &str = include_str!("security_queries.json");

pub fn load_nvd() -> EvalResult<EvalDataset> {
    let root = cache_root("nvd")?;
    let feed_2023 = ensure_nvd_feed(&root, NVD_2023_URL)?;
    let feed_2024 = ensure_nvd_feed(&root, NVD_2024_URL)?;
    load_nvd_from_corpus_paths_with_bundled_queries("nvd", &[feed_2023, feed_2024])
}

pub fn load_nvd_corpus_with_queries(
    corpus_path: &Path,
    queries_path: &Path,
) -> EvalResult<EvalDataset> {
    let documents = load_nvd_documents(corpus_path)?;
    let query_file: SecurityQueriesFile = serde_json::from_str(&fs::read_to_string(queries_path)?)
        .map_err(|err| {
            EvalError::MalformedDataset(format!(
                "failed to parse security query set {}: {}",
                queries_path.display(),
                err
            ))
        })?;

    let corpus_ids: HashSet<String> = documents.iter().map(|doc| doc.id.clone()).collect();
    let query_ids: HashSet<String> = query_file
        .queries
        .iter()
        .map(|query| query.id.clone())
        .collect();
    let mut queries = Vec::with_capacity(query_ids.len());
    for query in &query_file.queries {
        queries.push(EvalQuery {
            id: query.id.clone(),
            text: query.text.clone(),
        });
    }

    let mut qrels = Vec::with_capacity(query_file.qrels.len());
    let mut seen_query_ids = HashSet::new();
    for qrel in query_file.qrels {
        if !query_ids.contains(&qrel.query_id) {
            return Err(EvalError::MalformedDataset(format!(
                "security query qrel references missing query id {}",
                qrel.query_id
            )));
        }
        if !corpus_ids.contains(&qrel.doc_id) {
            return Err(EvalError::MalformedDataset(format!(
                "security query qrel references missing document id {}",
                qrel.doc_id
            )));
        }
        seen_query_ids.insert(qrel.query_id.clone());
        qrels.push(Qrel {
            query_id: qrel.query_id,
            doc_id: qrel.doc_id,
            relevance: qrel.relevance,
        });
    }

    if seen_query_ids.len() != query_ids.len() {
        let missing: Vec<_> = query_ids.difference(&seen_query_ids).cloned().collect();
        return Err(EvalError::MalformedDataset(format!(
            "security query set contains queries without qrels: {:?}",
            missing
        )));
    }

    Ok(EvalDataset {
        name: "nvd".to_string(),
        documents,
        queries,
        qrels,
    })
}

fn load_nvd_from_corpus_paths_with_bundled_queries(
    name: &str,
    paths: &[PathBuf],
) -> EvalResult<EvalDataset> {
    let mut documents = Vec::new();
    for path in paths {
        documents.extend(load_nvd_documents(path)?);
    }

    let query_file: SecurityQueriesFile = serde_json::from_str(BUNDLED_SECURITY_QUERIES)
        .map_err(|e| EvalError::MalformedDataset(format!("bundled security_queries.json: {e}")))?;

    let corpus_ids: HashSet<String> = documents.iter().map(|d| d.id.clone()).collect();
    let mut queries = Vec::with_capacity(query_file.queries.len());
    for q in &query_file.queries {
        queries.push(EvalQuery {
            id: q.id.clone(),
            text: q.text.clone(),
        });
    }

    // Bundled qrels reference CVEs across both feed years; skip any not in this corpus slice.
    let mut qrels = Vec::new();
    for qrel in &query_file.qrels {
        if corpus_ids.contains(&qrel.doc_id) {
            qrels.push(Qrel {
                query_id: qrel.query_id.clone(),
                doc_id: qrel.doc_id.clone(),
                relevance: qrel.relevance,
            });
        }
    }

    // Drop queries that have no matching qrels (corpus doesn't contain their target docs)
    let active_query_ids: HashSet<&str> = qrels.iter().map(|q| q.query_id.as_str()).collect();
    queries.retain(|q| active_query_ids.contains(q.id.as_str()));

    Ok(EvalDataset {
        name: name.to_string(),
        documents,
        queries,
        qrels,
    })
}

fn load_nvd_documents(path: &Path) -> EvalResult<Vec<EvalDocument>> {
    let mut sources = Vec::new();
    if path.is_file() {
        sources.push(path.to_path_buf());
    } else if path.is_dir() {
        discover_feed_files(path, &mut sources)?;
        sources.sort();
    } else {
        return Err(EvalError::MalformedDataset(format!(
            "NVD corpus path does not exist: {}",
            path.display()
        )));
    }

    let mut documents = Vec::new();
    for source in sources {
        let feed: NvdFeed = read_gz_json(&source)?;
        for vulnerability in feed.vulnerabilities {
            let Some(cve) = vulnerability.cve else {
                continue;
            };
            let Some(id) = cve.id else {
                continue;
            };
            let english = english_descriptions(&cve.descriptions);
            if english.is_empty() {
                continue;
            }
            let text = english.join("\n");
            let title = text.chars().take(80).collect::<String>();
            documents.push(EvalDocument {
                id,
                title: Some(title),
                text,
            });
        }
    }

    Ok(documents)
}

fn ensure_nvd_feed(root: &Path, url: &str) -> EvalResult<PathBuf> {
    let filename = file_name_from_url(url)?;
    let path = root.join(&filename);
    let checksum_path = checksum_sidecar(&path);

    if path.exists() {
        if checksum_path.exists() {
            let expected = fs::read_to_string(&checksum_path)?.trim().to_string();
            let actual = sha256_file(&path)?;
            if actual == expected {
                return Ok(path);
            }
            fs::remove_file(&path)?;
        } else {
            let digest = sha256_file(&path)?;
            fs::write(&checksum_path, format!("{digest}\n"))?;
            return Ok(path);
        }
    }

    download_to_path(url, &path, None)?;
    let digest = sha256_file(&path)?;
    fs::write(&checksum_path, format!("{digest}\n"))?;
    Ok(path)
}

fn checksum_sidecar(path: &Path) -> PathBuf {
    let file_name = path.file_name().expect("feed file name");
    path.with_file_name(format!("{}.sha256", file_name.to_string_lossy()))
}

fn discover_feed_files(dir: &Path, out: &mut Vec<PathBuf>) -> EvalResult<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            discover_feed_files(&path, out)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".json.gz"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn english_descriptions(rows: &[NvdDescription]) -> Vec<String> {
    rows.iter()
        .filter(|description| description.lang.eq_ignore_ascii_case("en"))
        .map(|description| description.value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SecurityQueriesFile {
    #[serde(default)]
    queries: Vec<SecurityQueryRow>,
    #[serde(default)]
    qrels: Vec<SecurityQrelRow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SecurityQueryRow {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SecurityQrelRow {
    query_id: String,
    doc_id: String,
    relevance: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/datasets/nvd_mini.json.gz")
    }

    #[test]
    fn loads_offline_fixture() {
        let dataset = load_nvd_documents(&fixture_path()).unwrap();
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset[0].id, "CVE-2023-11111");
        let expected_title =
            "CVE-2023-11111: buffer overflow in demo parser allows remote code execution when"
                .to_string();
        assert_eq!(dataset[0].title.as_deref(), Some(expected_title.as_str()));
        assert!(dataset[2].text.contains('\n'));
    }

    #[test]
    fn joins_security_queries_and_qrels() {
        let dir = tempfile::tempdir().unwrap();
        let queries_path = dir.path().join("security-queries.json");
        fs::write(
            &queries_path,
            r#"{
  "queries": [
    {"id":"q-1","text":"find demo parser overflow"},
    {"id":"q-2","text":"jwt signing weakness"}
  ],
  "qrels": [
    {"query_id":"q-1","doc_id":"CVE-2023-11111","relevance":2},
    {"query_id":"q-2","doc_id":"CVE-2023-22222","relevance":1}
  ]
}"#,
        )
        .unwrap();

        let dataset = load_nvd_corpus_with_queries(&fixture_path(), &queries_path).unwrap();
        assert_eq!(dataset.name, "nvd");
        assert_eq!(dataset.documents.len(), 5);
        assert_eq!(dataset.queries.len(), 2);
        assert_eq!(dataset.qrels.len(), 2);
        assert_eq!(dataset.queries[1].text, "jwt signing weakness");
        assert_eq!(dataset.qrels[0].relevance, 2);
    }

    #[test]
    fn load_rejects_missing_query_ids() {
        let dir = tempfile::tempdir().unwrap();
        let queries_path = dir.path().join("security-queries.json");
        fs::write(
            &queries_path,
            r#"{
  "queries": [{"id":"q-1","text":"find demo parser overflow"}],
  "qrels": [{"query_id":"q-2","doc_id":"CVE-2023-11111","relevance":2}]
}"#,
        )
        .unwrap();

        let err = load_nvd_corpus_with_queries(&fixture_path(), &queries_path).unwrap_err();
        assert!(err.to_string().contains("missing query id"));
    }

    #[test]
    fn load_nvd_includes_bundled_queries() {
        let dataset =
            load_nvd_from_corpus_paths_with_bundled_queries("nvd-test", &[fixture_path()]).unwrap();
        assert_eq!(dataset.queries.len(), 10);
        assert_eq!(dataset.qrels.len(), 10);
        assert!(dataset.queries.iter().any(|q| q.id == "nvd-q01"));
    }
}
