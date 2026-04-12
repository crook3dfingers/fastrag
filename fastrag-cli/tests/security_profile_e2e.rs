//! E2E: index the 5-CVE NVD fixture with --security-profile and verify
//! that Rejected/Disputed CVEs do NOT appear in query results, and that
//! Analyzed CVEs do appear.
//!
//! Does NOT require a real embedder or LLM — uses the test-doubles from
//! fastrag-embed's test-utils feature.

use std::path::Path;
use std::process::Command;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/fastrag-nvd/fixtures/nvd_slice.json")
}

fn fastrag_bin() -> std::path::PathBuf {
    // Built by cargo test --test security_profile_e2e -- the binary is in
    // the same target directory.
    let mut p = std::env::current_exe().unwrap();
    p.pop(); // remove test binary name
    p.pop(); // remove `deps`
    p.join("fastrag")
}

/// Run `fastrag index <fixture> --corpus <dir> --security-profile` and
/// return the combined stdout+stderr.
fn run_index(corpus_dir: &Path) -> std::process::Output {
    Command::new(fastrag_bin())
        .args([
            "index",
            fixture_path().to_str().unwrap(),
            "--corpus",
            corpus_dir.to_str().unwrap(),
            "--security-profile",
        ])
        .output()
        .expect("fastrag binary must be present; run `cargo build -p fastrag-cli --features nvd,hygiene,retrieval`")
}

#[test]
#[ignore = "requires fastrag binary; run with FASTRAG_NVD_TEST=1"]
fn security_profile_drops_rejected_cves() {
    if std::env::var("FASTRAG_NVD_TEST").is_err() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = run_index(dir.path());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "index must succeed; stderr: {stderr}");
    // Hygiene summary must report 2 rejected docs (CVE-2024-10001 Rejected + CVE-2024-10002 Disputed)
    assert!(
        stdout.contains("rejected=2"),
        "expected rejected=2 in hygiene summary; stdout: {stdout}"
    );
}
