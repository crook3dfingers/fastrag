//! The server must refuse to start when --bundle-path points at an invalid
//! bundle directory. Catching this at startup (before accepting traffic) is
//! the whole point of the BundleState preflight.

use assert_cmd::Command;
use tempfile::tempdir;

#[test]
fn serve_http_refuses_invalid_bundle() {
    let tmp = tempdir().unwrap();
    let bundle = tmp.path().join("not-a-bundle");
    std::fs::create_dir_all(&bundle).unwrap();
    // No bundle.json, no corpora — invalid.

    let corpus = tmp.path().join("corpus");
    std::fs::create_dir_all(&corpus).unwrap();

    let assert = Command::cargo_bin("fastrag")
        .unwrap()
        .args([
            "serve-http",
            "--corpus",
            corpus.to_str().unwrap(),
            "--bundle-path",
            bundle.to_str().unwrap(),
            "--port",
            "0",
        ])
        .timeout(std::time::Duration::from_secs(10))
        .assert()
        .failure();
    let output = assert.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        stderr.contains("bundle") || stderr.contains("manifest"),
        "expected bundle/manifest error, got stderr: {stderr}"
    );
}
