//! Integration test: baseline::diff over checked-in report + baseline fixtures.

use std::path::PathBuf;

use fastrag_eval::baseline::{diff, load_baseline};
use fastrag_eval::matrix::MatrixReport;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_report(name: &str) -> MatrixReport {
    let bytes = std::fs::read(fixture(name)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn good_run_passes_baseline_gate() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_good.json");
    let d = diff(&report, &baseline).unwrap();
    assert!(
        !d.has_regressions(),
        "expected no regressions, got: {:?}",
        d.regressions
    );
}

#[test]
fn bad_run_produces_primary_hit5_regression() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_bad.json");
    let d = diff(&report, &baseline).unwrap();
    assert!(d.has_regressions());
    let primary_hit5 = d
        .regressions
        .iter()
        .find(|r| format!("{:?}", r.variant) == "Primary" && r.metric == "hit@5");
    assert!(
        primary_hit5.is_some(),
        "expected a Primary hit@5 regression, got: {:?}",
        d.regressions
    );
}

#[test]
fn bad_run_renders_markdown_mentioning_primary() {
    let baseline = load_baseline(&fixture("baseline_current.json")).unwrap();
    let report = load_report("report_bad.json");
    let d = diff(&report, &baseline).unwrap();
    let rendered = d.render_report();
    assert!(rendered.contains("Primary"));
    assert!(rendered.contains("hit@5"));
}
