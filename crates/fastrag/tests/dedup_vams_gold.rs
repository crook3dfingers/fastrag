//! Real-data dedup benchmark — labeled pairs from VAMS scanner output.
//! Runs under `FASTRAG_DEDUP_GOLD=1`, matching `FASTRAG_NVD_TEST` /
//! `FASTRAG_RERANK_TEST` patterns.
#![cfg(feature = "retrieval")]

use fastrag::corpus::verify;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Pair {
    a: String,
    b: String,
    is_duplicate: bool,
}

fn load_pairs() -> Vec<Pair> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dedup/vams_pairs.jsonl");
    let raw = std::fs::read_to_string(&path).expect("vams_pairs.jsonl present");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Pair>(l).unwrap_or_else(|e| panic!("bad row {l}: {e}")))
        .collect()
}

#[test]
#[ignore = "requires FASTRAG_DEDUP_GOLD=1 and labeled VAMS pairs"]
fn vams_dedup_precision_recall() {
    if std::env::var("FASTRAG_DEDUP_GOLD").ok().as_deref() != Some("1") {
        eprintln!("FASTRAG_DEDUP_GOLD not set; skipping");
        return;
    }
    let pairs = load_pairs();
    assert!(
        !pairs.is_empty(),
        "vams_pairs.jsonl must have at least one row"
    );

    let threshold = 0.7f32;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;

    for p in &pairs {
        let sa = verify::signature_of(&p.a);
        let sb = verify::signature_of(&p.b);
        let j = verify::jaccard(&sa, &sb);
        let predicted_dup = j >= threshold;
        match (predicted_dup, p.is_duplicate) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }

    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    eprintln!(
        "VAMS dedup: p={precision:.3} r={recall:.3} f1={f1:.3} tn={tn} (n={})",
        pairs.len()
    );
    // No hard assertion — this test reports numbers for the ADR writeup.
    // A hard gate can be added once the fixture is populated with real labels.
}
