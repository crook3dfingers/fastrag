//! Offline taxonomy regeneration tool. Parses a MITRE CWE XML catalog and
//! writes the descendant closure as JSON.
//!
//! Usage:
//!   cargo run -p fastrag-cwe --features compile-tool --bin compile-taxonomy -- \
//!     --in path/to/cwec_v4.16.xml \
//!     --out crates/fastrag-cwe/data/cwe-tree-v4.16.json

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut view = String::from("1000");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--in" => input = args.next().map(PathBuf::from),
            "--out" => output = args.next().map(PathBuf::from),
            "--view" => view = args.next().unwrap_or(view),
            "--help" | "-h" => {
                println!("Usage: compile-taxonomy --in INPUT.xml --out OUTPUT.json [--view 1000]");
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let input = input.ok_or("--in required")?;
    let output = output.ok_or("--out required")?;

    let xml = std::fs::read(&input)?;
    let taxonomy = fastrag_cwe::compile::build_closure(&xml, &view)?;
    let json = serde_json::to_string_pretty(&taxonomy)?;
    std::fs::write(&output, json)?;
    println!(
        "wrote taxonomy (version={}, view={}) to {}",
        taxonomy.version(),
        taxonomy.view(),
        output.display()
    );
    Ok(())
}
