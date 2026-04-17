#!/usr/bin/env bash
# scripts/build-bundle.sh — assemble the vams-lookup-v1 bundle.
#
# Stages (all idempotent):
#   1. Fetch CISA KEV catalog (SHA-pinned).
#   2. Emit CWE + KEV JSONL.
#   3. Run `fastrag index` → corpora/cwe and corpora/kev.
#   4. Copy CWE taxonomy artifact.
#   5. (Optional) Ingest vams-findings JSONL into corpora/vams-findings.
#   6. Write bundle.json manifest.
#
# Prerequisites:
#   - `cargo build --release` has produced target/release/fastrag.
#   - A MITRE CWE XML catalog is available at $CWE_XML and the matching
#     SHA-256 pin at $CWE_SHA_FILE. The simplest way to provision both:
#         scripts/fetch-cwe.sh data v4.19.1
#     which downloads + SHA-verifies the MITRE zip and extracts the XML.
#     Airgap operators who pre-staged the XML must also ship the SHA pin.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-bundles/vams-lookup-v1}"
CWE_VERSION="${CWE_VERSION:-v4.19.1}"
CWE_XML="${CWE_XML:-data/cwec_${CWE_VERSION}.xml}"
CWE_SHA_FILE="${CWE_SHA_FILE:-data/cwe-${CWE_VERSION}.sha256}"
FASTRAG="${FASTRAG:-target/release/fastrag}"
DATA_DIR="$(mktemp -d -t vams-lookup-build-XXXXXX)"
trap 'rm -rf "$DATA_DIR"' EXIT

if [[ ! -f "$CWE_XML" ]]; then
  echo "CWE XML not found at $CWE_XML" >&2
  echo "Run 'scripts/fetch-cwe.sh data ${CWE_VERSION}' to download + SHA-verify," >&2
  echo "or set CWE_XML=/path/to/catalog.xml if you already staged it (airgap)." >&2
  exit 1
fi

if [[ ! -f "$CWE_SHA_FILE" ]]; then
  echo "CWE SHA pin missing at $CWE_SHA_FILE" >&2
  echo "Run 'scripts/fetch-cwe.sh data ${CWE_VERSION}' to pin it." >&2
  exit 1
fi
CWE_SHA="$(cat "$CWE_SHA_FILE")"

if [[ ! -x "$FASTRAG" ]]; then
  echo "fastrag binary not found at $FASTRAG — run 'cargo build --release' first" >&2
  exit 1
fi

mkdir -p "$BUNDLE_DIR/taxonomy"

# 1. Fetch KEV
scripts/fetch-kev.sh "$DATA_DIR"

# 2. Emit JSONL
python3 scripts/build_taxonomy_corpus.py \
    --cwe-xml "$CWE_XML" \
    --cwe-out "$DATA_DIR/cwe.jsonl" \
    --kev-catalog "$DATA_DIR/known_exploited_vulnerabilities.json" \
    --kev-out "$DATA_DIR/kev.jsonl"

# 3. Index
rm -rf "$BUNDLE_DIR/corpora/cwe" "$BUNDLE_DIR/corpora/kev"
"$FASTRAG" index "$DATA_DIR/cwe.jsonl" \
    --corpus "$BUNDLE_DIR/corpora/cwe" \
    --embedder bge \
    --format jsonl \
    --text-fields name,description,extended_description \
    --id-field cwe_id \
    --metadata-fields cwe_id,parents,children,applicable_platforms \
    --array-fields parents,children,applicable_platforms \
    --cwe-field cwe_id

"$FASTRAG" index "$DATA_DIR/kev.jsonl" \
    --corpus "$BUNDLE_DIR/corpora/kev" \
    --embedder bge \
    --format jsonl \
    --text-fields vulnerability_name,short_description,required_action \
    --id-field cve_id \
    --metadata-fields cve_id,vendor_project,product,date_added,due_date,known_ransomware_campaign_use \
    --metadata-types date_added=date,due_date=date,known_ransomware_campaign_use=bool

# 4. Taxonomy (compiled closure JSON — already in the repo)
cp crates/fastrag-cwe/data/cwe-tree-v4.19.1.json "$BUNDLE_DIR/taxonomy/cwe-taxonomy.json"

# 5. vams-findings (optional — skip if JSONL not present, e.g. Track C not yet complete)
VAMS_FINDINGS_JSONL="${VAMS_FINDINGS_JSONL:-../vams/data/synthetic-findings/all.jsonl}"
if [[ -f "$VAMS_FINDINGS_JSONL" ]]; then
  rm -rf "$BUNDLE_DIR/corpora/vams-findings"
  "$FASTRAG" index "$VAMS_FINDINGS_JSONL" \
      --corpus "$BUNDLE_DIR/corpora/vams-findings" \
      --embedder bge \
      --format jsonl \
      --preset tarmo-finding \
      --metadata-fields origin,language,ai_confidence,ai_reasoning,analyst_outcome,analyst_confidence,rejection_reason,rejection_reason_text,synthesis_source
  echo "vams-findings corpus ready"
else
  echo "skipping vams-findings (no JSONL at $VAMS_FINDINGS_JSONL)"
fi

# 6. Manifest
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
KEV_SHA="$(cat "$DATA_DIR/kev.sha256")"
cat > "$BUNDLE_DIR/bundle.json" <<EOF
{
  "schema_version": 1,
  "bundle_id": "vams-lookup-v1",
  "built_at": "$BUILT_AT",
  "corpora": ["cwe", "kev"],
  "taxonomy": "cwe-taxonomy.json",
  "sources": {
    "cwe": {"type": "mitre-xml", "version": "$CWE_VERSION", "sha256": "$CWE_SHA"},
    "kev": {"type": "cisa-feed", "sha256": "$KEV_SHA"}
  }
}
EOF

echo "Bundle ready at $BUNDLE_DIR"
