#!/usr/bin/env bash
# scripts/fetch-cwe.sh — download MITRE CWE XML catalog with SHA pinning.
#
# Usage: scripts/fetch-cwe.sh <dest-dir> [version]
#   version defaults to v4.19.1 (matches data/cwe.sha256 pin).
#
# Writes <dest-dir>/cwec_<version>.xml and <dest-dir>/cwe-<version>.sha256.
# The SHA is computed over the distributed zip, not the extracted XML,
# because the zip is what MITRE publishes and the only thing whose bytes
# are stable across unzip implementations. On subsequent runs the SHA is
# re-verified; a mismatch fails loudly (MITRE does not republish a version
# once released, so a mismatch is a supply-chain signal worth investigating).

set -euo pipefail

DEST="${1:-data}"
VERSION="${2:-v4.19.1}"
URL="https://cwe.mitre.org/data/xml/cwec_${VERSION}.xml.zip"
SHA_FILE="$DEST/cwe-${VERSION}.sha256"
ZIP_TMP="$DEST/cwec_${VERSION}.xml.zip.tmp"
XML_OUT="$DEST/cwec_${VERSION}.xml"

mkdir -p "$DEST"

# Atomic write: a curl interrupted by Ctrl+C or network drop must not leave
# a partial zip that would then be SHA-pinned on next run.
curl -sSf --max-time 120 -o "$ZIP_TMP" "$URL"

ACTUAL="$(sha256sum "$ZIP_TMP" | awk '{print $1}')"

if [[ -f "$SHA_FILE" ]]; then
  PINNED="$(cat "$SHA_FILE")"
  if [[ "$ACTUAL" != "$PINNED" ]]; then
    rm -f "$ZIP_TMP"
    echo "CWE SHA mismatch: pinned=$PINNED actual=$ACTUAL" >&2
    echo "Either MITRE republished ${VERSION} (unexpected — investigate)" >&2
    echo "or the download was corrupted." >&2
    exit 1
  fi
else
  echo "$ACTUAL" > "$SHA_FILE"
  echo "Pinned CWE SHA: $ACTUAL" >&2
fi

# unzip -p writes to stdout; pipe through an atomic rename so a broken
# extraction does not leave a half-written XML at the final path.
unzip -p "$ZIP_TMP" "cwec_${VERSION}.xml" > "$XML_OUT.tmp"
mv "$XML_OUT.tmp" "$XML_OUT"
rm -f "$ZIP_TMP"

echo "CWE catalog ready at $XML_OUT ($ACTUAL)"
