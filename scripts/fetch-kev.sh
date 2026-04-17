#!/usr/bin/env bash
# scripts/fetch-kev.sh — download CISA KEV catalog with SHA pinning.
#
# Usage: scripts/fetch-kev.sh <dest-dir>
#
# Writes <dest-dir>/known_exploited_vulnerabilities.json and
# <dest-dir>/kev.sha256. On subsequent runs, re-downloads and verifies
# the SHA matches the pinned version (fails if CISA published a new
# version since last pin — forces an explicit re-pin).

set -euo pipefail

DEST="${1:-data}"
URL="https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
SHA_FILE="$DEST/kev.sha256"
OUT="$DEST/known_exploited_vulnerabilities.json"

mkdir -p "$DEST"
curl -sSf -o "$OUT" "$URL"

ACTUAL="$(sha256sum "$OUT" | awk '{print $1}')"

if [[ -f "$SHA_FILE" ]]; then
  PINNED="$(cat "$SHA_FILE")"
  if [[ "$ACTUAL" != "$PINNED" ]]; then
    echo "KEV SHA mismatch: pinned=$PINNED actual=$ACTUAL" >&2
    echo "Either CISA published a new catalog (expected — update $SHA_FILE)" >&2
    echo "or the download was corrupted." >&2
    exit 1
  fi
else
  echo "$ACTUAL" > "$SHA_FILE"
  echo "Pinned KEV SHA: $ACTUAL" >&2
fi

echo "KEV catalog ready at $OUT ($ACTUAL)"
