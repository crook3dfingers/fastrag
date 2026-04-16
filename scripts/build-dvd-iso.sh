#!/usr/bin/env bash
#
# Pack the airgap deliverables onto a single-layer DVD image.
#
# Inputs (produced by `make airgap-save`):
#   dist/fastrag-${TAG}.tar.gz   — `docker save | gzip` of the airgap image
#   dist/SHA256SUMS              — sha256 of the tarball, as consumed by
#                                  `sha256sum -c`
#
# Output:
#   ${OUT}                       — ISO-9660 + Joliet + Rock Ridge disc
#
# Layout inside the ISO:
#   /image/fastrag-*.tar.gz
#   /image/SHA256SUMS
#   /bundles/fastrag-sample/...  (or a placeholder if no fixture exists)
#   /README.md                   (a copy of docs/airgap-install.md)
#
# The 4.4 GiB gate stays ~300 MiB below the 4.7 GiB physical single-layer
# ceiling so the disc writes cleanly on real hardware without us having to
# chase the exact overhead of each burner.
set -euo pipefail

TAG="${1:?usage: $0 IMAGE_TAG ISO_OUT}"
OUT="${2:?usage: $0 IMAGE_TAG ISO_OUT}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

TARBALL="dist/fastrag-${TAG}.tar.gz"
SUMS="dist/SHA256SUMS"
if [[ ! -f "$TARBALL" ]]; then
    echo "[dvd-iso] FAIL: $TARBALL not found — run 'make airgap-save' first" >&2
    exit 1
fi
if [[ ! -f "$SUMS" ]]; then
    echo "[dvd-iso] FAIL: $SUMS not found — run 'make airgap-save' first" >&2
    exit 1
fi

mkdir -p "$WORK/image" "$WORK/bundles/fastrag-sample"
cp "$TARBALL" "$WORK/image/"
cp "$SUMS"    "$WORK/image/"
cp "docs/airgap-install.md" "$WORK/README.md"

if [[ -d tests/fixtures/bundles/sample ]]; then
    cp -r tests/fixtures/bundles/sample/. "$WORK/bundles/fastrag-sample/"
else
    echo "[dvd-iso] WARN: no sample bundle at tests/fixtures/bundles/sample — shipping empty placeholder" >&2
    touch "$WORK/bundles/fastrag-sample/.placeholder"
fi

mkdir -p "$(dirname "$OUT")"

# ISO-9660 volume IDs are capped at 32 characters. Truncate if the tag pushes
# us over (`v0.1.0-367-gc17cad6-dirty` + `FASTRAG-` already leaves ~6 chars).
VOLID="FASTRAG-${TAG}"
if (( ${#VOLID} > 32 )); then
    VOLID="${VOLID:0:32}"
fi

genisoimage -quiet -r -J -V "$VOLID" -o "$OUT" "$WORK"

size=$(stat -c%s "$OUT")
MAX=$((4400 * 1024 * 1024))
if (( size > MAX )); then
    printf '[dvd-iso] FAIL: %s is %d bytes, max %d\n' "$OUT" "$size" "$MAX" >&2
    exit 1
fi
printf '[dvd-iso] OK: %s (%d bytes, under %d)\n' "$OUT" "$size" "$MAX"
