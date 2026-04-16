#!/usr/bin/env bash
# Enforce the 1.5 GiB image-size budget on the fastrag airgap image.
set -euo pipefail
IMAGE="${1:?usage: $0 IMAGE}"
MAX_BYTES=$((1500 * 1024 * 1024))   # 1.5 GiB

size=$(docker image inspect --format='{{.Size}}' "$IMAGE")
if (( size > MAX_BYTES )); then
    printf "[size-gate] FAIL: %s is %d bytes, max %d\n" "$IMAGE" "$size" "$MAX_BYTES" >&2
    exit 1
fi
printf "[size-gate] OK: %s is %d bytes (under %d)\n" "$IMAGE" "$size" "$MAX_BYTES"
