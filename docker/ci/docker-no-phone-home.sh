#!/usr/bin/env bash
# Boot the fastrag airgap image with --network=none and verify it never
# attempts outbound traffic. Expected behaviour: the container exits quickly
# because BUNDLE_NAME is unset, and its logs contain no DNS / connect-failure
# symptoms.
set -euo pipefail
IMAGE="${1:?usage: $0 IMAGE}"

CID=$(docker run -d --rm --network=none \
    -e BUNDLE_NAME= \
    "$IMAGE" || true)

if [[ -z "${CID}" ]]; then
    echo "[phone-home] FAIL: container failed to start" >&2
    exit 1
fi

# Give the entrypoint a moment to emit its error and exit.
sleep 2

running=$(docker ps -q --filter "id=${CID}" || true)
logs=$(docker logs "${CID}" 2>&1 || true)

# Clean up regardless.
if [[ -n "${running}" ]]; then
    docker stop "${CID}" >/dev/null || true
fi
docker rm -f "${CID}" >/dev/null 2>&1 || true

if grep -Eqi 'dns|name resolution|could not resolve|connection refused.*[0-9]+\.[0-9]+' <<<"${logs}"; then
    echo "[phone-home] FAIL: outbound attempt detected in logs" >&2
    echo "--- container logs ---" >&2
    printf '%s\n' "${logs}" >&2
    exit 1
fi

echo "[phone-home] OK: no outbound traffic attempted"
