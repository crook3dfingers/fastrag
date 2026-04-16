#!/usr/bin/env bash
# End-to-end smoke test for the fastrag airgap image.
#
# Builds a minimal bundle fixture on the fly (bundle.json schema v1 + a
# schema-v2 taxonomy + three empty corpora with the identity.model_id that
# matches --embedder qwen3-q8), mounts it, starts the container, then probes
# /health, /ready, and /cwe/relation. Pass tokens differ to exercise the
# admin/read split.
set -euo pipefail
IMAGE="${1:?usage: $0 IMAGE}"
CONTAINER="${CONTAINER:-fastrag-smoke}"
HOST_PORT="${HOST_PORT:-18080}"
READY_TIMEOUT="${READY_TIMEOUT:-240}"

BUNDLE_DIR=$(mktemp -d)
cleanup() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    rm -rf "${BUNDLE_DIR}"
}
trap cleanup EXIT

# --- Generate a minimal smoke-test bundle.
mkdir -p "${BUNDLE_DIR}/taxonomy" \
         "${BUNDLE_DIR}/corpora/cve" \
         "${BUNDLE_DIR}/corpora/cwe" \
         "${BUNDLE_DIR}/corpora/kev"

cat > "${BUNDLE_DIR}/bundle.json" <<'JSON'
{
    "schema_version": 1,
    "bundle_id": "smoke",
    "built_at": "2026-04-16T00:00:00Z",
    "corpora": ["cve", "cwe", "kev"],
    "taxonomy": "cwe-taxonomy.json"
}
JSON

cat > "${BUNDLE_DIR}/taxonomy/cwe-taxonomy.json" <<'JSON'
{"schema_version":2,"version":"4.15","view":"1000","closure":{"89":[89]},"parents":{"89":[]}}
JSON

# Minimal corpus manifest — embed_loader only reads identity.model_id for
# read-path auto-detection. The smoke path never actually queries a corpus,
# so empty index.bin / entries.bin are fine.
for c in cve cwe kev; do
    cat > "${BUNDLE_DIR}/corpora/${c}/manifest.json" <<'JSON'
{"identity":{"model_id":"Qwen/Qwen3-Embedding-0.6B-GGUF@Q8_0"}}
JSON
    : > "${BUNDLE_DIR}/corpora/${c}/index.bin"
    : > "${BUNDLE_DIR}/corpora/${c}/entries.bin"
done

# --- Launch.
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER}" \
    -p "${HOST_PORT}:8080" \
    -v "${BUNDLE_DIR}:/var/lib/fastrag/bundles/smoke:ro" \
    -e BUNDLE_NAME=smoke \
    -e FASTRAG_ADMIN_TOKEN=smoke-admin \
    -e FASTRAG_TOKEN=smoke-read \
    "${IMAGE}" >/dev/null

# --- Wait for readiness. llama-server subprocesses take time to boot.
for (( i = 0; i < READY_TIMEOUT; i++ )); do
    if curl -sf "http://127.0.0.1:${HOST_PORT}/ready" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

# --- Probe endpoints.
http_code() {
    curl -fsS -o /dev/null -w '%{http_code}' "$@"
}

if [[ "$(http_code "http://127.0.0.1:${HOST_PORT}/health")" != "200" ]]; then
    echo "[smoke] FAIL: /health did not return 200" >&2
    docker logs "${CONTAINER}" >&2 || true
    exit 1
fi

if [[ "$(http_code "http://127.0.0.1:${HOST_PORT}/ready")" != "200" ]]; then
    echo "[smoke] FAIL: /ready did not return 200 within ${READY_TIMEOUT}s" >&2
    curl -sS "http://127.0.0.1:${HOST_PORT}/ready" >&2 || true
    docker logs "${CONTAINER}" >&2 || true
    exit 1
fi

if [[ "$(http_code -H 'x-fastrag-token: smoke-read' \
        "http://127.0.0.1:${HOST_PORT}/cwe/relation?cwe_id=89")" != "200" ]]; then
    echo "[smoke] FAIL: /cwe/relation did not return 200" >&2
    docker logs "${CONTAINER}" >&2 || true
    exit 1
fi

echo "[smoke] OK"
