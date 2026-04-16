#!/usr/bin/env bash
# FastRAG airgap entrypoint.
#
# Requires:
#   BUNDLE_NAME           — directory under /var/lib/fastrag/bundles/ to load.
#   FASTRAG_ADMIN_TOKEN   — admin token for /admin/reload (optional).
#   FASTRAG_TOKEN         — read token for /query, /cve, /cwe, etc. (optional).
#
# fastrag spawns its own llama-server subprocesses for the Qwen3 embedder and
# the BGE-reranker-v2-m3 reranker, resolving GGUF files via $FASTRAG_MODEL_DIR.
# No network access is attempted at start-up.

set -euo pipefail

BUNDLE_NAME="${BUNDLE_NAME:-}"
BUNDLES_DIR="${BUNDLES_DIR:-/var/lib/fastrag/bundles}"
PORT="${PORT:-8080}"

# fastrag serve-http binds to 127.0.0.1 by default; inside a container the
# docker port mapping can only reach it via the container interface, so
# listen on all interfaces here.
export FASTRAG_HOST="${FASTRAG_HOST:-0.0.0.0}"

if [[ -z "${BUNDLE_NAME}" ]]; then
    echo "[entrypoint] BUNDLE_NAME env var required" >&2
    exit 1
fi

BUNDLE_PATH="${BUNDLES_DIR}/${BUNDLE_NAME}"
if [[ ! -d "${BUNDLE_PATH}" ]]; then
    echo "[entrypoint] bundle directory not found: ${BUNDLE_PATH}" >&2
    exit 1
fi

# --corpus is required by serve-http; register every corpus the bundle
# supplies as a named entry so /query and /similar route by name.
corpus_args=()
for cname in cve cwe kev; do
    cdir="${BUNDLE_PATH}/corpora/${cname}"
    if [[ -d "${cdir}" ]]; then
        corpus_args+=(--corpus "${cname}=${cdir}")
    fi
done
if [[ ${#corpus_args[@]} -eq 0 ]]; then
    echo "[entrypoint] bundle at ${BUNDLE_PATH} has no corpora/ subdirs" >&2
    exit 1
fi

auth_args=()
if [[ -n "${FASTRAG_TOKEN:-}" ]]; then
    auth_args+=(--token "${FASTRAG_TOKEN}")
fi
if [[ -n "${FASTRAG_ADMIN_TOKEN:-}" ]]; then
    auth_args+=(--admin-token "${FASTRAG_ADMIN_TOKEN}")
fi

exec fastrag serve-http \
    "${corpus_args[@]}" \
    --bundle-path "${BUNDLE_PATH}" \
    --bundles-dir "${BUNDLES_DIR}" \
    --embedder qwen3-q8 \
    --rerank llama-cpp \
    --port "${PORT}" \
    "${auth_args[@]}"
