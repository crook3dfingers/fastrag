#!/usr/bin/env bash
# scripts/smoke_viper_assist.sh — run the VIPER Assist retrieval smoke test.
#
# Indexes a curated 11-row VIPER corpus subset with the `viper-assist` preset
# + Nomic Embed Text v1.5 (llama-cpp backend), runs the issue-#74 prompts,
# and asserts each top-k contains an allowlisted VIPER page id.
#
# Required:
#   - GGUF at $VIPER_NOMIC_GGUF (default
#     /var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf).
#   - llama-server in PATH (or LLAMA_SERVER_PATH set).
#
# Usage:
#   scripts/smoke_viper_assist.sh           # run the smoke
#   VIPER_NOMIC_GGUF=/abs/path scripts/smoke_viper_assist.sh
#   FASTRAG_LOG=debug scripts/smoke_viper_assist.sh   # verbose
set -euo pipefail

DEFAULT_GGUF="/var/lib/fastrag/models/nomic-embed-text-v1.5.Q5_K_M.gguf"
GGUF_PATH="${VIPER_NOMIC_GGUF:-$DEFAULT_GGUF}"

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "error: Nomic GGUF not found at: $GGUF_PATH" >&2
    echo "       set VIPER_NOMIC_GGUF=/abs/path or place the file there." >&2
    exit 2
fi

export FASTRAG_LLAMA_TEST=1
export VIPER_NOMIC_GGUF="$GGUF_PATH"

cd "$(dirname "$0")/.."

exec cargo test \
    -p fastrag-cli \
    --features retrieval,store \
    --test viper_assist_smoke \
    -- --ignored --nocapture
