#!/usr/bin/env bash
set -euo pipefail

INPUT=$(cat)
STOP_HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')

# Prevent infinite loops — allow stop if already in hook-driven continuation
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
if [ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ]; then
  exit 0
fi

# Get the last 50 lines of the transcript (enough to cover the current turn)
TAIL=$(tail -50 "$TRANSCRIPT")

# Check if doc-editor skill was invoked recently
if echo "$TAIL" | grep -q '"doc-editor"'; then
  # Check if Edit or Write tool was called after the skill
  LAST_DOC_EDITOR_LINE=$(echo "$TAIL" | grep -n '"doc-editor"' | tail -1 | cut -d: -f1)
  REMAINING=$(echo "$TAIL" | tail -n +"$LAST_DOC_EDITOR_LINE")

  if echo "$REMAINING" | grep -qE '"name"\s*:\s*"(Edit|Write)"'; then
    # Edit/Write found after doc-editor — allow stop
    exit 0
  else
    # doc-editor called but no Edit/Write followed — block stop
    echo '{"decision":"block","reason":"doc-editor skill was invoked but no Edit or Write followed. Apply the cleaned prose to the file now."}'
    exit 0
  fi
fi

# No doc-editor in recent transcript — allow stop
exit 0
