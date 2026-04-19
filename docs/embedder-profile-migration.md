# Embedder Profile Migration

FastRAG retrieval commands now expect embedder configuration to live in
`fastrag.toml` under `[embedder.profiles.<name>]`. The operator-facing change is
simple: define the backend once, then select it with `--embedder-profile`.

## Removed surface

| Old surface | Use now |
|---|---|
| `--embedder bge` | Define a BGE profile and run with `--embedder-profile <name>` |
| `--model-path <path>` | Define a `backend = "llama-cpp"` profile with `model = "/path/to/model.gguf"`; install `llama-server` or set `LLAMA_SERVER_PATH` |
| `--embedder openai` | Define an OpenAI profile with `model = "text-embedding-3-small"` or `"text-embedding-3-large"` |
| `--embedder ollama --ollama-model <model>` | Define an Ollama profile with `backend = "ollama"` and `model = "<model>"` |
| `--ollama-url <url>` as the primary setup path | Put `base_url = "<url>"` in the selected Ollama profile |
| `--embedder qwen3-q8` | Define a `backend = "llama-cpp"` profile with `model = "/path/to/model.gguf"`; install `llama-server` or set `LLAMA_SERVER_PATH` |

Minimal replacement shape:

```toml
[embedder]
default_profile = "prod"

[embedder.profiles.prod]
backend = "ollama"
model = "mixedbread-ai/mxbai-embed-large-v1"
base_url = "http://ollama:11434"
use_catalog_defaults = true
```

Then run:

```bash
fastrag index ./docs --corpus ./corpus --config ./fastrag.toml --embedder-profile prod
fastrag query "payment terms" --corpus ./corpus --config ./fastrag.toml --embedder-profile prod
```

## VAMS update summary

- Ship `fastrag.toml` with the deployment and mount it into the fastrag
  container.
- Put the VAMS retrieval embedder in `[embedder.profiles.vams]`.
- Start `serve-http` with `--config /etc/fastrag/fastrag.toml --embedder-profile vams`.
- Keep the findings corpus mount separate from the bundle mount; only the
  embedder selection moved.

## Downstream issue draft: pentest-scribe

Title: migrate fastrag integration to embedder profiles

Body:

`fastrag` retrieval commands no longer use direct preset flags as the primary
operator surface. Update the deployment to ship a `fastrag.toml`, define a
named embedder profile, and switch the runtime command to
`--config <path> --embedder-profile <name>`. Remove hard-coded `--embedder`,
`--ollama-model`, and Qwen preset assumptions from compose/scripts/docs.

## Downstream issue draft: pentest-storm

Title: switch fastrag runtime to profile-based embedder config

Body:

FastRAG now resolves retrieval embedders from `fastrag.toml`. Add a checked-in
example config, mount it into the service, and update any `index`, `query`, or
`serve-http` invocations to pass `--config` plus `--embedder-profile`. Treat
endpoint/model selection as profile data, not CLI preset wiring.
