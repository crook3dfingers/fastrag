# fastrag airgap image

Debian-12 slim runtime with `fastrag`, `llama-server`, and pre-staged GGUFs
for a completely offline security lookup + retrieval service. Built by
`docker/Dockerfile.airgap`.

## Layout at runtime

```
/usr/local/bin/fastrag              # fastrag CLI (ENTRYPOINT via tini)
/usr/local/bin/llama-server         # spawned as a subprocess per role
/opt/fastrag/lib/                   # libonnxruntime.so.* (on LD_LIBRARY_PATH)
/opt/fastrag/models/                # $FASTRAG_MODEL_DIR
    Qwen3-Embedding-0.6B-Q8_0.gguf  # embedder (qwen3-q8)
    bge-reranker-v2-m3-q8_0.gguf    # reranker (llama-cpp)
/var/lib/fastrag/bundles/           # mount your bundles here
```

## Environment variables

| Variable                | Required | Purpose                                                 |
|-------------------------|----------|---------------------------------------------------------|
| `BUNDLE_NAME`           | yes      | Directory under `/var/lib/fastrag/bundles/` to load.    |
| `FASTRAG_TOKEN`         | no       | Read token for `/query`, `/cve`, `/cwe`, etc.           |
| `FASTRAG_ADMIN_TOKEN`   | no       | Admin token for `/admin/reload`. Must differ from read. |
| `BUNDLES_DIR`           | no       | Override bundles root (default `/var/lib/fastrag/bundles`). |
| `PORT`                  | no       | Listen port inside the container (default `8080`).      |
| `FASTRAG_MODEL_DIR`     | preset   | Points at the pre-staged GGUFs. Do not override.        |

## Run

```bash
docker run --rm -p 8080:8080 \
    -v /path/to/bundles:/var/lib/fastrag/bundles:ro \
    -e BUNDLE_NAME=fastrag-20260416 \
    -e FASTRAG_TOKEN=<read-token> \
    -e FASTRAG_ADMIN_TOKEN=<admin-token> \
    fastrag:<tag>
```

## Build, size, audit, smoke

All three gates are wired into `make`:

```bash
make airgap-image            # docker build
make airgap-size             # fails if image > 1.5 GiB
make airgap-no-phone-home    # boots with --network=none, checks logs
make airgap-smoke            # fixture bundle + /health /ready /cwe/relation
```
