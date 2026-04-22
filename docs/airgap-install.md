# Airgap installation

The DVD contains the fastrag Docker image, a sample bundle, and this README. The operator flow:

If you are rebuilding the standard FastRAG image in a TLS-inspected enterprise
network before producing the DVD payload, place the operator root CA at
`enterprise-certs/ca-bundle.crt` in the repo before `docker build`. The
standard Dockerfile now installs that optional CA into the build container so
`apt`, Cargo, and git-backed fetches trust the same root bundle.

## 1. Mount the disc

```bash
sudo mount /dev/sr0 /mnt/dvd
```

If `/dev/sr0` is not the right device, `lsblk` lists optical drives with their `rom` mount targets.

## 2. Verify and load the image

```bash
cd /mnt/dvd/image
sha256sum -c SHA256SUMS
docker load < fastrag-*.tar.gz
```

`docker load` accepts the gzipped tarball as-is. The loaded image tag is printed on the last line — use it in step 4.

## 3. Stage the sample bundle

```bash
sudo mkdir -p /var/lib/fastrag/bundles
sudo cp -r /mnt/dvd/bundles/fastrag-sample /var/lib/fastrag/bundles/
```

For production bundles shipped on a later disc, copy them into the same directory alongside the sample.

## 4. Generate tokens and start the container

```bash
READ_TOKEN=$(openssl rand -hex 32)
ADMIN_TOKEN=$(openssl rand -hex 32)

docker run -d --name fastrag --restart unless-stopped -p 8080:8080 \
    -v /var/lib/fastrag/bundles:/var/lib/fastrag/bundles:ro \
    -e BUNDLE_NAME=fastrag-sample \
    -e FASTRAG_TOKEN="$READ_TOKEN" \
    -e FASTRAG_ADMIN_TOKEN="$ADMIN_TOKEN" \
    fastrag:X.Y.Z
```

Replace `fastrag:X.Y.Z` with the tag printed by `docker load`. Store `$READ_TOKEN` and `$ADMIN_TOKEN` somewhere durable — regenerating them requires a container restart.

The published airgap image now writes a temporary `fastrag.toml` at startup and
uses an internal `airgap` embedder profile that points at the bundled
Qwen3 GGUF through the llama-cpp backend. Operators do not need to mount a
separate config file for this DVD path unless they are rebuilding the image or
changing the embedded model/runtime behavior.

## 5. Confirm readiness

```bash
curl -fsS http://localhost:8080/ready
```

`/ready` returns 200 once the bundle, embedder, and reranker subprocesses are all live. Initial start-up takes ~30–120 s while `llama-server` loads the GGUFs.

## Updating bundles

When a new bundle arrives on DVD or USB:

1. Copy the new bundle directory alongside the existing one under `/var/lib/fastrag/bundles/`.
2. Issue an admin reload with the new directory name:

   ```bash
   curl -X POST http://localhost:8080/admin/reload \
       -H "x-fastrag-admin-token: $ADMIN_TOKEN" \
       -H 'content-type: application/json' \
       -d '{"bundle_path":"fastrag-20260501"}'
   ```

The reload is atomic: in-flight queries finish against the prior bundle, and the swap happens under a mutex so concurrent reloads serialize. Prior bundle directories can stay on disk for rollback — re-issue `/admin/reload` against the older directory name to revert.

`bundle_path` is resolved relative to the container's `/var/lib/fastrag/bundles`. Absolute paths and any form of `..` are rejected with HTTP 400 `path_escape`.

## Sizing

Peak memory during reload is roughly 2× the resident bundle size, since the new index loads before the old one is released. Size the host with at least 4 GiB of RAM free above the steady-state bundle footprint.
