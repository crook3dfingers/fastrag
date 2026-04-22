#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

grep -Fq "FROM public.ecr.aws/docker/library/rust:1.89-trixie AS builder" "$ROOT_DIR/Dockerfile"
grep -Fq "FROM public.ecr.aws/docker/library/debian:trixie-slim" "$ROOT_DIR/Dockerfile"
grep -Fq "FROM public.ecr.aws/docker/library/rust:1.89-trixie AS rust-builder" "$ROOT_DIR/docker/Dockerfile.airgap"
grep -Fq "FROM public.ecr.aws/docker/library/debian:trixie-slim AS llama-builder" "$ROOT_DIR/docker/Dockerfile.airgap"
grep -Fq "FROM public.ecr.aws/docker/library/debian:trixie-slim" "$ROOT_DIR/docker/Dockerfile.airgap"

echo "PASS: FastRAG Dockerfiles use mirrored official base images"
