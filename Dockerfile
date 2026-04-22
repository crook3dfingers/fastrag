# Multi-stage build for fastrag-cli.
# Final image is a distroless static base — no shell, no package manager.

FROM public.ecr.aws/docker/library/rust:1.89-trixie AS builder
WORKDIR /build
COPY enterprise-certs/ /tmp/enterprise-certs/
RUN set -eu; \
    if [ -f /tmp/enterprise-certs/ca-bundle.crt ]; then \
        cp /tmp/enterprise-certs/ca-bundle.crt /usr/local/share/ca-certificates/tarmo-operator-root.crt; \
        update-ca-certificates; \
    fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CARGO_HTTP_CAINFO=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release -p fastrag-cli --no-default-features \
        --features language-detection,retrieval,store

FROM public.ecr.aws/docker/library/debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libbz2-1.0 \
        libgcc-s1 \
        liblzma5 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/fastrag /usr/local/bin/fastrag
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
USER nobody:nogroup
EXPOSE 8081
ENV FASTRAG_LOG=info \
    FASTRAG_LOG_FORMAT=json \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENTRYPOINT ["/usr/local/bin/fastrag"]
CMD ["serve-http", "--corpus", "/var/lib/fastrag/corpus", "--port", "8081"]
