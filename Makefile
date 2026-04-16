IMAGE_NAME ?= fastrag
IMAGE_TAG  ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
ISO_OUT    ?= dist/fastrag-airgap.iso

.PHONY: airgap-image
airgap-image:
	DOCKER_BUILDKIT=1 docker build \
		-f docker/Dockerfile.airgap -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: airgap-save
airgap-save: airgap-image
	mkdir -p dist
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | gzip -9 > dist/fastrag-$(IMAGE_TAG).tar.gz
	cd dist && sha256sum fastrag-$(IMAGE_TAG).tar.gz > SHA256SUMS

.PHONY: dvd-iso
dvd-iso: airgap-save
	@bash scripts/build-dvd-iso.sh $(IMAGE_TAG) $(ISO_OUT)

.PHONY: airgap-smoke
airgap-smoke: airgap-image
	bash docker/ci/docker-smoke.sh $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: airgap-size
airgap-size: airgap-image
	bash docker/ci/docker-build-size.sh $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: airgap-no-phone-home
airgap-no-phone-home: airgap-image
	bash docker/ci/docker-no-phone-home.sh $(IMAGE_NAME):$(IMAGE_TAG)
