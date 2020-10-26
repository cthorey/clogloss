SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:

REPO = vegai
DOCKERFILE = ./build/Dockerfile
CUDAGL_TAG='10.1-devel-ubuntu18.04'
# supported NVIDIA Architecture
# Should be smth torch_cuda_arch_list="6.1;7.5" - where the architecture can be
# found using sudo nvidia-container-cli --load-kmods info
TORCH_CUDA_ARCH_LIST="6.1;7.5"

GIT_COMMIT := $(shell git rev-parse HEAD)
GIT_COMMIT_SHORT := $(shell git rev-parse --short HEAD)
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_TAG := $(shell git tag --points-at $(GIT_COMMIT))


VERSION = dev

NO_CACHE =
ifdef DOCKER_NO_CACHE
	ifeq ($(DOCKER_NO_CACHE), true)
		NO_CACHE = --no-cache
	endif
endif
BUILD_ARGS_0 = --build-arg TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST)
BUILD_ARGS_1 = --build-arg CUDAGL_TAG=$(CUDAGL_TAG)
DOCKER_BUILD_ARGS = $(BUILD_ARGS_0) $(BUILD_ARGS_1) $(NO_CACHE)

.PHONY: help
help:
	$(info Available make targets:)
	@egrep '^(.+)\:\ ##\ (.+)' ${MAKEFILE_LIST} | column -t -c 2 -s ':#'

.PHONY: version
version:  ## Display the current version
	@echo $(VERSION)

.PHONY: build
build: ## Build docker image
	$(info *** Building docker image: xihelm/$(REPO):$(VERSION))
	@docker build \
		$(DOCKER_BUILD_ARGS) \
		--tag xihelm/$(REPO):$(VERSION) \
		--label COMMIT=$(GIT_COMMIT) \
		--label BRANCH=$(GIT_BRANCH) \
		--file $(DOCKERFILE) \
		.

.PHONY: notebook
notebook: ## Launch a notebook server on 8890
	$(info *** Launch a notebook server on 8890)
	@./scripts/notebook.py --port=8890

