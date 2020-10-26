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
RELEASE_NUMBER := $(shell echo $(GIT_BRANCH) | sed -nE 's:releases/(.*):\1:p')
ifeq ($(RELEASE_NUMBER),)
	RELEASE_NUMBER := $(shell echo $(GIT_TAG) | sed -nE 's:final/(.*):\1:p')
endif


TAG_DEV = dev
TAG_RC = rc
TAG_RELEASE = release
TAG_DEFAULT = latest

NO_CACHE =
ifdef DOCKER_NO_CACHE
	ifeq ($(DOCKER_NO_CACHE), true)
		NO_CACHE = --no-cache
	endif
endif
BUILD_ARGS_0 = --build-arg TORCH_CUDA_ARCH_LIST=$(TORCH_CUDA_ARCH_LIST)
BUILD_ARGS_1 = --build-arg CUDAGL_TAG=$(CUDAGL_TAG)
DOCKER_BUILD_ARGS = $(BUILD_ARGS_0) $(BUILD_ARGS_1) $(NO_CACHE)

ifeq ($(GIT_BRANCH), dev)
	VERSION := $(TAG_DEV)
else ifneq ($(RELEASE_NUMBER),)
  ifeq ($(GIT_TAG),)
		VERSION := $(TAG_RC)-$(RELEASE_NUMBER).$(GIT_SHORT_COMMIT)
	else
		VERSION := $(TAG_RELEASE)-$(RELEASE_NUMBER)
	endif
else
	VERSION := $(TAG_DEFAULT)
endif



.PHONY: help
help:
	$(info Available make targets:)
	@egrep '^(.+)\:\ ##\ (.+)' ${MAKEFILE_LIST} | column -t -c 2 -s ':#'

.PHONY: build-args
build-args: ## Output build args
	@echo $(DOCKER_BUILD_ARGS)


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

.PHONY: serve
serve: ## Launch a serving server on requested port
	$(info *** Launch a serving server on requested port)
	@docker run --rm -ti \
		--label COMMIT=$(GIT_COMMIT) \
		--label BRANCH=$(GIT_BRANCH) \
		--volume /mnt/hdd/omatai/models:/workdir/models \
		--volume ~/.aws:/root/.aws \
		--volume ~/.pgpass:/root/.pgpass \
		--publish 5555:5555 \
		xihelm/$(REPO):$(VERSION) python3 -u ./vegai/scripts/serve.py spawn_server --host=* --port=5555

.PHONY: ci-push
ci-push: ## Push docker image
	$(info *** Pushing docker image: xihelm/$(REPO):$(VERSION))
	@docker push xihelm/$(REPO):$(VERSION)

