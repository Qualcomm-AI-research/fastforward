##
## We use the `make` tool for simple tasks. For a gentle introduction see:
## https://opensource.com/article/18/8/what-how-makefile .
##

SHELL = /bin/bash

## Variables with their default value:

DOCKER_REGISTRY ?=

VER_PYTHON ?= 3.10
IMAGE_NAME ?= $(if $(DOCKER_REGISTRY),$(DOCKER_REGISTRY)/users/$(USER)/)fastforward-py$(VER_PYTHON)
IMAGE_TAG ?= latest

CONTAINER_NAME ?= fastforward_$(USER)
SSH_PORT ?= 54320
OTHER_SSHD_FLAGS ?=

VERIFY_TARGET ?= --lint --format --mypy --tests

# for support of "make boot-attached GPUS=2,3" for example
GPUS ?= all

## Prepare flags for docker command lines:

VOLUME_FLAGS ?=
COMMON_FLAGS = $$($(CURDIR)/scripts/clone-user $(IMAGE_NAME):$(IMAGE_TAG)) $(VOLUME_FLAGS) -v $$HOME:$$HOME -v $(CURDIR):$(CURDIR) -w $(CURDIR) --gpus $(GPUS) --name $(CONTAINER_NAME) --network=host --rm $(IMAGE_NAME):$(IMAGE_TAG)
ATTACHED_MODE_FLAGS = -t -i
DETACHED_MODE_FLAGS = -d

BOOT_DEPENDENCIES = stop

## List of phony targets: their recipe is always executed when invoked:
.PHONY: build run sshd test stop

build:
	docker build --build-arg VER_PYTHON="$(VER_PYTHON)" --progress=plain --pull --file docker/Dockerfile --tag $(IMAGE_NAME):$(IMAGE_TAG) $(CURDIR)
	@echo "Successfully built the docker image $(IMAGE_NAME):$(IMAGE_TAG)"
	docker images --format "table {{.Size}}\t{{.Repository}}\t{{.Tag}}" "$(IMAGE_NAME):$(IMAGE_TAG)"

push:
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

## Targets for booting a container and attach to it:
run: $(BOOT_DEPENDENCIES)
	docker run $(ATTACHED_MODE_FLAGS) $(COMMON_FLAGS)

## Targets for booting a container but not attach to it:
sshd: $(BOOT_DEPENDENCIES)
	docker run -e MY_ENV=test $(DETACHED_MODE_FLAGS) $(COMMON_FLAGS) sh /usr/local/bin/run_sshd -o AuthorizedKeysFile=$$HOME/.ssh/authorized_keys -o Port=$(SSH_PORT) -D

# Verify your code before a commit for example
test: $(BOOT_DEPENDENCIES)
	docker run $(COMMON_FLAGS) scripts/verify $(VERIFY_TARGET)


## Targets for stopping a container and perform cleanup:
stop:
	@docker stop $(CONTAINER_NAME) &> /dev/null || true
	@docker rm $(CONTAINER_NAME) &> /dev/null || true
	@echo 'Ready to boot new container'

