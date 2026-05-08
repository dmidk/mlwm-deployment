#!/bin/bash

# Container application (defuault to podman if not set)
CONTAINER_APP=${CONTAINER_APP:-podman}

# Configuration
MLWM_LOG_LEVEL=DEBUG
MLWM_IMAGE_NAME="anna:latest"

HTTP_PROXY=""
HTTPS_PROXY=""

# Set MLWM_PULL_PROXY before running this script, e.g.:
#   export MLWM_PULL_PROXY="your.proxy.server:port"
if [ -z "$MLWM_PULL_PROXY" ]; then
	echo "Info: MLWM_PULL_PROXY is not set. Using public DockerHub."
	MLWM_PULL_PROXY=""
    CR_URL="docker.io"
else
	echo "Info: Using proxy $MLWM_PULL_PROXY and internal DockerHub."
    CR_URL="dockerhub.dmi.dk"
fi

# if we're on ARM architecture, use the ARM base image
if [ "$(uname -m)" = "aarch64" ]; then
	echo "Info: Detected ARM architecture. Using ARM base image."
	# dockerhub doesn't have an official pytorch image for ARM, so we use NVIDIA's NGC registry
	if [ -z "$CR_URL" ] || [ "$CR_URL" = "docker.io" ]; then
		CR_URL="nvcr.io"
	fi
	MLWM_BASE_IMAGE="${CR_URL}/nvidia/pytorch:26.01-py3"
else
	echo "Info: Using x86_64 base image."
	MLWM_BASE_IMAGE="${CR_URL}/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
fi

# Check AWS credentials if S3 access is needed
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
	echo "Error: AWS_ACCESS_KEY_ID is not set. Please set it before running this script."
	exit 1
fi
if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
	echo "Error: AWS_SECRET_ACCESS_KEY is not set. Please set it before running this script."
	exit 1
fi
if [ -z "$AWS_DEFAULT_REGION" ]; then
	echo "Error: AWS_DEFAULT_REGION is not set. We set it automatically to eu-central-1."
	AWS_DEFAULT_REGION="eu-central-1"
fi

# Pull base image with proxy
HTTP_PROXY="$MLWM_PULL_PROXY" HTTPS_PROXY="$MLWM_PULL_PROXY" ${CONTAINER_APP} --log-level="$MLWM_LOG_LEVEL" pull "$MLWM_BASE_IMAGE"

# Build image with AWS credentials as build arguments
echo "Running ${CONTAINER_APP} build to create image $MLWM_IMAGE_NAME ..."
${CONTAINER_APP} build \
	--build-arg BASE_IMAGE="$MLWM_BASE_IMAGE" \
	--build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
	--build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
	-t "$MLWM_IMAGE_NAME" \
	-f Containerfile \
	.
