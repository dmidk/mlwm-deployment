#!/bin/bash

# Configuration
MLWM_LOG_LEVEL=DEBUG
MLWM_IMAGE_NAME="surface-dummy-model_dini:latest"

HTTP_PROXY=""
HTTPS_PROXY=""

# Set MLWM_PULL_PROXY before running this script, e.g.:
#   export MLWM_PULL_PROXY="your.proxy.server:port"
if [ -z "$MLWM_PULL_PROXY" ]; then
	echo "Info: MLWM_PULL_PROXY is not set. Using public DockerHub."
	MLWM_PULL_PROXY=""
    CR_URL="dockerhub.com"
else
	echo "Info: Using proxy $MLWM_PULL_PROXY and internal DockerHub."
    CR_URL="dockerhub.dmi.dk"
fi

MLWM_BASE_IMAGE="$CR_URL/pytorchlightning/pytorch_lightning:base-cuda-py3.12-torch2.6-cuda12.4.1"

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
HTTP_PROXY="$MLWM_PULL_PROXY" HTTPS_PROXY="$MLWM_PULL_PROXY" podman --log-level="$MLWM_LOG_LEVEL" pull "$MLWM_BASE_IMAGE"

# Build image with AWS credentials as build arguments
podman build \
	--build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
	--build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    --build-arg AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
	-t "$MLWM_IMAGE_NAME" \
	-f Containerfile
