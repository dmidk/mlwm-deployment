#!/bin/bash

# This script runs the inference container using initial conditions from DINI
# stored on AWS

# The script takes only one argument: the analysis time to use for inference,
# in ISO8601 format (e.g. 2025-11-05T090000Z). If "Z" is omitted, UTC is
# assumed. An optional second argument can be provided to specify the forecast
# duration in ISO8601 duration format (e.g. PT18H for 18 hours). If not
# provided, the default is PT18H.

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ] ; then
    echo "Usage: $0 <ANALYSIS_TIME> [<FORECAST_DURATION>]" >&2
    echo "" >&2
    echo "  ANALYSIS_TIME: the analysis time to start the forecast from in ISO8601 format" >&2
    echo "  FORECAST_DURATION: the duration of the forecast in ISO8601 duration format (default PT18H)" >&2
    exit 1
fi
ANALYSIS_TIME="$1"
if [ "$#" -eq 2 ] ; then
    FORECAST_DURATION="$2"
else
    FORECAST_DURATION="PT18H"
fi

# function to format analysis time to remove colons and ensure UTC 'Z' suffix
format_analysis_time() {
  local iso="$1"

  if [[ -z "$iso" ]]; then
    echo "format_analysis_time: missing ISO8601 datetime" >&2
    return 1
  fi

  if date -u -d "1970-01-01T00:00:00Z" >/dev/null 2>&1; then
    # GNU date (Linux)
    date -u -d "$iso" +"%Y-%m-%dT%H%M%SZ" || return 1
  else
    # macOS / BSD fallback using Python stdlib
    python3 - <<'EOF' "$iso"
from datetime import datetime, timezone
import sys

dt = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
dt = dt.astimezone(timezone.utc)
print(dt.strftime("%Y-%m-%dT%H%M%SZ"))
EOF
  fi
}

# Create the inference working directory if it doesn't exist
mkdir -p ./inference_workdir/

# prepare environment variables for container
ANALYSIS_TIME=$(format_analysis_time "${ANALYSIS_TIME}")
DINI_ZARR="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/single_levels.zarr/"
DATASTORE_INPUT_PATHS="danra.danra_surface=${DINI_ZARR},danra.danra_static=${DINI_ZARR}"
TIME_DIMENSIONS="time"
INFERENCE_WORKDIR="$(pwd)/inference_workdir/"

# Check if hostname starts with "spark" to determine GPU device mounting strategy
if [[ $HOSTNAME == spark* ]]; then
  # On spark devices: use --gpus all flag
  podman run --rm \
    --gpus all \
    --shm-size=32g \
    -v ${INFERENCE_WORKDIR}:/workspace/inference_workdir:Z \
    -e DATASTORE_INPUT_PATHS="${DATASTORE_INPUT_PATHS}" \
    -e TIME_DIMENSIONS="${TIME_DIMENSIONS}" \
    -e ANALYSIS_TIME="${ANALYSIS_TIME}" \
    -e FORECAST_DURATION="${FORECAST_DURATION}" \
    localhost/surface-dummy-model_dini:latest
else
  # On non-spark devices: use manual device mounting
  podman run --rm \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools \
    --device /dev/nvidia-modeset \
    -v /lib/x86_64-linux-gnu/libcuda.so.1:/lib/x86_64-linux-gnu/libcuda.so.1:ro \
    -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
    -v /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
    --shm-size=32g \
    -v ${INFERENCE_WORKDIR}:/workspace/inference_workdir:Z \
    -e DATASTORE_INPUT_PATHS="${DATASTORE_INPUT_PATHS}" \
    -e TIME_DIMENSIONS="${TIME_DIMENSIONS}" \
    -e ANALYSIS_TIME="${ANALYSIS_TIME}" \
    -e FORECAST_DURATION="${FORECAST_DURATION}" \
    localhost/surface-dummy-model_dini:latest
fi