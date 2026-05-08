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
dt = dt.astim$ezone(timezone.utc)
print(dt.strftime("%Y-%m-%dT%H%M%SZ"))
EOF
  fi
}

# Create the inference working directory if it doesn't exist
mkdir -p ./inference_workdir/

# danra.danra_sl_state=https://object-store.os-api.cci1.ecmwf.int/danra/v0.6.0dev1/single_levels.zarr/,\
# danra.danra_static=https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/single_levels.zarr/"}

# prepare environment variables for container
ANALYSIS_TIME=$(format_analysis_time "${ANALYSIS_TIME}")
DINI_ZARR="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/single_levels.zarr/"
DINI_ZARR_PL="s3://harmonie-zarr/dini/control/${ANALYSIS_TIME}/pressure_levels.zarr/"

#DINI_ZARR="https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/single_levels.zarr/"
#DINI_ZARR_PL="https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/pressure_levels.zarr/"

# Overwrite input paths to datastore. The container will use these paths instead of those in the datastore config file when creating the inference dataset.
DATASTORE_INPUT_PATHS="danra_model1_config.danra_sl_state=${DINI_ZARR},danra_model1_config.danra_static=${DINI_ZARR},danra_model1_config.danra_pl_state=${DINI_ZARR_PL},danra_model1_config.danra_forcing=${DINI_ZARR}"
DATASTORE_RENAME_VARIABLES="${DATASTORE_RENAME_VARIABLES:-}"
TIME_DIMENSIONS="time"
INFERENCE_WORKDIR="$(pwd)/inference_workdir/"

podman run --rm \
    --device nvidia.com/gpu=all \
    --shm-size=32g \
    -v ${INFERENCE_WORKDIR}:/workspace/inference_workdir:Z \
    -v $(pwd)/src:/workspace/src:Z \
    -e DATASTORE_INPUT_PATHS="${DATASTORE_INPUT_PATHS}" \
    -e DATASTORE_RENAME_VARIABLES="${DATASTORE_RENAME_VARIABLES}" \
    -e TIME_DIMENSIONS="${TIME_DIMENSIONS}" \
    -e ANALYSIS_TIME="${ANALYSIS_TIME}" \
    -e FORECAST_DURATION="${FORECAST_DURATION}" \
    localhost/anna:latest
