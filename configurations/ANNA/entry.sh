#!/usr/bin/env bash
# This script is used to run the inference for the ANNA model.
#
# This script is intended to be run in a container, and assumes that during the
# container image build that the inference artifact was unpacked to
# inference_artifact/. You can also run this script interactively if you have
# extracted the inference artifact yourself.
#
# The selection of datasets to use for input to the model, analysis time and
# forecast duration is controller by the following environment variables:
# DATASTORE_INPUT_PATHS, ANALYSIS_TIME, FORECAST_DURATION and NUM_EVAL_STEPS
# (the latter should be inferred from FORECAST_DURATION, but that is TODO)
#
# - DATASTORE_INPUT_PATHS is a comma-separated list of mappings of
#   {datastore_name}.{input_name}={input_path}
# - ANALYSIS_TIME is the analysis time to start the forecast from is ISO8601
#   format
# - FORECAST_DURATION is the duration of the forecast in ISO8601 duration
#   format and effects the length of the produced inference dataset
# - NUM_EVAL_STEPS is the number of autoregressive steps to run during
#   inference. This should be consistent with FORECAST_DURATION and the model
#   configuration (e.g. if the model was trained on 3-hourly data and
#   FORECAST_DURATION is PT18H then NUM_EVAL_STEPS should be 6

# make this script fail on any error
set -e

## Runtime configuration (variable expected to change on every execution)
# enable use of .env so that during development we can set environment (e.g.
# paths to replace in datastore config)
if [ -f .env ] ; then
    echo "Sourcing local .env file"
    set -a && source .env && set +a
fi

USE_UV=${USE_UV:-true}
if [ "$USE_UV" = true ] ; then
    echo "Using uv to run commands"
    UV_CMD="uv run"
else
    echo "Not using uv to run commands, using plain python"
    UV_CMD=""
fi

# print CUDA debug info
${UV_CMD} python - <<'PY'
import torch, subprocess, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("device capability:", cap)
    print("name:", torch.cuda.get_device_name(0))
    try:
        torch.randn(2, device="cuda")
        print("cuda op: OK")
    except Exception as e:
        print("cuda op failed:", e)
PY

## Model specific inference configuration (same across all executions)
NUM_HIDDEN_DIMS=300
NUM_HIDDEN_DIMS_GRID=150 # To merge from Joels fork.
GRAPH_NAME="hi_lam"
HIEARCHICAL_GRAPH=true
MODEL_TIMESTEP_HOURS=3


# set default override of input paths in the datastore config used for creating the
# inference dataset if environment variable isn't set

# DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS:-"\
# danra.danra_surface=https://object-store.os-api.cci1.ecmwf.int/danra/v0.6.0dev1/single_levels.zarr/,\
# danra.danra_static=https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/single_levels.zarr/"}

# Overwrite paths to use DANRA under development
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS:-"\
danra.danra_sl_state=https://object-store.os-api.cci1.ecmwf.int/danra/v0.6.0dev1/single_levels.zarr/,\
danra.danra_static=https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/single_levels.zarr/"}

TIME_DIMENSIONS=${TIME_DIMENSIONS:-"analysis_time,elapsed_forecast_duration"}
ANALYSIS_TIME=${ANALYSIS_TIME:-"2019-02-04T12:00"}  # assumed to be in UTC
# default forecast duration of 18 hours
FORECAST_DURATION=${FORECAST_DURATION:-"PT18H"}

# compute number of eval steps from forecast duration
if [ -z "${NUM_EVAL_STEPS}" ] ; then
    # check that FORECAST_DURATION is in expected format PT{N}H
    if [[ "${FORECAST_DURATION}" =~ ^PT([0-9]+)H$ ]] ; then
        HOURS="${BASH_REMATCH[1]}"
        NUM_EVAL_STEPS=$((HOURS / MODEL_TIMESTEP_HOURS))
        echo "Inferred NUM_EVAL_STEPS=${NUM_EVAL_STEPS} from FORECAST_DURATION=${FORECAST_DURATION}"
    else
        echo "ERROR: Cannot infer NUM_EVAL_STEPS from FORECAST_DURATION='${FORECAST_DURATION}', please set NUM_EVAL_STEPS explicitly"
        exit 1
    fi
fi

# All working directories (for input data, output data, intermediate files)
# will be created under INFERENCE_WORKDIR
INFERENCE_WORKDIR=${INFERENCE_WORKDIR:-"./inference_workdir"}

echo "Creating forecast using following runtime args:"
echo "  DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS}"
echo "  TIME_DIMENSIONS=${TIME_DIMENSIONS}"
echo "  ANALYSIS_TIME=${ANALYSIS_TIME}"
echo "  FORECAST_DURATION=${FORECAST_DURATION}"
echo "  NUM_EVAL_STEPS=${NUM_EVAL_STEPS}"
echo "  INFERENCE_WORKDIR=${INFERENCE_WORKDIR}"

# set cli argument for creating hierarchical graph if needed
if [ "$HIEARCHICAL_GRAPH" = true ] ; then
    CREATE_GRAPH_ARG="--hierarchical"
else
    CREATE_GRAPH_ARG=""
fi

## Setup working directories
INFERENCE_ARTIFACT_PATH="./inference_artifact"
INPUT_DATASETS_ROOT_PATH="${INFERENCE_WORKDIR}/inputs"
OUTPUT_DATASETS_ROOT_PATH="${INFERENCE_WORKDIR}/outputs"
mkdir -p ${OUTPUT_DATASETS_ROOT_PATH}

# disable weights and biases logging, without this --eval with neural-lam fails
# because it tries to set up the logging and there is no WANDB_API_KEY set
${UV_CMD} wandb disabled

## 1. Create inference dataset
# This uses a cli stored within mlwm to called mllam-data-prep to create the
# inference dataset. The inference dataset is created by modifying the
# configuration used during training to
# a) change the paths to the input datasets,
# b) include the statistics from the training dataset and
# c) set the dimensions in the configuration to have `analysis_time` and
#    `elapsed_forecast_duration` instead of just `time`.
echo "Creating inference dataset\n"
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS} \
ANALYSIS_TIME=${ANALYSIS_TIME} \
FORECAST_DURATION=${FORECAST_DURATION} \
TIME_DIMENSIONS=${TIME_DIMENSIONS} \
INFERENCE_WORKDIR=${INFERENCE_WORKDIR} \
${UV_CMD} python src/create_inference_dataset.py
