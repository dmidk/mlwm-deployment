#!/usr/bin/env bash
# This script is used to run the inference for the surface-dummy-model_DINI model.
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

# set default override of input paths in the datastore config used for creating the
# inference dataset if environment variable isn't set
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS:-"\
danra.danra_surface=https://object-store.os-api.cci1.ecmwf.int/danra/v0.6.0dev1/single_levels.zarr/,\
danra.danra_static=https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/single_levels.zarr/"}
TIME_DIMENSIONS=${TIME_DIMENSIONS:-"analysis_time,elapsed_forecast_duration"}
ANALYSIS_TIME=${ANALYSIS_TIME:-"2019-02-04T12:00"}  # assumed to be in UTC
# forecast out to 18 hours, which means 6 steps of 3 hours each (the model was
# trained on 3-hourly analysis data)
FORECAST_DURATION=${FORECAST_DURATION:-"PT18H"}
NUM_EVAL_STEPS=${NUM_EVAL_STEPS:-6}
INFERENCE_WORKDIR=${INFERENCE_WORKDIR:-"./inference_workdir"}

echo "Creating forecast using following runtime args:"
echo "  DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS}"
echo "  TIME_DIMENSIONS=${TIME_DIMENSIONS}"
echo "  ANALYSIS_TIME=${ANALYSIS_TIME}"
echo "  FORECAST_DURATION=${FORECAST_DURATION}"
echo "  NUM_EVAL_STEPS=${NUM_EVAL_STEPS}"
echo "  INFERENCE_WORKDIR=${INFERENCE_WORKDIR}"

## Model specific inference configuration (same across all executions)
NUM_HIDDEN_DIMS=2
GRAPH_NAME="multiscale"
HIEARCHICAL_GRAPH=false

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
uv run wandb disabled

## 1. Create inference dataset
# This uses a cli stored within mlwm to called mllam-data-prep to create the
# inference dataset. The inference dataset is created by modifying the
# configuration used during training to
# a) change the paths to the input datasets,
# b) include the statistics from the training dataset and
# c) set the dimensions in the configuration to have `analysis_time` and
#    `elapsed_forecast_duration` instead of just `time`.
DATASTORE_INPUT_PATHS=${DATASTORE_INPUT_PATHS} \
ANALYSIS_TIME=${ANALYSIS_TIME} \
FORECAST_DURATION=${FORECAST_DURATION} \
TIME_DIMENSIONS=${TIME_DIMENSIONS} \
INFERENCE_WORKDIR=${INFERENCE_WORKDIR} \
uv run python src/create_inference_dataset.py

## 2. Create graph
# TODO: could cache this, although that isn't implemented at the moment
uv run python -m neural_lam.create_graph --config_path ${INFERENCE_WORKDIR}/config.yaml \
    --name ${GRAPH_NAME} ${CREATE_GRAPH_ARG}

## 3. Run inference
uv run python -m neural_lam.train_model --config_path ${INFERENCE_WORKDIR}/config.yaml \
    --eval test\
    --graph ${GRAPH_NAME} \
    --hidden_dim ${NUM_HIDDEN_DIMS} \
    --ar_steps_eval ${NUM_EVAL_STEPS} \
    --val_steps_to_log  \
    --load ${INFERENCE_ARTIFACT_PATH}/checkpoint.pkl \
    --save_eval_to_zarr_path ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr

## 4. Transform inference output back to original grid and variables
# TODO: this will result in {input_name}.zarr dataset for each input that is
# used for constructing state-variables that the model should predict. This
# means that we will have `danra_surface.zarr` in this case. We rename name
# that manually here but maybe mllam-data-prep should be able to merge inputs
# originating from the same zarr dataset path?
uv run python -m mllam_data_prep.recreate_inputs \
    --config-path ${INFERENCE_WORKDIR}/danra.datastore.yaml \
    --output-path-format "${OUTPUT_DATASETS_ROOT_PATH}/{input_name}.zarr" \
    ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr

echo "Renaming ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr to ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr"
mv ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr
