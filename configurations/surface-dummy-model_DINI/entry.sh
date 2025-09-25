#!/usr/bin/env bash
# This script is used to run the inference for the surface-dummy-model_DINI model.
#
# This script is intended to be run in a container, and assumes that during the
# container image build that the inference artifact was unpacked to
# inference_artifact/

# make this script fail on any error
set -e

# forecast out to 18 hours, which means 6 steps of 3 hours each (the model was
# trained on 3-hourly analysis data)
NUM_EVAL_STEPS=6

# model specific parameters, ideally these would come from some config
NUM_HIDDEN_DIMS=2
GRAPH_NAME="multiscale"
HIEARCHICAL_GRAPH=false

if [ "$HIEARCHICAL_GRAPH" = true ] ; then
    CREATE_GRAPH_ARG="--hierarchical"
else
    CREATE_GRAPH_ARG=""
fi

INFERENCE_ARTIFACT_PATH="./inference_artifact"
INFERENCE_WORK_PATH="./inference_workdir"

# XXX: these mount points could come from config.yaml for the model run configuration
INPUT_DATASETS_ROOT_PATH="${INFERENCE_WORK_PATH}/inputs"
OUTPUT_DATASETS_ROOT_PATH="${INFERENCE_WORK_PATH}/outputs"

mkdir -p ${OUTPUT_DATASETS_ROOT_PATH}

## 1. Create inference dataset
# This uses a cli stored within mlwm to called mllam-data-prep to create the
# inference dataset. The inference dataset is created by modifying the
# configuration used during training to
# a) change the paths to the input datasets,
# b) include the statistics from the training dataset and
# c) set the dimensions in the configuration to have `analysis_time` and
#    `elapsed_forecast_duration` instead of just `time`.
uv run python src/create_inference_dataset.py

## 2. Create graph
# TODO: could cache this, although that isn't implemented at the moment
# uv run python -m neural_lam.create_graph --config_path ${INFERENCE_WORK_PATH}/config.yaml \
#     --name ${GRAPH_NAME} ${CREATE_GRAPH_ARG}

## 3. Run inference
uv run python -m neural_lam.train_model --config_path ${INFERENCE_WORK_PATH}/config.yaml \
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
    --config-path ${INFERENCE_WORK_PATH}/danra.datastore.yaml \
    --output-path-format "${OUTPUT_DATASETS_ROOT_PATH}/{input_name}.zarr" \
    ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr

echo "Renaming ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr to ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr"
mv ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr
