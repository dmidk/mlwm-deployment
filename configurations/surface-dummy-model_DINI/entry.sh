#!/usr/bin/env bash
# This script is used to run the inference for the surface-dummy-model_DINI model.
#
# This script is intended to be run in a container, and assumes that during the
# container image build that the inference artifact was unpacked to
# inference_artifact/


INFERENCE_ARTIFACT_PATH="./inference_artifact"
INPUT_DATASETS_ROOT_PATH="/volume/inputs"
OUTPUT_DATASETS_ROOT_PATH="/volume/outputs"

# 1. Create inference dataset
uv run python -m mlwm.create_inference_dataset \
    --config_path ${INFERENCE_ARTIFACT_PATH}/config.yaml \
    --override_input_paths \
    danra_surface=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr \
    danra_surface_forcing=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr \
    danra_static=${INPUT_DATASETS_ROOT_PATH}/single_levels.zarr

# 2. Create graph
uv run python -m neural_lam.create_graph --config_path ${INFERENCE_ARTIFACT_PATH}/config.yaml

# 3. Run inference
uv run python -m neural_lam.inference --config_path ${INFERENCE_ARTIFACT_PATH}/config.yaml \
    --load ${INFERENCE_ARTIFACT_PATH}/checkpoint.ckpt \
    --save_eval_to_zarr_path ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr

# 4. Transform inference output back to original grid and variables
# TODO: this will result in {input_name}.zarr dataset for each input that is
# used for constructing state-variables that the model should predict. This
# means that we will have `danra_surface.zarr` in this case. We rename name
# that manually here but maybe mllam-data-prep should be able to merge inputs
# originating from the same zarr dataset path?
uv run python -m mllam_data_prep.recreate_inputs ${OUTPUT_DATASETS_ROOT_PATH}/inference_output.zarr
rename ${OUTPUT_DATASETS_ROOT_PATH}/danra_surface.zarr ${OUTPUT_DATASETS_ROOT_PATH}/single_levels.zarr
