import copy
import datetime
from pathlib import Path
from typing import Dict

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import pytorch_lightning as pl
import torch
import xarray as xr
from loguru import logger
from neural_lam import models as nl_models
from neural_lam.config import NeuralLAMConfig, load_config_and_datastore
from neural_lam.weather_dataset import WeatherDataModule

FP_TRAINING_CONFIG = "inference_artifact/configs/config.yaml"
FP_TRAINING_DATASTORE_STATS = (
    "inference_artifact/stats/danra.datastore.stats.zarr"
)
FP_TRAINING_DATASTORE_CONFIG = (
    "inference_artifact/configs/danra.datastore.yaml"
)

# XXX: Parameters from training that aren't currently saved to the config, we
# have to hardcode these for now
NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1
MODEL_CLASS = nl_models.GraphLAM
# Inference system dependent parameters (larger batch size may require more
# memory, and more workers may require more CPU cores)
BATCH_SIZE = 4
NUM_WORKERS = 2

S3_BUCKET_URL = "https://object-store.os-api.cci1.ecmwf.int/danra"
OVERWRITE_INPUT_PATHS = dict(
    danra_surface=f"{S3_BUCKET_URL}/v0.6.0dev1/single_levels.zarr/",
    danra_static=f"{S3_BUCKET_URL}/v0.5.0/single_levels.zarr/",
)
ANALYSIS_TIME = "2019-02-04T12:00"
FORECAST_DURATION = datetime.timedelta(hours=6)

# the path below describes where to save the inference datastore config,
# inference zarr dataset and the inference config for neural-lam itself
FP_INFERENCE_WORKDIR = "inference_workdir"
FP_INFERENCE_DATASTORE_CONFIG = f"{FP_INFERENCE_WORKDIR}/danra.datastore.yaml"
FP_INFERENCE_CONFIG = f"{FP_INFERENCE_WORKDIR}/config.yaml"


def _create_inference_datastore_config(
    training_config: mdp.Config,
    forecast_analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    overwrite_input_paths: Dict[str, str] = {},
    sampling_dim: str = "time",
) -> mdp.Config:
    """
    From a training datastore config, create an inference datastore config that:
    - samples along a new sampling dimension `sampling_dim` (default:
      `analysis_time`) instead of `time`
    - has a single split called "test" with a single time slice given by the
      `forecast_analysis_time` argument
    - optionally overwrites input paths with the `overwrite_input_paths` argument
    - ensures that the output variables have the correct dimensions, i.e.
      replacing `time` with [`analysis_time`, `elapsed_forecast_duration`]
    - ensures that the input datasets have the correct dimensions and dim_mappings,
      i.e. replacing `time` with [`analysis_time`, `elapsed_forecast_duration`

    Parameters
    ----------
    training_config : mdp.Config
        The training config to base the inference config on
    forecast_analysis_time : datetime.datetime
        The analysis time to use for the inference config
    forecast_duration : datetime.timedelta
        The forecast duration to use for the inference config
    overwrite_input_paths : Dict[str, str], optional
        A dictionary of input names and paths to overwrite in the training config,
        by default {}
    sampling_dim : str, optional
        The new sampling dimension to use, by default "time"

    Returns
    -------
    mdp.Config
        The inference config
    """
    # the new sampling dimension is `analysis_time`
    old_sampling_dim = "time"
    sampling_dim = "analysis_time"
    # instead of only having `time` as dimension, the input forecast datasets
    # have two dimensions that describe the time value [analysis_time,
    # elapsed_forecast_duration]
    dim_replacements = dict(
        time=["analysis_time", "elapsed_forecast_duration"],
    )
    # there will be a single split called "test"
    # split_name = "test"
    # which will have a single time slice, given by the analysis time argument
    # to the script
    sampling_coord_range = dict(
        start=forecast_analysis_time,
        end=forecast_analysis_time + forecast_duration,
    )

    inference_config = copy.deepcopy(training_config)

    if len(overwrite_input_paths) > 0:
        for key, value in overwrite_input_paths.items():
            if key not in training_config.inputs:
                raise ValueError(
                    f"Key {key} not found in config inputs. "
                    f"Available keys are: {list(training_config.inputs.keys())}"
                )
            logger.info(
                f"Overwriting input path for {key} with {value} previously "
                f"{training_config.inputs[key].path}"
            )
            inference_config.inputs[key].path = value

    # setup the split (test) for the dataset with a coordinate range along the
    # sampling dimension (analysis_time) of length 1
    # inference_config.output.splitting = mdp_config.Splitting(
    #     dim=sampling_dim,
    #     splits={split_name: mdp_config.Split(**sampling_coord_range)},
    # )

    # XXX: currently (as of 0.4.0) neural-lam requires that `train`, `val` and
    # `test` splits are always present, even if they are not used. So we
    # create empty `train` and `val` splits here
    inference_config.output.splitting = mdp_config.Splitting(
        dim="time",
        splits={
            "train": mdp_config.Split(
                start=forecast_analysis_time, end=forecast_analysis_time
            ),
            "val": mdp_config.Split(
                start=forecast_analysis_time, end=forecast_analysis_time
            ),
            "test": mdp_config.Split(
                start=forecast_analysis_time,
                end=forecast_analysis_time + forecast_duration,
            ),
        },
    )

    # ensure the output data is sampled along the sampling dimension
    # (analysis_time) too
    inference_config.output.coord_ranges = {
        sampling_dim: mdp_config.Range(**sampling_coord_range)
    }

    inference_config.output.chunking = {sampling_dim: 1}

    # replace old sampling_dimension (time) dimension in outputs with
    # [`analysis_time`, `elapsed_forecast_time`]
    for variable, dims in training_config.output.variables.items():
        if old_sampling_dim in dims:
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            inference_config.output.variables[variable] = dims
            logger.info(
                f"Replaced {old_sampling_dim} dimension with"
                f" {dim_replacements[old_sampling_dim]} for {variable}"
            )

    # these dimensions should also be "renamed" from the input datasets
    for input_name in training_config.inputs.keys():
        if "time" in training_config.inputs[input_name].dim_mapping:
            dims = training_config.inputs[input_name].dims
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            inference_config.inputs[input_name].dims = dims

            del inference_config.inputs[input_name].dim_mapping[
                old_sampling_dim
            ]

            # add new "rename" dim-mappins for `analysis_time` and
            # `elapsed_forecast_duration`
            for dim in dim_replacements[old_sampling_dim]:
                inference_config.inputs[input_name].dim_mapping[
                    dim
                ] = mdp_config.DimMapping(method="rename", dim=dim)

    return inference_config


def _prepare_inference_dataset_zarr() -> str:
    """
    Prepare the inference dataset.

    Returns
    -------
    str
        The path to the inference datastore config file. The inference dataset
        is saved as a zarr store in the same directory as the config file, with
        the same name but with a .zarr extension instead of .yaml.
    """
    if Path(FP_INFERENCE_DATASTORE_CONFIG).exists():
        logger.info(
            f"Found existing inference datastore config at "
            f"{FP_INFERENCE_DATASTORE_CONFIG}, skipping dataset creation"
        )
        return FP_INFERENCE_DATASTORE_CONFIG

    ds_stats = xr.open_dataset(FP_TRAINING_DATASTORE_STATS)
    logger.debug(f"Opened stats dataset: {ds_stats}")

    logger.debug(
        f"Loading training datastore config from {FP_TRAINING_DATASTORE_CONFIG}"
    )
    datastore_training_config = mdp.Config.from_yaml_file(
        FP_TRAINING_DATASTORE_CONFIG
    )

    inference_config = _create_inference_datastore_config(
        training_config=datastore_training_config,
        forecast_analysis_time=datetime.datetime.fromisoformat(ANALYSIS_TIME),
        forecast_duration=FORECAST_DURATION,
        overwrite_input_paths=OVERWRITE_INPUT_PATHS,
        sampling_dim="analysis_time",
    )

    ds = mdp.create_dataset(config=inference_config, ds_stats=ds_stats)

    # neural-lam's convention is to have the same name for the zarr store
    # as the config file, but with .zarr extension
    fp_dataset = FP_INFERENCE_DATASTORE_CONFIG.replace(".yaml", ".zarr")

    Path(FP_INFERENCE_DATASTORE_CONFIG).parent.mkdir(
        parents=True, exist_ok=True
    )
    inference_config.to_yaml_file(FP_INFERENCE_DATASTORE_CONFIG)
    ds.to_zarr(fp_dataset)
    logger.info(f"Saved inference dataset to {fp_dataset}")

    return FP_INFERENCE_DATASTORE_CONFIG


def _create_inference_config(fp_inference_datastore_config: str) -> str:
    training_config = NeuralLAMConfig.from_yaml_file(FP_TRAINING_CONFIG)
    inference_config = copy.deepcopy(training_config)

    # overwrite the path to the datastore config, to point to the
    # inference datastore config
    inference_config.datastore.config_path = Path(
        fp_inference_datastore_config
    ).relative_to(Path(FP_INFERENCE_CONFIG).parent)

    # XXX: There is a bug in neural-lam here that means that the datastore kind
    # doesn't correctly get serialised to a string in the config file when
    # saved to yaml
    inference_config.datastore.kind = "mdp"

    inference_config.to_yaml_file(FP_INFERENCE_CONFIG)
    logger.info(f"Saved inference config to {FP_INFERENCE_CONFIG}")

    return FP_INFERENCE_CONFIG


@logger.catch(reraise=True)
def main():
    fp_inference_datastore_config = _prepare_inference_dataset_zarr()
    fp_inference_config = _create_inference_config(
        fp_inference_datastore_config=fp_inference_datastore_config
    )

    # Load neural-lam configuration and datastore to use
    config, datastore = load_config_and_datastore(
        config_path=fp_inference_config
    )

    # XXX: hardcoded timestep from DANRA right now, this should be inferred
    # from the dataset itself probably. neural-lam wants to know the number of
    # steps for the autoregressive prediction, not the total duration.
    ar_steps_eval = FORECAST_DURATION / isodate.parse_duration("PT3H")

    # Create datamodule
    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=0,
        ar_steps_eval=ar_steps_eval,
        standardize=True,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    devices = "auto"

    class ModelArgs:
        output_std = None
        # XXX: we shouldn't have to set a loss function when we're only doing
        # inference, but neural-lam currently requires it
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        lr = None

        graph = "inference-graph"
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        val_steps_to_log = [1, 3]
        metrics_watch = []
        num_past_forcing_steps = NUM_PAST_FORCING_STEPS
        num_future_forcing_steps = NUM_FUTURE_FORCING_STEPS

    model_args = ModelArgs()
    model = MODEL_CLASS(model_args, config=config, datastore=datastore)

    assert data_module.eval_dataloader() is not None
    assert device_name is not None
    assert devices is not None

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        devices=devices,
        log_every_n_steps=1,
        # use `detect_anomaly` to ensure that we don't have NaNs popping up
        # during inference
        detect_anomaly=True,
    )

    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main()
