import copy
import datetime
import os
from pathlib import Path
from typing import Dict

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import xarray as xr
from loguru import logger
from neural_lam.config import DatastoreSelection, NeuralLAMConfig

FP_TRAINING_CONFIG = "inference_artifact/configs/config.yaml"
# the path below describes where to save the inference datastore config,
# inference zarr dataset and the inference config for neural-lam itself
FP_INFERENCE_WORKDIR = "inference_workdir"
FP_INFERENCE_CONFIG = f"{FP_INFERENCE_WORKDIR}/config.yaml"


def _parse_datastore_input_paths(s: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a comma-separated list of {datastore_name}.{input_name}={input_path}
    into a dictionary of dictionaries.

    Parameters
    ----------
    s : str
        The string to parse.

    Returns
    -------
    Dict[str, Dict[str, str]]
        A dictionary of dictionaries.
    """
    result = {}
    for item in s.split(","):
        try:
            datastore_input, input_path = item.split("=")
            datastore_name, input_name = datastore_input.split(".")
        except ValueError:
            raise ValueError(
                f"Invalid format for DATASTORE_INPUT_PATHS item: {item}. "
                f"Expected format is {{datastore_name}}:{{input_name}}={{input_path}}"
            )
        if datastore_name not in result:
            result[datastore_name] = {}
        result[datastore_name][input_name] = input_path
    return result


REQUIRED_ENV_VARS = {
    # comma-separated list of {datastore_name}:{input_name}={input_path}
    "DATASTORE_INPUT_PATHS": _parse_datastore_input_paths,
    # iso8601 datetime string, e.g. 2019-02-04T12:00+0000
    "ANALYSIS_TIME": isodate.parse_datetime,
    # iso8160 duration string, e.g. PT6H for 6 hours
    "FORECAST_DURATION": isodate.parse_duration,
    # comma-separated list of time dimensions to replace, e.g.
    # time,forecast_reference_time
    "TIME_DIMENSIONS": lambda s: s.split(","),
}


def _parse_env_vars() -> Dict[str, any]:
    """
    Parse and validate required environment variables.

    Returns
    -------
    Dict[str, any]
        A dictionary of parsed environment variables.
    """
    env_vars = {}
    for var, parser in REQUIRED_ENV_VARS.items():
        value = os.getenv(var)
        if value is None:
            raise EnvironmentError(f"Environment variable {var} is not set.")
        try:
            env_vars[var] = parser(value)
        except Exception as e:
            raise ValueError(f"Error parsing environment variable {var}: {e}")
    return env_vars


def _create_inference_datastore_config(
    training_config: mdp.Config,
    forecast_analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    time_dimensions: list[str],
    overwrite_input_paths: Dict[str, str] = {},
) -> mdp.Config:
    """
    From a training datastore config, create an inference datastore config that:
    - samples along a new sampling dimension `sampling_dim` (default:
      `analysis_time`) instead of `time`
    - has a single split called "test" with a single time slice given by the
      `forecast_analysis_time` argument
    - optionally overwrites input paths with the `overwrite_input_paths` argument
    - ensures that the output variables have the correct dimensions, for example
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
    time_dimensions : list[str], optional
        The list of time dimensions to replace `time` with, for example
        replacing `time` with [`analysis_time`, `elapsed_forecast_duration`],
        the first dimension is assumed to be the sampling dimension (e.g. the
        analysis time)
    overwrite_input_paths : Dict[str, str], optional
        A dictionary of input names and paths to overwrite in the training config,
        by default {}

    Returns
    -------
    mdp.Config
        The inference config
    """
    # the new sampling dimension is `analysis_time`
    old_sampling_dim = "time"
    if not isinstance(time_dimensions, list) or len(time_dimensions) == 0:
        raise ValueError(
            "time_dimensions must be a non-empty list of strings, got "
            f"{time_dimensions}"
        )
    sampling_dim = time_dimensions[0]
    # instead of only having `time` as dimension, the input forecast datasets
    # have two dimensions that describe the time value [analysis_time,
    # elapsed_forecast_duration]
    dim_replacements = dict(
        time=time_dimensions,
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


def _prepare_inference_dataset_zarr(
    datastore_name: str,
    datastore_input_paths: Dict[str, str],
    analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    time_dimensions: list[str],
) -> str:
    """
    Prepare the inference dataset for a single datastore.

    Parameters
    ----------
    datastore_name : str
        The name of the datastore to prepare the inference dataset for, this
        sets the expected path of the training datastore config and stats.
    datastore_input_paths : Dict[str, str]
        A dictionary of input names and paths to overwrite in the training
        config.
    analysis_time : datetime.datetime
        The analysis time to use for the inference dataset.
    forecast_duration : datetime.timedelta
        The forecast duration to use for the inference dataset.
    time_dimensions : list[str]
        The list of time dimensions to replace `time` with, for example
        replacing `time` with [`analysis_time`, `elapsed_forecast_duration`]

    Returns
    -------
    str
        The path to the inference datastore config file. The inference dataset
        is saved as a zarr store in the same directory as the config file, with
        the same name but with a .zarr extension instead of .yaml.
    """
    fp_training_datastore_stats = (
        f"inference_artifact/stats/{datastore_name}.datastore.stats.zarr"
    )
    ds_stats = xr.open_dataset(fp_training_datastore_stats)
    logger.debug(f"Opened stats dataset: {ds_stats}")

    fp_training_datastore_config = (
        f"inference_artifact/configs/{datastore_name}.datastore.yaml"
    )

    logger.debug(
        f"Loading training datastore config from {fp_training_datastore_config}"
    )
    datastore_training_config = mdp.Config.from_yaml_file(
        fp_training_datastore_config
    )

    inference_config = _create_inference_datastore_config(
        training_config=datastore_training_config,
        forecast_analysis_time=analysis_time,
        forecast_duration=forecast_duration,
        overwrite_input_paths=datastore_input_paths,
        time_dimensions=time_dimensions,
    )

    fp_inference_datastore_config = (
        f"{FP_INFERENCE_WORKDIR}/{datastore_name}.datastore.yaml"
    )

    Path(fp_inference_datastore_config).parent.mkdir(
        parents=True, exist_ok=True
    )
    logger.info(
        f"Saving inference datastore config to {fp_inference_datastore_config}"
    )

    # neural-lam's convention is to have the same name for the zarr store
    # as the config file, but with .zarr extension
    fp_dataset = fp_inference_datastore_config.replace(".yaml", ".zarr")
    inference_config.to_yaml_file(fp_inference_datastore_config)

    ds = mdp.create_dataset(config=inference_config, ds_stats=ds_stats)
    ds.to_zarr(fp_dataset)
    logger.info(f"Saved inference dataset to {fp_dataset}")

    return fp_inference_datastore_config


def _prepare_all_inference_dataset_zarr(
    analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    datastore_input_paths: Dict[str, Dict[str, str]],
    time_dimensions: list[str],
) -> str:
    """
    Prepare the inference dataset.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The analysis time to use for the inference dataset(s).
    forecast_duration : datetime.timedelta
        The forecast duration to use for the inference dataset(s).
    datastore_input_paths : Dict[str, Dict[str,str]]
        A dictionary of datastore names and their corresponding input names
        and paths to overwrite in the training config.
    time_dimensions : list[str]
        The list of time dimensions to replace `time` with, for example
        replacing `time` with [`analysis_time`, `elapsed_forecast_duration`]

    Returns
    -------
    Dict[str, str]
        A dictionary of datastore names and the path to their corresponding
        inference datastore config file. The inference dataset is saved as a
        zarr store in the same directory as the config file, with the same
        name but with a .zarr extension instead of .yaml.
    """
    fps_datastore_configs = {}
    for datastore_name, input_paths in datastore_input_paths.items():
        logger.info(f"Processing {datastore_name} datastore for inference")
        fp_training_datastore_config = _prepare_inference_dataset_zarr(
            datastore_name=datastore_name,
            datastore_input_paths=input_paths,
            analysis_time=analysis_time,
            forecast_duration=forecast_duration,
            time_dimensions=time_dimensions,
        )

        fps_datastore_configs[datastore_name] = fp_training_datastore_config

    return fps_datastore_configs


def _create_inference_config(
    fps_inference_datastore_config: Dict[str, str]
) -> str:
    training_config = NeuralLAMConfig.from_yaml_file(FP_TRAINING_CONFIG)
    inference_config = copy.deepcopy(training_config)

    def _set_datastore_config_path(node: DatastoreSelection, fp: str):
        node.config_path = Path(fp).relative_to(
            Path(FP_INFERENCE_CONFIG).parent
        )
        # XXX: There is a bug in neural-lam here that means that the datastore kind
        # doesn't correctly get serialised to a string in the config file when
        # saved to yaml
        node.kind = str(node.kind)

    # see if the neural-lam config was for single or multiple datastores
    if hasattr(training_config, "datastores"):
        # using multiple datastores
        for (
            datastore_name,
            fp_datastore_config,
        ) in fps_inference_datastore_config.items():
            if datastore_name not in inference_config.datastores:
                raise ValueError(
                    f"Datastore {datastore_name} not found in training config. "
                    f"Available datastores are: "
                    f"{list(inference_config.datastores.keys())}"
                )
            _set_datastore_config_path(
                node=inference_config.datastores[datastore_name],
                fp=fp_datastore_config,
            )
    else:
        fp_datastore_config = list(fps_inference_datastore_config.values())[0]
        # using a single datastore
        _set_datastore_config_path(
            node=inference_config.datastore, fp=fp_datastore_config
        )

    inference_config.to_yaml_file(FP_INFERENCE_CONFIG)
    logger.info(f"Saved inference config to {FP_INFERENCE_CONFIG}")

    return FP_INFERENCE_CONFIG


@logger.catch(reraise=True)
def main():
    env_vars = _parse_env_vars()
    # convert analysis time to UTC and strip timezone info
    analysis_time = (
        env_vars["ANALYSIS_TIME"]
        .astimezone(datetime.timezone.utc)
        .replace(tzinfo=None)
    )

    fps_inference_datastore_config = _prepare_all_inference_dataset_zarr(
        analysis_time=analysis_time,
        forecast_duration=env_vars["FORECAST_DURATION"],
        datastore_input_paths=env_vars["DATASTORE_INPUT_PATHS"],
        time_dimensions=env_vars["TIME_DIMENSIONS"],
    )
    _create_inference_config(
        fps_inference_datastore_config=fps_inference_datastore_config
    )


if __name__ == "__main__":
    with_debugger = os.getenv("MLWM_DEBUGGER", "0")
    if with_debugger == "ipdb":
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
