import copy
import datetime
from typing import Dict

import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import xarray as xr
from loguru import logger


@logger.catch(reraise=True)
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
    split_name = "test"
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
    inference_config.output.splitting = mdp_config.Splitting(
        dim=sampling_dim,
        splits={split_name: mdp_config.Split(**sampling_coord_range)},
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


def main():
    fp_stats = "inference_artifact/stats/danra.datastore.stats.zarr"
    fp_training_datastore = "inference_artifact/configs/danra.datastore.yaml"

    S3_BUCKET_URL = "https://object-store.os-api.cci1.ecmwf.int/danra"
    overwrite_input_paths = dict(
        danra_surface=f"{S3_BUCKET_URL}/v0.6.0dev1/single_levels.zarr/",
        danra_static=f"{S3_BUCKET_URL}/v0.5.0/single_levels.zarr/",
    )
    analysis_time = "2019-02-04T12:00"
    forecast_duration = datetime.timedelta(hours=6)

    inference_datastore_config_output_fp = "danra.inference.datastore.yaml"

    ds_stats = xr.open_dataset(fp_stats)
    logger.debug(f"Opened stats dataset: {ds_stats}")

    logger.debug(
        f"Loading training datastore config from {fp_training_datastore}"
    )
    datastore_training_config = mdp.Config.from_yaml_file(
        fp_training_datastore
    )

    inference_config = _create_inference_datastore_config(
        training_config=datastore_training_config,
        forecast_analysis_time=datetime.datetime.fromisoformat(analysis_time),
        forecast_duration=forecast_duration,
        overwrite_input_paths=overwrite_input_paths,
        sampling_dim="analysis_time",
    )

    # save inference config to file
    inference_config.to_yaml_file(inference_datastore_config_output_fp)
    logger.info(
        f"Saved inference datastore config to {inference_datastore_config_output_fp}"
    )

    ds = mdp.create_dataset(config=inference_config, ds_stats=ds_stats)
    print(ds)


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main()
