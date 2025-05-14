import datetime
import tempfile
from pathlib import Path

import isodate
import rich
import xarray as xr
from loguru import logger

from . import config as mdp_config
from .create_dataset import create_dataset


def parse_key_value_arg(arg: str) -> tuple[str, str]:
    """
    Parse a single key=value argument into a dictionary.

    This function is intended for use with argparse's `type=` argument when parsing
    command-line arguments like: --overwrite-source-paths key1=value1 key2=value2

    Args:
        arg (str): A string in the format key=value.

    Returns:
        Dict[str, str]: A dictionary with one key-value pair parsed from the
        input string.

    Raises:
        argparse.ArgumentTypeError: If the argument is not in key=value format.
    """
    import argparse

    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{arg}'. Expected key=value."
        )

    key, value = arg.split("=", 1)
    return (key, value)


def create_forecast_inference_dataset(
    fp_config: str,
    analysis_time: datetime.datetime,
    overwrite_input_paths: dict,
    use_stats_from_path: str,
):
    """
    Create forecasting prediction dataset derived from a config file used during
    training. In creating the inference dataset, it is assumed that the `time`
    dimension of all input datasets used should be replaced by the
    `analysis_time` and `elapsed_forecast_duration` dimensions.
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
        start=analysis_time,
        end=analysis_time,
    )

    # load and modify the original config file
    config = mdp_config.Config.from_yaml_file(file=fp_config)

    if overwrite_input_paths:
        for key, value in overwrite_input_paths.items():
            if key not in config.inputs:
                raise ValueError(
                    f"Key {key} not found in config inputs. "
                    f"Available keys are: {list(config.inputs.keys())}"
                )
            logger.info(
                f"Overwriting input path for {key} with {value} previously "
                f"{config.inputs[key].path}"
            )
            config.inputs[key].path = value

    # setup the split (test) for the dataset with a coordinate range along the
    # sampling dimension (analysis_time) of length 1
    config.output.splitting = mdp_config.Splitting(
        dim=sampling_dim,
        splits={split_name: mdp_config.Split(**sampling_coord_range)},
    )

    # ensure the output data is sampled along the sampling dimension
    # (analysis_time) too
    config.output.coord_ranges = {
        sampling_dim: analysis_time  # mdp_config.Range(**sampling_coord_range)
    }

    config.output.chunking = {sampling_dim: 1}

    # replace old sampling_dimension (time) dimension in outputs with
    # [`analysis_time`, `elapsed_forecast_time`]
    for variable, dims in config.output.variables.items():
        if old_sampling_dim in dims:
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            config.output.variables[variable] = dims
            logger.info(
                f"Replaced {old_sampling_dim} dimension with "
                f"{dim_replacements[old_sampling_dim]} for {variable}"
            )

    # these dimensions should also be "renamed" from the input datasets
    for input_name in config.inputs.keys():
        if "time" in config.inputs[input_name].dim_mapping:
            dims = config.inputs[input_name].dims
            orig_sampling_dim_index = dims.index(old_sampling_dim)
            dims.remove(old_sampling_dim)
            for dim in dim_replacements[old_sampling_dim][::-1]:
                dims.insert(orig_sampling_dim_index, dim)
            config.inputs[input_name].dims = dims

            del config.inputs[input_name].dim_mapping[old_sampling_dim]

            # add new "rename" dim-mappins for `analysis_time` and
            # `elapsed_forecast_duration`
            for dim in dim_replacements[old_sampling_dim]:
                config.inputs[input_name].dim_mapping[
                    dim
                ] = mdp_config.DimMapping(method="rename", dim=dim)

    # save config to temporary filepath
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    fp_config_temp = Path(tmpfile.name)
    config.to_yaml_file(fp_config_temp)
    logger.info(f"Temporary config file created at {fp_config_temp}")

    rich.print(config)

    if use_stats_from_path is not None:
        ds_stats = xr.open_zarr(use_stats_from_path)

    ds = create_dataset(config=config, ds_stats=ds_stats)

    return ds, config


def cli(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", help="Path to the config file", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output zarr file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite-input-paths",
        nargs="*",
        type=parse_key_value_arg,
        help=(
            "List of key=value pairs used to overwrite input paths of named "
            "inputs in the config file. For example: --overwrite-input-paths "
            "danra_surface=s3://mybucket/2025-05-01T1200Z/danra_surface.zarr "
            "danra_height_levels=s3://mybucket/2025-05-01T1200Z/"
            "danra_height_levels.zarr"
        ),
    )
    parser.add_argument(
        "--analysis_time",
        required=True,
        help="Analysis time to use for the dataset. This is used to select the "
        "correct time slice from the input data.",
        type=isodate.parse_datetime,
    )
    parser.add_argument(
        "--use-stats-from-path",
        help="Path to zarr dataset with stats to use in the new dataset. "
        "Using the option will cause mllam-data-prep to skip calculating "
        "stats and instead use the stats from the provided path.",
        type=Path,
        default=None,
    )

    args = parser.parse_args(argv)

    analysis_time = args.analysis_time
    use_stats_from_path = args.use_stats_from_path
    fp_config = Path(args.config)

    # Convert the list of tuples to a dictionary
    overwrite_input_paths = dict(args.overwrite_input_paths)

    ds, config = create_forecast_inference_dataset(
        analysis_time=analysis_time,
        fp_config=fp_config,
        overwrite_input_paths=overwrite_input_paths,
        use_stats_from_path=use_stats_from_path,
    )

    dataset_name = (
        f"{Path(fp_config).name}."
        f"{isodate.datetime_isoformat(analysis_time).replace(':','')}"
    )
    fp_config_inference = f"{dataset_name}.yaml"
    fp_dataset_inference = f"{dataset_name}.zarr"

    logger.info(f"Writing inference dataset to {fp_dataset_inference}")
    ds.to_zarr(fp_dataset_inference, mode="w", consolidated=True)

    logger.info(f"Writing inference config to {fp_config_inference}")
    config.to_yaml_file(fp_config_inference)


if __name__ == "__main__":
    cli()
