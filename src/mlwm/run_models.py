"""
The cli implemented in this module is used to run the MLWM models configured in
this repository. The script works by locally launching docker containers for
each configured model, while preparing the input and output volumes required
for each before each run.

In pseudocode, the script does the following:

- for each model configured in this repository (stored in
  `configurations/{model_name}`):
    - read the model configuration .yaml file (stored in
      `configurations/{model_name}/config.yaml`)
    - copy the required input zarr datasets from the s3 locations described in
      the configuration file, this will be to
      `$WORKDIR/input/{model_name}/{input_name}`. These will be parameterised
      by the `analysis_time` of the execution
    - create the output directorie(s) for the model, this will be
      `$WORKDIR/output/{model_name}/{output_name}`
    - launch the docker container for the model, mounting the input and output
      directories as volumes to the container
    - wait for the container to finish running
    - copy the output directories to the s3 locations given in the model
      configuration, again these will be parameterised by the `analysis_time`

The following environment variables are used to configure the script (the
values of which can be overridden with dodenv by setting them in a file named
.env):
- `WORKDIR`: the working directory for the script, this is where the input
  and output directories will be created. This should be a local directory on
  the machine running the script.
"""
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, List

import dotenv
from loguru import logger
from tqdm import tqdm
from upath import UPath

from . import __version__, config_spec
from .paths import create_path as create_dataset_path

dotenv.load_dotenv()

WORKDIR = os.getenv("WORKDIR", None)
assert WORKDIR is not None, "WORKDIR environment variable must be set"
MODEL_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configurations"


def launch_docker_container(
    image: str, volumes: Dict[str, str], command: str = None
):
    """
    Launch a docker container with the specified image and volumes.

    Parameters
    ----------
    image : str
        The docker image to use for the container.
    volumes : Dict[str, str]
        A dictionary mapping host paths to container paths for volume mounts.
    command : str, optional
        The command to run inside the container. If None, the default entrypoint
        of the image will be used.

    Returns
    -------
    None
    """
    pass  # Placeholder for the actual implementation


def construct_s3_uri(
    data_path_config: config_spec.DataPathConfig,
    analysis_time: datetime.datetime,
):
    """
    Construct the S3 URI for the dataset based on the configuration and analysis time.

    Parameters
    ----------
    data_path_config : config_spec.DataPathConfig
        The configuration object for the dataset.
    analysis_time : datetime.datetime
        The analysis time for the dataset.

    Returns
    -------
    UPath
        The constructed source path as a UPath object.
    """
    uri_args = data_path_config.uri_args
    prefix = create_dataset_path(
        model_name=uri_args.model_name,
        model_config=uri_args.model_config,
        bbox=uri_args.bbox.to_dict(),
        resolution=uri_args.resolution.to_dict(),
        analysis_time=analysis_time,
        data_kind=uri_args.data_kind,
    )
    uri = UPath(f"s3://{uri_args.bucket_name}/{prefix}")
    return uri


def copy_directory_to_s3(src_dir, dst_dir):
    # Find all files in the local directory recursively
    files = [f for f in src_dir.rglob("*") if f.is_file()]

    if len(files) == 0:
        logger.warning(f"No files found in {src_dir}. Skipping copy.")
        return

    # Set up progress bar
    for file in tqdm(files, desc="Copying files"):
        # Compute relative path from local_dir to file
        relative_path = file.relative_to(src_dir)
        # Construct destination path in S3
        dest_path = dst_dir / relative_path

        # Ensure destination directory exists (no-op for S3, but useful for
        # local testing)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        dest_path.write_bytes(file.read_bytes())


def _prepare_inputs(
    model_name: str,
    model_config: config_spec.Config,
    analysis_time: datetime.datetime,
    model_workdir: Path,
):
    """
    Prepare the input datasets for the model.

    This includes:
    - Copying input datasets from S3 to local directories.

    Parameters
    ----------
    model_name : str
        The name of the model to run.
    model_config : Config
        The configuration object for the model.
    analysis_time : datetime.datetime
        The analysis time for the datasets.
    model_workdir : Path
        The working directory for the model.

    Returns
    -------
    input_data_volume_mounts : Dict[str, str]

    """

    volume_mounts = {}
    for input_name, data_path_config in model_config.inputs.items():
        logger.info(f"Preparing input dataset for {input_name}")

        # copy the input dataset from S3 to local directory
        source_uri = construct_s3_uri(
            data_path_config=data_path_config, analysis_time=analysis_time
        )
        input_workdir = model_workdir / "input" / model_name / input_name
        logger.info(
            f"Copying input dataset from {source_uri} to {input_workdir}"
        )
        copy_directory_to_s3(src_dir=source_uri, dst_dir=input_workdir)

        volume_mounts[input_workdir] = data_path_config.internal_path

    output_uris = {}
    for output_name, data_path_config in model_config.outputs.items():
        logger.info(f"Setting up output path for {output_name}")

        # create the output directory
        output_workdir = model_workdir / "output" / model_name / output_name
        logger.info(f"Creating output directory: {output_workdir}")
        output_workdir.mkdir(parents=True, exist_ok=True)

        # add the output directory to the volume mounts
        volume_mounts[output_workdir] = data_path_config.internal_path
        output_uris[output_workdir] = construct_s3_uri(
            data_path_config=data_path_config, analysis_time=analysis_time
        )

    return volume_mounts, output_uris


def prep_and_run_model(
    model_name: str,
    model_config: config_spec.Config,
    analysis_time: datetime.datetime,
):
    """
    Prepare and run the model using the specified configuration.

    This includes:
    - Copying input datasets from S3 to local directories.
    - Creating output directories.
    - Launching the docker container for the model.
    - Waiting for the container to finish running.
    - Copying output datasets from local directories to S3.
    - Cleaning up local directories.
    - Logging the results.

    Parameters
    ----------
    model_name : str
        The name of the model to run.
    model_config : Config
        The configuration object for the model.

    Returns
    -------
    None
    """
    model_workdir = Path(f"{WORKDIR}/input/{model_name}")
    logger.info(f"Creating working directory: {model_workdir}")
    volume_mounts, output_uris = _prepare_inputs(
        model_name=model_name,
        model_config=model_config,
        analysis_time=analysis_time,
        model_workdir=model_workdir,
    )

    logger.info(f"Launching docker container for model: {model_name}")
    launch_docker_container(
        image=model_config.docker_image,
        volumes=volume_mounts,
    )

    for output_workdir, output_uri in output_uris.items():
        logger.info(f"Copying output from {output_workdir} to {output_uri}")
        copy_directory_to_s3(src_dir=output_workdir, dst_dir=output_uri)

    # clean up workdir
    logger.info(f"Cleaning up workdir: {model_workdir}")
    shutil.rmtree(model_workdir)


def find_model_configurations() -> List[str]:
    """
    Find all model configurations in the `configurations` directory.

    Returns
    -------
    List[str]
        A list of paths to the model configuration files.
    """
    fps_model_config_paths = list(MODEL_CONFIGS_PATH.glob("*"))
    model_configs = {
        fp.name: config_spec.Config.from_yaml_file(fp / "config.yaml")
        for fp in fps_model_config_paths
    }
    return model_configs


def cli():
    """
    Command line interface for running the MLWM models.

    This function sets up the command line interface using Click. It allows
    users to run the models with various options and parameters.

    Returns
    -------
    None
    """
    logger.info(
        f"Starting MLWM model run (with configuration version: {__version__})..."
    )
    model_configurations = find_model_configurations()
    logger.info(f"Using working directory: {WORKDIR}")
    logger.info(f"Found {len(model_configurations)} model configuration(s).")

    # hardcode analysis time to 2024-01-01T00:00:00Z for now
    analysis_time = datetime.datetime(2024, 1, 1, 0, 0, 0)

    for model_name, model_config in model_configurations.items():
        logger.info(f"Running model: {model_name}")
        prep_and_run_model(
            model_name=model_name,
            model_config=model_config,
            analysis_time=analysis_time,
        )
        logger.info(f"Finished running model: {model_name}")


if __name__ == "__main__":
    cli()
