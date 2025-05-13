"""
Contents required when package to an inference artifact:

- Model checkpoint, provided by a path to the CLI. Written to `checkpoint.pkl`
- Extraction of statistics from training dataset. Paths to datasets inferred
  from datastore configs link reference in neural-lam config provided to cli.
  Written to `stats/{datastore_name}.stats.zarr`
- Command line arguments that training was run with. Expected to be in
  `training_cli_args.yaml`
"""
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import dotenv
import mllam_data_prep as mdp
import xarray as xr
import yaml
from loguru import logger
from tqdm import tqdm
from upath import UPath

from . import __version__

DEFAULT_ARTIFACT_PATH_FORMAT = (
    "s3://mlwm-artifacts/inference-artifacts/{artifact_name}.zip"
)

dotenv.load_dotenv()

ARTIFACT_PATH_FORMAT = os.environ.get(
    "ARTIFACT_PATH_FORMAT", DEFAULT_ARTIFACT_PATH_FORMAT
)

TRAINING_CLI_ARGS_FILE = "training_cli_args.yaml"


def _find_datastore_paths(nl_config_path: str) -> Dict[str, str]:
    """
    Find the paths to the datastore configuration files in the neural-lam. This
    function returns a dictionary because the research branch of neural-lam
    uses more than one datastore (one for the domain interior and one for the
    boundary). By supporting the new config structure we can also support
    multiple datastores in the future.

    Parameters
    ----------
    nl_config_path : str
        Path to the neural-lam config file. This is used to make the paths to
        the datastore configuration files absolute if they are not already.

    Returns
    -------
    datastore_paths : Dict[str, str]
        A dictionary with the name of the datastore as the key and the path
        to the datastore configuration file as the value. If there is only
        one datastore, the key is None.
    """
    with open(nl_config_path, "r") as f:
        nl_config = yaml.safe_load(f)

    def _make_abspath(datastore_path: str):
        """
        Make the path absolute if it is not already. If the path is relative,
        it is made absolute with respect to the neural-lam config file.
        """
        # if the path is absolute, return it as is
        if Path(datastore_path).is_absolute():
            return datastore_path
        return Path(nl_config_path).parent / datastore_path

    datastore_paths = {}
    if "datastore" in nl_config:
        datastore_paths[None] = _make_abspath(
            nl_config["datastore"]["config_path"]
        )
    elif "datastores" in nl_config:
        for name, datastore_config in nl_config["datastores"].items():
            datastore_paths[name] = _make_abspath(
                datastore_config["config_path"]
            )
    else:
        raise ValueError("No datastore found in the config file.")

    return datastore_paths


def _extract_stats(datastore_config_path: str, artifact_path: str):
    """
    Extract statistics from the training dataset created from the datastore
    configuration file. The statistics are saved to a file named
    {datastore_name}.stats.zarr in the artifact output directory.

    Parameters
    ----------
    datastore_config_path : str
        Path to the datastore configuration file. The name of the
        datastore is assumed to be the name of the file without the extension, i.e.
        {datastore_name}.yaml.
    artifact_path : str
        Path to the artifact output directory. The statistics will be saved
        to a file named {datastore_name}.stats.zarr in this directory.
    """
    fp_stats = (
        Path(artifact_path)
        / Path(datastore_config_path).with_suffix(".stats.zarr").name
    )

    dataset_path = Path(datastore_config_path).with_suffix(".zarr")
    ds = xr.open_zarr(dataset_path)

    ds_stats = xr.Dataset()

    # XXX: this is a massive hack, but for now we will just assume that any
    # variable that has "__" in its name twice is a statistic. We can do this because
    # the statistics variables follow the format `{var_name}__{split}__{statistic}`.
    # (see
    # https://github.com/mllam/mllam-data-prep/blob/v0.6.0/mllam_data_prep/create_dataset.py#L279)
    # This could be done better by exposing in the mllam-data-prep how the
    # statistics variables are stored...

    for var_name in ds.data_vars:
        if len(var_name.split("__")) == 3:
            # this is a statistic
            ds_stats[var_name] = ds[var_name].compute()

    # copy the attributes too
    ds_stats.attrs.update(ds.attrs)

    logger.info(f"Saving stats for {datastore_config_path} to {fp_stats}")
    fp_stats.parent.mkdir(parents=True, exist_ok=True)
    ds_stats.to_zarr(str(fp_stats), mode="w", consolidated=True)


def _extract_stats_for_all_datastores(nl_config_path, artifact_output_path):
    datastore_paths = _find_datastore_paths(nl_config_path)

    for datastore_path in datastore_paths.values():
        _extract_stats(datastore_path, Path(artifact_output_path) / "stats")


def _copy_checkpoint(checkpoint_path: str, fp_checkpoint_dst: str):
    """
    Copy the checkpoint file to the artifact output directory.
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    artifact_output_path : str
        Path to the artifact output directory.
    """
    fp_checkpoint_dst = Path(fp_checkpoint_dst) / "checkpoint.pkl"
    shutil.copyfile(checkpoint_path, fp_checkpoint_dst)
    logger.info(f"Copied checkpoint {checkpoint_path} -> {fp_checkpoint_dst}")

    # XXX: we should really validate the checkpoint here somehow...


def _copy_yaml_configs(nl_config_path: str, artifact_output_path: str):
    """
    Copy the yaml config files to the artifact output directory. This includes
    the datastore configuration files that the neural-lam configuration file
    points to. The config files are copied to a subdirectory called "configs"
    in the artifact output directory.

    NB: The `config_path` field of the datastore(s) in the neural-lam config
    file is/are modified to be relative to the neural-lam config file.
    This is done to make it easier to package the config files together with
    the artifact.

    Parameters
    ----------
    nl_config_path : str
        Path to the yaml config file.
    artifact_output_path : str
        Path to the artifact output directory.
    """
    artifact_output_path = Path(artifact_output_path) / "configs"
    artifact_output_path.mkdir(parents=True, exist_ok=True)

    nl_config = yaml.safe_load(open(nl_config_path, "r"))

    datastore_config_paths = _find_datastore_paths(nl_config_path)

    for (
        datastore_name,
        datastore_config_path,
    ) in datastore_config_paths.items():
        fn_datastore_config = Path(datastore_config_path).name
        fp_datastore_config_dst = (
            artifact_output_path / Path(datastore_config_path).name
        )
        shutil.copy(datastore_config_path, fp_datastore_config_dst)
        logger.debug(
            f"Copied {datastore_config_path} -> {fp_datastore_config_dst}"
        )

        if datastore_name is not None:
            # update the config path in the neural-lam config file to just have
            # the name of the file
            nl_config["datastores"][datastore_name][
                "config_path"
            ] = fn_datastore_config

    fn_config = Path(nl_config_path).name
    fp_config_dst = artifact_output_path / fn_config

    # save the neural-lam config file
    with open(fp_config_dst, "w") as f:
        yaml.dump(nl_config, f)

    logger.debug(f"Copied {nl_config_path} -> {fp_config_dst}")


def _check_and_copy_training_cli_args(artifact_output_path: str):
    """
    Check for the presence of a file called `training_cli_args.yaml` in the
    current directory, check that it is a valid yaml file, and copy it to the
    artifact output directory.

    Parameters
    ----------
    artifact_output_path : str
        Path to the artifact output directory.
    """
    if not Path(TRAINING_CLI_ARGS_FILE).exists():
        raise FileNotFoundError(
            f"File {TRAINING_CLI_ARGS_FILE} not found. Please create this file "
            "with the command line arguments that were used to train the "
            "model."
        )

    try:
        with open(TRAINING_CLI_ARGS_FILE, "r") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"File {TRAINING_CLI_ARGS_FILE} is not a valid yaml file. Please "
            "check the file."
        ) from e
    # copy the file to the artifact output directory
    shutil.copyfile(
        TRAINING_CLI_ARGS_FILE,
        Path(artifact_output_path) / TRAINING_CLI_ARGS_FILE,
    )
    logger.info(
        f"Copied {TRAINING_CLI_ARGS_FILE} to "
        f"{Path(artifact_output_path) / TRAINING_CLI_ARGS_FILE}"
    )


def _create_artifact_meta(artifact_output_path, nl_config_path, args):
    """
    Create a metadata file for the artifact. This includes the name of the
    artifact, the version of the code, the path to the checkpoint file,
    the path to the neural-lam config file, and the command line arguments
    that were used to create the artifact.

    Parameters
    ----------
    artifact_output_path : str
        Path to the artifact output directory.
    nl_config_path : str
        Path to the neural-lam config file.
    args : Namespace
        Command line arguments that were used to create the artifact.
    """

    fp_artifact_meta_yaml = artifact_output_path / "artifact.yaml"

    meta = dict(
        artifact_name=artifact_output_path.name,
        mlwm_version=__version__,
        checkpoint_path=args.checkpoint,
        config_path=nl_config_path,
        argv=sys.argv,
        # hostname
        hostname=os.uname()[1],
        # current working directory
        cwd=os.getcwd(),
        # python version
        python_version=sys.version,
        # mllam-data-pipeline version
        mllam_data_prep_version=mdp.__version__,
    )

    with open(fp_artifact_meta_yaml, "w") as f:
        yaml.dump(meta, f)
    logger.info(f"Saved artifact metadata to {fp_artifact_meta_yaml}")


@logger.catch(reraise=True)
def main():
    import argparse

    argparser = argparse.ArgumentParser(
        description="Extract statistics from a training dataset."
    )
    argparser.add_argument(
        "artifact_name",
        type=str,
        help="Path to the mdp datastore configuration file.",
    )
    argparser.add_argument(
        "--nl_config",
        type=str,
        default="config.yaml",
        help="Path to neural-lam, the datastore path(s) will be parsed from here",
    )
    argparser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to the model checkpoint.",
    )

    args = argparser.parse_args()

    nl_config_path = args.nl_config
    workdir = Path(tempfile.mkdtemp())
    artifact_output_path = workdir / args.artifact_name

    logger.info(
        f"Building inference artifact for {args.artifact_name} with checkpoint "
        f"{args.checkpoint} with mlwm version `{__version__}` in {workdir}"
    )

    _extract_stats_for_all_datastores(
        nl_config_path=nl_config_path,
        artifact_output_path=artifact_output_path,
    )

    _copy_checkpoint(
        checkpoint_path=args.checkpoint, fp_checkpoint_dst=artifact_output_path
    )

    _copy_yaml_configs(
        nl_config_path=nl_config_path,
        artifact_output_path=artifact_output_path,
    )

    _create_artifact_meta(
        artifact_output_path=artifact_output_path,
        nl_config_path=nl_config_path,
        args=args,
    )

    _check_and_copy_training_cli_args(
        artifact_output_path=artifact_output_path
    )

    # create a zip file with everything in it
    fp_artifact_local = workdir / f"{args.artifact_name}.zip"

    with zipfile.ZipFile(fp_artifact_local, "w") as zipf:
        files_to_zip = list(artifact_output_path.rglob("*"))
        for fp in tqdm(files_to_zip, desc="Zipping files", unit="file"):
            zipf.write(fp, fp.relative_to(artifact_output_path))

    fp_artifact_target = UPath(
        ARTIFACT_PATH_FORMAT.format(artifact_name=args.artifact_name)
    )
    with open(fp_artifact_local, "rb") as f:
        # Upload the zip file to the target location
        logger.info(f"Uploading artifact to {fp_artifact_target}")
        fp_artifact_target.write_bytes(f.read())


if __name__ == "__main__":
    main()
