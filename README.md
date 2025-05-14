# MLWM Deployment
Deployment repository for DMIs MLWMs (Machine Learning Weather Model).

This repository contains the deployment scripts and configuration files for deploying the MLWM models. All models are stored as container images and have an assumption of being deployed in a containerized environment. The containers are assuming the same structure of input data.

## Data Directory Structure
Input data is following a structure that contains a list of traditional weather model identifiers:

- `<model_name>`: Name of the model
- `<model_config>`: Name of the model configuration
- `<bbox>`: Bounding box of the model
- `<member>`: Ensemble number of run
- `<resolution>`: Resolution of the model
- `<analysis_time>`: Analysis time of the model run in [ISO8601 format](https://en.wikipedia.org/wiki/ISO_8601), `YYYY-MM-DDThhmmZ` (i.e. without colons ":" which is still valid ISO8601 format, with `Z` to indicate UTC time)
- `<data_kind>`: Kind of data [e.g. "pressure_levels", "surface_levels"]

A path is then constructed as follows:
```
<model_name>/<model_config>/<bbox>/<resolution>/<analysis_time>/<member>/<data_kind>.zarr
```
- `<model_name>` is a string that contains the name of the model, e.g. `harmonie_cy46`, `ifs_cy50`, etc.
- `<bbox>` is a string that contains the coordinates of the bounding box in the format `w<lon_min>_s<lat_min>_e<lon_max>_n<lat_max>`.
- `<resolution>` is a string that contains the resolution of the model in the format `dx<lon_resolution><unit>_dy<lat_resolution><unit>`.

All floats (`lon_min`, `lat_min`, `lon_max`, `lat_max`, `lon_resolution`,
`lat_resolution`) are formatted with 'p' in place of the decimal point to avoid
having dots in the paths. For example, `0.1` becomes `0p1`.

Functions to construct and parse p-number strings, resolution strings, bbox strings and path strings are provided in the `mlwm.paths` module. E.g.:

```python
import mlwm.paths as mpaths
import datetime

path = mpaths.create_path(
    model_name="harmonie_cy46",
    model_config="danra",
    bbox=dict(lon_min=12.5, lat_min=45.65, lon_max=24.52, lat_max=64.40),
    resolution=dict(lon_resolution=2.5, lat_resolution=2.5, units="km"),
    analysis_time=datetime.datetime(2023, 10, 1, 12, 0),
    data_kind="pressure_levels"
)

parsed_components = mlwm_paths.parse_path(path)
```

More examples can be found in [`mlwm/tests/test_paths.py`](src/mlwm/tests/test_paths.py).

`<member>` is the number of the ensemble member, with 0 being the control run following the format `member<member>`, like `member0`.


## Building inference artifact on machine that was trained on

The purpose of building an **inference artifact** is to collect all the information that is needed to run model inference later.

This includes:

- model checkpoint (with the weights)
- model configuration (i.e. the neural-lam and datastore config yaml-files)
- the training arguments (i.e. command line arguments used during training)
- statistics of the training dataset (used for standardization)

With the `mlwm.build_inference_artifact` command, you can build an inference artifact zip-file and upload this to the DMI S3 bucket for inference artifacts (`s3://mlwm-artifacts/inference-artifacts/`). Before running this command you must make a yaml-file containing the command-line arguments you used during training, e.g.:

```yaml
# training_cli_args.yaml
- num_workers: 6
- precision: bf16-mixed
- batch_size: 1
- hidden_dim: 300
- hidden_dim_grid: 150
- time_delta_enc_dim: 32
- config_path: ${CONFIG}
- model: hi_lam
- processor_layers: 2
- graph_name: 2dfeat_7deg_tri_9s_hi3
- num_nodes: $SLURM_JOB_NUM_NODES
- epochs: 80
- ar_steps_train: 1
- lr: 0.001
- min_lr: 0.001
- val_interval: 5
- ar_steps_eval: 4
- val_steps_to_log: 1 2 4
```

When you build the inference artifact you need to give it a name (in the example below this is `gefion-1`), the resulting artifact zip file will then be named `<name>.zip`. This name is simply used to identify the artifact and is not necessarily related to any forecasting container image built later.

To build the inference artifact you run `mlwm.build_inference_artifact`, for example:

```bash
uv run python -m mlwm.build_inference_artifact gefion-1 --nl_config config.yaml --checkpoint train-graph_lam-4x2-01_24_14-5078/min_val_loss.ckp
```

The contents of the zip-file will be:

```bash
gefion-1/
├── artifact.yaml
├── checkpoint.pkl
├── configs
│   ├── config.yaml
│   └── danra.datastore.yaml
├── stats
│   └── danra.datastore.stats.zarr
└── training_cli_args.yaml
```

`artifact.yaml` contains information about how the inference artifact was built, i.e. the command line arguments used to build the artifact, e.g. system information, and the command line arguments used to invocate the cli.
