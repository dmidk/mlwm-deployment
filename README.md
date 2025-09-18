# MLWM Deployment
Deployment repository for DMIs MLWMs (Machine Learning Weather Model).

> WARNING: This repository is under active development and the information
> provided here is subject to change.

This repository was designed with the following constraints in mind:

- Individual forecast runs must be executed within isolated container
  environments.
- This repository must contain the configuration of which pretrained models to
  run for each forecasting run, with a clear process for how to add new pretrained
  models.
- This repository must contain a python package (named `mlwm`) to be used
  for
    1. packaging up everything needed for inference from the computer where a
       model was trained on (we call this creating an *inference artifact*)
    2. continuous integration tooling that builds container images that will be
       used for inference
    3. CLI for executing all model inferences configured in this repository.

## Adding a new pretrained model for scheduled forecasting

The process of adding a new model to be executed as part of the set of models
run for each forecast cycle is as follows:

1. Create an **inference artifact** and upload to the mlwm inference artifact
   repository (`s3://mlwm-artifacts/`) on the machine where you trained the
   model. This contains everything needed to run the forecast inference (model
   weights, model configuration, dataset statistics). This is done with
   `wlwm.build_inference_artifact`. If you are on an airgapped system you can
   upload the inference artifact zip-file manually.
2. Add configuration to mlwm deployment repository (the repository you are
   looking at right now) which describes how to build an inference container
   image from the inference artifact you created above.

This repository contains the deployment scripts and configuration files for deploying the MLWM models. All models are stored as container images and have an assumption of being deployed in a containerized environment. The containers are assuming the same structure of input data.

## Building inference images
Inference are build during the GitHub Actions workflow. The workflow is triggered on every push to the `main` branch. The workflow builds the inference images and pushes them to the ghcr.io registry.

It is also possible to build the inference images manually. This can be done by using the `build_image.sh` script, that is located in the configuration directory of the model you want to build. For example, to build the `surface-dummy-model_DINI` image, you can run the following command:

```bash
cd configurations/surface-dummy-model_DINI
AWS_ACCESS_KEY_ID=<your_aws_access_key_id> AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key> MLWM_PULL_PROXY=<proxy_for_container_registry> ./build_image.sh
```
If you don't have a proxy for the container registry, you can leave the `MLWM_PULL_PROXY` variable empty.


### Building inference artifact on machine that was trained on

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
uv run python -m mlwm.build_inference_artifact gefion-1 \
--nl_config /dcai/projects/cu_0003/user_space/hinkas/git-repos/ablation-studies/configs/danra_model1/7deg_config.yaml \
--cli_training_args_filepath /dcai/projects/cu_0003/user_space/hinkas/git-repos/mllam/mlwm-deployment/config.yaml \
--checkpoint /dcai/projects/cu_0003/user_space/hinkas/models_for_inf/train-hi_lam-2x300-02_27_15-4034/last.ckpt
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

### Adding a new model configuration

> TBC: This functionality is not yet complete, so the information written below is subject to change.

Model configurations are stored in this repository (in `configurations/{model_name}`). Each configuration should be comprised of three things:
1. A README describing the inference setup. Ideally, this will include things like notes on how the model was trained and what makes the model special.
2. The inference artifact URI (this contains all information required from training to execute the model for inference).
3. A Containerfile to build the inference container image from.

### Executing the configured forecasting models

> TBC: This functionality is not yet complete, so the information written below is subject to change.

To execute the forecasting models configured in this repo run:

`uv run python -m mlwm.run_models`

Internally, the execution works as follows:

- for each model configured in this repository (stored in `configurations/{model_name}`):
    - read the model configuration .yaml file (stored in `configurations/{model_name}/config.yaml`)
    - copy the required input zarr datasets from the s3 locations described in the configuration file, this will be to `$WORKDIR/input/{model_name}/{input_name}`. These will be parameterised by the `analysis_time` of the execution
    - create the output directorie(s) for the model, this will be `$WORKDIR/output/{model_name}/{output_name}`
    - launch the docker container for the model, mounting the input and output directories as volumes to the container
    - wait for the container to finish running
    - copy the output directories to the s3 locations given in the model configuration, again these will be parameterised by the `analysis_time`


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
