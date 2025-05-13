# MLWM Deployment
Deployment repository for DMIs MLWMs (Machine Learning Weather Model).

This repository contains the deployment scripts and configuration files for deploying the MLWM models. All models are stored as container images and have an assumption of being deployed in a containerized environment. The containers are assuming the same structure of input data.

## Data Directory Structure
Input data is following a structure that contains a list of traditional weather model identifiers:

- `<model_name>`: Name of the model
- `<model_config>`: Name of the model configuration
- `<bbox>`: Bounding box of the model
- `<resolution>`: Resolution of the model
- `<analysis_time>`: Analysis time of the model run in [ISO8601 format](https://en.wikipedia.org/wiki/ISO_8601) (without colons ":" which is
  still valid ISO8601 format)
- `<data_kind>`: Kind of data [e.g. "pressure_levels", "surface_levels"]

A path is then constructed as follows:
```
<model_name>/<model_config>/<bbox>/<resolution>/<analysis_time>/<data_kind>.zarr
```
- `<model_name>` is a string that contains the name of the model, e.g. `harmonie_cy46`, `ifs_cy50`, etc.
- `<bbox>` is a string that contains the coordinates of the bounding box in the format `w<lon_min>_s<lat_min>_e<lon_max>_n<lat_max>`.
- `<resolution>` is a string that contains the resolution of the model in the format `dx<lon_resolution><unit>_dy<lat_resolution><unit>`.

All floats (`lon_min`, `lat_min`, `lon_max`, `lat_max`, `lon_resolution`,
`lat_resolution`) are formatted with 'p' in place of the decimal point to avoid
having dots in the paths. For example, `dx0.1` becomes `dx0p1`.
