# ANNA

The ANNA artifact is "s3://mlwm-artifacts/inference-artifacts/gefion-1.zip", which contains a model trained on the DANRA dataset on Gefion.

## Building image
`AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_access_key> CONTAINER_APP=podman ./build_image.sh`

## Running inference
`AWS_ACCESS_KEY_ID=<access_key> AWS_SECRET_ACCESS_KEY=<secret_access_key> ./run_inference_container.sh 2026-02-04T00:00:00Z`


## Training cli args

```yaml
- datastore:
  - config_path: /dcai/projects/cu_0003/user_space/hinkas/git-repos/ablation-studies/configs/danra_model1/7deg_config.yaml 
- num_workers: 6 
- precision: bf16-mixed 
- batch_size: 1
- hidden_dim: 300 
- hidden_dim_grid: 150 
- time_delta_enc_dim: 32 
- config_path: /dcai/projects/cu_0003/user_space/hinkas/git-repos/ablation-studies/configs/danra_model1/7deg_config.yaml 
- model: hi_lam 
- processor_layers: 2 
- graph_name: 7deg_rect_hi3 
- num_nodes: 2 
- epochs: 80
- ar_steps_train: 1
- lr: 0.001 
- min_lr: 0.001 
- val_interval: 5 
- ar_steps_eval: 4
- val_steps_to_log: 1 2 4
```

## mllam-data-prep config

```yaml
schema_version: v0.5.0
dataset_version: v0.1.0

output:
  variables:
    static: [grid_index, static_feature]
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  coord_ranges:
    time:
      start: 2000-01-01T00:00
      end: 2020-10-29T00:00
      step: PT3H
  chunking:
    time: 1
    state_feature: 20
  splitting:
    dim: time
    splits:
      train:
        start: 2000-01-01T00:00
        end: 2018-10-29T00:00
        compute_statistics:
          ops: [mean, std, diff_mean, diff_std]
          dims: [grid_index, time]
      val:
        start: 2018-11-05T00:00
        end: 2019-10-22T00:00
      test:
        start: 2019-10-29T00:00
        end: 2020-10-29T00:00

inputs:
  danra_sl_state:
    path: /dcai/projects/cu_0003/data/sources/danra/v0.5.0/single_levels.zarr/
    dims: [time, x, y]
    variables:
      - pres_seasurface
      - t2m
      - u10m
      - v10m
      - pres0m
      - lwavr0m
      - swavr0m
    dim_mapping:
      time:
        method: rename
        dim: time
      grid_index:
        method: stack
        dims: [x, y]
      state_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
    target_output_variable: state

  danra_pl_state:
    path: /dcai/projects/cu_0003/data/sources/danra/v0.5.0/pressure_levels.zarr/
    dims: [time, x, y, pressure]
    variables:
      z:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
      t:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
      r:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
      u:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
      v:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
      tw:
        pressure:
          values: [100, 200, 400, 600, 700, 850, 925, 1000,]
          units: hPa
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [pressure]
        name_format: "{var_name}{pressure}"
      grid_index:
        method: stack
        dims: [x, y]
    target_output_variable: state

  danra_static:
    path: /dcai/projects/cu_0003/data/sources/danra/v0.5.0/single_levels.zarr/
    dims: [x, y]
    variables:
      - lsm
      - orography
    dim_mapping:
      grid_index:
        method: stack
        dims: [x, y]
      static_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
    target_output_variable: static

  danra_forcing:
    path: /dcai/projects/cu_0003/data/sources/danra/v0.5.0/single_levels.zarr/
    dims: [time, x, y]
    derived_variables:
      # derive variables to be used as forcings
      toa_radiation:
        kwargs:
          time: ds_input.time
          lat: ds_input.lat
          lon: ds_input.lon
        function: mllam_data_prep.ops.derive_variable.physical_field.calculate_toa_radiation
      hour_of_day_sin:
        kwargs:
          time: ds_input.time
          component: sin
        function: mllam_data_prep.ops.derive_variable.time_components.calculate_hour_of_day
      hour_of_day_cos:
        kwargs:
          time: ds_input.time
          component: cos
        function: mllam_data_prep.ops.derive_variable.time_components.calculate_hour_of_day
      day_of_year_sin:
        kwargs:
          time: ds_input.time
          component: sin
        function: mllam_data_prep.ops.derive_variable.time_components.calculate_day_of_year
      day_of_year_cos:
        kwargs:
          time: ds_input.time
          component: cos
        function: mllam_data_prep.ops.derive_variable.time_components.calculate_day_of_year
    dim_mapping:
      time:
        method: rename
        dim: time
      grid_index:
        method: stack
        dims: [x, y]
      forcing_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
    target_output_variable: forcing

extra:
  projection:
    class_name: LambertConformal
    kwargs:
      central_longitude: 25.0
      central_latitude: 56.7
      standard_parallels: [56.7, 56.7]
      globe:
        semimajor_axis: 6367470.0
        semiminor_axis: 6367470.0
```

## Neural-lam config

```yaml
datastore:
  config_path: /dcai/projects/cu_0003/user_space/hinkas/git-repos/ablation-studies/configs/danra_model1/danra_model1_config.yaml
  kind: mdp
datastore_boundary:
  config_path: /dcai/projects/cu_0003/user_space/hinkas/git-repos/ablation-studies/configs/era_forcing/era_7deg_model1_config.yaml
  kind: mdp
training:
  excluded_intervals:
  - - 2002-11-19T00
    - 2002-11-19T06
  - - 2007-08-26T00
    - 2007-08-26T21
  - - 2017-11-25T15
    - 2017-11-25T15
  output_clamping:
    lower:
      r100: 0
      r1000: 0
      r200: 0
      r400: 0
      r600: 0
      r700: 0
      r850: 0
      r925: 0
    upper:
      r100: 1
      r1000: 1
      r200: 1
      r400: 1
      r600: 1
      r700: 1
      r850: 1
      r925: 1
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      lwavr0m: 1.0
      pres0m: 1.0
      pres_seasurface: 1.0
      r100: 0.125
      r1000: 0.125
      r200: 0.125
      r400: 0.125
      r600: 0.125
      r700: 0.125
      r850: 0.125
      r925: 0.125
      swavr0m: 1.0
      t100: 0.125
      t1000: 0.125
      t200: 0.125
      t2m: 1.0
      t400: 0.125
      t600: 0.125
      t700: 0.125
      t850: 0.125
      t925: 0.125
      tw100: 0.125
      tw1000: 0.125
      tw200: 0.125
      tw400: 0.125
      tw600: 0.125
      tw700: 0.125
      tw850: 0.125
      tw925: 0.125
      u100: 0.125
      u1000: 0.125
      u10m: 1.0
      u200: 0.125
      u400: 0.125
      u600: 0.125
      u700: 0.125
      u850: 0.125
      u925: 0.125
      v100: 0.125
      v1000: 0.125
      v10m: 1.0
      v200: 0.125
      v400: 0.125
      v600: 0.125
      v700: 0.125
      v850: 0.125
      v925: 0.125
      z100: 0.125
      z1000: 0.125
      z200: 0.125
      z400: 0.125
      z600: 0.125
      z700: 0.125
      z850: 0.125
      z925: 0.125
```
