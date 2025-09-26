# surface-dummy-model_DINI

The model configuration in this directory is a dummy model that was trained on
surface variables from DANRA, only 10 days of data and only trained 10
epochs. It is intended only as a demonstration of the inference pipeline and is
expected to give very poor results.

## Upstream package change requirements

Relative to the `main` branch on both github.com/mllam/mllam-data-prep and
github.com/mllam/neural-lam and number of pieces of functionality are currently
required to run this configuration:

**mllam-data-prep**:

using branch `feat/inference-cli-args` on
https://github.com/leifdenby/mllam-data-prep@feat/inference-cli-args, which adds:

- functionality to invert datasets created by `mllam-data-prep` back to the
  structure of the input datasets that we were used. In the current
  configuration that is used to restructure the forecast zarr dataset that
  `neural-lam` outputs during inference back to the structure of the input
  forecast dataset.

  - also in seperate branch and PR: https://github.com/leifdenby/mllam-data-prep/tree/feat/inverse-ops

- use of cf-compliant encoding of `xarray/pandas` `MultiIndex` coordinates to
  store stacked coordinates. This is required since we `MultiIndex` coordinates
  can't natively be stored in zarr/netcdf files, but fortunately `cf_xarray`
  have implemented the cf-compliant way of handling this (see
  https://cf-xarray.readthedocs.io/en/latest/coding.html)

  - needs its own branch and PR

- support for supplying statistics from the training dataset during creation of
  the inference dataset, so that the inference dataset can be normalised in the
  same way as the training dataset.

  - needs its own branch and PR

- support for selecting only a single value from a variable/coordinate in the
  configuration. This is used to select only a single analysis time during
  creation of the inference dataset.

  - needs its own branch and PR


**neural-lam**:

using branch `dev/first-inference-image` on
https://github.com/leifdenby/neural-lam/tree/dev/first-inference-image, which
adds:

- support for decoding cf-compliant `MultiIndex` encoded coordinates when reading
  datasets produced with mllam-data-prep.

  - this needs its own branch and PR, and needs to be implemented so datasets
    made with previous versions of `mllam-data-prep` are still usable in `neural-lam`

- support for writing output from inference (i.e. `--eval` mode) to a zarr
  dataset. Needs to be merged after the multiindex decoding above.

  - also in seperate branch and PR: https://github.com/leifdenby/neural-lam/tree/feat/write-to-zarr

- support for using forecast data in in mllam-data-prep datastore (`MDPDatastore`)

  - needs its own branch and PR

- make logging of validation steps optional in the training CLI (i.e. `--eval` mode)

  - needs its own branch and PR
