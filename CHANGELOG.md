# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## unreleased

### Added

- add GitHub Actions workflow for building and pushing inference images to ghcr.io, [\#11](https://github.com/dmidk/mlwm-deployment/pull/11), @khintz
- add CLI arg to inference artifact building to skip upload [\#16](https://github.com/dmidk/mlwm-deployment/pull/16), @leifdenby
- add model running CLI, [\#12](https://github.com/dmidk/mlwm-deployment/pull/12), @leifdenby
- add inferernce container entrypoint, to be run when an inference container is started, [\#13](https://github.com/dmidk/mlwm-deployment/pull/13), @leifdenby
- add functions for constructing and parsing S3 forecast data paths (`mlwm.paths`) [\#2](https://github.com/dmidk/mlwm-deployment/pull/2), @leifdenby

### Modified

- make ensemble member part of the folder structure [\#6](https://github.com/dmidk/mlwm-deployment/pull/6), @observingClouds

## [v0.1.0](https://github.com/dmidk/mlwm-deployment/releases/tag/v0.1.0)

### Added

- cli tool for building (and uploading to S3 bucket) inference artifacts `mlwm.build_inference_artifact` containing model checkpoint, training/data config, training cli args and training dataset statistics, [\#1](https://github.com/dmidk/mlwm-deployment/pull/1)
