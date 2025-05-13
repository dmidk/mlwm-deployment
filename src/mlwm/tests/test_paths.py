import datetime

import mlwm.paths as mlwm_paths


def test_round_trip_bbox():
    original_bbox = {
        "lon_min": -10.5,
        "lat_min": 35.0,
        "lon_max": 10.5,
        "lat_max": 45.0,
    }
    bbox_str = mlwm_paths.format_bbox(**original_bbox)
    parsed_bbox = mlwm_paths.parse_bbox(bbox_str)
    assert (
        parsed_bbox == original_bbox
    ), f"Expected {original_bbox}, got {parsed_bbox}"


def test_round_trip_resolution():
    original_resolution = {
        "lon_resolution": 0.1,
        "lat_resolution": 0.2,
        "unit": "deg",
    }
    resolution_str = mlwm_paths.format_resolution(**original_resolution)
    parsed_resolution = mlwm_paths.parse_resolution(resolution_str)
    assert (
        parsed_resolution == original_resolution
    ), f"Expected {original_resolution}, got {parsed_resolution}"


def test_round_trip_path():
    model_name = "harmonie_cy46"
    model_config = "default"
    bbox = {
        "lon_min": -10.5,
        "lat_min": 35.0,
        "lon_max": 10.5,
        "lat_max": 45.0,
    }
    resolution = {
        "lon_resolution": 0.1,
        "lat_resolution": 0.2,
        "unit": "deg",
    }
    analysis_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    data_kind = "surface_levels"

    path = mlwm_paths.create_path(
        model_name, model_config, bbox, resolution, analysis_time, data_kind
    )

    parsed_components = mlwm_paths.parse_path(path)

    assert (
        parsed_components["model_name"] == model_name
    ), f"Expected {model_name}, got {parsed_components['model_name']}"
    assert (
        parsed_components["model_config"] == model_config
    ), f"Expected {model_config}, got {parsed_components['model_config']}"
    assert (
        parsed_components["bbox"] == bbox
    ), f"Expected {bbox}, got {parsed_components['bbox']}"
    assert (
        parsed_components["resolution"] == resolution
    ), f"Expected {resolution}, got {parsed_components['resolution']}"
    assert (
        parsed_components["analysis_time"] == analysis_time
    ), f"Expected {analysis_time}, got {parsed_components['analysis_time']}"
    assert (
        parsed_components["data_kind"] == data_kind
    ), f"Expected {data_kind}, got {parsed_components['data_kind']}"
