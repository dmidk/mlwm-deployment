"""
This module implements functions to create and parse paths for model data.

The format is specified in README.md in this repo

"""
import datetime
from typing import Dict, Union

import parse

BBOX_STRING_FORMAT = "w{lon_min}_s{lat_min}_e{lon_max}_n{lat_max}"
RESOLUTION_STRING_FORMAT = "dx{lon_resolution}{unit}_dy{lat_resolution}{unit}"
NUMBER_STRING_FORMAT = "{integer_part}p{decimal_part}"
VALID_LENGTH_UNITS = ["m", "km", "deg"]
DATETIME_FORMAT = r"%Y-%m-%dT%H%MZ"  # ISO8601 format without colons or seconds
PATH_FORMAT = (
    "{model_name}/{model_config}/{bbox}/{resolution}/"
    "{analysis_time:" + DATETIME_FORMAT + "}/member{member:d}/{data_kind}.zarr"
)


def format_number(number: Union[float, int]) -> str:
    """
    Format a float number to a string with 'p' in place of the decimal point.

    Parameters
    ----------
    number : Union[float, int]
        The number to format. It can be either an integer or a float.

    Returns
    -------
    str
        The formatted number as a string. If the input is an integer, it
        returns the integer as a string. If the input is a float, it replaces
        the decimal point with 'p'.
    """
    if isinstance(number, int):
        return str(number)
    if isinstance(number, float):
        # Split the number into integer and decimal parts
        integer_part, decimal_part = str(number).split(".")
        # Format the number with 'p' in place of the decimal point
        return NUMBER_STRING_FORMAT.format(
            integer_part=integer_part,
            decimal_part=decimal_part,
        )
    else:
        raise ValueError(
            f"Unsupported type: {type(number)}. Expected int or float."
        )


def parse_number(number: str) -> Union[float, int]:
    """
    Parse a string number formatted with 'p' in place of the decimal point back
    to a float or int.

    Parameters
    ----------
    number : str
        The formatted number as a string. It can be either an integer or a float.

    Returns
    -------
    Union[float, int]
        The parsed number as a float or int. If the input is an integer, it
        returns the integer. If the input is a float, it replaces 'p' with '.'
        to convert it back to a float.
    """
    if "p" in number:
        # Split the number into integer and decimal parts
        integer_part, decimal_part = number.split("p")
        # Convert back to float
        return float(f"{integer_part}.{decimal_part}")
    else:
        return int(number)


def format_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> str:
    """
    Format the bounding box coordinates as a string.

    Parameters
    ----------
    lon_min : float
        The minimum longitude.
    lat_min : float
        The minimum latitude.
    lon_max : float
        The maximum longitude.
    lat_max : float
        The maximum latitude.

    Returns
    -------
    str
        The formatted bounding box as a string in the format
        `w<lon_min>_s<lat_min>_e<lon_max>_n<lat_max>`.
    """
    return BBOX_STRING_FORMAT.format(
        lon_min=format_number(lon_min),
        lat_min=format_number(lat_min),
        lon_max=format_number(lon_max),
        lat_max=format_number(lat_max),
    )


def parse_bbox(bbox: str) -> Dict[str, float]:
    """
    Parse a bounding box string into its components.

    Parameters
    ----------
    bbox : str
        The bounding box string in the format
        `w<lon_min>_s<lat_min>_e<lon_max>_n<lat_max>`.

    Returns
    -------
    dict
        A dictionary containing the bounding box coordinates. The keys are:
        - 'lon_min': The minimum longitude.
        - 'lat_min': The minimum latitude.
        - 'lon_max': The maximum longitude.
        - 'lat_max': The maximum latitude.
    """
    parts = parse.parse(BBOX_STRING_FORMAT, bbox)
    if parts is None:
        raise ValueError(
            f"Invalid bbox format: {bbox}. Expected format: {BBOX_STRING_FORMAT}"
        )

    return dict(
        lon_min=parse_number(parts["lon_min"]),
        lat_min=parse_number(parts["lat_min"]),
        lon_max=parse_number(parts["lon_max"]),
        lat_max=parse_number(parts["lat_max"]),
    )


def format_resolution(
    lon_resolution: Union[float, int],
    lat_resolution: Union[float, int],
    unit: str,
) -> str:
    """
    Format the resolution as a string.

    Parameters
    ----------
    lon_resolution : Union[float, int]
        The longitude resolution.
    lat_resolution : Union[float, int]
        The latitude resolution.
    unit : str
        The unit of the resolution. Must be one of ['m', 'km', 'deg'].

    Returns
    -------
    str
        The formatted resolution as a string in the format
        `dx<lon_resolution><unit>_dy<lat_resolution><unit>`.

    Raises
    ------
    ValueError
        If the unit is not one of the valid length units.
    """
    if unit not in VALID_LENGTH_UNITS:
        raise ValueError(
            f"Invalid unit: {unit}. Must be one of {VALID_LENGTH_UNITS}."
        )

    return RESOLUTION_STRING_FORMAT.format(
        lon_resolution=format_number(lon_resolution),
        lat_resolution=format_number(lat_resolution),
        unit=unit,
    )


def parse_resolution(resolution: str) -> Dict[str, Union[float, int]]:
    """
    Parse a resolution string into its components.

    Parameters
    ----------
    resolution : str
        The resolution string in the format
        `dx<lon_resolution><unit>_dy<lat_resolution><unit>`.

    Returns
    -------
    dict
        A dictionary containing the resolution components. The keys are:
        - 'lon_resolution': The longitude resolution.
        - 'lat_resolution': The latitude resolution.
        - 'unit': The unit of the resolution.

    Raises
    ------
    ValueError
        If the unit is not one of the valid length units.
    """
    parts = parse.parse(RESOLUTION_STRING_FORMAT, resolution)
    if parts is None:
        raise ValueError(
            f"Invalid resolution format: {resolution}. "
            f"Expected format: {RESOLUTION_STRING_FORMAT}"
        )

    return dict(
        lon_resolution=parse_number(parts["lon_resolution"]),
        lat_resolution=parse_number(parts["lat_resolution"]),
        unit=parts["unit"],
    )


def create_path(
    model_name: str,
    model_config: str,
    bbox: dict,
    resolution: dict,
    analysis_time: datetime.datetime,
    data_kind: str,
    member: int = 0,
) -> str:
    """
    Create a path for the model data.

    Parameters
    ----------
    model_name : str
        The name of the model.
    model_config : str
        The name of the model configuration.
    bbox : dict
        The bounding box coordinates as a dictionary with keys 'lon_min',
        'lat_min', 'lon_max', 'lat_max'.
    resolution : dict
        The resolution as a dictionary with keys 'lon_resolution',
        'lat_resolution', and 'unit'.
    analysis_time : str
        The analysis time in ISO8601 format.
    data_kind : str
        The kind of data (e.g. "pressure_levels", "surface_levels").
    member : int, optional
        Ensemble member number. Default is 0 which is the control member.
    Returns
    -------
    str
        The constructed path as a string.
    """

    bbox_str = format_bbox(**bbox)
    resolution_str = format_resolution(**resolution)

    return PATH_FORMAT.format(
        model_name=model_name,
        model_config=model_config,
        bbox=bbox_str,
        resolution=resolution_str,
        analysis_time=analysis_time,
        data_kind=data_kind,
        member=member,
    )


def parse_path(path: str):
    """
    Parse a path into its components.

    Parameters
    ----------
    path : str
        The path to parse.

    Returns
    -------
    dict
        A dictionary containing the components of the path. The keys are:
        - 'model_name': The name of the model.
        - 'model_config': The name of the model configuration.
        - 'bbox': The bounding box as a dictionary with keys 'lon_min',
          'lat_min', 'lon_max', 'lat_max'.
        - 'resolution': The resolution as a dictionary with keys
          'lon_resolution', 'lat_resolution', and 'unit'.
        - 'member': The ensemble member number.
        - 'analysis_time': The analysis time as a datetime object.
        - 'data_kind': The kind of data (e.g. "pressure_levels", "surface_levels").
    """
    parts = parse.parse(PATH_FORMAT, path)

    if parts is None:
        for n in range(10):
            s1 = "/".join(PATH_FORMAT.split("/")[:n])
            s2 = "/".join(path.split("/")[:n])
            print(s1, s2)
            print(parse.parse(s1, s2))
        raise ValueError(
            f"Invalid path format: {path}. Expected format: {PATH_FORMAT}"
        )

    model_name = parts["model_name"]
    model_config = parts["model_config"]
    bbox = parse_bbox(parts["bbox"])
    resolution = parse_resolution(parts["resolution"])
    analysis_time = parts["analysis_time"]
    data_kind = parts["data_kind"]
    member = parts["member"]

    return dict(
        model_name=model_name,
        model_config=model_config,
        bbox=bbox,
        resolution=resolution,
        analysis_time=analysis_time,
        data_kind=data_kind,
        member=member,
    )
