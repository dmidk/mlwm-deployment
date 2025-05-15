from dataclasses import dataclass
from enum import Enum
from typing import Dict

import dataclass_wizard as dw


class Unit(str, Enum):
    """
    Enum to represent valid units for resolution.
    """

    METER = "m"
    KILOMETER = "km"
    DEGREE = "deg"


@dataclass
class BoundingBox:
    """
    Represents the bounding box for a dataset.
    """

    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the BoundingBox instance to a dictionary.

        Returns
        -------
        Dict[str, float]
            A dictionary representation of the bounding box.
        """
        return {
            "lon_min": self.lon_min,
            "lat_min": self.lat_min,
            "lon_max": self.lon_max,
            "lat_max": self.lat_max,
        }


@dataclass
class Resolution:
    """
    Represents the resolution of the dataset.
    """

    lon_resolution: float
    lat_resolution: float
    unit: Unit  # Enforce valid units using the Unit Enum

    def to_dict(self) -> Dict[str, str | float]:
        """
        Convert the Resolution instance to a dictionary.

        Returns
        -------
        Dict[str, str | float]
            A dictionary representation of the resolution.
        """
        return {
            "lon_resolution": self.lon_resolution,
            "lat_resolution": self.lat_resolution,
            # Use `.value` to get the string representation of the Enum
            "unit": self.unit.value,
        }


@dataclass
class UriArgs:
    """
    Represents the arguments used to construct the URI for a dataset.
    """

    bbox: BoundingBox
    resolution: Resolution
    data_kind: str
    model_name: str
    model_config: str
    bucket_name: str


@dataclass
class DataPathConfig:
    """
    Represents an input or output dataset configuration.
    """

    uri_args: UriArgs
    internal_path: str


@dataclass
class Config(dw.YAMLWizard):
    """
    Represents the overall configuration for the YAML file.
    """

    inputs: Dict[str, DataPathConfig]
    outputs: Dict[str, DataPathConfig]
    docker_image: str
