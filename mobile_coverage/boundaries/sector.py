from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from mobile_coverage.common.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.sector")

_EARTH_RADIUS_M = 6_371_000.0
_DEFAULT_N_ARC_POINTS = 32


def _destination_point(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """
    Compute a destination point given a start location, bearing, and distance.

    Uses the spherical earth destination-point formula so the arc follows
    the Earth's surface rather than a flat-plane approximation.

    Args:
        lat_deg: Start latitude in decimal degrees (WGS84).
        lon_deg: Start longitude in decimal degrees (WGS84).
        bearing_deg: Bearing clockwise from North in degrees.
        distance_m: Distance in metres.

    Returns:
        (latitude_deg, longitude_deg) of the destination point.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    bearing = np.radians(bearing_deg)
    delta = distance_m / _EARTH_RADIUS_M

    sin_lat2 = (
        np.sin(lat) * np.cos(delta)
        + np.cos(lat) * np.sin(delta) * np.cos(bearing)
    )
    lat2 = np.arcsin(sin_lat2)
    lon2 = lon + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat),
        np.cos(delta) - np.sin(lat) * sin_lat2,
    )
    return np.degrees(lat2), np.degrees(lon2)


def generate_sector_polygon_geom(
    cell_lon: float,
    cell_lat: float,
    azimuth_deg: float,
    horizontal_beam_deg: float,
    radius_m: float,
    n_arc_points: int = _DEFAULT_N_ARC_POINTS,
) -> Optional[BaseGeometry]:
    """
    Build a sector (pizza-slice) polygon from cell antenna parameters.

    For directional cells the polygon runs from the tower centre out to
    an arc spanning azimuth ± horizontal_beam/2 at the given radius.
    For omnidirectional cells (horizontal_beam >= 360) a full circle is
    returned.

    Args:
        cell_lon: Tower longitude in decimal degrees (WGS84).
        cell_lat: Tower latitude in decimal degrees (WGS84).
        azimuth_deg: Antenna boresight bearing clockwise from North (degrees).
        horizontal_beam_deg: Total horizontal beamwidth (degrees).
        radius_m: Coverage radius in metres (use radii_90 for maximum extent).
        n_arc_points: Number of points sampled along the arc; higher gives a
            smoother boundary at the cost of polygon complexity.

    Returns:
        Shapely Polygon representing the planned coverage sector, or None if
        the geometry cannot be constructed.
    """
    try:
        is_omni = horizontal_beam_deg >= 360.0

        half_beam = horizontal_beam_deg / 2.0
        start_bearing = 0.0 if is_omni else azimuth_deg - half_beam
        end_bearing = 360.0 if is_omni else azimuth_deg + half_beam

        bearings = np.linspace(start_bearing, end_bearing, n_arc_points)

        arc_coords = [
            # _destination_point returns (lat, lon); Shapely expects (lon, lat)
            (pt[1], pt[0])
            for b in bearings
            for pt in (_destination_point(cell_lat, cell_lon, b, radius_m),)
        ]

        if is_omni:
            coords = arc_coords + [arc_coords[0]]
        else:
            centre = (cell_lon, cell_lat)
            coords = [centre] + arc_coords + [centre]

        polygon = Polygon(coords)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon if polygon.is_valid else None

    except Exception as exc:
        log.warning(f"Failed to generate sector polygon: {exc}")
        return None


def generate_sector_polygon_from_row(
    row: pd.Series,
    radius_col: str = "radii_90",
    n_arc_points: int = _DEFAULT_N_ARC_POINTS,
) -> Optional[BaseGeometry]:
    """
    Convenience wrapper that builds a sector polygon from a cell-details row.

    Args:
        row: A row from the DataFrame returned by get_cell_details().
        radius_col: Which radius column to use. One of 'radii_70', 'radii_80',
            or 'radii_90'. Defaults to 'radii_90' (maximum planned extent).
        n_arc_points: Number of points sampled along the arc.

    Returns:
        Shapely Polygon or None on failure.
    """
    return generate_sector_polygon_geom(
        cell_lon=float(row["cell_longitude"]),
        cell_lat=float(row["cell_latitude"]),
        azimuth_deg=float(row["azimuth"]),
        horizontal_beam_deg=float(row["horizontal_beam"]),
        radius_m=float(row[radius_col]),
        n_arc_points=n_arc_points,
    )
