from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from mobile_coverage.geometry import grid_to_polygons, make_grid, prepare_coords
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.kde")


def generate_kde_boundary_geom(
    df: pd.DataFrame,
    *,
    bandwidth: float = 0.001,
    kernel: str = "gaussian",
    quantile: float = 0.05,
    resolution: int = 400,
    margin_factor: float = 1.0,
    **kwargs,
) -> Optional[BaseGeometry]:
    """
    Build a boundary from a KernelDensity estimator.
    """
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError as exc:
        log.error(f"Error: KernelDensity unavailable ({exc}).")
        return None

    coords = prepare_coords(df)
    if coords is None:
        return None

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kwargs)
    try:
        kde.fit(coords)
    except Exception as exc:
        log.error(f"Error during KernelDensity fitting: {exc}")
        return None

    try:
        train_scores = kde.score_samples(coords)
    except Exception as exc:
        log.error(f"Error evaluating KernelDensity on training data: {exc}")
        return None

    threshold = np.quantile(train_scores, quantile)

    xx, yy, grid = make_grid(
        coords, resolution=resolution, margin_factor=margin_factor)
    try:
        grid_scores = kde.score_samples(grid).reshape(xx.shape)
    except Exception as exc:
        log.error(f"Error evaluating KernelDensity grid scores: {exc}")
        return None

    def score_points(points: np.ndarray) -> np.ndarray:
        return kde.score_samples(points)

    return grid_to_polygons(
        xx, yy, grid_scores, level=threshold, scorer=score_points
    )
