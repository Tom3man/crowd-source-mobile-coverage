from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from mobile_coverage.geometry import grid_to_polygons, make_grid, prepare_coords
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.gmm")


def generate_gmm_boundary_geom(
    df: pd.DataFrame,
    *,
    n_components: int = 3,
    covariance_type: str = "full",
    quantile: float = 0.05,
    resolution: int = 400,
    margin_factor: float = 1.0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Optional[BaseGeometry]:
    """
    Build a boundary using a GaussianMixture density level-set.
    Keeps the top (1 - quantile) probability mass.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        log.error(f"Error: GaussianMixture unavailable ({exc}).")
        return None

    coords = prepare_coords(df)
    if coords is None:
        return None

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        **kwargs,
    )
    try:
        gmm.fit(coords)
    except Exception as exc:
        log.error(f"Error during GaussianMixture training: {exc}")
        return None

    try:
        train_scores = gmm.score_samples(coords)
    except Exception as exc:
        log.error(f"Error evaluating GaussianMixture on training data: {exc}")
        return None

    threshold = np.quantile(train_scores, quantile)

    xx, yy, grid = make_grid(
        coords, resolution=resolution, margin_factor=margin_factor)
    try:
        grid_scores = gmm.score_samples(grid).reshape(xx.shape)
    except Exception as exc:
        log.error(f"Error evaluating GaussianMixture grid scores: {exc}")
        return None

    def score_points(points: np.ndarray) -> np.ndarray:
        return gmm.score_samples(points)

    return grid_to_polygons(
        xx, yy, grid_scores, level=threshold, scorer=score_points)
