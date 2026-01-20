from __future__ import annotations

from typing import Optional

import pandas as pd
from shapely.geometry.base import BaseGeometry

from mobile_coverage.geometry import grid_to_polygons, make_grid, prepare_coords
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.isolation_forest")


def generate_isolation_forest_boundary_geom(
    df: pd.DataFrame,
    *,
    resolution: int = 400,
    margin_factor: float = 1.0,
    level: float = 0.0,
    random_state: int = 42,
    verbose: bool = False,
    **kwargs,
) -> Optional[BaseGeometry]:
    """
    Generate a spatial coverage boundary using an IsolationForest anomaly detector.

    The IsolationForest learns a non-parametric boundary that encloses the
    inlier (coverage) region in longitude-latitude space. The resulting decision
    surface is converted to a shapely MultiPolygon by contouring the level-set
    f(x) = level.

    Args:
        df: DataFrame containing at least 'longitude' and 'latitude' columns.
        resolution: Number of grid divisions for boundary contouring.
        margin_factor: Relative margin around the training point extent.
        level: Decision threshold (typically 0.0; higher = tighter boundary).
        random_state: Seed for deterministic reproducibility.
        verbose: If True, prints additional diagnostic info.
        **kwargs: Passed through to sklearn.ensemble.IsolationForest.

    Returns:
        A shapely MultiPolygon representing the inferred coverage boundary,
        or None if the model failed to converge or produced invalid geometry.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        if verbose:
            log.error(f"[IsolationForest] ImportError: {exc}")
        return None

    coords = prepare_coords(df)
    if coords is None or len(coords) < 3:
        if verbose:
            log.error("[IsolationForest] Insufficient or invalid coordinates.")
        return None

    # Initialise with safe defaults (consistent reproducibility)
    clf = IsolationForest(random_state=random_state, **kwargs)

    try:
        clf.fit(coords)
    except Exception as exc:
        if verbose:
            log.error(f"[IsolationForest] Training failed: {exc}")
        return None

    # Build regular evaluation grid around data extent
    xx, yy, grid = make_grid(
        coords, resolution=resolution, margin_factor=margin_factor
    )

    try:
        grid_scores = clf.decision_function(grid).reshape(xx.shape)
    except Exception as exc:
        if verbose:
            log.error(f"[IsolationForest] Decision function failed: {exc}")
        return None

    def score_points(points):
        return clf.decision_function(points)

    # Convert score level-set to polygons
    result = grid_to_polygons(
        xx, yy, grid_scores, level=level, scorer=score_points
    )

    if result is None and verbose:
        log.error("[IsolationForest] Failed to generate a valid boundary.")

    return result
