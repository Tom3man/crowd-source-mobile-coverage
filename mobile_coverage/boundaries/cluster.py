from __future__ import annotations

import warnings
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from mobile_coverage.geometry import prepare_coords
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.cluster")


def generate_cluster_hull_geom(
    df: pd.DataFrame,
    *,
    clusterer: str = "dbscan",
    hull_method: str = "convex",
    alpha: Optional[float] = None,
    min_cluster_points: int = 3,
    allow_singletons: bool = False,
    **kwargs,
) -> Optional[BaseGeometry]:
    """
    Cluster points (DBSCAN or HDBSCAN) and union cluster hulls.

    Args:
        df: DataFrame with longitude/latitude columns.
        clusterer: 'dbscan' or 'hdbscan'.
        hull_method: 'convex' or 'alpha'.
        alpha: Alpha parameter for alpha shape (requires alphashape package).
        min_cluster_points: Minimum number of points required for a hull.
    allow_singletons:
            If True, include singleton clusters as buffered points.
        kwargs: forwarded to the underlying cluster estimator.
    """
    coords = prepare_coords(df)
    if coords is None:
        return None

    labels: Optional[np.ndarray] = None

    if clusterer.lower() == "dbscan":
        try:
            from sklearn.cluster import DBSCAN
        except ImportError as exc:
            log.error(f"DBSCAN unavailable ({exc}).")
            return None
        model = DBSCAN(**kwargs)
        try:
            labels = model.fit_predict(coords)
        except Exception as exc:
            log.error(f"Error during DBSCAN fitting: {exc}")
            return None
    elif clusterer.lower() == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError as exc:
            log.error(f"hdbscan package unavailable ({exc}).")
            return None
        model = hdbscan.HDBSCAN(**kwargs)  # type: ignore
        try:
            labels = model.fit_predict(coords)
        except Exception as exc:
            log.error(f"Error during HDBSCAN fitting: {exc}")
            return None
    else:
        log.error(f"Unsupported clusterer '{clusterer}'.")
        return None

    if labels is None:
        return None

    unique_labels = sorted(i for i in np.unique(labels) if i != -1)
    polygons: list[Polygon] = []

    use_alpha = hull_method.lower() == "alpha"
    alphashape_fn: Optional[Callable[[Iterable[Point], float], BaseGeometry]] = None
    if use_alpha:
        try:
            import alphashape  # type: ignore

            alphashape_fn = alphashape.alphashape  # type: ignore
        except ImportError:
            warnings.warn(
                "alphashape library not available; falling back to convex hull"
            )
            use_alpha = False

    for label in unique_labels:
        cluster_points = coords[labels == label]
        if len(cluster_points) < min_cluster_points:
            if not allow_singletons:
                continue
            cluster_points = coords[labels == label]
        pts_geom = [Point(xy) for xy in cluster_points]
        if len(pts_geom) == 0:
            continue
        if use_alpha and alpha is not None and alphashape_fn is not None:
            try:
                hull = alphashape_fn(pts_geom, alpha)
            except Exception as exc:
                log.warning(f"Alpha shape failed for cluster {label}: {exc}")
                hull = MultiPoint(pts_geom).convex_hull
        else:
            hull = MultiPoint(pts_geom).convex_hull

        if hull.is_empty:
            continue

        if len(pts_geom) == 1 and allow_singletons:
            hull = hull.buffer(0.00001)

        hull = hull if hull.is_valid else hull.buffer(0)
        if isinstance(hull, Polygon) and hull.is_valid:
            polygons.append(hull)
        elif isinstance(hull, MultiPolygon):
            polygons.extend(
                [g for g in hull.geoms if isinstance(g, Polygon)
                 and g.is_valid])

    if not polygons:
        log.warning("No cluster hulls were generated.")
        return None

    try:
        result = unary_union(polygons)
    except Exception as exc:
        log.warning(f"unary_union failed for cluster hulls ({exc}).")
        result = MultiPolygon(polygons)

    if isinstance(result, Polygon):
        result = MultiPolygon([result])
    elif isinstance(result, MultiPolygon):
        result = MultiPolygon(
            [g for g in result.geoms if isinstance(g, Polygon) and
             not g.is_empty])
    elif hasattr(result, "geoms"):
        polys = [g for g in result.geoms if isinstance(g, Polygon) and
                 not g.is_empty]
        result = MultiPolygon(polys)
    else:
        result = None

    if result is not None and result.is_valid:
        return result

    if result is not None:
        buffered = result.buffer(0)
        if isinstance(buffered, Polygon):
            buffered = MultiPolygon([buffered])
        if isinstance(buffered, MultiPolygon) and buffered.is_valid:
            log.info("Cluster hull boundary fixed with buffer(0).")
            return buffered

    log.warning("Failed to obtain a valid cluster hull boundary.")
    return None
