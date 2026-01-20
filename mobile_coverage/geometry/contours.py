from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.geometry.contours")


def grid_to_polygons(
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    level: float,
    scorer: Callable[[np.ndarray], np.ndarray],
) -> Optional[BaseGeometry]:
    """
    Convert a level-set on a score grid into a MultiPolygon.
    """
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xx, yy, values, levels=[level])
    except Exception as exc:
        log.error(f"Error during contour generation: {exc}")
        plt.close(fig)
        return None
    finally:
        plt.close(fig)

    if not cs.allsegs or not cs.allsegs[0]:
        log.warning("No contour lines found at requested level.")
        return None

    lines = [seg for seg in cs.allsegs[0] if len(seg) >= 3]
    if not lines:
        log.warning("Contour segmentation did not yield valid lines.")
        return None

    line_geoms = []
    for seg in lines:
        try:
            line_geoms.append(LineString(seg))
        except Exception:
            continue

    if not line_geoms:
        log.warning(
            "Failed to build shapely LineStrings from contour segments.")
        return None

    try:
        all_polygons = list(polygonize(line_geoms))
    except Exception as exc:
        log.warning(f"Error during polygonization: {exc}")
        return None

    if not all_polygons:
        log.warning("Polygonization did not yield any polygons.")
        return None

    positive_polygons: list[Polygon] = []
    for poly in all_polygons:
        if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 0:
            continue
        rep = poly.representative_point()
        try:
            score = scorer(np.array([[rep.x, rep.y]]))[0]
        except Exception as exc:
            log.warning(
                f"Failed to evaluate score at representative point: {exc}")
            continue
        if score >= level:
            cleaned = poly if poly.is_valid else poly.buffer(0)
            if isinstance(cleaned, Polygon) and cleaned.is_valid:
                positive_polygons.append(cleaned)

    if not positive_polygons:
        log.warning("No polygons classified as inside the boundary.")
        return None

    try:
        result = unary_union(positive_polygons)
    except Exception as exc:
        log.warning(
            f"unary_union failed ({exc}); falling back to MultiPolygon.")
        result = MultiPolygon([p for p in positive_polygons if p.is_valid])

    if isinstance(result, Polygon):
        result = MultiPolygon([result])
    elif isinstance(result, MultiPolygon):
        result = MultiPolygon(
            [p for p in result.geoms if isinstance(p, Polygon) and
             not p.is_empty])
    elif hasattr(result, "geoms"):
        polys = [g for g in result.geoms if isinstance(g, Polygon) and
                 g.is_valid and not g.is_empty]
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
            log.info("Boundary fixed with buffer(0).")
            return buffered

    log.warning("Failed to obtain a valid MultiPolygon boundary.")
    return None
