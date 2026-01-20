from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from sklearn import svm

from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.svm")


def generate_svm_boundary_geom(df, **args) -> BaseGeometry:
    """
    Generate a valid MultiPolygon with true cut-out holes from One-Class SVM
    boundary. Converts input longitude/latitude columns to float type if they
    are not already.

    Args:
        df (pd.DataFrame): DataFrame with 'longitude' and 'latitude' columns.
                        These columns can contain numbers or Decimal objects.
        **args: Keyword arguments passed directly to svm.OneClassSVM.

    Returns:
        BaseGeometry:
            A valid MultiPolygon geometry representing the SVM boundary,
            or None if the boundary could not be created.
    """

    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ['longitude', 'latitude']):
        log.error("Error: Input df must be a pandas DataFrame with 'longitude' and 'latitude' columns.")
        return None

    if len(df) < 2:
        log.warning("Need at least 2 data points for SVM.")
        return None

    # --- Convert coordinate columns to float type ---
    # This resolves the Decimal vs float TypeError
    try:
        df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame
        df_copy['longitude'] = df_copy['longitude'].astype(float)
        df_copy['latitude'] = df_copy['latitude'].astype(float)
        coords = df_copy[['longitude', 'latitude']].values
    except (TypeError, ValueError) as e:
        log.error(f"Error converting coordinate columns to float: {e}")
        return None

    # --- 1. Train the SVM ---
    try:
        clf = svm.OneClassSVM(**args)
        clf.fit(coords)
    except Exception as e:
        log.error(f"Error during SVM training: {e}")
        return None

    # --- 2. Create mesh grid for contouring ---
    # Now calculations will use standard floats
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 1 if x_range > 1e-9 else 0.1
    y_margin = y_range * 1 if y_range > 1e-9 else 0.1

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    resolution = 500
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    try:
        Z = clf.decision_function(grid_points).reshape(xx.shape)
    except Exception as e:
        log.error(f"Error during SVM decision function evaluation: {e}")
        return None

    # --- 3. Extract contour lines at level 0 (the boundary) ---
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xx, yy, Z, levels=[0])
    except Exception as e:
        log.error(f"Error during contour generation: {e}")
        plt.close(fig)
        return None
    plt.close(fig)

    if not cs.allsegs or not cs.allsegs[0]:
        log.warning("No contour lines found at level 0.")
        return None

    segments = cs.allsegs[0]
    lines = [LineString(seg) for seg in segments if len(seg) >= 2]

    if not lines:
        log.warning("No valid LineStrings created from contour segments.")
        return None

    # --- 4. Polygonize the lines ---
    try:
        all_polygons = list(polygonize(lines))
    except Exception as e:
        log.error(f"Error during polygonization: {e}")
        return None

    if not all_polygons:
        log.warning("Polygonisation did not yield any polygons.")
        return None

    # --- 5. Classify polygons and perform unary union ---
    positive_polygons = []
    for p in all_polygons:
        if p.is_valid and p.area > 1e-9:
            rep_point = p.representative_point()
            try:
                decision_val = clf.decision_function([[rep_point.x, rep_point.y]])[0]
                if decision_val >= 0:
                    # Ensure the polygon added is valid - unary_union can struggle with invalid inputs
                    if p.is_valid:
                        positive_polygons.append(p)
                    else:
                        # Attempt to buffer by 0 to fix potential self-intersections
                        buffered_p = p.buffer(0)
                        if buffered_p.is_valid and isinstance(buffered_p, Polygon):
                            positive_polygons.append(buffered_p)
                        else:
                            log.warning(f"Skipping invalid polygon generated during classification step even after buffer(0). Area: {p.area}")

            except Exception as e:
                log.warning(f"Error checking decision function for a polygon point: {e}")

    if not positive_polygons:
        log.warning("No valid polygons were classified as inside the SVM boundary.")
        return None

    # --- 6. Unary Union to merge positive polygons and create holes ---
    try:
        # Filter again for validity just before union, as buffer(0) might create MultiPolygons
        valid_positive_polygons = [poly for poly in positive_polygons if poly.is_valid and isinstance(poly, Polygon)]
        if not valid_positive_polygons:
            log.warning("No valid polygons remaining before unary union.")
            return None
        result_geom = unary_union(valid_positive_polygons)

    except Exception as e:
        # Catch potential errors during unary_union (often related to complex topology)
        log.error(f"Error during unary union: {e}")
        # As a fallback, try creating a MultiPolygon directly from the valid positive polygons
        # This might result in overlaps instead of proper union, but is better than nothing.
        log.warning(
            "Attempting fallback: creating MultiPolygon from individual"
            "positive polygons.")
        try:
            result_geom = MultiPolygon(valid_positive_polygons)
            if not result_geom.is_valid:
                log.warning("Fallback MultiPolygon is invalid.")
                # Try buffer(0) on the multipolygon as a last resort
                buffered_result = result_geom.buffer(0)
                if buffered_result.is_valid:
                    log.warning("Fallback MultiPolygon fixed with buffer(0).")
                    result_geom = buffered_result
                else:
                    log.error(
                        "Fallback MultiPolygon remains invalid even after"
                        "buffer(0). Cannot proceed.")
                    return None
        except Exception as fallback_e:
            log.error(
                f"Error during fallback MultiPolygon creation: {fallback_e}")
            return None

    # --- 7. Format output as MultiPolygon GeoJSON mapping ---
    final_multi_poly = None
    if result_geom is None:
        log.error("Error: Resulting geometry is None after union/fallback.")
        return None

    # Simplify handling by ensuring result_geom is always iterable
    geoms_to_wrap = []
    if isinstance(result_geom, Polygon):
        if result_geom.is_valid:
            geoms_to_wrap = [result_geom]
    elif isinstance(result_geom, MultiPolygon):
        # Filter out invalid geoms within the MultiPolygon if any
        geoms_to_wrap = [g for g in result_geom.geoms if g.is_valid and
                         isinstance(g, Polygon)]
    elif hasattr(result_geom, 'geoms'):
        log.warning(
            "unary_union resulted in a GeometryCollection. "
            "Filtering for valid Polygons.")
        geoms_to_wrap = [g for g in result_geom.geoms if g.is_valid and
                         isinstance(g, Polygon)]

    if not geoms_to_wrap:
        log.warning(
            "No valid polygons found in the final geometry after union/cleanup"
        )
        return None

    # Create the final MultiPolygon
    final_multi_poly = MultiPolygon(geoms_to_wrap)

    # Final validity check
    if final_multi_poly.is_valid:
        return final_multi_poly
    else:
        # Try one last buffer(0) fix
        log.warning("Final MultiPolygon is invalid. Attempting buffer(0) fix.")
        buffered_final = final_multi_poly.buffer(0)
        if buffered_final.is_valid and isinstance(buffered_final,
                                                  (Polygon, MultiPolygon)):
            # Re-wrap if buffer resulted in a single Polygon
            if isinstance(buffered_final, Polygon):
                final_multi_poly = MultiPolygon([buffered_final])
            else:
                final_multi_poly = buffered_final
            log.warning("Final MultiPolygon fixed with buffer(0).")
            return final_multi_poly
        else:
            log.error(
                "Final MultiPolygon remains invalid even after buffer(0).")
            return None
