from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample


def _truth_polygon_from_points(
    df_true_pts: pd.DataFrame,
    work_crs: int | str = 27700,
    method: str = "hull",
    buffer_m: float = 0.0,
):
    """
    Build a ground-truth polygon from positives (this cell in target levels).
    """

    if df_true_pts.empty:
        return None

    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            df_true_pts["longitude"], df_true_pts["latitude"]
        ),
        crs=4326
    ).to_crs(work_crs)

    if len(gdf) < 3:
        # not enough to form polygon
        return None

    if method == "alpha":
        # lightweight alpha-shape (falls back to hull if unstable)
        try:
            from math import dist
            mp = MultiPoint(list(gdf.geometry.values))
            tris = list(shapely.ops.triangulate(mp))
            if not tris:
                geom = mp.convex_hull
            else:
                # pick alpha via heuristic on median edge length
                edges = []
                for t in tris:
                    a, b, c = t.exterior.coords[:3]
                    edges += [dist(a, b), dist(b, c), dist(c, a)]
                med = np.median(edges) if edges else 1.0
                alpha = max(1e-9, 3.0/med)  # heuristic
                inv_a = 1.0/alpha
                keep = []
                for tri in tris:
                    a, b, c = tri.exterior.coords[:3]
                    la, lb, lc = dist(a, b), dist(b, c), dist(c, a)
                    s = 0.5*(la+lb+lc)
                    area2 = max(s*(s-la)*(s-lb)*(s-lc), 0.0)
                    if area2 == 0:
                        continue
                    R = (la*lb*lc)/(4.0*np.sqrt(area2))
                    if R <= inv_a:
                        keep.append(tri)
                geom = unary_union(keep).buffer(0) if keep else mp.convex_hull
        except Exception:
            geom = gdf.unary_union.convex_hull
    else:
        geom = gdf.unary_union.convex_hull

    if buffer_m and buffer_m != 0:
        geom = gpd.GeoSeries([geom], crs=work_crs).buffer(buffer_m).iloc[0]

    return gpd.GeoSeries([geom], crs=work_crs)


def _area_overlap_metrics(
    poly_pred_wgs84,
    truth_poly_work_gs: gpd.GeoSeries,
    work_crs: int | str = 27700,
):
    """
    Compute area precision/recall/F1 using projected areas.
    """

    if (poly_pred_wgs84 is None) or getattr(poly_pred_wgs84, "is_empty", True) or truth_poly_work_gs is None:
        return dict(area_precision=0.0, area_recall=0.0, area_f1=0.0)

    # project predicted polygon to work CRS
    pred = gpd.GeoSeries([poly_pred_wgs84], crs=4326).to_crs(work_crs).iloc[0]
    truth = truth_poly_work_gs.iloc[0]

    if pred.is_empty or truth.is_empty:
        return dict(area_precision=0.0, area_recall=0.0, area_f1=0.0)

    inter = pred.intersection(truth)
    inter_area = inter.area if not inter.is_empty else 0.0
    pred_area = max(pred.area, 1e-12)
    true_area = max(truth.area, 1e-12)

    ap = inter_area / pred_area
    ar = inter_area / true_area
    af1 = (2*ap*ar/(ap+ar)) if (ap+ar) > 0 else 0.0
    return dict(area_precision=ap, area_recall=ar, area_f1=af1)


def spatial_point_metrics(
    poly_pred_wgs84,
    df_test: pd.DataFrame,
    cell_id: str,
    work_crs: int | str = 27700,
    truth_shape: str = "hull",
    truth_buffer_m: float = 0.0,
    neg_ratio: float = 1.0,
    rng: int = 42
):
    """
    Unified evaluator: returns point- and area-based metrics + hybrid.
    - poly_pred_wgs84: shapely Polygon/MultiPolygon in EPSG:4326
    - df_test: your filtered test window (inside AOI), ALL cells for that month
    - cell_id: the cell we're evaluating
    - target_levels: which signal categories count as 'positive'
    """
    # ---------- AREA METRICS ----------
    df_true_pts = df_test[(df_test["unique_cell"] == cell_id)]

    truth_poly_gs = _truth_polygon_from_points(
        df_true_pts, work_crs=work_crs,
        method=truth_shape, buffer_m=truth_buffer_m
    )
    area_metrics = _area_overlap_metrics(
        poly_pred_wgs84, truth_poly_gs, work_crs=work_crs)

    # ---------- POINT METRICS (balanced) ----------
    if df_test.empty:
        point_metrics = dict(
            point_precision=0.0, point_recall=0.0, point_f1=0.0)
    else:
        pos = df_test[df_test["unique_cell"] == cell_id]
        neg = df_test[df_test["unique_cell"] != cell_id]

        if pos.empty:
            point_metrics = dict(
                point_precision=0.0, point_recall=0.0, point_f1=0.0)
        else:
            n_pos = len(pos)
            n_neg = min(len(neg), int(np.ceil(neg_ratio * n_pos)))
            neg_sample = resample(
                neg, replace=False, n_samples=n_neg, random_state=rng
            ) if n_neg > 0 else neg.iloc[0:0]

            eval_df = pd.concat([pos, neg_sample], ignore_index=True)

            # labels by membership
            y_true = (eval_df["unique_cell"] == cell_id).to_numpy()

            # predictions: polygon contains?
            if (poly_pred_wgs84 is None) or getattr(poly_pred_wgs84, "is_empty", True):
                y_pred = np.zeros(len(eval_df), dtype=bool)
            else:
                P = prep(poly_pred_wgs84)
                y_pred = np.array(
                    [P.contains(Point(x, y))
                     for x, y in zip(
                         eval_df["longitude"],
                         eval_df["latitude"])],
                    dtype=bool)

            pp = precision_score(y_true, y_pred, zero_division=0)
            pr = recall_score(y_true, y_pred, zero_division=0)
            pf1 = f1_score(y_true, y_pred, zero_division=0)
            point_metrics = dict(
                point_precision=pp, point_recall=pr, point_f1=pf1)

    # ---------- HYBRID ----------
    hybrid = 0.5 * (point_metrics["point_f1"] + area_metrics["area_f1"])

    out = dict(**point_metrics, **area_metrics, hybrid_f1=hybrid)
    return out
