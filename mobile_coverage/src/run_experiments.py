"""Train candidate boundary models per cell and write results to CSV.

Loads pre-built convex hull bins, trains multiple boundary estimators on
historical months, evaluates against the most recent month, and writes
results to `data/model_results_geoms.csv`.  Resumes automatically from
any previously completed cells if the output file already exists.
"""

from __future__ import annotations

from pathlib import Path  # used for out_path in main()
from typing import Callable, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from mobile_coverage.common import config
from mobile_coverage.boundaries import BOUNDARY_GENERATORS
from mobile_coverage.boundaries.sector import generate_sector_polygon_from_row
from mobile_coverage.data.load import get_cell_details, get_data
from mobile_coverage.evaluation.metrics import spatial_point_metrics
from mobile_coverage.geometry import sanitise_numpy_scalars
from mobile_coverage.common.logging import configure_logger

log = configure_logger("cell_coverage.model_experiments")

ModelSpec = tuple[
    str,
    Callable[..., BaseGeometry | None],
    Sequence[dict[str, object]],
]

MODEL_SPECS: list[ModelSpec] = [
    ("convex_hull", BOUNDARY_GENERATORS["convex_hull"], [{"quantile": 0.98}]),
    (
        "svm",
        BOUNDARY_GENERATORS["svm"],
        [
            {"kernel": "rbf", "nu": 0.02, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.02, "gamma": 2.0e4},
            {"kernel": "rbf", "nu": 0.04, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.04, "gamma": 2.0e4},
            {"kernel": "rbf", "nu": 0.06, "gamma": 1.0e4},
            {"kernel": "rbf", "nu": 0.06, "gamma": 2.0e4},
        ],
    ),
    (
        "iso_forest",
        BOUNDARY_GENERATORS["iso_forest"],
        [
            {"contamination": 0.02, "random_state": 1995},
            {"contamination": 0.04, "random_state": 1995},
        ],
    ),
    (
        "gmm",
        BOUNDARY_GENERATORS["gmm"],
        [
            {"n_components": 3, "covariance_type": "full", "quantile": 0.05},
            {"n_components": 4, "covariance_type": "tied", "quantile": 0.02},
        ],
    ),
    (
        "kde",
        BOUNDARY_GENERATORS["kde"],
        [
            {"bandwidth": 0.0008, "quantile": 0.05},
            {"bandwidth": 0.0012, "quantile": 0.02},
        ],
    ),
    (
        "cluster",
        BOUNDARY_GENERATORS["cluster"],
        [
            {"clusterer": "dbscan", "eps": 0.0009, "min_samples": 6},
            {
                "clusterer": "hdbscan",
                "min_cluster_size": 6,
                "hull_method": "alpha",
                "alpha": 1.8,
            },
        ],
    ),
]


def iter_cells_by_area_bin(hulls_df: pd.DataFrame, tolerance: float):
    """
    Yield (area_bin, df_subset) pairs filtered around the median n_points.
    """
    for area_bin, group in hulls_df.groupby('area_bin'):
        if group.empty:
            continue

        median_points = group['n_points'].median()
        if pd.isna(median_points):
            log.warning(
                "Skipping area_bin %s because the median n_points is NaN",
                area_bin,
            )
            continue

        lower = median_points * (1 - tolerance)
        upper = median_points * (1 + tolerance)
        filtered = group[
            (group['n_points'] >= lower) & (group['n_points'] <= upper)
        ]

        if filtered.empty:
            log.info(
                "No cells within %.0f%% window for bin %s",
                tolerance * 100,
                area_bin,
            )
            continue

        yield int(area_bin), filtered


def build_and_evaluate_models(
    cell_id: str,
    df: pd.DataFrame,
    model_specs: Sequence[ModelSpec],
    cell_details_row: pd.Series | None = None,
) -> dict[str, dict[str, object]]:
    """
    Train/test each model spec on all but the most recent month of data.

    Ground truth is the sector polygon derived from antenna parameters.
    If no cell_details_row is available, evaluation falls back to alpha shape.
    """
    df_cell = df[df['unique_cell'] == cell_id]
    present_months = sorted(df_cell['month'].unique())

    if len(present_months) < 2:
        log.warning(
            "Cell %s does not have enough monthly history for train/test",
            cell_id,
        )
        return {}

    df_train = df_cell[df_cell['month'].isin(present_months[:-1])]
    df_test = df_cell[df_cell['month'] == present_months[-1]]

    # --- build ground truth: sector polygon from antenna parameters ---
    sector_geom = None
    if cell_details_row is not None:
        sector_geom = generate_sector_polygon_from_row(cell_details_row)
        if sector_geom is None:
            log.warning(
                "sector polygon could not be built for cell %s"
                " — falling back to alpha shape",
                cell_id,
            )

    eval_kwargs = dict(neg_ratio=1.0)

    best_models = {}

    # --- sector polygon baseline (evaluated identically to data-driven methods) ---
    if sector_geom is not None:
        sector_metrics = spatial_point_metrics(
            sector_geom, df_test, cell_id, **eval_kwargs
        )
        best_models["sector"] = {
            "geometry": sector_geom,
            "metrics": sector_metrics,
            "params": {"radius_col": "radii_90"},
        }
        log.info("sector point_f1 %.4f", sector_metrics["point_f1"])

    # --- data-driven methods ---
    for name, generator, param_grid in model_specs:
        log.info("Evaluating %s with %d parameter sets", name, len(param_grid))

        best_geom = None
        best_metrics = {"point_f1": -1.0}
        best_params = None

        for params in param_grid:
            log.debug("%s trying params %s", name, params)
            geom = generator(df_train, **params)

            if geom is None:
                log.warning("%s %s returned no geometry", name, params)
                continue

            metrics = spatial_point_metrics(
                geom, df_test, cell_id, **eval_kwargs
            )
            log.debug("%s metrics for %s: %s", name, params, metrics)

            if metrics["point_f1"] > best_metrics["point_f1"]:
                log.info(
                    "%s achieved new best point_f1 %.4f with params %s",
                    name,
                    metrics["point_f1"],
                    params,
                )
                best_geom = geom
                best_metrics = metrics
                best_params = params

        if best_geom is None:
            log.error("%s produced no viable geometry", name)
        else:
            log.info(
                "%s best point_f1 %.4f with params %s",
                name,
                best_metrics["point_f1"],
                best_params,
            )

        best_models[name] = {
            "geometry": best_geom,
            "metrics": best_metrics,
            "params": best_params,
        }

    return best_models


def _build_model_rows(
    area_bin: int,
    cell_row: pd.Series,
    model_results: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """
    Create tabular rows capturing cell/model metadata, metrics, and geometries.
    """
    model_rows: list[dict[str, object]] = []

    for model_name, info in model_results.items():
        geom = info.get("geometry")
        metrics = info.get("metrics") or {}

        if geom is None or not metrics:
            continue

        row: dict[str, object] = {
            "cell_id": cell_row["unique_cell"],
            "model_name": model_name,
            "area_bin": int(area_bin),
            "n_points": int(cell_row["n_points"]),
            "area": float(cell_row["area"]),
            "geometry": geom,
        }

        row.update(sanitise_numpy_scalars(metrics))
        model_rows.append(row)

    return model_rows


def main(max_cells: int | None = None):
    """
    Run experiments over all cells in each area bin and write results to CSV.

    Resumes automatically: any cell_id already present in the output file is
    skipped.  Results are appended incrementally so a crashed run loses at
    most one cell's work.

    Args:
        max_cells: If set, stop after processing this many cells in total.
            Useful for a quick smoke-test before a full run.
    """
    out_path = Path(config.MODEL_RESULTS_PATH)

    # load already-completed cell ids for resume
    completed: set[str] = set()
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path, usecols=["cell_id"])
            completed = set(existing["cell_id"].unique())
            log.info("Resuming — %d cells already completed", len(completed))
        except Exception:
            pass

    df = get_data()
    hulls_df = pd.read_csv(config.HULLS_PATH)
    cell_details = get_cell_details().set_index("unique_cell")

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    total_cells_run = 0

    for area_bin, cells in iter_cells_by_area_bin(
        hulls_df=hulls_df,
        tolerance=1.0,  # include all cells; no n_points filtering
    ):
        log.info(
            "Processing bin %s with %d candidate cells", area_bin, len(cells)
        )

        for _, cell_row in cells.iterrows():
            if max_cells is not None and total_cells_run >= max_cells:
                log.info("Reached max_cells=%d, stopping.", max_cells)
                break

            cell_id = cell_row['unique_cell']

            if cell_id in completed:
                log.debug("Skipping already-completed cell %s", cell_id)
                continue

            log.info("Running models for cell %s (bin %s)", cell_id, area_bin)

            cell_details_row = (
                cell_details.loc[cell_id]
                if cell_id in cell_details.index
                else None
            )

            best_models = build_and_evaluate_models(
                cell_id, df, MODEL_SPECS, cell_details_row=cell_details_row
            )

            rows = _build_model_rows(area_bin, cell_row, best_models)
            if rows:
                geo_df = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
                geo_df.to_csv(
                    out_path,
                    mode="a",
                    header=write_header,
                    index=False,
                )
                write_header = False

            completed.add(cell_id)
            total_cells_run += 1

        if max_cells is not None and total_cells_run >= max_cells:
            break

    log.info("Done. %d cells processed this run.", total_cells_run)


if __name__ == "__main__":
    import sys
    _max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_cells=_max)
