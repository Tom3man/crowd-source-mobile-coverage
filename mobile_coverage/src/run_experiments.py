"""Train candidate boundary models per cell and log the best configs/results.

Loads pre-built convex hull bins, filters cells by point-count window, trains
multiple boundary estimators on historical months, evaluates against the most
recent month, logs metrics to MLflow, and writes winning geometries to
`data/model_results_geoms.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import geopandas as gpd
import mlflow
import pandas as pd
from shapely.geometry.base import BaseGeometry

from mobile_coverage import config
from mobile_coverage.boundaries import BOUNDARY_GENERATORS
from mobile_coverage.data.load import get_data
from mobile_coverage.evaluation.metrics import spatial_point_metrics
from mobile_coverage.geometry import sanitise_numpy_scalars
from mobile_coverage.logging import configure_logger

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


def setup_mlflow(experiment_name: str = "cell_hull_bin_eval") -> None:
    """
    Ensure MLflow logs locally under ./mlruns and point to the desired experiment.
    """
    tracking_dir = Path(__file__).resolve().parent / "mlruns"
    tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment(experiment_name)


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
) -> dict[str, dict[str, object]]:
    """
    Train/test each model spec on all but the most recent month of data.
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

    best_models = {}
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
                geom,
                df_test,
                cell_id,
                work_crs=27700,
                truth_shape="hull",
                truth_buffer_m=0.0,
                neg_ratio=1.0,
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


def log_runs_for_cell(
    area_bin: int,
    cell_row: pd.Series,
    model_results: dict[str, dict[str, object]],
) -> None:
    """
    Log one MLflow run per cell/model pair containing shared + model params.
    """
    if not model_results:
        log.warning(
            "Skipping MLflow logging for cell %s because no models ran",
            cell_row['unique_cell'],
        )
        return

    base_params = {
        "cell_id": cell_row['unique_cell'],
        "area_bin": int(area_bin),
        "n_points": int(cell_row['n_points']),
        "area": float(cell_row['area']),
    }

    for model_name, info in model_results.items():
        metrics = sanitise_numpy_scalars(info.get("metrics") or {})
        params = info.get("params") or {}

        if not metrics:
            log.info(
                "No metrics to log for %s on cell %s",
                model_name,
                cell_row['unique_cell'],
            )
            continue

        all_params = sanitise_numpy_scalars(
            {"model_name": model_name, **base_params, **params}
        )
        run_name = f"{cell_row['unique_cell']}_{model_name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(all_params)
            mlflow.log_metrics(metrics)


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


def main():
    """
    Run experiments over all cells in each area bin and log results.
    """

    setup_mlflow()

    df = get_data()
    hulls_df = pd.read_csv(config.HULLS_PATH)
    model_records: list[dict[str, object]] = []

    for area_bin, cells in iter_cells_by_area_bin(
        hulls_df=hulls_df,
        tolerance=config.POINT_TOLERANCE,
    ):
        log.info(
            "Processing bin %s with %d candidate cells", area_bin, len(cells)
        )

        for _, cell_row in cells.iterrows():
            cell_id = cell_row['unique_cell']
            log.info("Running models for cell %s (bin %s)", cell_id, area_bin)

            best_models = build_and_evaluate_models(cell_id, df, MODEL_SPECS)
            log_runs_for_cell(area_bin, cell_row, best_models)
            model_records.extend(
                _build_model_rows(area_bin, cell_row, best_models)
            )

    if model_records:
        geo_df = gpd.GeoDataFrame(model_records, geometry="geometry", crs=4326)
        geo_df.to_csv(config.MODEL_RESULTS_PATH, index=False)
        log.info(
            "Saved %d model geometry rows to %s",
            len(geo_df),
            config.MODEL_RESULTS_PATH,
        )
    else:
        log.warning(
            "No model geometries were produced; GeoDataFrame not created."
        )


if __name__ == "__main__":
    main()
