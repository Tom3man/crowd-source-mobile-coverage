"""
Build per-cell convex hulls from crowd-sourced GPS data and bucket by area.

Aggregates raw point observations per `unique_cell`, fits a convex hull as a
coarse coverage shape, computes simple metadata (area, point count), assigns
each hull to a logarithmic area bin, and writes the result to
`data/cell_hulls.csv` for downstream modelling.
"""

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from mobile_coverage.common import config
from mobile_coverage.boundaries.convex import generate_convex_hull_geom
from mobile_coverage.data.load import get_data
from mobile_coverage.common.logging import configure_logger

log = configure_logger("cell_coverage.build_cell_hulls")


def build_hulls_dataframe(
    df: pd.DataFrame,
    quantile: float = 0.99,
) -> gpd.GeoDataFrame:
    """
    Aggregate points per cell and derive convex hull geometry plus metadata.

    Uses DuckDB to pre-aggregate point counts per cell (fast), then iterates
    over cells to build hull geometries in Python — avoiding the slow
    groupby.apply pattern on large DataFrames.
    """
    # Drop non-coordinate columns — DuckDB can't handle period[M] and
    # smaller groups make the hull iteration faster.
    coords_df = df[["unique_cell", "longitude", "latitude"]]
    cell_counts = duckdb.query("""
        SELECT unique_cell, COUNT(*) AS n_points
        FROM coords_df
        WHERE longitude IS NOT NULL AND latitude IS NOT NULL
        GROUP BY unique_cell
    """).to_df()

    log.info("Building hulls for %d cells", len(cell_counts))
    n_points_map = (
        cell_counts.set_index("unique_cell")["n_points"].to_dict()
    )

    records = []
    for i, (cell_id, cell_df) in enumerate(coords_df.groupby("unique_cell")):
        if i % 500 == 0:
            log.info("Progress: %d / %d cells", i, len(n_points_map))

        geom = generate_convex_hull_geom(cell_df, quantile=quantile)
        records.append({
            "unique_cell": cell_id,
            "geometry": geom,
            "n_points": n_points_map.get(cell_id, len(cell_df)),
        })

    hulls_df = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    hulls_df = hulls_df[hulls_df.geometry.notna()]
    hulls_df = hulls_df.assign(area=hulls_df.geometry.area)
    hulls_df = hulls_df.sort_values(by="area", ascending=True)

    areas = hulls_df["area"].clip(lower=1e-12)
    log_areas = np.log10(areas)

    hulls_df["area_bin"] = pd.cut(
        log_areas, bins=10, labels=range(1, 11)
    ).astype("Int64")

    return hulls_df


def main() -> None:
    df = get_data()
    hulls_df = build_hulls_dataframe(df)

    config.HULLS_PATH.parent.mkdir(exist_ok=True)
    hulls_df.to_csv(config.HULLS_PATH, index=False)
    log.info("Wrote %d hull rows to %s", len(hulls_df), config.HULLS_PATH)


if __name__ == "__main__":
    main()
