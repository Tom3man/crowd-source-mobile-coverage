"""
Build per-cell convex hulls from crowd-sourced GPS data and bucket by area.

Aggregates raw point observations per `unique_cell`, fits a convex hull as a
coarse coverage shape, computes simple metadata (area, point count), assigns
each hull to a logarithmic area bin, and writes the result to
`data/cell_hulls.csv` for downstream modelling.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from mobile_coverage import config
from mobile_coverage.boundaries.convex import generate_convex_hull_geom
from mobile_coverage.data.load import get_data
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.build_cell_hulls")


def build_hulls_dataframe(
    df: pd.DataFrame,
    quantile: float = 0.99,
) -> gpd.GeoDataFrame:
    """
    Aggregate points per cell and derive convex hull geometry plus metadata.
    """

    def _hull_and_count(g: pd.DataFrame) -> pd.Series:
        geom = generate_convex_hull_geom(g, quantile=quantile)
        return pd.Series({"geometry": geom, "n_points": len(g)})

    hulls_df = (
        df.groupby("unique_cell", group_keys=False)
        .apply(_hull_and_count)
        .reset_index()
    )

    hulls_df = gpd.GeoDataFrame(hulls_df, geometry="geometry", crs="EPSG:4326")
    hulls_df = hulls_df.assign(area=hulls_df.geometry.area)
    hulls_df = hulls_df.sort_values(by="area", ascending=True)

    areas = hulls_df["area"].clip(lower=1e-12)  # avoid log(0)
    log_areas = np.log10(areas)

    hulls_df["area_bin"] = pd.cut(
        log_areas, bins=10, labels=range(1, 11)).astype(
        "Int64"
    )

    return hulls_df


def main() -> None:
    df = get_data()
    hulls_df = build_hulls_dataframe(df)

    config.HULLS_PATH.parent.mkdir(exist_ok=True)
    hulls_df.to_csv(config.HULLS_PATH, index=False)
    log.info(f"Wrote {len(hulls_df)} hull rows to {config.HULLS_PATH}")


if __name__ == "__main__":
    main()
