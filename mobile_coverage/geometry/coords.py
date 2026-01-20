from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.geometry.coords")


def prepare_coords(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Validate a DataFrame and return coordinate array in (lon, lat) order.
    """
    if not isinstance(df, pd.DataFrame) or not all(col in df.columns for col in ["longitude", "latitude"]):
        log.error("Error: Input df must be a pandas DataFrame with 'longitude' and 'latitude' columns.")
        return None

    if len(df) < 2:
        log.warning("Warning: Need at least 2 data points.")
        return None

    try:
        df_copy = df.copy()
        df_copy["longitude"] = df_copy["longitude"].astype(float)
        df_copy["latitude"] = df_copy["latitude"].astype(float)
        return df_copy[["longitude", "latitude"]].to_numpy()
    except (TypeError, ValueError) as exc:
        log.error(f"Error converting coordinate columns to float: {exc}")
        return None
