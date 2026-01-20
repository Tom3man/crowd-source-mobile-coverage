from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from datasets import load_dataset
from dateutil import rrule

from mobile_coverage import config
from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.build_data")


def _coerce_date(value: datetime | str | None, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def get_data(
    start_date: datetime | str | None = None,
    end_date: datetime | str | None = None,
) -> pd.DataFrame:
    """
    Load the cell service dataset and filter it to well-represented cells.
    """
    start = _coerce_date(start_date, config.DEFAULT_START_DATE)
    end = _coerce_date(end_date, config.DEFAULT_END_DATE)

    all_months = [
        dt.strftime("%Y-%m-%d")
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)
    ]

    ds = load_dataset(
        "joefee/cell-service-data",
        data_files={
            "train": config.DATA_FILES,
        },
    )

    df = ds['train'].to_pandas()

    log.info(f"len of full df: {len(df)}")

    # Bin this into signal level categories
    df["signal_level_category"] = pd.cut(
        df["signal_level"],
        bins=[-np.inf, -105, -95, -82, -74, np.inf],
        labels=[
            "1. Very Weak",
            "2. Weak",
            "3. Moderate",
            "4. Strong",
            "5. Very Strong"
        ]
    )

    sufficient_data_cells = duckdb.query(f"""
    WITH monthly_count AS (
        SELECT
            unique_cell,
            date_trunc('month', CAST(timestamp AS timestamp)) as month,
            COUNT(*) as count
        FROM df
        WHERE signal_level IS NOT NULL
        GROUP BY unique_cell, month HAVING COUNT(*) >= {config.MIN_POINTS_REQUIRED}
    )
    SELECT unique_cell
    FROM monthly_count
    GROUP BY unique_cell
    HAVING COUNT(DISTINCT month) = {len(all_months)}
    """).to_df()['unique_cell'].tolist()

    sufficient_data_cells = [
        cell for cell in sufficient_data_cells if cell not in config.OUTLIER_CELLS
    ]

    df = df[df['unique_cell'].isin(sufficient_data_cells)]

    log.info(
        f"Number of cells with sufficient data: {len(sufficient_data_cells)}"
    )
    log.info(
        f"Total number of cells: {df['unique_cell'].nunique()}"
    )

    df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')

    return df
