from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mobile_coverage import DATA_PATH

# Data loading windows and filtering
HUGGING_FACE_DATA_PATH = "TomFreeman3/cell_service_data"
CELL_DETAILS_DATA_PATH = "TomFreeman3/cell_service_data"
CELL_DETAILS_FILE = "cell_tower_info.parquet"
DEFAULT_START_DATE = datetime(2025, 11, 1)
DEFAULT_END_DATE = datetime(2026, 2, 28)
MIN_POINTS_REQUIRED = 30
OUTLIER_CELLS: set[str] = set()
DATA_FILES = [
    "2025_11.parquet",
    "2025_12.parquet",
    "2026_01.parquet",
    "2026_02.parquet",
]

# Pipeline outputs
HULLS_PATH = Path(f"{DATA_PATH}/cell_hulls.csv")
MODEL_RESULTS_PATH = Path(f"{DATA_PATH}/model_results_geoms.csv")

# Experiment tuning
POINT_TOLERANCE = 0.2  # +/- 20% window around median n_points per bin
