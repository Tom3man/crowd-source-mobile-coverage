from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mobile_coverage import DATA_PATH

# Data loading windows and filtering
DEFAULT_START_DATE = datetime(2025, 3, 1)
DEFAULT_END_DATE = datetime(2025, 6, 30)
MIN_POINTS_REQUIRED = 30
OUTLIER_CELLS = {"4dc7c9ec434ed06502767136789763ec11d2c4b7"}
DATA_FILES = [
    "np_extract_part_1.csv",
    # "np_extract_part_2.csv",
    "np_extract_part_3.csv",
    "np_extract_part_4.csv",
    "np_extract_part_5.csv",
    "np_extract_part_6.csv",
    "np_extract_part_7.csv",
    "np_extract_part_8.csv",
    # "np_extract_part_9.csv",
]

# Pipeline outputs
HULLS_PATH = Path(f"{DATA_PATH}/cell_hulls.csv")
MODEL_RESULTS_PATH = Path(f"{DATA_PATH}/model_results_geoms.csv")

# Experiment tuning
POINT_TOLERANCE = 0.2  # +/- 20% window around median n_points per bin
