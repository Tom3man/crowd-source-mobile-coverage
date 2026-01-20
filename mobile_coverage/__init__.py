import os
from pathlib import Path

MODULE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
REPO_PATH = Path(os.path.dirname(MODULE_PATH))
DATA_PATH = REPO_PATH / "data"
