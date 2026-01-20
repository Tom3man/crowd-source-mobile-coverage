from __future__ import annotations

import numpy as np


def sanitise_numpy_scalars(values: dict[str, object]) -> dict[str, object]:
    """
    Convert numpy scalar types to native Python types for downstream consumers.
    """
    clean_values: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            clean_values[key] = value.item()
        else:
            clean_values[key] = value
    return clean_values
