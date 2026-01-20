from __future__ import annotations

import numpy as np


def make_grid(
    coords: np.ndarray,
    resolution: int = 400,
    margin_factor: float = 1.0,
    min_margin: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a mesh grid around the coordinate extent with configurable margin.
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * margin_factor if x_range > min_margin else min_margin
    y_margin = y_range * margin_factor if y_range > min_margin else min_margin

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid_points
