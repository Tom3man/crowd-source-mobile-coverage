from mobile_coverage.boundaries.cluster import generate_cluster_hull_geom
from mobile_coverage.boundaries.convex import generate_convex_hull_geom
from mobile_coverage.boundaries.gmm import generate_gmm_boundary_geom
from mobile_coverage.boundaries.isolation_forest import (
    generate_isolation_forest_boundary_geom,
)
from mobile_coverage.boundaries.kde import generate_kde_boundary_geom
from mobile_coverage.boundaries.svm import generate_svm_boundary_geom

BOUNDARY_GENERATORS = {
    "cluster": generate_cluster_hull_geom,
    "convex_hull": generate_convex_hull_geom,
    "gmm": generate_gmm_boundary_geom,
    "iso_forest": generate_isolation_forest_boundary_geom,
    "kde": generate_kde_boundary_geom,
    "svm": generate_svm_boundary_geom,
}

__all__ = [
    "BOUNDARY_GENERATORS",
    "generate_cluster_hull_geom",
    "generate_convex_hull_geom",
    "generate_gmm_boundary_geom",
    "generate_isolation_forest_boundary_geom",
    "generate_kde_boundary_geom",
    "generate_svm_boundary_geom",
]
