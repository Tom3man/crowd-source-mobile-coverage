import pandas as pd
from shapely.geometry import MultiPoint, Point
from shapely.geometry.base import BaseGeometry

from mobile_coverage.logging import configure_logger

log = configure_logger("cell_coverage.boundaries.convex")


def generate_convex_hull_geom(df, quantile: float = 0.95) -> BaseGeometry:
    """
    Generate a valid convex hull MultiPolygon from input DataFrame.
    Converts input longitude/latitude columns to float type if they
    are not already.

    Args:
        df (pd.DataFrame): DataFrame with 'longitude' and 'latitude' columns.
        **args: Currently unused, present for API consistency.
    Returns:
        BaseGeometry:
            Shapely geometry object representing the convex hull,
            or None if unsuccessful.
    """

    # Construct a convex hull with shapely using train_df
    points_train = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    # Find center of mass among these points
    multipoint = MultiPoint(points_train)
    center_of_mass = multipoint.centroid

    # For each point, calculate distance to center of mass
    distances = [point.distance(center_of_mass) for point in points_train]

    # Find the 95% percentile distance
    threshold_distance = pd.Series(distances).quantile(quantile)

    # Filter points to only those within the threshold distance
    filtered_points = [
        point for point, distance in zip(points_train, distances) if
        distance <= threshold_distance]

    # Create new multipoint from filtered points
    multipoint_filtered = MultiPoint(filtered_points)

    # Calculate the convex hull
    convex_hull = multipoint_filtered.convex_hull

    if convex_hull.is_valid:
        return convex_hull
    else:
        log.warning("Convex hull is invalid.")
        return None
