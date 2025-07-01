"""
functions.py

This file contains utility functions for geometric operations, such as calculating distances,
    creating boundary anchors, and extracting coordinates from points.
"""

from numpy import cos, sin, pi

from shapely import Point, LineString, Polygon
from shapely import distance, line_locate_point

from .anchors import Anchor
from .utils import get_normals_along_line


def get_distance(p1: tuple | Point | Anchor, p2: tuple | Point | Anchor) -> float:
    """ 
    Returns the Euclidean distance between two points.

    Args:
    ----
        p1 (tuple | Point | Anchor): The coordinates of the first point.
        p2 (tuple | Point | Anchor): The coordinates of the second point.

    Returns:
    -------
        float: The Euclidean distance between the two points.

    Example:
    -------
        >>> p1 = (0, 0)
        >>> p2 = (3, 4)
        >>> distance = get_distance(p1, p2)
        >>> print(distance)
            5.0
    """
    if isinstance(p1, tuple):
        p1 = Point(p1)
    elif isinstance(p1, Anchor):
        p1 = p1.point
    if isinstance(p2, Point):
        p2 = Point(p2)
    elif isinstance(p2, Anchor):
        p2 = p2.point
    return distance(p1, p2)


def extract_coords_from_point(point_any_type: tuple | Point | Anchor) -> tuple:
    """ 
    Returns coordinates from a point of any type.

    Args:
    ----
        point_any_type: A point of type tuple, Point, or Anchor.

    Returns:
    -------
        tuple: The coordinates of the point.

    Raises:
    ------
        TypeError: If the provided point is not of type tuple, Point, or Anchor.
    """
    if isinstance(point_any_type, Anchor):
        return point_any_type.coords
    if isinstance(point_any_type, Point):
        return list(point_any_type.coords)[0]
    if isinstance(point_any_type, tuple):
        return point_any_type
    raise TypeError("only tuple, Point and Anchor types are supported")


def create_boundary_anchors(polygon: Polygon, locs_cfg: list) -> list:
    """ 
    Returns anchors on the boundary of the polygon with normal to the surface
        orientation and given offset.

    Args:
    ----
        polygon (Polygon): anchors will be located on the boundary of this polygon
        locs_cfg (list): list containing config dicts for the new anchors with the following items:
            label (str): anchor label
            loc (tuple): xy coordinates for the point of intersection with boundary line
            dir (str): direction of the anchor
                options are 'top', 'bottom', 'left', and 'right'
            offset (float): distance from the boundary of the anchor

    Returns:
    -------
        list: A list of Anchor objects, each containing the point, normal angle, and label

    Raises:
    ------
        TypeError: If the direction is not one of the allowed directions.
    """

    allowed_dirs = ["top", "bottom", "right", "left"]

    # properties
    line = polygon.boundary
    xmin, ymin, xmax, ymax = line.bounds
    cm = line.centroid
    x0, y0 = (cm.x, cm.y)

    anchors = []

    for label, loc, dir, offset in locs_cfg:
        if dir == "bottom":
            baseline = LineString([(loc, ymin - 10), (loc, y0)])
        elif dir == "top":
            baseline = LineString([(loc, y0), (loc, ymax + 10)])
        elif dir == "left":
            baseline = LineString([(xmin - 10, loc), (x0, loc)])
        elif dir == "right":
            baseline = LineString([(x0, loc), (xmax + 10, loc)])
        else:
            raise TypeError(f"incorrect {dir} / allowed directions are: {allowed_dirs}")

        pt = line.intersection(baseline)
        pt_loc = line_locate_point(line, pt, normalized=True)
        norm_angle = get_normals_along_line(line, pt_loc)
        pt = Point(pt.x + offset * cos(norm_angle * pi/180),
                   pt.y + offset * sin(norm_angle * pi/180))

        anchors.append(Anchor(pt, norm_angle, label))

    return anchors