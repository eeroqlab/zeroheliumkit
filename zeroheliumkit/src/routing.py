"""
routing.py

This file contains a collection of functions for routing in a network.
Provides methods for creating fillet routing lines and Bezier curves between two anchor points.
"""

import numpy as np
from scipy.interpolate import BPoly
from numpy import deg2rad
from shapely import Point, LineString, affinity

from .utils import append_line, fmodnew
from .anchors import Anchor
from .functions import get_distance
from .errors import RouteError


def ArcLine(centerx: float,
            centery: float,
            radius: float,
            start_angle: float,
            end_angle: float,
            numsegments: int=10) -> LineString:
    """ 
    Returns LineString representing an arc.

    Args:
    ----
        centerx (float): center.x of arcline
        centery (float): center.y of arcline
        radius (float): radius of the arcline
        start_angle (float): starting angle
        end_angle (float): end angle
        numsegments (int, optional): number of the segments. Defaults to 10.

    Returns:
    -------
        LineString: A LineString representing the arc.

    Example:
    -------
        >>> ArcLine(centerx=0, centery=0, radius=5, start_angle=0, end_angle=180)
    """
    theta = np.radians(np.linspace(start_angle, end_angle, num=numsegments, endpoint=True))
    x = centerx + radius * np.cos(theta)
    y = centery + radius * np.sin(theta)
    return LineString(np.column_stack([x, y]))


def get_fillet_params(anchor: Anchor, radius: float) -> tuple:
    """ 
    Calculate the lengths required for fillet routing and return fillet parameters.
        The origin of fillet line is at (0,0).

    Args:
    ----
        anchor (Anchor): The anchor (second) point for the fillet.
        radius (float): The radius of the fillet curve.

    Returns:
    -------
        tuple: A tuple containing the calculated lengths and the anchor direction.
    """
    rad = anchor.direction * np.pi/180

    # calculating lengths
    if np.cos(rad)==1:
        length2 = anchor.y - radius
    else:
        length2 = (np.sign(rad)
                    * (anchor.y - np.sign(rad) * radius * (1 - np.cos(rad)))
                    / np.sqrt(1 - np.cos(rad)**2))

    length1 = anchor.x - 1 * length2 * np.cos(rad) - np.sign(rad) * radius * np.sin(rad)

    return length1, length2, anchor.direction


def make_fillet_line(length1: float,
                     length2: float,
                     direction: float,
                     radius: float,
                     num_segments: int) -> LineString:
    """ 
    Returns a fillet shape LineString.

    Args:
    ----
        length1 (float): length of the first section
        length2 (float): length of the second section placed after arcline
        direction (float): orientation of the endsegment
        radius (float): radius if the arcline
        num_segments (int): num of segments in the arcline

    Returns:
    -------
        LineString: A LineString in a fillet shape.
    """

    direction = fmodnew(direction)
    rad = direction * np.pi/180

    # creating first section
    pts = [(0, 0), (length1, 0)]
    skeletone = LineString(pts)

    # creating arcline section
    if direction > 0:
        x0, y0, phi0 = (0, radius, 270)
    else:
        x0, y0, phi0 = (0, -radius, 90)
    skeletone = append_line(skeletone, ArcLine(x0, y0, radius, phi0, phi0 + direction, num_segments))

    # creating second straight line section
    skeletone = append_line(skeletone, LineString([(0, 0), (length2 * np.cos(rad), length2 * np.sin(rad))]))

    return skeletone


def normalize_anchors(anchor1: Anchor, anchor2: Anchor) -> Anchor:
    """ 
    Returns normalizied anchor: 
        anchor1 moves to (0,0) and rotates to dir=0 and anchor2 transforms accordingly.

    Args:
    ----
        anchor1 (Anchor): first anchor
        anchor2 (Anchor): second anchor

    Returns:
    -------
        Anchor: Normalized anchor with coordinates (0,0) and direction 0.
    """

    direction = anchor2.direction - anchor1.direction
    point = affinity.rotate(Point((anchor2.x - anchor1.x, anchor2.y - anchor1.y)),
                            angle=-anchor1.direction,
                            origin=(0,0))
    norm_anchor = Anchor(point, direction)

    return norm_anchor


def snap_line_to_anchor(anchor: Anchor, line: LineString) -> LineString:
    """ 
    Snaps the linestring to the anchor position and direction

    Args:
    ----
        anchor (Anchor): line start point will be snapped to this anchor
        line (LineString): linestring to be snapped

    Returns:
    -------
        LineString: A linestring snapped to the anchor position and direction.
    """

    line = affinity.rotate(line, anchor.direction, origin=(0,0))
    line = affinity.translate(line, xoff=anchor.x, yoff=anchor.y)

    return line


def create_route(a1: Anchor,
                 a2: Anchor,
                 radius: float,
                 num_segments: int=10,
                 print_status: bool=False,
                 bezier_cfg: dict={"extension_fraction": 0.1, "depth_factor": 3}) -> LineString:
    """ 
    Returns a fillet routing linestring

    Args:
    ----
        anchor (Anchor): _description_
        radius (float): _description_
        num_segments (int, optional): _description_. Defaults to 10.
        print_status (bool, optional): If True, prints the status of the route construction. 
            Defaults to False.
        bezier_cfg (dict, optional): Configuration for Bezier curve construction. 
            Defaults to {"extension_fraction": 0.1, "depth_factor": 3}.

    Returns:
    -------
        LineString: A linestring representing the route between two anchors.

    Raises:
    ------
        RouteError: If the route cannot be constructed between the two anchors.
    """

    # normalizing anchors: a1 moves to (0,0) and rotates to dir=0 and a2 transforms accordingly
    anchor_norm = normalize_anchors(a1, a2)
    length1, length2, direction = get_fillet_params(anchor_norm, radius)
    rad = direction * np.pi/180

    # constructing route depending on the params
    if length2 < 0 and anchor_norm.y == 0 and direction == 0:
        # straight line
        route_line = LineString([(0,0), (anchor_norm.x, 0)])
        if print_status:
            print(f"route between {a1.label} and {a2.label}: straight line")
        return snap_line_to_anchor(a1, route_line)
    
    elif length2 < 0 and anchor_norm.y == 0 and direction == 180:
        raise RouteError(f"Cannot construct route between {a1.label} and {a2.label}")

    elif (length1 > 0 and length2 > 0) and np.sign(anchor_norm.y)==np.sign(np.sin(rad)):
        # valid fillet
        route_line = make_fillet_line(length1,
                                      length2,
                                      direction,
                                      radius,
                                      num_segments)

        # correction of the last point -> adjusting to anchor point
        route_line = LineString(list(route_line.coords[:-1]) + [anchor_norm.coords])
        if print_status:
            print(f"route between {a1.label} and {a2.label}: fillet curve")
        return snap_line_to_anchor(a1, route_line)

    elif np.sign(anchor_norm.y)==np.sign(np.sin(rad)) and length1 > 0:
        # correcting the radius of the round section to make valid fillet

        print("using smaller radius for routing")
        while length2 < 0:
            radius = radius * 0.6
            length1, length2, direction = get_fillet_params(anchor_norm, radius)
        route_line = make_fillet_line(length1,
                                      length2,
                                      direction,
                                      radius,
                                      num_segments)

        # correction of the last point -> adjusting to anchor point
        route_line = LineString(list(route_line.coords[:-1]) + [anchor_norm.coords])
        if print_status:
            print(f"route between {a1.label} and {a2.label}: fillet curve, mod radius is {radius}")
        return snap_line_to_anchor(a1, route_line)

    else:
        # using bezier construct for invalid fillet params

        route_line = make_bezier_line(a1,
                                      a2,
                                      num_segments,
                                      bezier_cfg["extension_fraction"],
                                      bezier_cfg["depth_factor"])
        if print_status:
            print(f"route between {a1.label} and {a2.label}: bezier curve")
        return route_line


def make_bezier_line(a1: Anchor,
                     a2: Anchor,
                     num_segments: int=20,
                     extension_fraction: float=0.1,
                     depth_factor: float=3) -> LineString:
    """ 
    Returns a Bezier routing LineString.

    Args:
    ----
        a1 (Anchor): The first anchor.
        a2 (Anchor): The second anchor.
        num_segments (int, optional): The number of segments in the arcline. Defaults to 10.
        extension_fraction (float, optional): The fraction of the length between a1 and a2 excluded from construction of bezier curve. 
            Defaults to 0.1. Choose betweer 0 and 1.
        depth_factor (float, optional): 
            The factor to control the depth of the bezier curve. Defaults to 3. Choose between (2, 4).

    Returns:
    -------
        LineString: A Bezier curve LineString between two anchors.
    """
    
    d = get_distance(a1, a2) * extension_fraction
    dx1, dy1 = d * np.cos(deg2rad(a1.direction)), d * np.sin(deg2rad(a1.direction))
    dx2, dy2 = d * np.cos(deg2rad(a2.direction)), d * np.sin(deg2rad(a2.direction))

    x = [a1.x + dx1, a1.x + depth_factor * dx1, a2.x - depth_factor * dx2, a2.x - dx2]
    y = [a1.y + dy1, a1.y + depth_factor * dy1, a2.y - depth_factor * dy2, a2.y - dy2]
    # nodes = np.asfortranarray([x, y])
    # curve = bezier.Curve(nodes, degree=len(x)-1)
    # pts_bezier = curve.evaluate_multi(np.linspace(0.0, 1.0, num_segments))
    nodes = np.asarray(list(zip(x, y)))
    curve = BPoly(nodes[:, np.newaxis, :], [0, 1])
    pts_bezier = curve(np.linspace(0, 1, num_segments)).T
    line = LineString([(a1.x, a1.y)] + list(zip(*pts_bezier)) + [(a2.x, a2.y)])

    return line
