import numpy as np
import bezier

from shapely import Point, LineString
from shapely import affinity, get_coordinates

from . import Anchor
from .functions import modFMOD, add_line, get_length_between_points
from .basics import ArcLine


def calc_fillet_params(anchor: Anchor, radius: float) -> tuple:
        """ calculates fillet parameters

        Args:
            anchor (Anchor): endpoint of the fillet
            radius (float): radius of the curved section

        Returns:
            tuple: length1, length2, direction
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


def valid_fillet(length1: float,
                 length2: float,
                 direction: float,
                 radius: float,
                 num_segments: int) -> LineString:
    """ create a fillet linestring

    Args:
        length1 (float): length of the first section
        length2 (float): length of the second section placed after arcline
        direction (float): orientation of the endsegment
        radius (float): radius if the arcline
        num_segments (int): num of segments in the arcline
    
    Returns:
        skeletone (Linestring): fillet line
    """

    direction = modFMOD(direction)
    rad = direction * np.pi/180

    # creating first section
    pts = [(0, 0), (length1, 0)]
    skeletone = LineString(pts)

    # creating arcline section
    if direction > 0:
        x0, y0, phi0 = (0, radius, 270)
    else:
        x0, y0, phi0 = (0, -radius, 90)
    skeletone = add_line(skeletone, ArcLine(x0, y0, radius, phi0, phi0 + direction, num_segments))

    # creating second straight line section
    skeletone = add_line(skeletone, LineString([(0, 0), (length2 * np.cos(rad), length2 * np.sin(rad))]))

    return skeletone


def route_fillet(anchor1: Anchor,
                 anchor2: Anchor,
                 radius: float,
                 num_segments: int=10,
                 starting_length_scale: float=10) -> LineString:
    """ creating a fillet routing linestring

    Args:
        anchor (Anchor): _description_
        radius (float): _description_
        num_segments (int, optional): _description_. Defaults to 10.

    Returns:
        LineString: _description_
    """

    direction = anchor2.direction - anchor1.direction
    point = affinity.rotate(Point((anchor2.x - anchor1.x, anchor2.y - anchor1.y)),
                            angle=-anchor1.direction,
                            origin=(0,0))
    anchor = Anchor(point, direction)
    length1, length2, direction = calc_fillet_params(anchor, radius)
    rad = direction * np.pi/180

    if (length1 > 0 and length2 > 0) and np.sign(anchor.y)==np.sign(np.sin(rad)):
        # valid fillet
        skeletone = valid_fillet(length1, length2, direction, radius, num_segments)

        # correction of the last point -> adjusting to anchor point
        skeletone = LineString(list(skeletone.coords[:-1]) + [anchor.coords])
        valid = True

    elif np.sign(anchor.y)==np.sign(np.sin(rad)) and length1 > 0:
        # correcting the radius of the round section to make valid fillet

        print("using smaller radius for routing")
        while length2 < 0:
            radius = radius * 0.6
            length1, length2, direction = calc_fillet_params(anchor, radius)
        skeletone = valid_fillet(length1, length2, direction, radius, num_segments)

        skeletone = LineString(list(skeletone.coords[:-1]) + [anchor.coords])
        valid = True

    else:
        # using bezier construct for invalid fillet params

        dist = get_length_between_points(anchor1, anchor2)/starting_length_scale
        skeletone = LineString([Point(dist,0), Point(3*dist,0),
                                Point(anchor.x - 3*dist*np.cos(rad),anchor.y - 3*dist*np.sin(rad)),
                                Point(anchor.x - dist * np.cos(rad),anchor.y - dist * np.sin(rad))])
        coords = get_coordinates(skeletone)
        nodes = np.asfortranarray([coords[:,0], coords[:,1]])
        curve = bezier.Curve(nodes, degree=len(coords)-1)
        pts_bezier = curve.evaluate_multi(np.linspace(0.0, 1.0, num_segments))
        skeletone = LineString([(0,0)] + list(zip(*pts_bezier)) + [anchor.coords])
        valid = False

    skeletone = affinity.rotate(skeletone, anchor1.direction, origin=(0,0))
    skeletone = affinity.translate(skeletone, xoff=anchor1.x, yoff=anchor1.y)

    return skeletone, valid