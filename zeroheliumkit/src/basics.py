"""
Collection of different simple 1D/2D Geometries
"""

import numpy as np

from shapely import Point, LineString, Polygon, affinity

from .core import Entity


def Rectangle(width: float,
              height: float,
              location: tuple | Point=None,
              direction: float=None) -> Polygon:
    """ creates a Rectangle polygon

    Args:
        width (float)
        height (float)

    Returns:
        Polygon
    """
    poly = Polygon([(-width/2, -height/2),
                    (-width/2, height/2),
                    (width/2, height/2),
                    (width/2, -height/2)])

    if direction:
        poly = affinity.rotate(poly, direction, origin=(0,0))

    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(poly, *location)

    return  poly


def Square(size: float, location: tuple | Point=None, direction: float=None) -> Polygon:
    """ creates a Square polygon

    Args:
        size (float)

    Returns:
        Polygon
    """
    return Rectangle(size, size, location, direction)


def RegularPolygon(edge: float=None,
                   radius: float=None,
                   location: tuple | Point=None,
                   num_edges: int=6) -> Polygon:
    """ creates a Regular polygon with specified number of edges.
        Size of the polygon defined by radius or edge size.

    Args:
        edge (float): size of edge
        radius (float): radius of the Regular polygon
        num_edges (int): number of edges

    Returns:
        Polygon
    """
    coords = []
    angle = 2 * np.pi / num_edges
    if edge:
        radius = edge/np.sin(angle/2) * 0.5
    for i in range(num_edges):
        xcoor = radius * np.cos(i * angle)
        ycoor = radius * np.sin(i * angle)
        coords.append((xcoor, ycoor))

    poly = Polygon(coords)

    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(poly, *location)

    return poly


def Circle(radius: float, location: tuple | Point=None, num_edges: int=100) -> Polygon:
    """ creates a Circle polygon

    Args:
        radius (float): radius of the circle

    Returns:
        Polygon
    """
    return RegularPolygon(radius=radius, location=location, num_edges=num_edges)


def ArcLine(centerx: float,
            centery: float,
            radius: float,
            start_angle: float,
            end_angle: float,
            numsegments=10) -> LineString:
    """creates a 1D arcline

    Args:
        centerx (float): center.x of arcline
        centery (float): center.x of arcline
        radius (float): radius of the arcline
        start_angle (float): starting angle
        end_angle (float): end angle
        numsegments (int, optional): number of the segments. Defaults to 10.

    Returns:
        LineString
    """
    theta = np.radians(np.linspace(start_angle, end_angle, num=numsegments, endpoint=True))
    xcoor = centerx + radius * np.cos(theta)
    ycoor = centery + radius * np.sin(theta)

    return LineString(np.column_stack([xcoor, ycoor]))


def Meander(length: float,
            radius: float,
            direction: float,
            num_segments: int=100) -> LineString:
    """ creates a 1D full Meander line

    Args:
        length (float): length of the straight section
        radius (float): radius of the round section
        direction (float): rotates the meander by given value in the end
        num_segments (int, optional): number of segments in round section
                                      Defaults to 100

    Returns:
        LineString
    """

    coord_init = [(0,0), (0,length/2)]
    meander_entity = Entity()
    meander_entity.add_line(LineString(coord_init))
    meander_entity.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))
    meander_entity.add_line(LineString([(0,0), (0,-length)]))
    meander_entity.add_line(ArcLine(radius, 0, radius, 180, 360, num_segments))
    meander_entity.add_line(LineString([(0,0), (0,length/2)]))
    meander_entity.rotate(direction, origin=(0,0))

    return meander_entity.skeletone


def MeanderHalf(length: float,
                radius: float,
                direction: float,
                num_segments: int=100) -> LineString:
    """ creates a 1D half Meander line

    Args:
        length (float): length of the straight section
        radius (float): radius of the round section
        direction (float): rotates the meander by given value in the end
        num_segments (int, optional): number of segments in round section
                                      Defaults to 100

    Returns:
        LineString
    """

    coord_init = [(0,0), (0,length/2)]
    meander_entity = Entity()
    meander_entity.add_line(LineString(coord_init))
    meander_entity.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))
    meander_entity.add_line(LineString([(0,0), (0,-length/2)]))
    meander_entity.rotate(direction, origin=(0,0))

    return meander_entity.skeletone


def PinchGate(arm_w: float,
              arm_l: float,
              length: float,
              width: float) -> Polygon:
    """ creates a pinch gate polygon

    Args:
        arm_w (float): arm width
        arm_l (float): arm length
        length (float): length of the pinch gate
        width (float): width of the pinch gate

    Returns:
        Polygon
    """

    pts = [(-arm_w/2, arm_w/2),
            (arm_l - length/32, arm_w/2),
            (arm_l, length/8),
            (arm_l, 7*length/16),
            (arm_l + width/3, length/2),
            (arm_l + width, length/2),
            (arm_l + width, -length/2),
            (arm_l + width/3, -length/2),
            (arm_l, -7*length/16),
            (arm_l, -length/8),
            (arm_l - length/32, -arm_w/2),
            (-arm_w/2, -arm_w/2)
            ]

    return Polygon(pts)
