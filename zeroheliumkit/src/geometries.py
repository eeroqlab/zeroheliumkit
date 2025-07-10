""" This submodule contains most frequently used geometries """

from math import pi, tanh, cos, tan
import numpy as np

from shapely import Point, LineString, Polygon
from shapely import affinity, unary_union, box

from .core import Entity, Structure
from .anchors import Anchor
from .utils import azimuth, buffer_along_path, round_polygon
from .functions import extract_coords_from_point
from .routing import get_fillet_params, make_fillet_line, normalize_anchors
from .settings import GRID_SIZE


# -------------------------------------
# Collection of simple geometries
# -------------------------------------

def Rectangle(width: float,
              height: float,
              location: tuple | Point=None,
              direction: float=None,
              round_radius: float=None,
              **kwargs) -> Polygon:
    """ Returns a rectangle Polygon

    Args:
    ----
        - width (float): width of the rectangle
        - height (float): height of the rectangle

    Example:
    -------
        >>> Rectangle(4, 2)
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
        poly = affinity.translate(poly, *location)
    if round_radius:
        poly = round_polygon(poly, round_radius, **kwargs)
    return  poly


def Square(size: float, location: tuple | Point=None, direction: float=None, round_radius: float=None, **kwargs) -> Polygon:
    """ Returns a square Polygon

    Args:
    ----
        - size (float): size of the square

    Example:
    -------
        >>> Square(3)
    """
    return Rectangle(size, size, location, direction, round_radius, **kwargs)


def RegularPolygon(edge_size: float=None,
                   radius: float=None,
                   location: tuple | Point=None,
                   num_edges: int=6) -> Polygon:
    """ Returns a regular Polygon with specified number of edges.
        Size of the polygon defined by radius or edge size.

    Args:
    ----
        - edge (float): size of edge
        - radius (float): radius of the Regular polygon
        - num_edges (int): number of edges

    Example:
    -------
        >>> RegularPolygon(edge=4, num_edges=6, location=(0,0))
    """
    coords = []
    angle = 2 * np.pi / num_edges
    if edge_size:
        radius = edge_size/np.sin(angle/2) * 0.5
    for i in range(num_edges):
        x = radius * np.cos(i * angle)
        y = radius * np.sin(i * angle)
        coords.append((x, y))
    polygon = Polygon(coords)
    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(polygon, *location)
    return polygon


def Circle(radius: float, location: tuple | Point=None, num_edges: int=100) -> Polygon:
    """ Returns a circle shape Polygon

    Args:
    ----
        - radius (float): radius of the circle

    Example:
    ------
        >>> Circle(radius=5)
    """
    return RegularPolygon(radius=radius, location=location, num_edges=num_edges)


def CircleSegment(radius: float=1,
                  start_angle: float=0,
                  end_angle: float=90,
                  location: tuple | Point=None,
                  num_edges: int=25) -> Polygon:
    """ Returns a polygon representing a circular segment.

    Args:
    ----
        - radius (float): The radius of the circle segment.
        - start_angle (float): The starting angle of the circular segment in degrees.
        - end_angle (float): The ending angle of the circular segment in degrees.
        - location (tuple or Point, optional): The location of the circular segment. If provided,
            the circular segment will be translated to this location. Defaults to None.
        - num_edges (int, optional): The number of edges used to approximate the circular segment.
            Defaults to 25.

    Example:
        >>> segment = CircleSegment(radius=5, start_angle=45, end_angle=135)
        >>> print(segment)
        POLYGON ((3.535533905932737 3.535533905932737, 4.045084971874737 4.045084971874737, ...))
    """
    coords = [(0,0)]
    iteration_angle = (end_angle - start_angle) / (num_edges - 1)

    for i in range(num_edges):
        x = radius * np.cos(np.deg2rad(start_angle + i * iteration_angle))
        y = radius * np.sin(np.deg2rad(start_angle + i * iteration_angle))
        coords.append((x, y))
    polygon = Polygon(coords)
    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(polygon, *location)
    return polygon


def Ring(inner_radius: float, outer_radius: float, location: tuple | Point=None, num_edges: int=100) -> Polygon:
    """ Returns a ring shape Polygon

    Args:
    ----
        - inner_radius (float): inner radius of the ring
        - outer_radius (float): outer radius of the ring
        - location (tuple or Point, optional): The location of the ring. If provided, the ring will be translated to this location. Defaults to None.
        - num_edges (int, optional): The number of edges used to approximate the ring. Defaults to 100.

    Example:
    -------
        >>> Ring(3, 5)
    """
    inner = Circle(inner_radius, num_edges=num_edges)
    outer = Circle(outer_radius, num_edges=num_edges)
    ring = outer.difference(inner)
    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(ring, *location)
    return ring


def RingSector(inner_radius: float,
               outer_radius: float,
               start_angle: float,
               end_angle: float,
               location: tuple | Point=None,
               num_edges: int=100):
    """ Returns the intersection between a ring and a circular sector.

    Args:
    ----
        - inner_radius (float): The inner radius of the ring.
        - outer_radius (float): The outer radius of the ring.
        - start_angle (float): The starting angle of the sector in degrees.
        - end_angle (float): The ending angle of the sector in degrees.
        - location (tuple | Point, optional): The location of the center of the ring. Defaults to None.
        - num_edges (int, optional): The number of edges used to approximate the ring and sector. Defaults to 100.

    Example:
    -------
        >>> inner_radius = 2.0
        >>> outer_radius = 4.0
        >>> start_angle = 0.0
        >>> end_angle = math.pi / 2
        >>> location = (0, 0)
        >>> num_edges = 100
        >>> result = RingSector(inner_radius, outer_radius, start_angle, end_angle, location, num_edges)
        >>> print(result)
        [(4.0, 0.0), (3.9999999999999996, 0.040000000000000036), (3.9999999999999996, 0.08000000000000007), ...]
    """
    
    coords = []
    iteration_angle = (end_angle - start_angle) / (num_edges - 1)

    for i in range(num_edges):
        x = inner_radius * np.cos(np.deg2rad(start_angle + i * iteration_angle))
        y = inner_radius * np.sin(np.deg2rad(start_angle + i * iteration_angle))
        coords.append((x, y))
    for i in range(num_edges):
        x = outer_radius * np.cos(np.deg2rad(end_angle - i * iteration_angle))
        y = outer_radius * np.sin(np.deg2rad(end_angle - i * iteration_angle))
        coords.append((x, y))
    polygon = Polygon(coords)
    if location:
        if isinstance(location, Point):
            location = (location.x, location.y)
        return affinity.translate(polygon, *location)
    return polygon


def ArcLine(centerx: float,
            centery: float,
            radius: float,
            start_angle: float,
            end_angle: float,
            numsegments: int=10) -> LineString:
    """ Returns LineString representing an arc.

    Args:
    ----
        - centerx (float): center.x of arcline
        - centery (float): center.y of arcline
        - radius (float): radius of the arcline
        - start_angle (float): starting angle
        - end_angle (float): end angle
        - numsegments (int, optional): number of the segments. Defaults to 10.

    Example:
    -------
        >>> ArcLine(centerx=0, centery=0, radius=5, start_angle=0, end_angle=180)
    """
    theta = np.radians(np.linspace(start_angle, end_angle, num=numsegments, endpoint=True))
    x = centerx + radius * np.cos(theta)
    y = centery + radius * np.sin(theta)
    return LineString(np.column_stack([x, y]))


def Meander(length: float=100,
            radius: float=50,
            direction: float=None,
            num_segments: int=100,
            input_radius: float=None,
            output_radius: float=None,
            mirror: str=None) -> LineString:
    """ Returns a 1D full Meander line.

    Args:
    ----
        - length (float): Length of the straight section.
        - radius (float): Radius of the round section.
        - direction (float): Rotates the meander by the given value in the end.
        - num_segments (int, optional): Number of segments in the round section. Defaults to 100.

    Example:
    -------
        >>> meander = Meander(10, 2, 45, 50)
        >>> print(meander)
    """
    e = Entity()
    if input_radius:
        assert input_radius < length/2, "input_radius should be less than length/2"
        e.add_line(ArcLine(0, input_radius, input_radius, -90, 0, int(num_segments/2)))
        e.add_line(LineString([(0,0), (0,length/2 - input_radius)]))
    else:
        e.add_line(LineString([(0,0), (0,length/2)]))

    e.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))
    e.add_line(LineString([(0,0), (0,-length)]))
    e.add_line(ArcLine(radius, 0, radius, 180, 360, num_segments))

    if output_radius:
        assert output_radius < length/2, "output_radius should be less than length/2"
        e.add_line(LineString([(0,0), (0,length/2 - output_radius)]))
        e.add_line(ArcLine(output_radius, 0, output_radius, 180, 90, int(num_segments/2)))
    else:
        e.add_line(LineString([(0,0), (0,length/2)]))

    if mirror:
        e.mirror(aroundaxis=mirror, keep_original=False)
    if direction:
        e.rotate(direction, origin=(0,0))

    return e.skeletone.lines


def MeanderHalf(length: float=100,
                radius: float=50,
                direction: float=None,
                num_segments: int=100,
                input_radius: float=None,
                output_radius: float=None,
                mirror: str=None) -> LineString:
    """ Returns a 1D half Meander line.

    Args:
    ----
        - length (float): length of the straight section
        - radius (float): radius of the round section
        - direction (float): rotates the meander by given value in the end
        - num_segments (int, optional): number of segments in round section. Defaults to 100

    Example:
    -------
        >>> MeanderHalf(10, 5, 45, 50)
    """
    e = Entity()

    if input_radius:
        assert input_radius < length/2, "input_radius should be less than length/2"
        e.add_line(ArcLine(0, input_radius, input_radius, -90, 0, int(num_segments/2)))
        e.add_line(LineString([(0,0), (0,length/2 - input_radius)]))
    else:
        e.add_line(LineString([(0,0), (0,length/2)]))

    e.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))

    if output_radius:
        assert output_radius < length/2, "output_radius should be less than length/2"
        e.add_line(LineString([(0,0), (0,-length/2 + output_radius)]))
        e.add_line(ArcLine(output_radius, 0, output_radius, 180, 270, int(num_segments/2)))
    else:
        e.add_line(LineString([(0,0), (0,-length/2)]))
    
    if mirror:
        e.mirror(aroundaxis=mirror, keep_original=False)
    if direction:
        e.rotate(direction, origin=(0,0))

    return e.skeletone.lines


def PinchGate(arm_w: float,
              arm_l: float,
              length: float,
              width: float) -> Polygon:
    """ Returns a Polygon representing a pinch gate.

    Args:
    ----
        - arm_w (float): arm width
        - arm_l (float): arm length
        - length (float): length of the pinch gate
        - width (float): width of the pinch gate

    Example:
    -------
        >>> PinchGate(2, 5, 10, 3)
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


def LineExtrudedRectangle(point: tuple | Point | Anchor,
                          width: float,
                          length: float,
                          direction: float=0) -> Polygon:
    """ Returns a rectangle that is extruded from a point in specific direction.

    Args:
    ----
        - point (tuple | Point | Anchor): The starting point of the extrusion. Can be a tuple, Point object, or Anchor object.
        - width (float): The width of the rectangle.
        - length (float): The length of the rectangle.
        - direction (float): The direction of the extrusion in degrees. Default is 0.

    Example:
    -------
        >>> LineExtrudedRectangle(Anchor((10,5), 60, "a"), 2, 20)
    """
    rect = Rectangle(length, width, (length/2, 0))
    if isinstance(point, (tuple, Point)):
        rect = affinity.rotate(rect, direction, origin=(0,0))
        if isinstance(point, Point):
            point = (point.x, point.y)
        rect = affinity.translate(rect, *point)
    elif isinstance(point, Anchor):
        rect = affinity.rotate(rect, point.direction, origin=(0,0))
        rect = affinity.translate(rect, *point.coords)
    else:
        raise ValueError("point should be a tuple, Point, or Anchor object")
    return rect


def CornerCutterPolygon(radius: float=10, num_segments: int=7):
    """ Generates a polygon which is used to cut for corner rounding.

    Args:
    ----
        - radius (float): The radius of the rounded corners. Default is 10.
        - num_segments (int): The number of segments used to approximate the rounded corners. Must be greater than 2. Default is 7.
    """

    if num_segments < 2:
        raise ValueError("Number of segments must be greater than 2")
    coords = [(0,0), (0,radius)]
    start_angle = 180
    iteration_angle = 90 / (num_segments - 1)

    for i in range(num_segments - 2):
        x = radius * (1 + np.cos(np.deg2rad(start_angle + (i + 1) * iteration_angle)))
        y = radius * (1 + np.sin(np.deg2rad(start_angle + (i + 1) * iteration_angle)))
        coords.append((x, y))
    coords.append((radius, 0))
    return Polygon(coords)


def CornerRounder(corner: tuple | Point | Anchor, radius: float=10, angle: float=0, num_segments: int=7, margin: float=0.1):
    """ Creates a Polygon to Round a corner (use for cuts). Works only with 90 degree corners.

    Args:
    ----
        - corner (tuple | Point | Anchor): The corner to be rounded.
        - radius (float): The radius of the rounded corner. Default is 10.
        - angle (float): The angle of the corner. Default is 0.
        - num_segments (int): The number of segments used to approximate the rounded corner. Must be greater than 2. Default is 7.
        - margin (float): The margin to be added to the corner. Default is 0.1.
    """

    if isinstance(corner, tuple):
        corner = Point(corner)
    if isinstance(corner, Anchor):
        corner = corner.point
    width = margin * radius
    corner_polygon = CornerCutterPolygon(radius, num_segments)
    cutter = unary_union([corner_polygon,
                          box(-width, -width, radius+width, 0),
                          box(-width, -width, 0, radius+width)])
    cutter = affinity.rotate(cutter, angle, origin=(0,0))
    cutter = affinity.translate(cutter, xoff=corner.x, yoff=corner.y)
    return cutter


# -------------------------------------
# Multi-layer Geometry Classes
# -------------------------------------

class StraightLine(Structure):
    """ Represents a straight line 'polygons' in different layers.

    Args:
    ----
        - anchors (tuple, optional): Two anchors that define the start and end of the line.
        - lendir (tuple, optional): A tuple containing the length and direction (in degrees) of the line.
        - layers (dict, optional): A dictionary containing the names of the layers and their corresponding widths.
        - alabel (tuple, optional): A tuple containing labels for the start and end anchors.
        - cap_style (str, optional): The style of line ending. Valid options are 'square', 'round', or 'flat'.

    Raises:
    ------
    NameError: If the provided cap_style is not one of 'square', 'round', or 'flat'.
    AttributeError: If neither 'anchors' nor 'lendir' is provided, or if both are provided.

    Examples:
    --------
        >>> # Create a straight line with anchors
        >>> line = StraightLine(anchors=((0, 0), (10, 10)), cap_style='round')
        >>> # Create a straight line with length and direction
        >>> line = StraightLine(lendir=(5, 45), cap_style='square')
        >>> # Create a straight line with layers and anchor labels
        >>> line = StraightLine(anchors=((0, 0), (10, 10)), layers={'layer1': 1, 'layer2': 2}, alabel=('start', 'end'))
    """

    def __init__(self,
                 anchors: tuple=None,
                 lendir: tuple=None,
                 layers: dict=None,
                 alabel: tuple=None,
                 cap_style: str='square',
                 **kwargs):

        super().__init__()
        if cap_style not in ["square", "round", "flat"]:
            raise NameError("please choose line_ending from square, round or flat")

        if anchors:
            p1 = extract_coords_from_point(anchors[0])
            p2 = extract_coords_from_point(anchors[1])

        elif lendir:
            length, direction = lendir
            p1 = (0, 0)
            p2 = (length * np.cos(direction * np.pi/180),
                  length * np.sin(direction * np.pi/180))
        else:
            raise AttributeError("provide only one argument: 'anchors' or 'lendir'")

        # create skeletone
        self.skeletone.lines = LineString([p1, p2])

        # create polygons
        if layers:
            for k, width in layers.items():
                self.buffer_line(name=k, offset=width/2, cap_style=cap_style, **kwargs)

        # create anchors
        if alabel:
            angle = azimuth(p1, p2)
            self.add_anchor([Anchor(point=p1, direction=angle, label=alabel[0]),
                             Anchor(point=p2, direction=angle, label=alabel[1])])


class ArbitraryLine(Structure):
    """ Represents an arbitrary line path (list of points) and
        polygons created along this path and width values on the points.

        Args:
        ----
            - points (list): list of points along which a polygon will be constructed
            - layers (dict): layers info, where the keys are the layer names and the values are the corresponding widths
            - alabel (tuple): labels of the start and end-points.

        Example:
        -------
            >>> points = [(0, 0), (1, 1), (2, 0)]
            >>> layers = {'layer1': 0.1, 'layer2': 0.2}
            >>> alabel = ('start', 'end')
            >>> line = ArbitraryLine(points, layers, alabel)
    """
    def __init__(self,
                 points: list,
                 layers: dict=None,
                 alabel: tuple=None):

        super().__init__()

        # create skeletone
        self.skeletone.lines = LineString(points)

        # create polygons
        if layers:
            for k, width in layers.items():
                polygon = buffer_along_path(points, width)
                self.add_layer(k, polygon)

        # create anchors
        if alabel:
            input_angle = azimuth(points[0], points[1])
            output_angle = azimuth(points[-2], points[-1])
            self.add_anchor([Anchor(points[0], input_angle, alabel[0]),
                             Anchor(points[-1], output_angle, alabel[1])])


class Taper(ArbitraryLine):
    """ Represents a multilayer Taper geometry.

    Args:
    ----
        - length (float): length of the tapered section
        - layers (dict, optional): layer info. Dictionary values must be a (input width, output width) tuple. Defaults to None.
        - alabel (tuple, optional): labels of the start and end-points. Defaults to None.

    Examples:
    --------
        >>> length = 10.0
        >>> layers = {'layer1': (0.1, 0.2), 'layer2': (0.2, 0.5)}
        >>> alabel = ('start', 'end')
        >>> taper = Taper(length, layers, alabel)
    """
    def __init__(self,
                 length: float,
                 layers: dict=None,
                 alabel: tuple=None,
                 cap_style: str='square'):

        # preparing dictionary for supplying it into ArbitraryLine class
        for k, v in layers.items():
            w1 = v[0]   # input side width
            w2 = v[1]   # output side width
            if w1==w2:
                # 1. if width are the same -> adding small difference
                # in order to avoid 'division by zero' error
                # 2. later during "setattr" operation this difference will be eliminated
                # by "set_precision" (by default)
                w2 = w2 + GRID_SIZE/10
            layers[k] = np.asarray([w1, w1, w2, w2])

        pts = [(-length/2 - w1, 0),
               (-length/2, 0),
               (length/2, 0),
               (length/2 + w2, 0)
               ]
        if cap_style == "flat":
            pts = pts[1:-1]
            layers = {k: v[1:-1] for k, v in layers.items()}
        elif cap_style == "square":
            pass
        else:
            raise NameError("please choose cap_style from square or flat")
        super().__init__(pts, layers, alabel)


class Fillet(Structure):
    """ Represents a multilayer Fillet geometry.

    Args:
    ----
        - anchor (Anchor | tuple[Anchor, Anchor]): The anchor point(s) of the fillet. If a tuple is provided, it represents the start and end anchors of the fillet.
        - radius (float): The radius of the fillet.
        - num_segments (int): The number of line segments used to approximate the fillet curve.
        - layers (dict, optional): A dictionary mapping layer names to widths for creating polygons.
        - alabel (bool, optional): Adds anchors if True. Defaults to True.

    Example:
    -------
        >>> # Create a fillet from anchor A to anchor B with a radius of 10 and 8 line segments
        >>> fillet = Fillet((anchor_A, anchor_B), radius=10, layers={'layer1': 0.5, 'layer2': 0.3})
        >>> # Create a normalized fillet
        >>> fillet = Fillet(Anchor(anchor_A, radius=10, layers={'layer1': 0.5, 'layer2': 0.3})
    """

    def __init__(self,
                 anchor: Anchor | tuple[Anchor, Anchor],
                 radius: float,
                 num_segments: int=20,
                 layers: dict=None,
                 alabel: bool=True):

        super().__init__()

        # determine to make a normalized fillet (start from origin) or
        # make a fillet from the first anchor to the second anchor
        if isinstance(anchor, tuple):
            anchor_norm = normalize_anchors(anchor[0], anchor[1])
        else:
            anchor_norm = anchor

        # get fillet params 
        params = get_fillet_params(anchor_norm, radius)

        # create skeletone
        self.skeletone.lines = make_fillet_line(*params, radius, num_segments)

        # create polygons
        if layers:
            for k, width in layers.items():
                self.buffer_line(name=k,
                                 offset=width/2,
                                 cap_style='square')

        # snap to first anchor if relevant and add anchors to structure
        if isinstance(anchor, tuple):
            self.rotate(anchor[0].direction).moveby(anchor[0].coords)
            if alabel:
                self.add_anchor([anchor[0], anchor[1]])
        else:
            if alabel:
                self.add_anchor([Anchor((0,0), 0, "origin"), anchor])


class MicroChannels(Structure):
    """ Creates microchannels for eHe or can be used to create IDC.

    Args:
    ----
        - length (float): The length of the microchannels.
        - spacing (float): The spacing between each microchannel.
        - num (int): The number of microchannels.
        - angle (float): The angle of the microchannels in degrees.
        - layers (dict): A dictionary containing the names and widths of the layers.
        - alabel (tuple, optional): A tuple containing the labels for the anchors.

    Example:
    -------
        >>> # Create a MicroChannels object with length 10, spacing 1, 3 microchannels,
        >>> # angle 45 degrees, layers {'layer1': 0.5, 'layer2': 0.3}, and anchors ('A', 'B').
        >>> mc = MicroChannels(length=10, spacing=1, num=3, angle=45,
        >>>                    layers={'layer1': 0.5, 'layer2': 0.3}, alabel=('A', 'B'))
    """

    def __init__(self,
                 length: float,
                 spacing: float,
                 num: int,
                 angle: float,
                 layers: dict,
                 alabel: tuple=None):
        super().__init__()

        # create skeletone
        slope = tan(angle * pi/180)
        l = length - spacing * slope * (num - 1)
        pts = lambda i: [(0, 0),
                         (0, l + slope * spacing * i),
                         (spacing, l + slope * spacing * (i + 1)),
                         (spacing, 0)]
        for i in range(num - 1):
            self.add_line(LineString(pts(i)))
        self.add_line(LineString([(0, 0), (0, length), (spacing/2, length)]))

        # create polygon
        for k, width in layers.items():
            self.buffer_line(name=k,
                             offset=width/2,
                             cap_style='flat',
                             join_style='round',
                             quad_segs=3)

        # create anchors
        if alabel:
            first, last = self.skeletone.lines.boundary.geoms
            self.add_anchor([Anchor(point=first, direction=90, label=alabel[0]),
                             Anchor(point=last, direction=0, label=alabel[1])])


class SpiralInductor(Entity):
    """ Represents a spiral inductor.

    Args:
        - size (float): The size of the inductor.
        - width (float): The width of each turn in the spiral.
        - gap (float): The gap between each turn in the spiral.
        - num_turns (int): The number of turns in the spiral.
        - smallest_section_length (float): The length of the smallest section in the spiral.
        - layers (dict): A dictionary mapping layer names to their respective widths.
        - alabel (dict): A dictionary containing labels for the first and last anchor points.

    Example:
    -------
        >>> # Create a spiral inductor
        >>> inductor = SpiralInductor(size=10, width=1, gap=0.5, num_turns=3,
        >>>                           smallest_section_length=0.2,
        >>>                           layers={'layer1': 0.1, 'layer2': 0.2},
        >>>                           alabel=('start', 'end'))
    """

    def __init__(self,
                 size: float,
                 width: float,
                 gap: float,
                 num_turns: int=5,
                 smallest_section_length: float=0.1,
                 layers: dict={"layer1": 0.1},
                 alabel: tuple=None):
        super().__init__()

        # create skeletone
        eps = 0.1 # this removes some artifacts at the corner of the central_pad unary_union process
        radius = width/2
        self._gap = gap
        self._width = width
        coord_init = [(0, 0), (0, -size/2 - width/2),
                      (size/2 + width + gap, -size/2 - width/2)]
        self.skeletone.lines = LineString(coord_init)

        # create polygons
        self.__construct_spiral(size, radius, num_turns, smallest_section_length)
        for k, w in layers.items():
            self.buffer_line(name=k, offset=w/2, cap_style='round',
                             join_style='mitre', quad_segs=2)

        central_pad = box(-size/2, -size/2, size/2, size/2).buffer(self._width+eps, join_style=1)
        central_pad.simplify(0.2, preserve_topology=True)
        for k in layers.keys():
            united = unary_union([getattr(self, k), central_pad])
            self.layers.append(k)
            setattr(self, k, united)

        # create anchors
        if alabel:
            first, last = self.skeletone.lines.boundary.geoms
            self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]),
                            Anchor(point=last, direction=0, label=alabel[1])])

    def __num_segments(self, R: float, smallest_segment: float):
        # limits the maximum number of segments in arc
        return int(10*tanh(pi * R/2/smallest_segment/20))

    def __construct_spiral(self, size: float, radius: float, num_turns:int, ls: float):
        # create the spiral line
        self.add_line(ArcLine(0, radius, radius, 270, 360, self.__num_segments(radius, ls)))
        self.add_line(LineString([(0, 0), (0, size/2)]))
        for _ in range(num_turns):
            radius = radius + self._gap + self._width
            self.add_line(LineString([(0, 0), (0, size/2)]))
            self.add_line(ArcLine(-radius, 0, radius, 0, 90, self.__num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (-size, 0)]))
            self.add_line(ArcLine(0, -radius, radius, 90, 180, self.__num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (0, -size)]))
            self.add_line(ArcLine(radius, 0, radius, 180, 270, self.__num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (size + self._width + self._gap, 0)]))
            self.add_line(ArcLine(0, radius, radius, 270, 360, self.__num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (0, size/2)]))
        self.add_line(LineString([(0, 0), (2*self._gap + 2*self._width, 0)]))


class IDC(Entity):
    """ Represents a special IDC (symmetrical).

    Args:
    ----
        - length (float): The length of the IDC.
        - spacing (float): The spacing between each IDC.
        - num (int): The number of IDCs to create.
        - layers (dict): A dictionary mapping layer names to their widths.
        - alabel (tuple): A tuple containing two labels for the anchors.

    Example:
    -------
        >>> length = 10.0
        >>> spacing = 5.0
        >>> num = 3
        >>> layers = {'layer1': 0.5, 'layer2': 0.3}
        >>> alabel = ('start', 'end')
        >>> idc = IDC(length, spacing, num, layers, alabel)
    """
    def __init__(self,
                 length: float,
                 spacing: float,
                 num: int,
                 layers: dict,
                 alabel: tuple):
        super().__init__()

        pts = [(0, 0), (spacing/2, 0), (spacing/2,length),
               (spacing/2, -length), (spacing/2, 0), (spacing, 0)]
        self.skeletone.lines = LineString(pts)
        for _ in range(num):
            self.add_line(LineString(pts))

        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square',
                             join_style='round', quad_segs=2)

        # create anchors
        if alabel:
            first, last = self.skeletone.lines.boundary.geoms
            self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]),
                            Anchor(point=last, direction=0, label=alabel[1])])
