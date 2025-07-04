import numpy as np
import warnings
from typing import Tuple, List
from math import fmod, sqrt

from shapely import Polygon, MultiPolygon, LineString, Point, MultiLineString
from shapely import ops, affinity, unary_union
from shapely import get_num_interior_rings
from shapely import (centroid, line_interpolate_point, intersection,
                     is_empty, crosses, remove_repeated_points)

from .errors import RouteError, TopologyError
from .settings import GRID_SIZE
from .fonts import _glyph, _indentX, _indentY


def check_point_equality(p1: tuple | Point, p2: tuple | Point) -> None:
    if isinstance(p1, Point) and isinstance(p2, Point):
        if p1.x == p2.x and p1.y == p2.y:
            raise ValueError("p1 and p2 are the same value.")
    elif p1 == p2:
        raise ValueError("p1 and p2 are the same value.")


def fmodnew(angle: float | int) -> float:
    """
    Returns a Modified modulo calculations for angles in degrees.
        The lower branch always has a negative sign.

    Args:
    -----
        angle (float | int): The angle in degrees.
    
    Returns:
    --------
        float: The angle in degrees, normalized to the range [-180, 180).

    Example:
    --------
        >>> result = fmodnew(370)
        >>> print(result)
            -350.0
    """
    if np.abs(angle) % 360 == 180:
        return 180
    if np.abs(angle) % 360 < 180:
        return fmod(angle, 360)
    if np.sign(angle) > 0:
        return angle % 360 - 360
    return angle % 360


def flatten_lines(line1: LineString, line2: LineString, bypass_alignment: bool=True) -> LineString:
    """ 
    Appends line2 to line1 and returning a new LineString object.
    The last point of line1 and the first point of line2 are assumed to be the same.

    Args:
    -----
        line1 (LineString): The first LineString object.
        line2 (LineString): The second LineString object.
        bypass_alignment (bool): If True, the function will not check that 
            the last point of line1 and the first point of line2 are the same. Defaults to False.

    Returns:
    --------
        LineString: A new LineString object formed by flattening line1 with line2.

    Raises:
    -------
        ValueError: If the last point of the first line and the first point of the second line are not the same

    Example:
    --------
        >>> line1 = LineString([(0, 0), (1, 1)])
        >>> line2 = LineString([(2, 2), (3, 3)])
        >>> result = flatten_lines(line1, line2)
        >>> print(result)
            LINESTRING (0 0, 1 1, 2 2, 3 3)
    """
    # doesn't consider line points that are EXTREMELY close to one another - ask Niyaz if there should be features in place to "bypass" this
    if not bypass_alignment:
        print(line1.coords[-1])
        print(line2.coords[0])
        if line1.coords[-1] != line2.coords[0]:
            raise ValueError("The last point of the first line and the first point of the second line are not the same.")

    if line1.is_empty:
        return line2
    elif line2.is_empty:
        return line1
    else:
        coords1 = np.asarray(list(line1.coords))
        coords2 = np.asarray(list(line2.coords))
        n1 = len(coords1)
        n2 = len(coords2)
        coords_new = np.zeros((n1 + n2 - 1, 2), dtype=float)
        coords_new[:n1] = coords1
        coords_new[n1:] = coords2[1:]
        return LineString(coords_new)


def append_line(line1: LineString,
                line2: LineString,
                direction: float=None,
                ignore_crossing=False,
                chaining=True) -> LineString:
    """
    Appends arbitrary line to another arbitrary line and Return the result.

    Args:
    -----
        line1 (LineString): The original LineString.
        line2 (LineString): The LineString to be appended.
        direction (float, optional): The angle in degrees to rotate line2 before appending. Defaults to None.
        ignore_crossing (bool, optional): Whether to ignore crossing between line1 and line2. Defaults to False.
        chaining (bool, optional): Whether to chain line2 to the end of line1 or perform a union. Defaults to True.

    Raises:
    -------
        RouteError: If the appended line crosses the skeleton and ignore_crossing is False.

    Example:
    --------
        >>> line1 = LineString([(0, 0), (1, 1), (2, 2)])
        >>> line2 = LineString([(2, 2), (3, 3), (4, 4)])
        >>> result = append_line(line1, line2, direction=45, ignore_crossing=True, chaining=False)
        >>> print(result)
            LINESTRING (0 0, 1 1, 2 2, 3 3, 4 4)
    """

    if direction:
        line2 = affinity.rotate(line2, angle=direction, origin=(0,0))

    if is_empty(line1):
        return line2
    elif is_empty(line2):
        return line1
    else:
        if chaining:
            end_point = line1.boundary.geoms[-1]
            line2 = affinity.translate(line2, xoff = end_point.x, yoff = end_point.y)
            line3 = flatten_lines(line1, line2)
        else:
            line3 = unary_union([line1, line2])

        if not ignore_crossing:
            if crosses(line1, line2):
                raise RouteError("""Appending line crosses the skeleton.
                                    If crossing is intended use 'ignore_crossing=True'""")
        return line3


def combine_lines(line1: LineString,
                  line2: LineString,
                  tol: float=1e-6) -> LineString:
    """
    Combines two LineStrings by joining them together at their endpoints
        if they are within a distance defined by the tolerance.

    Args:
    -----
        line1 (LineString): The first LineString.
        line2 (LineString): The second LineString.
        tol (float, optional): The distance within which to merge the lines. Defaults to 1e-6.

    Raises:
    -------
        ValueError: If the distance between all boundary points is not within the tolerance.

    Example:
    --------
        >>> line1 = LineString([(0, 0), (1, 1)])
        >>> line2 = LineString([(1, 1), (2, 2)])
        >>> result = merge_lines_with_tolerance(line1, line2, tol=0.5)
        >>> print(result)
            LINESTRING (0 0, 1 1, 2 2)
    """
    a1, a2 = list(line1.boundary.geoms)
    b1, b2 = list(line2.boundary.geoms)
    if a1.equals_exact(b1, tolerance=tol):
        pts = list(line2.coords).reverse() + list(line1.coords)[:1]
    elif a1.equals_exact(b2, tolerance=tol):
        pts = list(line2.coords) + list(line1.coords)[:1]
    elif a2.equals_exact(b1, tolerance=tol):
        pts = list(line1.coords) + list(line2.coords)[1:]
    elif a2.equals_exact(b2, tolerance=tol):
        pts = list(line1.coords)[:-1] + list(reversed(list(line2.coords)))
    else:
        raise ValueError(f"lines cannot be merged within tolerance {tol}")

    return LineString(pts)


def azimuth(p1: tuple | Point, p2: tuple | Point) -> float:
    """
    Returns the azimuth angle between two points (from x-axis).

    Args:
    -----
        p1 (tuple or Point): The coordinates of the first point.
        p2 (tuple or Point): The coordinates of the second point.

    Returns:
    --------
        float: The azimuth angle in degrees, measured clockwise from the positive x-axis.

    Raises:
    -------
        ValueError: If p1 and p2 are the same Point object or have the same values.

    Example:
    --------
        >>> p1 = (0, 0)
        >>> p2 = (1, -1)
        >>> result = azimuth(p1, p2)
        >>> print(result)
            -45.0
    """
    check_point_equality(p1, p2)

    if isinstance(p1, Point):
        p1 = (p1.x, p1.y)
    if isinstance(p2, Point):
        p2 = (p2.x, p2.y)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return np.degrees(angle)


def offset_point(point: tuple | Point, offset: float, angle: float) -> Point:
    """
    Offsets a point by a given distance and angle.

    Args:
    -----
        point (tuple or Point): The point to be offset. Can be a tuple (x, y) or a Point object.
        offset (float): The distance by which the point should be offset.
        angle (float): The angle (in degrees) at which the point should be offset.

    Returns:
    --------
        Point: A new Point object representing the offset point.

    Example:
    --------
        >>> result = offset_point((0, 0), 5, 45) 
            Point(3.5355339059327378, 3.5355339059327378)
    """
    if isinstance(point, tuple):
        point = Point(point)
    p = affinity.translate(point,
                           xoff=offset * np.cos(np.radians(angle)),
                           yoff=offset * np.sin(np.radians(angle)))
    return p


def get_abc_line(p1: tuple | Point, p2: tuple | Point) -> tuple:
    """ 
    Calculates the coefficients (a, b, c) of the line equation Ax + By + C = 0
        that passes through two points p1 and p2.

    Args:
    -----
        p1 (tuple | Point): The first point on the line.
        p2 (tuple | Point): The second point on the line.

    Returns:
    --------
        tuple: A tuple (a, b, c) representing the coefficients of the line equation.

    Raises:
    -------
        ValueError: If p1 and p2 are the same Point object or have the same values.

    Example:
    --------
        >>> p1 = (1, 2)
        >>> p2 = (3, 4)
        >>> result = get_abc_line(p1, p2)    
            (-2, 2, 2)
    """
    check_point_equality(p1, p2)

    if not isinstance(p1, Point):
        p1 = Point(p1)
    if not isinstance(p2, Point):
        p2 = Point(p2)
    a = p1.y - p2.y
    b = p2.x - p1.x
    c = -a * p1.x - b * p1.y
    return a, b, c


def get_intersection_point(abc1: tuple, abc2: tuple) -> Point:
    """ 
    Calculates the intersection point of two lines represented by their coefficients.

    Args:
    -----
        abc1 (tuple): Coefficients of the first line in the form (a1, b1, c1).
        abc2 (tuple): Coefficients of the second line in the form (a2, b2, c2).

    Returns:
    --------
        Point: The intersection point of the two lines.

    Raises:
    -------
        ZeroDivisionError: If the lines are parallel and do not intersect.

    Example:
    --------
        >>> abc1 = (2, 3, 4)
        >>> abc2 = (5, 6, 7)
        >>> result = get_intersection_point(abc1, abc2)
            Point(1.0, 2.0)
    """
    a1, b1, c1 = abc1
    a2, b2, c2 = abc2
    denominator = (a1 * b2 - a2 * b1)
    if denominator == 0:
        raise ZeroDivisionError("Constructed parallel lines do not intersect.")
    x = (b1 * c2 - b2 * c1) / denominator
    y = (a2 * c1 - a1 * c2) / denominator
    return Point(x, y)


def get_intersection_point_bruteforce(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    """
    Calculates the intersection point between two line segments using a brute-force approach.

    Args:
    -----
        p1 (Point): The starting point of the first line segment.
        p2 (Point): The ending point of the first line segment.
        p3 (Point): The starting point of the second line segment.
        p4 (Point): The ending point of the second line segment.

    Returns:
    --------
        Point: The intersection point of the two line segments.

    Raises:
    -------
        TopologyError: If the constructed lines (p1,p2) and (p3,p4) do not intersect.

    Example:
    --------
        >>> p1 = Point(0, 0)
        >>> p2 = Point(2, 2)
        >>> p3 = Point(0, 2)
        >>> p4 = Point(2, 0)
        >>> result = get_intersection_point_bruteforce(p1, p2, p3, p4)
        >>> print(intersection_point)  
            POINT (1 1)
    """
    if not isinstance(p1, Point):
        p1 = Point(p1)
    if not isinstance(p2, Point):
        p2 = Point(p2)
    if not isinstance(p3, Point):
        p3 = Point(p3)
    if not isinstance(p4, Point):
        p4 = Point(p4)
    intersec = intersection(LineString([p1, p2]), LineString([p3, p4]))
    if intersec.is_empty:
        raise TopologyError("constructed lines (p1,p2) and (p3,p4) do not intersect")
    else:
        return intersec.centroid


def get_normals_along_line(line: LineString | MultiLineString,
                           locs: float | list) -> list:
    """ 
    Calculates normal angles of a line at desired locations.

    Args:
    -----
        line (LineString | MultiLineString): The given line.
        locs (float | list): The point locations along the line. It should be normalized.

    Returns:
    --------
        normal_angles(list): A list of normal angles at the specified locations.

    Example:
    --------
        >>> line = LineString([(0, 0), (1, 1), (2, 0)])
        >>> locs = [0.25, 0.5, 0.75]
        >>> result = get_normals_along_line(line, locs)
        >>> print(result)  
            [45.0, 45.0, 45.0]
    """
    float_indicator = isinstance(locs, (float, int))
    if isinstance(locs, list):
        locs = np.asarray(locs)
    elif float_indicator:
        locs = np.asarray([locs])

    epsilon_up = np.full(shape=len(locs), fill_value=GRID_SIZE)
    epsilon_down = np.full(shape=len(locs), fill_value=GRID_SIZE)
    if locs[0]==0:
        epsilon_down[0] = 0.0
    elif locs[-1]==1:
        epsilon_up[-1] = 0.0

    pts_up = line_interpolate_point(line, locs + epsilon_up, normalized=True).tolist()
    pts_down = line_interpolate_point(line, locs - epsilon_down, normalized=True).tolist()
    normal_angles = np.asarray(list(map(azimuth, pts_down, pts_up))) + 90

    if not float_indicator:
        return normal_angles
    return normal_angles[0]


def midpoint(p1: Point, p2: Point, alpha: float=0.5) -> Point:
    """
    Calculates the midpoint between two points.

    Args:
    -----
        p1 (Point): The first point.
        p2 (Point): The second point.
        alpha (float, optional): The weight of the 'mid' point in the calculation. Defaults to 0.5.

    Returns:
    --------
        Point: The midpoint between p1 and p2, weighted by alpha.

    Example:
    --------
        >>> p1 = Point(0, 0)
        >>> p2 = Point(2, 4)
        >>> result = midpoint(p1, p2)
        >>> print(result)
            POINT (1.0 2.0)
    """


    return Point(p1.x + alpha * (p2.x - p1.x), p1.y + alpha * (p2.y - p1.y))


def create_list_geoms(geometry: list) -> list:
    """ 
    Returns a list of geometries from a given geometry object.
    
    Args:
    -----
        geometry: A geometry object.

    Returns:
    --------
        list: A list of geometries.

    Example:
    --------
        >>> point = Point(1, 2)
        >>> result = create_list_geoms(point)
        >>> print(result)
            [POINT (1 2)]
    """
    if hasattr(geometry, "geoms"):
        # working with multi-geometries
        return list(geometry.geoms)
    # working with single-geometries
    return [geometry]


def has_interior(p: Polygon) -> bool:
    """ 
    Determines if a polygon has any interior.

    Args:
    -----
        p (Polygon): The polygon to check.

    Returns:
    --------
        bool: True if the polygon has interiors, False otherwise.

    Example:
    --------
        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], 
                              interiors=[[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]])
        >>> result = has_interior(polygon)
        >>> print(result)
            True
    """
    return False if not list(p.interiors) else True


def flatten_polygon(p: Polygon) -> MultiPolygon:
    """
    Creates a cut line along the centroid of each hole and dissects the polygon.
    1e6 is the length of the cut line. ## (is this a weird way to say it?)

    Args:
    -----
        p (Polygon): The input polygon to be flattened.

    Returns:
    --------
        MultiPolygon: A MultiPolygon object containing the dissected polygons.

    Example:
    --------
        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], 
                              interiors=[[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]])
        >>> result = flatten_polygon(polygon)
        >>> print(result)
            MULTIPOLYGON ...
    """
    YCOORD = 1e6 

    multipolygon = MultiPolygon([p])

    if has_interior(p):
        disected_all = []
        for interior in p.interiors:
            com = centroid(interior)
            cut_line = LineString([(com.x, -YCOORD), (com.x, YCOORD)])
            disected = ops.split(multipolygon, cut_line)
            multipolygonlist = []
            for geom in list(disected.geoms):
                if isinstance(geom, Polygon):
                    multipolygonlist += [geom]
            multipolygon = MultiPolygon(multipolygonlist)
            disected_all += list(multipolygon.geoms)
        return multipolygon
    return multipolygon


def flatten_multipolygon(mp: MultiPolygon) -> MultiPolygon:
    """ 
    Removes holes from a MultiPolygon object containing Polygons with holes.

    Args:
    -----
        mp (MultiPolygon): The input MultiPolygon object.

    Returns:
    -------
        MultiPolygon: A MultiPolygon object containing the polygons without holes.

    Example:
    --------
        >>> mp = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)],    
                                interiors=[[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]])]) 
        >>> result = flatten_multipolygon(mp)
        >>> print(result)
            MULTIPOLYGON ...
    """
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])
    p_list = []
    for p in mp.geoms:
        polys_with_no_holes = flatten_polygon(p)
        p_list += list(polys_with_no_holes.geoms)
    return MultiPolygon(p_list)


def polygonize_text(text: str="abcdef", size: float=1000) -> MultiPolygon:
    """ 
    Converts text to a MultiPolygon geometry.

    Args:
    -----
        text (str, optional): text in str format. Defaults to "abcdef".
        size (float, optional): defines the size of the text. Defaults to 1000.

    Returns:
    --------
        MultiPolygon: A MultiPolygon object representing the text.
    
    Raises:
    -------
        ValueError: If a character in the text does not have a corresponding geometry.

    Example:
    --------
        >>> result = polygonize_text("Hello World", size=1000)
        >>> print(result)
            MULTIPOLYGON ...
    """
    scaling = size/1000
    xoffset = 0
    yoffset = 0
    MULTIPOLY = []

    for line in text.split("\n"):

        for c in line:
            ascii_val = ord(c)

            if c==" ":
                xoffset += 500 * scaling

            elif (33 <= ascii_val <= 126) or (ascii_val == 181):
                multipolygon = []
                for poly in _glyph.get(ascii_val):
                    coords = np.array(poly) * scaling
                    coords[:, 0] += xoffset
                    coords[:, 1] += yoffset
                    multipolygon.append(Polygon(coords))

                mpolygon = unary_union(MultiPolygon(multipolygon))
                _, _, xmax, _ = mpolygon.bounds
                xoffset = xmax + _indentX * scaling
                MULTIPOLY.append(mpolygon)
            else:
                valid_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"

                raise ValueError(
                        'Warning, no geometry for character "%s" with ascii value %s. '
                        "Valid characters: %s"
                        % (chr(ascii_val), ascii_val, valid_chars)
                    )

        yoffset -= _indentY * scaling
        xoffset = 0
    
    poly = unary_union(MULTIPOLY)
    xmin, ymin, xmax, ymax = poly.bounds
    center_loc_x = xmax/2 + xmin/2
    center_loc_y = ymax/2 + ymin/2

    return affinity.translate(poly, -center_loc_x, -center_loc_y)


def get_intersection_withoffset(pts: Tuple[Point, Point, Point],
                                width: Tuple[float, float, float]) -> Point:
    """
    Given three points and offset distances, calculates the coordinate of the offset point
        from the middle point.

    Args:
    -----
        pts (Tuple[Point, Point, Point]): A tuple of three points.
        width (Tuple[float, float, float]): A tuple of three offset distances.

    Returns:
    --------
        Point: The intersection point of the offset lines.

    Example:
    --------
        >>> pts = (Point(0, 0), Point(1, 1), Point(2, 0))
        >>> width = (1, 2, 1)
        >>> result = get_centerpoint_offset(pts, width)
        >>> print(result)
            POINT (1.5 0.5)
    """
    
    angle1 = fmodnew(azimuth(pts[0], pts[1]) + 90)
    angle2 = fmodnew(azimuth(pts[1], pts[2]) + 90)
    p1 = offset_point(pts[0], width[0], angle1)
    p2 = offset_point(pts[1], width[1], angle1)
    p3 = offset_point(pts[1], width[1], angle2)
    p4 = offset_point(pts[2], width[2], angle2)

    try:
        p = get_intersection_point_bruteforce(p1, p2, p3, p4)
        #p = get_intersection_point(get_abc_line(p1, p2), get_abc_line(p3, p4))
    except TopologyError:
        p = get_intersection_point(get_abc_line(p1, p2), get_abc_line(p3, p4))
    return p


def round_polygon(polygon: Polygon, round_radius: float, **kwargs) -> Polygon:
    """
    Rounds the corners of a polygon by applying a buffer operation.

    Args:
    -----
        polygon (Polygon): The input polygon to round.
        round_radius (float): The radius of the rounding.
        **kwargs: Additional keyword arguments to pass to the buffer method.
            See [Shapely buffer docs](https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html) for additional keyword arguments.

    Returns:
    --------
        Polygon: A new polygon with rounded corners.

    Example:
    --------
        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> rounded_polygon = round_polygon(polygon, round_radius=0.1)
        >>> print(rounded_polygon)
            POLYGON ((0.1 0, 0.9 0, 1 0.1, 1 0.9, 0.9 1, 0.1 1, 0 0.9, 0 0.1, 0.1 0))
    """
    return polygon.buffer(round_radius,**kwargs).buffer(-2*round_radius,**kwargs).buffer(round_radius,**kwargs)


def buffer_along_path(points: List[tuple | Point], widths: list | float) -> Polygon:
    """ 
    Calculates a polygon (aka "buffer") along the path defined by the list of point and widths.

    Args:
    -----
        points (List[Union[tuple, Point]]): The points along which the polygon structure is constructed.
        widths (Union[list, float]): The widths of the polygon defined by a list or a single value (uniform widths).

    Returns:
    --------
        Polygon: A polygon object representing the buffer along the path.

    Raises:
    -------
        ValueError: If the number of points and widths do not match.

    Example:
    --------
        >>> points = [(0, 0), (1, 1), (2, 0)]
        >>> widths = [1, 2, 1]
        >>> result = make_polygon_along_path(points, widths)
        >>> print(result)
            POLYGON ((-0.5 0, 0.5 0, 1.5 1, 2.5 0, 1.5 -1, 0.5 -1, -0.5 0))
    """
    if isinstance(widths, (int, float)):
        widths = np.full(shape=len(points), fill_value=widths, dtype=np.float32)
    elif isinstance(widths, list):
        if len(widths) != len(points):
            raise ValueError("The number of points and widths do not match.")

    rotation_angle = fmodnew(azimuth(points[0], points[1]) + 90)
    start_p1 = offset_point(points[0], widths[0]/2, rotation_angle)
    start_p2 = offset_point(points[0], -widths[0]/2, rotation_angle)
    points1 = [start_p1]
    points2 = [start_p2]

    pts_stack = zip(points[:-2], points[1:-1], points[2:])
    wdt_stack = np.column_stack((widths[:-2], widths[1:-1], widths[2:]))

    for p, w in zip(pts_stack, wdt_stack):
        points1.append(get_intersection_withoffset(p, w/2))
        points2.append(get_intersection_withoffset(p, -w/2))

    rotation_angle = fmodnew(azimuth(points[-2], points[-1]) + 90)
    end_p1 = offset_point(points[-1], widths[-1]/2, rotation_angle)
    end_p2 = offset_point(points[-1], -widths[-1]/2, rotation_angle)
    points1.append(end_p1)
    points2.append(end_p2)

    pts = points1 + points2[::-1]

    polygon = Polygon(pts)

    if polygon.is_simple:
        return polygon
    else:
        # TODO: handle the case where the polygon is not simple
        boundary = []
        l1 = unary_union(LineString(points1))
        for l in l1.geoms:
            if not l.is_ring:
                boundary.extend(list(l.coords))
        l2 = unary_union(LineString(points2[::-1]))
        for l in l2.geoms:
            if not l.is_ring:
                boundary.extend(list(l.coords))
        return remove_repeated_points(Polygon(boundary))


def buffer_line_with_variable_width(line: LineString,
                                    distance: list,
                                    widths: list,
                                    normalized: bool,
                                    join_style: str='flat') -> Polygon:
    """ 
    Returns a buffered a line with variable widths along its length.

    Args:
    -----
        line (LineString): The input line to buffer.
        distance (list): A list of distances along the line where the widths are defined.
        widths (list): A list of widths corresponding to the distances.
        normalized (bool): Flag indicating whether the distances are normalized.
        join_style (str, optional): The style of joining the buffered polygons. 
            Valid options are 'flat' and 'round'. Defaults to 'flat'.

    Returns:
    --------
        polygon (Polygon): Buffered line with variable widths along its length.

    Raises:
    -------
        ValueError: If join_style is not either 'flat' or 'round'.

    Example:
    --------
        >>> line = LineString([(0, 0), (1, 1), (2, 0)])
        >>> distance = [0.2, 0.5, 0.8]
        >>> widths = [1, 2, 1]
        >>> normalized = True
        >>> join_style = 'flat'
        >>> polygon = buffer_line_with_variable_width(line, distance, widths, normalized, join_style)
    """
    match join_style:
        case 'flat':
            pass
        case 'round':
            pass
        case _:
            raise ValueError("Join style is not valid. Only 'flat' and 'round' are accepted.")

    points = line_interpolate_point(line, distance, normalized=normalized)
    polygon = Polygon()
    if join_style == 'round':
        for p1, p2, w1, w2 in zip(points, points[1:], widths, widths[1:]):
            base_poly = MultiPolygon([p1.buffer(w1/2, quad_segs=20), p2.buffer(w2/2, quad_segs=20)])
            poly = base_poly.convex_hull
            polygon = unary_union([polygon, poly])
    elif join_style == 'flat':
        polygon = buffer_along_path(points, widths)
            
    return polygon


def mirror(object: Polygon | LineString | MultiLineString | MultiPolygon,
           aroundaxis: str,
           origin: tuple=(0,0)) -> Polygon | LineString | MultiLineString | MultiPolygon:
    """ Returns a mirrored object along a given axis "x" or "y".

    Args:
    -----
    object (Polygon | LineString | MultiLineString | MultiPolygon): The object to be mirrored.
    axis (LineString): The axis of the mirror.
    origin (tuple, optional): The origin of the mirror. Defaults to (0, 0).

    Example:
    --------
        >>> object = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> mirrored_object = mirror(object, "x")
    """
    if aroundaxis == "x":
        return affinity.scale(object, xfact=1, yfact=-1, origin=origin)
    elif aroundaxis == "y":
        return affinity.scale(object, xfact=-1, yfact=1, origin=origin)
    else:
        raise ValueError("Invalid axis. Choose 'x' or 'y'")


def oriented_angle(p1: list[float,float], p2: list[float,float], p3: list[float,float]) -> float:
    """
    Calculate the oriented angle between vectors p1->p2 and p2->p3.

    Args:
    -----
        p1, p2, p3: NumPy arrays representing points (x, y).

    Returns:
    --------
        The oriented angle in radians.
    """
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(determinant, dot_product)
    if angle < 0:
        angle += 2 * np.pi
    return angle


def find_nearest_point_index(polygon: Polygon, point: Point) -> int:
    """
    Finds the index of the closest point in a Shapely polygon to a given point.

    Args:
    -----
        polygon (Polygon): The input Shapely polygon.
        point (Point): The Shapely point to compare.

    Returns:
    --------
        int: The index of the closest point in the polygon's exterior.
    """
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon.")
    if not isinstance(point, Point):
        raise ValueError("Input must be a Shapely Point.")
    
    # Get the exterior coordinates of the polygon
    coords = list(polygon.exterior.coords)
    
    # Calculate distances and find the index of the minimum
    distances = [point.distance(Point(coord)) for coord in coords]
    closest_index = distances.index(min(distances))
    
    return closest_index


def round_polygon_corner(polygon: Polygon, corner_index: int, radius: float=1, quad_segs: int=8) -> Polygon:
    """
    Rounds one corner of a Shapely polygon.

    Args:
    -----
        polygon (Polygon): The input Shapely polygon.
        corner_index (int): The index of the corner to round (0-based).
        radius (float): The radius of the rounded corner.
        quad_segs (int): Number of segments for the rounded corner arc.

    Returns:
    --------
        Polygon: A new polygon with the specified corner rounded.
    """
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon.")
    
    coords = list(polygon.exterior.coords)
    if corner_index < 0 or corner_index >= len(coords) - 1:
        raise ValueError("Invalid corner index.")
    
    # Get the corner point and its adjacent points
    if coords[0] == coords[-1] and corner_index == 0:
        prev_point = coords[corner_index - 2]
    else:
        prev_point = coords[corner_index - 1]
    corner_point = coords[corner_index]
    next_point = coords[(corner_index + 1) % len(coords)]

    theta = oriented_angle(prev_point, corner_point, next_point)
    if theta > np.pi:
        sign = -1
    else:
        sign = 1
    
    # rounding procedure
    line = LineString([prev_point, corner_point, next_point])
    line2 = line.offset_curve(-sign * radius, join_style=1, quad_segs=quad_segs)
    line3 = line2.offset_curve(sign * radius, join_style=1, quad_segs=quad_segs)

    # Get the arc points
    arc_points = list(line3.coords)

    # Replace the corner with the arc
    if corner_index == 0:
        new_coords = arc_points[1:-1] + coords[1:-1]
    else:
        new_coords = coords[:corner_index] + arc_points[1: -1] + coords[corner_index + 1:]
    return Polygon(new_coords)


def replace_closest_polygon(multipolygon: MultiPolygon, point: Point, new_polygon: Polygon) -> MultiPolygon:
    """
    Replaces the closest polygon in a MultiPolygon with a new polygon.

    Args:
    -----
        multipolygon (MultiPolygon): The input Shapely MultiPolygon.
        point (Point): The Shapely point to compare.
        new_polygon (Polygon): The new polygon to replace the closest one.

    Returns:
    --------
        MultiPolygon: A new MultiPolygon with the closest polygon replaced.
    """
    
    # Find the closest polygon
    closest_polygon = None
    min_distance = float('inf')
    for polygon in multipolygon.geoms:
        distance = point.distance(polygon)
        if distance < min_distance:
            min_distance = distance
            closest_polygon = polygon
    
    # Replace the closest polygon with the new polygon
    updated_polygons = [
        new_polygon if polygon == closest_polygon else polygon
        for polygon in multipolygon.geoms
    ]
    
    return MultiPolygon(updated_polygons)


def round_corner(multipolygon: MultiPolygon, around_point: Point, radius: float, **kwargs) -> MultiPolygon:
    """
    Rounds the corner of the closest polygon in a MultiPolygon around a given point with a specified radius.

    Args:
    -----
        multipolygon (MultiPolygon): The MultiPolygon object containing multiple polygons.
        around_point (Point): The point around which the corner needs to be rounded.
        radius (float): The radius of the rounded corner.
        **kwargs: Additional keyword arguments to be passed to the round_polygon_corner function.

    Returns:
    --------
        MultiPolygon: A new MultiPolygon object with the rounded corner.
    """
    if isinstance(multipolygon, Polygon):
        multipolygon = MultiPolygon([multipolygon])

    # Find the closest polygon in Multipolygon
    closest_polygon = None
    min_distance = float('inf')
    for polygon in multipolygon.geoms:
        distance = around_point.distance(polygon)
        if distance < min_distance:
            min_distance = distance
            closest_polygon = polygon
    
     # Find the closest polygon in Polugon Exterior and Interiors
    num_holes = get_num_interior_rings(closest_polygon)
    if num_holes > 0:
        closest_hole = None
        min_distance_hole = float('inf')
        id_hole = 0
        for i in range(num_holes):
            hole = closest_polygon.interiors[i]
            distance = around_point.distance(hole)
            if distance < min_distance_hole:
                min_distance_hole = distance
                closest_hole = hole
                id_hole = i
        
        # check the around_point is closer to the hole or exterior
        distance_exterior = around_point.distance(closest_polygon.exterior)
        if min_distance_hole < distance_exterior:
            polygon_to_be_rounded = Polygon(closest_hole)
            closer_to_hole = True
        else:
            polygon_to_be_rounded = closest_polygon
            closer_to_hole = False
    else:
        polygon_to_be_rounded = closest_polygon

    # Rounding of the corner happens here
    corner_point_id = find_nearest_point_index(polygon_to_be_rounded, around_point)
    rounded_polygon = round_polygon_corner(polygon_to_be_rounded, corner_index=corner_point_id, radius=radius, **kwargs)

    if num_holes > 0:
        if closer_to_hole:
            holes = list(closest_polygon.interiors)
            holes[id_hole] = rounded_polygon.exterior
            fixed_polygon = Polygon(shell=closest_polygon.exterior, holes=holes)
        else:
            fixed_polygon = Polygon(shell=rounded_polygon.exterior, holes=closest_polygon.interiors)
    else:
        fixed_polygon = rounded_polygon

    # Replace the closest polygon with the new polygon
    updated_polygons = [
        fixed_polygon if polygon == closest_polygon else polygon
        for polygon in multipolygon.geoms
    ]

    return MultiPolygon(updated_polygons)

def calculate_label_pos(x: float, y: float, centroid: Point, label_distance: float=0.5) -> tuple:
    """
    Calculates the position of the label in respect to the centroid of the polygon.

    Args:
    -----
        x (float): x coordinate of the point.
        y (float): y coordinate of the point.
        centroid (Point): Point object representing the centroid of the polygon.
        label_distance (float, optional): desired distance from the point to the label. Defaults to 0.5.

    Returns:
    --------
        tuple: Calculated (x, y) coordinates for this label.
    """
    dist_x = x - centroid.x 
    dist_y = y - centroid.y

    length = sqrt(dist_x**2 + dist_y**2)

    unit_x = dist_x / length
    unit_y = dist_y / length
    label_x = (unit_x * label_distance) + x
    label_y = (unit_y * label_distance) + y

    return (label_x, label_y)