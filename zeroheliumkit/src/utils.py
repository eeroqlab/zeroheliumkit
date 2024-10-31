import numpy as np
from typing import Tuple, List
from math import fmod

from shapely import Polygon, MultiPolygon, LineString, Point, MultiLineString
from shapely import ops, affinity, unary_union
from shapely import (centroid, line_interpolate_point, intersection,
                     is_empty, crosses, remove_repeated_points)

from .errors import RouteError, TopologyError
from .settings import GRID_SIZE
from .fonts import _glyph, _indentX, _indentY


def fmodnew(angle: float | int) -> float:
    """ Returns a Modified modulo calculations for angles in degrees.
        The lower branch always has a negative sign.

    Args:
    -----
    angle (float | int): The angle in degrees.
    """
    if np.abs(angle) % 360 == 180:
        return 180
    if np.abs(angle) % 360 < 180:
        return fmod(angle, 360)
    if np.sign(angle) > 0:
        return angle % 360 - 360
    return angle % 360


def flatten_lines(line1: LineString, line2: LineString) -> LineString:
    """ Appending line2 to line1 and returning a new LineString object.
        The last point of line1 and the first point of line2 are assumed to be the same.

    Args:
    ----
    line1 (LineString): The first LineString object.
    line2 (LineString): The second LineString object.

    Example:
    -------
        >>> line1 = LineString([(0, 0), (1, 1)])
        >>> line2 = LineString([(2, 2), (3, 3)])
        >>> flatten_lines(line1, line2)
    """
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
    """ Appending arbitrary line to another arbitrary line and Return the result.

    Args:
    ----
    line1 (LineString): The original LineString.
    line2 (LineString): The LineString to be appended.
    direction (float, optional): The angle in degrees to rotate line2 before appending. Defaults to None.
    ignore_crossing (bool, optional): Whether to ignore crossing between line1 and line2. Defaults to False.
    chaining (bool, optional): Whether to chain line2 to the end of line1 or perform a union. Defaults to True.

    Raises:
    ------
        RouteError: If the appended line crosses the skeleton and ignore_crossing is False.

    Example:
    -------
        >>> line1 = LineString([(0, 0), (1, 1), (2, 2)])
        >>> line2 = LineString([(2, 2), (3, 3), (4, 4)])
        >>> result = append_line(line1, line2, direction=45, ignore_crossing=True, chaining=False)
        >>> print(result)  # Output: LINESTRING (0 0, 1 1, 2 2, 3 3, 4 4)
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
    """ Returns a LineString formed by merging two lines within a given tolerance.
        The function combines two LineStrings by joining them together at their endpoints
        if they are within a distance defined by the tolerance.

    Args:
    ----
    line1 (LineString): The first LineString.
    line2 (LineString): The second LineString.
    tol (float, optional): The distance within which to merge the lines. Defaults to 1e-6.

    Raises:
    ------
        ValueError: If the distance between all boundary points is not within the tolerance.

    Example:
    -------
        >>> line1 = LineString([(0, 0), (1, 1)])
        >>> line2 = LineString([(1, 1), (2, 2)])
        >>> merged_line = merge_lines_with_tolerance(line1, line2, tol=0.5)
        >>> print(merged_line)
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
    """ Returns the azimuth angle between two points (from x-axis).
        The angle is defined by fmodnew

    Args:
    ----
    p1 (tuple or Point): The coordinates of the first point.
    p2 (tuple or Point): The coordinates of the second point.

    Example:
    -------
        >>> p1 = (0, 0)
        >>> p2 = (1, -1)
        >>> angle = azimuth(p1, p2)
        >>> print(angle)  # Output: -45.0
    """
    if isinstance(p1, Point):
        p1 = (p1.x, p1.y)
    if isinstance(p2, Point):
        p2 = (p2.x, p2.y)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return np.degrees(angle)


def offset_point(point: tuple | Point, offset: float, angle: float) -> Point:
    """ Returns a point offset by a given distance and angle.

    Args:
    ----
    point (tuple or Point): The point to be offset. Can be a tuple (x, y) or a Point object.
    offset (float): The distance by which the point should be offset.
    angle (float): The angle (in degrees) at which the point should be offset.

    Example:
    -------
        >>> offset_point((0, 0), 5, 45) # Output: Point(3.5355339059327378, 3.5355339059327378)
    """
    if isinstance(point, tuple):
        point = Point(point)
    p = affinity.translate(point,
                           xoff=offset * np.cos(np.radians(angle)),
                           yoff=offset * np.sin(np.radians(angle)))
    return p


def get_abc_line(p1: tuple | Point, p2: tuple | Point) -> tuple:
    """ Returns the coefficients (a, b, c) of the line equation Ax + By + C = 0
        that passes through two points p1 and p2.

    Args:
    ----
    p1 (tuple | Point): The first point on the line.
    p2 (tuple | Point): The second point on the line.

    Example:
    -------
        >>> p1 = (1, 2)
        >>> p2 = (3, 4)
        >>> get_abc_line(p1, p2)    # Output: (-2, 2, 2)
    """
    if not isinstance(p1, Point):
        p1 = Point(p1)
    if not isinstance(p2, Point):
        p2 = Point(p2)
    a = p1.y - p2.y
    b = p2.x - p1.x
    c = -a * p1.x - b * p1.y
    return a, b, c


def get_intersection_point(abc1: tuple, abc2: tuple) -> Point:
    """ Returns the intersection point of two lines represented by their coefficients.

    Args:
    ----
    abc1 (tuple): Coefficients of the first line in the form (a1, b1, c1).
    abc2 (tuple): Coefficients of the second line in the form (a2, b2, c2).

    Raises:
    ------
    ZeroDivisionError: If the lines are parallel and do not intersect.

    Example:
    -------
        >>> abc1 = (2, 3, 4)
        >>> abc2 = (5, 6, 7)
        >>> get_intersection_point(abc1, abc2)
    """
    a1, b1, c1 = abc1
    a2, b2, c2 = abc2
    denominator = (a1 * b2 - a2 * b1)
    if denominator == 0:
        raise ZeroDivisionError("Constructed parallel lines do not intersect.")
    x = (b1 * c2 - b2 * c1) / denominator
    y = (a2 * c1 - a1 * c2) / denominator
    return Point(x, y)


def get_intersection_point_bruteforce(p1: Point, p2: Point, p3: Point, p4: Point):
    """ Returns the intersection point between two line segments using a brute-force approach.

    Args:
    ----
    p1 (Point): The starting point of the first line segment.
    p2 (Point): The ending point of the first line segment.
    p3 (Point): The starting point of the second line segment.
    p4 (Point): The ending point of the second line segment.

    Raises:
    ------
    TopologyError: If the constructed lines (p1,p2) and (p3,p4) do not intersect.

    Example:
    -------
        >>> p1 = Point(0, 0)
        >>> p2 = Point(2, 2)
        >>> p3 = Point(0, 2)
        >>> p4 = Point(2, 0)
        >>> intersection_point = get_intersection_point_bruteforce(p1, p2, p3, p4)
        >>> print(intersection_point)  # Output: POINT (1 1)
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
    """ Returns normal angles of the line at given locations.

    Args:
    ----
    line (LineString | MultiLineString): The given line.
    locs (float | list): The point locations along the line. It should be normalized.

    Example:
    -------
        >>> line = LineString([(0, 0), (1, 1), (2, 0)])
        >>> locs = [0.25, 0.5, 0.75]
        >>> normals = get_normals_along_line(line, locs)
        >>> print(normals)  # Output: [45.0, 45.0, 45.0]
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


def midpoint(p1, p2, alpha=0.5):
    """ Returns the midpoint between two points.

    Args:
    ----
    p1 (Point): The first point.
    p2 (Point): The second point.
    alpha (float, optional): The weight of the 'mid' point in the calculation. Defaults to 0.5.

    Example:
    -------
        >>> p1 = Point(0, 0)
        >>> p2 = Point(2, 4)
        >>> mid = midpoint(p1, p2)
        >>> print(mid)  # Output: POINT (1.0 2.0)
    """
    return Point(p1.x + alpha * (p2.x - p1.x), p1.y + alpha * (p2.y - p1.y))


def create_list_geoms(geometry) -> list:
    """ Returns a list of geometries from a given geometry object.
        If the geometry object has multiple geometries, it returns a list of those geometries.
        If the geometry object has a single geometry, it returns a list containing that geometry.
    
    Args:
    ----
    geometry: A geometry object.
    """
    if hasattr(geometry, "geoms"):
        # working with multi-geometries
        return list(geometry.geoms)
    # working with single-geometries
    return [geometry]


def has_interior(p: Polygon) -> bool:
    """ Returns True if a polygon has any interior.

    Args:
    ----
    p (Polygon): The polygon to check.
    """
    return False if not list(p.interiors) else True


def flatten_polygon(p: Polygon) -> MultiPolygon:
    """ Returns a MultiPolygon constructed by splitting a polygon with holes
        into a set of polygons without any holes. The function creates a cut line
        along the centroid of each hole and dissects the polygon.

    Args:
    ----
    p (Polygon): The input polygon to be flattened.
    """
    YCOORD = 1e6    # defines the length of the cut line

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
    """ Convert a MultiPolygon with polygons containing holes
        into a MultiPolygon with no holes and Returns it.

    Args:
    ----
    mp (MultiPolygon): The input MultiPolygon object.
    """
    if isinstance(mp, Polygon):
        mp = MultiPolygon([mp])
    p_list = []
    for p in mp.geoms:
        polys_with_no_holes = flatten_polygon(p)
        p_list += list(polys_with_no_holes.geoms)
    return MultiPolygon(p_list)


def polygonize_text(text: str="abcdef", size: float=1000) -> MultiPolygon:
    """ Converts given text to a MultiPolygon geometry and Returns it

    Args:
    ----
    text (str, optional): text in str format. Defaults to "abcdef".
    size (float, optional): defines the size of the text. Defaults to 1000.
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
    """ Given three points and offset distances, calculates the coordinate of the offset point
        from the middle point.

    Args:
    ----
    pts (Tuple[Point, Point, Point]): A tuple of three points.
    width (Tuple[float, float, float]): A tuple of three offset distances.

    Example:
    -------
        >>> pts = (Point(0, 0), Point(1, 1), Point(2, 0))
        >>> width = (1, 2, 1)
        >>> p = get_centerpoint_offset(pts, width)
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
    """ Rounds the corners of a polygon by applying a buffer operation.

    Args:
    ----
    polygon (Polygon): The input polygon to round.
    round_radius (float): The radius of the rounding.
    """
    return polygon.buffer(round_radius,**kwargs).buffer(-2*round_radius,**kwargs).buffer(round_radius,**kwargs)


def buffer_along_path(points: List[tuple | Point], widths: list | float) -> Polygon:
    """ Returns a polygon (aka "buffer") along the path defined by the list of point and widths.

    Args:
    ----
    points (List[Union[tuple, Point]]): The points along which the polygon structure is constructed.
    widths (Union[list, float]): The widths of the polygon defined by a list or a single value (uniform widths).

    Raises:
    ------
    ValueError: If the number of points and widths do not match.

    Example:
    -------
        >>> points = [(0, 0), (1, 1), (2, 0)]
        >>> widths = [1, 2, 1]
        >>> polygon = make_polygon_along_path(points, widths)
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
    """ Returns a buffered a line with variable widths along its length.

    Args:
    ----
    line (LineString): The input line to buffer.
    distance (list): A list of distances along the line where the widths are defined.
    widths (list): A list of widths corresponding to the distances.
    normalized (bool): Flag indicating whether the distances are normalized.
    join_style (str, optional): The style of joining the buffered polygons. 
        Valid options are 'flat' (default) and 'round'.

    Example:
    -------
        >>> line = LineString([(0, 0), (1, 1), (2, 0)])
        >>> distance = [0.2, 0.5, 0.8]
        >>> widths = [1, 2, 1]
        >>> normalized = True
        >>> join_style = 'flat'
        >>> polygon = buffer_line_with_variable_width(line, distance, widths, normalized, join_style)
    """
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
