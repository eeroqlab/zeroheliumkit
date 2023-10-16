import pickle
from math import atan, pi, fmod
import numpy as np

from shapely import Polygon, MultiPolygon, LineString, Point, MultiLineString
from shapely import centroid, line_interpolate_point, line_locate_point, intersection
from shapely import ops, affinity, unary_union

from .anchors import Anchor
from .settings import GRID_SIZE
from .fonts import _glyph, _indentX, _indentY
from .errors import TopologyError


def modFMOD(angle: float | int) ->float:
    """ modified modulo calculations for angles in degrees
        lower branch always has negative sign

    Args:
        angle (float):

    Returns:
        angle: angle modulo 360
    """

    if np.abs(angle) % 360 == 180:
        return 180
    if np.abs(angle) % 360 < 180:
        return fmod(angle, 360)
    if np.sign(angle) > 0:
        return angle % 360 - 360
    return angle % 360


def merge_lines_with_tolerance(line1: LineString,
                               line2: LineString,
                               tol: float=1e-6) -> LineString:
    """Returns LineStrings formed by combining two lines. 
    
    Lines are joined together at their endpoints in case two lines are
    intersecting within a distance defined by tolerance.

    Args:
        line1 (LineString): first line
        line2 (LineString): second line
        tol (float, optional): distance within to merge. Defaults to 1e-6.

    Raises:
        ValueError: distance between all boundary points are not within tolerance

    Returns:
        LineString: merged line
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


def attach_line(base_line: LineString, line: LineString) -> None:
    """ appending line to an object """

    coords_obj_1 = np.asarray(list(base_line.coords))
    coords_obj_2 = np.asarray(list(line.coords))
    n1 = len(coords_obj_1)
    n2 = len(coords_obj_2)
    coords_obj_new = np.zeros((n1 + n2 - 1, 2), dtype=float)
    coords_obj_new[:n1] = coords_obj_1
    coords_obj_new[n1:] = coords_obj_2[1:]

    return LineString(coords_obj_new)


def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    if isinstance(point1, Point):
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    else:
        angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360


def offset_point(point: tuple | Point, offset: float, angle: float) -> Point:
    # fix this
    rotation_angle = modFMOD(angle - 90)

    if isinstance(point, Point):
        point_coord = (point.x, point.y)
    else:
        point_coord = point
    p_origin = Point(point_coord)
    p = affinity.translate(p_origin, xoff=offset, yoff=0)
    p = affinity.rotate(p, angle=rotation_angle, origin=p_origin)
    return p


def get_angle_between_points(p1: tuple | Point, p2: tuple | Point) -> float:
    if isinstance(p1, Point):
        p1 = (p1.x, p1.y)
    if isinstance(p2, Point):
        p2 = (p2.x, p2.y)
    if p2[0] - p1[0] == 0:
        if p2[1] - p1[1] > 0:
            return 90
        return 270

    angle = atan((p2[1] - p1[1])/(p2[0] - p1[0])) * 180/pi

    if p2[1] - p1[1] > 0 and p2[0] - p1[0] < 0:
        return angle + 180
    if p2[1] - p1[1] < 0 and p2[0] - p1[0] < 0:
        return angle + 180
    if p2[1] - p1[1] < 0 and p2[0] - p1[0] > 0:
        return angle
    return angle


def get_length_between_points(p1: tuple | Point, p2: tuple | Point) -> float:
    if isinstance(p1, Point):
        p1 = (p1.x, p1.y)
    if isinstance(p2, Point):
        p2 = (p2.x, p2.y)
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_abc_line(p1: tuple | Point, p2: tuple | Point) -> tuple:
    if not isinstance(p1, Point):
        p1 = Point(p1)
    if not isinstance(p2, Point):
        p2 = Point(p2)
    a = p1.y - p2.y
    b = p2.x - p1.x
    c = -a * p1.x - b * p1.y
    return a, b, c


def get_intersection_point(abc1: tuple, abc2: tuple) -> Point:
    a1, b1, c1 = abc1
    a2, b2, c2 = abc2
    denominator = (a1 * b2 - a2 * b1)
    if denominator==0:
        raise ZeroDivisionError("constructed parallel lines do not intersect, you idiot! :)")
    x = (b1 * c2 - b2 * c1)/(a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2)/(a1 * b2 - a2 * b1)
    return Point(x, y)


def get_intersection_point_bruteforce(p1: Point, p2: Point, p3: Point, p4: Point):
    if not isinstance(p1, Point):
        p1 = Point(p1)
    if not isinstance(p2, Point):
        p2 = Point(p2)
    if not isinstance(p3, Point):
        p3 = Point(p3)
    if not isinstance(p4, Point):
        p4 = Point(p4)
    intersec = intersection(LineString([p1, p2]), LineString([p3, p4]))
    #print(intersec)
    if intersec.is_empty:
        raise TopologyError("constructed lines (p1,p2) and (p3,p4) do not intersect")
    else:
        return intersec.centroid

    #denominator = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    #if denominator==0:
    #    raise ZeroDivisionError("constructed parallel lines do not intersect, you idiot! :)")
    #px = ((p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x))/denominator
    #py = ((p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x))/denominator
    #return Point(px, py)


def get_normals_along_line(line: LineString | MultiLineString,
                           locs: float | list) -> list:

    """ finds normal angles of the line at given points (locations)

    Args:
        line (LineString | MultiLineString): given line
        locs (float | list): point locations along the line, it should be normalized

    Returns:
        list: normal angles
    """

    float_indicator = isinstance(locs, (float, int))
    if isinstance(locs, list):
        locs = np.asarray(locs)
    elif isinstance(locs, (float, int)):
        locs = np.asarray([locs])

    epsilon_up = np.full(shape=len(locs), fill_value=GRID_SIZE)
    epsilon_down = np.full(shape=len(locs), fill_value=GRID_SIZE)
    if locs[0]==0:
        epsilon_down[0] = 0.0
    elif locs[-1]==1:
        epsilon_up[-1] = 0.0

    pts_up = line_interpolate_point(line, locs + epsilon_up, normalized=True).tolist()
    pts_down = line_interpolate_point(line, locs - epsilon_down, normalized=True).tolist()
    normal_angles = np.asarray(list(map(get_angle_between_points, pts_down, pts_up))) + 90

    if not float_indicator:
        return normal_angles
    return normal_angles[0]


def midpoint(p1, p2, alpha=0.5):
    return Point(p1.x + alpha * (p2.x - p1.x), p1.y + alpha * (p2.y - p1.y))


def save_geometries(geometries_dict, file_path):
    """ Saves geometry layout of the Entity/Structure in .pickle format

    Args:
        geometries_dict (Entity/Structure): geometry layout
        file_path: name and location of the file
    """

    try:
        with open(file_path, 'wb') as file:
            pickle.dump(geometries_dict, file)
        print("Geometries saved successfully.")
    except Exception as e:
        print(f"Error occurred while saving geometries: {e}")


def read_geometries(file_path):
    try:
        with open(file_path, 'rb') as file:
            geometries_dict = pickle.load(file)
        return geometries_dict
    except Exception as e:
        print(f"Error occurred while reading geometries: {e}")
        return {}


def create_list_geoms(geometry) -> list:
    if hasattr(geometry, "geoms"):
        # working with multi-geometries
        return list(geometry.geoms)
    # working with single-geometries
    return [geometry]


def has_interior(p: Polygon) -> bool:
    return False if not list(p.interiors) else True


def convert_polygon_with_holes_into_muiltipolygon(p: Polygon) -> list:
    """ Converts polygon with hole into MultiPolygon.
        From the CenterOfMass of the interior a line is constructed, 
        which cuts the polygon into MultiPolygon. Note: the cut is done vertically.

    Args:
        p (Polygon): Polygon, might contain holes

    Returns:
        list: MultiPolygon
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
                if isinstance(geom,Polygon):
                    multipolygonlist += [geom]
            multipolygon = MultiPolygon(multipolygonlist)
            disected_all += list(multipolygon.geoms)

        return multipolygon
    return multipolygon


def extract_coords_from_point(point_any_type: tuple | Point | Anchor):

    if isinstance(point_any_type, Anchor):
    # if Anchor class provided then extract coords
        return point_any_type.coords

    if isinstance(point_any_type, Point):
        # if Point class provided then extract coords
        return list(point_any_type.coords)[0]

    if isinstance(point_any_type, tuple):
        # if tuple is provided then return the same
        return point_any_type

    raise TypeError("only tuple, Point and Anchor tupes are supported")


def polygonize_text(text: str="abcdef", size: float=1000) -> MultiPolygon:
    """ Converts given text to a MultiPolygon object

    Args:
        text (str, optional): text in str format. Defaults to "abcdef".
        size (float, optional): defines the size of the text. Defaults to 1000.

    Returns:
        MultiPolygon: converted text into MultiPolygon object
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


def round_polygon(polygon: Polygon, round_radius: float) -> Polygon:
    return polygon.buffer(round_radius).buffer(-2*round_radius).buffer(round_radius)


def create_boundary_anchors(polygon: Polygon, locs_cfg: list) -> list:
    """ creates anchors on the boundary of the polygon with normal to the surface
        orientation and given offset

    Args:
        polygon (Polygon): anchors will be located on the boundary of this polygon
        locs_cfg (list): item - (label, xy coordinate, direction, offset)
            label - anchor label
            xy - depending on the direction this will create a vertical/horizontal line
                which will intersect with boundary line, and intersection point is anchor location
            direction - 'top', 'bottom', 'left', 'right'
            offset - how far from the boundary the anchor will be located

    Returns:
        list: list of anchors
    """

    allowed_dirs = ["top", "bottom", "right", "left"] # allowed locs_cfg

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
        pt = Point(pt.x + offset * np.cos(norm_angle * np.pi/180),
                   pt.y + offset * np.sin(norm_angle * np.pi/180))

        anchors.append(Anchor(pt, norm_angle, label))

    return anchors