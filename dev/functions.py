import pickle
import numpy as np

from shapely import Polygon, MultiPolygon, LineString, Point, MultiLineString
from shapely import centroid, line_interpolate_point, ops, affinity
from math import atan, pi, fmod

from ..settings import GRID_SIZE


def modFMOD(angle):
    if np.abs(angle) % 360 == 180:
        return 180
    elif np.abs(angle) % 360 < 180:
        return fmod(angle, 360)
    elif np.sign(angle) > 0:
        return angle % 360 - 360
    else:
        return angle % 360


def merge_lines_with_tolerance(line1: LineString, line2: LineString, tol: float=1e-6) -> LineString:
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


def attach_line(object: LineString, line: LineString) -> None:
    """ appending line to an object """

    coords_obj_1 = np.asarray(list(object.coords))
    coords_obj_2 = np.asarray(list(line.coords))
    n1 = len(coords_obj_1)
    n2 = len(coords_obj_2)
    coords_obj_new = np.zeros((n1 + n2 - 1, 2), dtype=float)
    coords_obj_new[:n1] = coords_obj_1
    coords_obj_new[n1:] = coords_obj_2[1:]

    return LineString(coords_obj_new)


def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    if type(point1) is Point:
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    else:
        angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360


def offset_point(point: tuple | Point, offset: float, angle: float) -> Point:
    rotation_angle = angle - 90
    
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
        else:
            return 270
    else:    
        angle = atan((p2[1] - p1[1])/(p2[0] - p1[0])) * 180/pi
        if p2[1] - p1[1] > 0 and p2[0] - p1[0] < 0:
            return angle + 180
        elif p2[1] - p1[1] < 0 and p2[0] - p1[0] < 0:
            return angle + 180
        elif p2[1] - p1[1] < 0 and p2[0] - p1[0] > 0:
            return angle
        else:
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
    denominator = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
    if denominator==0:
        raise ZeroDivisionError("constructed parallel lines do not intersect, you idiot! :)")
    px = ((p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x))/denominator
    py = ((p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x))/denominator
    return Point(px, py)


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

    pts_up = line_interpolate_point(line, locs + GRID_SIZE, normalized=True).tolist()
    pts_down = line_interpolate_point(line, locs - GRID_SIZE, normalized=True).tolist()
    normal_angles = np.asarray(list(map(get_angle_between_points, pts_down, pts_up))) + 90

    if not float_indicator:
        return normal_angles
    else:
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
    else:
        # working with single-geometries
        return [geometry]


def has_interior(p: Polygon) -> bool:
    return False if list(p.interiors)==[] else True


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
    else:
        return multipolygon