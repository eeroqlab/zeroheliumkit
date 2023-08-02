import pickle
import numpy as np

from shapely import Polygon, MultiPolygon, LineString
from shapely import unary_union, centroid, ops

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