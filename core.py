import numpy as np
import pickle
import copy

from math import sqrt, pi, tanh
import shapely
from shapely import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely import (affinity, unary_union, ops, 
                     difference, set_precision, centroid, 
                     is_empty, crosses, line_merge, 
                     intersection, set_coordinates)

from .helpers.plotting import *
from .errors import *
from .settings import *
from .functions import *


#---------------------------------------------
# some useful functions

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


#---------------------------------------------
# core classes

class _Base:
    """ class with defult methods"""
    _errors = None

    def copy(self):
        """ Returns  deepcopy of the class """
        return copy.deepcopy(self)
    
    def __setattr__(self, name, value):
        """ changing the behaviour of the __setattr__:
            geometries are snapped to the grid by set_persicion

        Args:
            name (str): class attribute name
            value (obj): shapely object
        Note:
            class attribute names with shapely geometries should not start with '_'
        """

        if name[:1] != '_':
            try:
                if hasattr(value, "geoms"):
                    geometries = []
                    for v in list(value.geoms):
                        geometries.append(set_precision(v, grid_size=GRID_SIZE))
                    geometry_type = getattr(shapely, value.geom_type)
                    geometry_on_grid = geometry_type(geometries)
                else:
                    geometry_on_grid = set_precision(value, grid_size=GRID_SIZE)
                self.__dict__[name] = geometry_on_grid
            except Exception as e:
                self._errors = value
                print("something went wrong with setting precision, check for _errors")
                print("the error is ", e)
        else:
            self.__dict__[name] = value
    
    def rename_layer(self, old_name, new_name):
        self.__dict__[new_name] = self.__dict__.pop(old_name)


class Entity(_Base):
    """ collection of shapely objects linked together

        Structure:
            skeletone (Linestring): path of the Entity
            anchors (MultiPoint): anchor points of the Entity
            layers (Polygon or MultiPolygon): buffered skeletone, defined by user (multi) 
    
    """

    def __init__(self):
        self.skeletone = LineString()
        self.anchors = MultiPoint()
        self._direction = np.asarray([0, 0], dtype='float64')    # [input, output] directions defined by angles in degeres
    
    def layer_names(self, geom_type: str=None) -> list:
        """ gets name of layers, containing geometry objects 

        Args:
            geom_type (str, optional): selects only layers with polygons. Defaults to None.

        Returns:
            list: name of layers with geometries
        """
        
        name_list = []
        for attribute in dir(self):
            if attribute[:1] != '_':
                value = getattr(self, attribute)
                if not callable(value) and (value is not None):
                    name_list.append(attribute)
        
        if not geom_type:
            return name_list
        elif geom_type=="polygon":
            polygons_only = [l for l in name_list if l not in ("skeletone", "anchors")]
            return polygons_only
        else:
            raise NameError("Currently geom_type support only None or 'polygon'")
    
    
    def rotate(self, angle: float, origin=(0,0)) -> None:
        """ rotates all objects in the class

        Args:
            angle (float): rotation angle
            origin (str, optional): rotations are made around this point. Defaults to (0,0).
        """

        attr_list = self.layer_names()
        for attr in attr_list:
            setattr(self, attr, affinity.rotate(getattr(self, attr), angle, origin))
        
        self._direction += angle

    def move(self, coord: tuple=None, point: Point=None, origin_anchor: int=None):
        """ translate all objects in the class

        Args:
            coord ((x,y), optional): offset coordinates. Defaults to (0,0).
            p (Point, optional): offsets with respect to the p. Defaults to Point(0,0).
            origin_anchor (int, optional): translates origin to specified ancher. Defaults to None.
        """
        if coord!=None:
            p = Point(coord)

        if point!=None:
            p = point

        if origin_anchor!=None:
            anchors = list(self.anchors.geoms)
            offset_x = anchors[origin_anchor].x
            offset_y = anchors[origin_anchor].y
            p = Point(0, 0)
        else:
            offset_x = 0
            offset_y = 0

        attr_list = self.layer_names()
        for attr in attr_list:
            translated_geometry = affinity.translate(getattr(self, attr), 
                                                     xoff = p.x - offset_x, 
                                                     yoff = p.y - offset_y)
            setattr(self, attr, translated_geometry)

    def scale(self, xfact=1.0, yfact=1.0, origin=(0,0)):
        """ scales all objects in the class

        Args:
            xfact (float, optional): scales along x-axis. Defaults to 1.0.
            yfact (float, optional): scales along y-axis. Defaults to 1.0.
            origin ((x,y), optional): scales with respect (x,y) point. Defaults to (0,0).
        """

        attr_list = self.layer_names()
        for attr in attr_list:
            setattr(self, attr, affinity.scale(getattr(self, attr), xfact, yfact, 1.0, origin))

    def mirror(self, aroundaxis: str):
        """ mirrors all objects in the class

        Args:
            axis (str): defines the mirror axis. 'x' or 'y' is only supported
        """

        if aroundaxis=='y':
            self.scale(-1, 1, origin=(0,0))
            self._direction = 180 - self._direction
        elif aroundaxis=='x':
            self.scale(1, -1, origin=(0,0))
            self._direction = - self._direction
        else:
            raise("choose x or y axis for mirroring")
    
    def add_buffer(self, name: str, offset: float, **kwargs) -> None:
        """ create a class attribute with a Polygon
            Polygon is created by buffering skeletone 

        Args:
            name (str): attribute name
            offset (float): buffering skeletone by offset
        """

        setattr(self, name, self.skeletone.buffer(offset, **kwargs))
        
    def add_line(self, object: LineString, direction: float=None, ignore_crossing=False) -> None:
        """ appending to skeletone a LineString """

        l1 = self.skeletone
        l2 = copy.deepcopy(object)

        if direction:
            l2 = affinity.rotate(l2, angle=direction, origin=(0,0))

        if not is_empty(l1):
            end = l1.boundary.geoms[-1]
            l2 = affinity.translate(l2, xoff = end.x, yoff = end.y)
            if not ignore_crossing:
                if crosses(l1, l2):
                    raise TopologyError("Appending line crosses the skeletone. If crossing is intended use 'ignore_crossing=True' argument")
            self.skeletone = attach_line(l1, l2)
        else:
            self.skeletone = l2
        
    def add_anchor(self, points: list[tuple]=[]):
        """ adding points to 'anchors' class attribute

        Args:
            points (list[tuple], optional): _description_. Defaults to [].
        """
        anchors = list(self.anchors.geoms)
        for p in points:
            if isinstance(p, Point):
                anchors.append(p)
            else:
                anchors.append(Point(p))
        self.anchors = MultiPoint(anchors)

    def add_polygon(self, lname: str, object: Polygon) -> None:
        """ appending to existing Polygon a new Polygon

        Args:
            body_name (str): attribute name with existing polygon
            object (Polygon): new polygon
        """

        body_list = [getattr(self, lname), object]
        setattr(self, lname, unary_union(body_list))
    
    def add_text(self, text: str="abcdef", size: float=1000, loc: tuple=(0,0), layer: str=None):
        """ Converts text into polygons and adds them into the Entity "layer"

        Args:
            text (str, optional): Defaults to "abcdef".
            size (float, optional): Defaults to 1000.
            loc (tuple, optional): Defaults to (0,0).
            layer (str, optional): Defaults to None.
        """
        text_polygons = polygonize_text(text, size)
        text_polygons = affinity.translate(text_polygons, *loc)
        self.add_polygon(layer, text_polygons)
    
    def add_layer(self, lname: str, geometry: Polygon | MultiPolygon):
        setattr(self, lname, geometry)
    
    def cut_polygon(self, lname: str, object: Polygon) -> None:
        """ cuts the object from main polygon in lname attribute

        Args:
            lname (str): name of the attribute, where main polygon is located
            object (Polygon): polygon used for cut
        """
        core_polygon = getattr(self, lname)
        setattr(self, lname, difference(core_polygon, object))
    
    def delete_dublicate_anchors(self):
        self.anchors = unary_union(self.anchors)
    
    def delete_anchors(self, idx_list: list):
        list_of_anchors = list(self.anchors.geoms)
        self.anchors = MultiPoint([anchor for idx, anchor in enumerate(list_of_anchors) if idx not in idx_list])
    
    def delete_layer(self, lname: str):
        if hasattr(self, lname):
            delattr(self, lname)
        else:
            raise TypeError(f"layer '{lname}' doesn't exist")
    
    def get_skeletone_boundary(self) -> list:
        """ finds first and last points of the skeletone

        Returns:
            list: list of two Points
        """
        coords = np.asarray(list(self.skeletone.coords))
        return [coords[0], coords[-1]]
    
    def crop_layer(self, lname: str, bbox: Polygon):
        """ crop polygons in layer by bbox

        Args:
            lname (str): layer name
            bbox (Polygon): cropping polygon
        """
        geoms = getattr(self, lname)
        
        # crop geoms by bbox
        cropped_geoms = intersection(bbox, geoms)

        if isinstance(cropped_geoms, (Point, MultiPoint)):
            cropped_geoms = MultiPolygon()
        elif isinstance(cropped_geoms, (LineString, MultiLineString)):
            cropped_geoms = MultiPolygon()
        elif isinstance(cropped_geoms, GeometryCollection):
            # select only polygons 
            polygon_list = [geom for geom in list(cropped_geoms.geoms) if isinstance(geom, Polygon)]
            cropped_geoms = MultiPolygon(polygon_list)
        
        setattr(self, lname, cropped_geoms)

    
    def crop_all(self, bbox: Polygon):
        """ crop polygons in all layers

        Args:
            bbox (Polygon): cropping polygon
        """
        layer_names = self.layer_names(geom_type="polygon")
        for lname in layer_names:
            self.crop_layer(lname, bbox)


    def modify_polygon_points(self, lname: str, obj_idx: int, points: dict):
        """ Update the point coordinates of an object in a layer

        Args:
            layer (str): layer name
            obj_idx (int): polygon index in multipolygon list
            points (dict): point index in a polygon. 
                Dictionary key corrresponds to the point idx in polygon exterior coord list.
                Dictionary value - list of new [x,y] coordinates
        """
        
        mpolygon = getattr(self, lname)
        polygon_list = list(mpolygon.geoms)
        polygon = polygon_list[obj_idx]
        coords = get_coordinates(polygon)

        # updating coordinates of the polygon
        points_to_be_changed = list(points.keys())
        for point in points_to_be_changed:
            coords[point, 0] = coords[point, 0] + points[point]['x']
            coords[point, 1] = coords[point, 1] + points[point]['y']

        polygon_list[obj_idx] = set_coordinates(polygon, coords)
        setattr(self, lname, polygon_list)

            
    def save_to_file(self, dirname, name):
        geom_names = self.layer_names()
        geom_values = [getattr(self, k) for k in geom_names]
        geom_dict = dict(zip(geom_names, geom_values))

        import os
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = dirname + f"{name}.pickle"
        save_geometries(geom_dict, filename)

    
    def export_to_gds(self, 
                      devname: str, 
                      filename: str, 
                      export_layers: list=None, 
                      layer_order: dict=None):
        """_summary_

        Args:
            devname (str): name of the device
            filename (str): name of the exported file
            export_layers (list, optional): defines, which layers to export. Defaults to None.
            add_text (tuple(text, layer, location, size), optional): adds text to the "layer" at "location" [x,y]. Defaults to None.
        """

        from phidl import Device
        import phidl.geometry as pg

        if export_layers==None:
            export_layers = self.layer_names(geom_type="polygon")

        D = Device(devname)

        for i, l in enumerate(export_layers):
            mp = getattr(self, l)

            if layer_order:
                gds_layer = layer_order.get(l)
            else:
                gds_layer = i

            if isinstance(mp, MultiPolygon):
                for p in list(mp.geoms):
                    prepared_polygons = convert_polygon_with_holes_into_muiltipolygon(p)
                    for pp in list(prepared_polygons.geoms):
                        xpts, ypts = zip(*list(pp.exterior.coords))
                        D.add_polygon( [xpts, ypts], layer = gds_layer)
            else:
                prepared_polygons = convert_polygon_with_holes_into_muiltipolygon(mp)
                for pp in list(prepared_polygons.geoms):
                    xpts, ypts = zip(*list(pp.exterior.coords))
                    D.add_polygon( [xpts, ypts], layer = gds_layer)
        
        D.write_gds(filename+'.gds')
    
    
    def plot(self, ax=None, layer: list=["all"], show_idx=False, color=None, alpha=1, **kwargs):
        if layer==["all"]:
            attr_list = self.layer_names()
            for i, attr in enumerate(attr_list):
                geometry = getattr(self, attr)
                if type(geometry) in PLG_CLASSES:
                    plot_geometry(geometry, 
                                  ax=ax, 
                                  show_idx=show_idx, 
                                  color=COLORS[i%len(COLORS)],
                                  alpha=1, 
                                  **kwargs)
        else:
            for l, c in zip(layer, color):
                if hasattr(self, l):
                    geometry = getattr(self, l)
                    plot_geometry(geometry,
                                  ax=ax,
                                  show_idx=show_idx,
                                  color=c,
                                  alpha=alpha,
                                  **kwargs)



class Structure(Entity):
    """ collection of geometries
        inherits internal structure(attributes) of appending Entities
    """

    def __init__(self):
        super().__init__()

    def append(self, structure: Entity, anchoring: list=None, direction: float=0, connection_type: dict=None) -> None:
        """ appends Entity or Structure to Structure

        Args:
            structure (Entity): Entity or Structure with collection of geometries (Points, LineStrings, Polygons etc.)
            anchoring (list, optional): snaps appending object given by list of points. 
                                        [StructureObj Point, AppendingObj Point] 
                                        Defaults to None.
            direction (float, optional): rotates AppendingObj by given direction angle. 
                                        Defaults to 0.
            connection_type (dict, optional): describes how geometries are connected. 
                                            available options:
                                                - None
                                                - skeletone: "linemerge"
                                                - polygons: "union"
                                                - polygon: ("gapped", gap, length, angle)
                                                - anchors: "p_union"
                                            Defaults to None.
        """

        s = structure.copy()
        attr_list_device = self.layer_names()
        attr_list_structure = s.layer_names()
        attr_list = list(set(attr_list_device + attr_list_structure))
        c_point = None

        if direction:
            s.rotate(direction, origin=(0, 0))

        if anchoring:
            c_point = self.anchors.geoms[anchoring[0]]
            a_point = s.anchors.geoms[anchoring[1]]
            offset = (c_point.x - a_point.x, c_point.y - a_point.y)
            s.move(coord=offset)

        for a in attr_list:
            if not hasattr(self, a):
                value = self._combine_objects(None, getattr(s, a), None, None)
            elif not hasattr(s, a):
                value = self._combine_objects(getattr(self, a), None, None, None)
            else:
                isKeyPresent_or_DictPresent = a in connection_type if connection_type else False
                connection = connection_type.get(a) if isKeyPresent_or_DictPresent else None
                value = self._combine_objects(getattr(self, a), getattr(s, a), connection, c_point)

            setattr(self, a, value)
        
    
    def _combine_objects(self, 
                        obj1: Point | LineString | Polygon| MultiPoint | MultiLineString | MultiPolygon | None, 
                        obj2: Point | LineString | Polygon| MultiPoint | MultiLineString | MultiPolygon | None,
                        connection: str,
                        point: Point
                        ):
        """ combines two geometries with given connection type and connection point

        Args:
            obj1 (Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon | None): shapely geometry
            obj2 (Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon | None): shapely geometry
            connection (str): connection type ["linemerge", "union", ("gapped", g, l, angle), "p_union"]
            point (Point): connection Point

        Raises:
            TypeError: raise if appending shapely object is not [Point | LineString | Polygon | MultiPoint | MultiLineString | MultiPolygon]
            ValueError: something wrong went with combining geometries.
                        call _errors to inspect problematic core_obj
        """

        core_objs = self.empty_multiGeometry(obj1, obj2)
        #print(core_objs)
        if obj1:
            core_objs = self._append_geometry(core_objs, obj1)
        if obj2:
            if connection:
                core_objs = self._append_geometry(core_objs, obj2, connection, point)
            else:
                core_objs = self._append_geometry(core_objs, obj2)
        
        return core_objs
    
    def empty_multiGeometry(self, obj1, obj2):
        """ creates an empty multi-geometry object based on the types of input geometries

        Raises:
            TypeError: if geometry types are not from supported list or do not match between each other

        Returns:
            empty Multi-Geometry
        """
        if (type(obj1) in PTS_CLASSES) or (type(obj2) in PTS_CLASSES):
            return MultiPoint()
        elif (type(obj1) in LINE_CLASSES) or (type(obj2) in LINE_CLASSES):
            return MultiLineString()
        elif (type(obj1) in PLG_CLASSES) or (type(obj2) in PLG_CLASSES):
            return MultiPolygon()
        else:
            raise TypeError("incorrect shapely object types")
    
    def _append_geometry(self, core_objs, appending_objs, connection=None, point=None):
        """ appends single or multi shapely geometries
            works with Point, LineString, Polygon and multi-geometries
            if connection is provided - performs operation on geometries

        Returns:
            Multi-Geometry
        """
        geom_list = list(core_objs.geoms) + create_list_geoms(appending_objs)
        geom_type = getattr(shapely, core_objs.geom_type)
        multi_geom = geom_type(geom_list)

        if connection:
            multi_geom = self._perform_operation(multi_geom, connection, point)
        
        return multi_geom 
    
    def _perform_operation(self, multi_geom, connection, point):
        """ performing operation on MultiGeometry by connection type

        Args:
            o1: shapely geometry
            o2: shapely geometry
            connection (str): connection type
            point (point): connection Point

        Returns:
            tuple: result of combining objects
        """
        if connection=="union":
            return unary_union(multi_geom)
        elif connection=="linemerge":
            return ops.linemerge(multi_geom)
        elif connection[0]=="gapped":
            return self._connect_with_gap(multi_geom, connection=connection, point=point)
        elif connection=="p_union":
            return unary_union(multi_geom)
    
    def _connect_with_gap(self, multi_geom, connection: tuple, point: Point):
        '''
            this function cuts MultiPolygon at the intersection point
            returns MultiPolygons
            connection: ("gapped", gap, length, angle)
        '''
        g = connection[1]      # gap
        l = connection[2]      # length
        angle = connection[3]  # angle

        line = LineString([(point.x - l/2, point.y), (point.x + l/2, point.y)])
        line = affinity.rotate(line, angle, origin=(point.x, point.y))
        return difference(multi_geom, line.buffer(g/2, cap_style='square'))
    
    def get_skeletone_boundary(self, geometry_index: int=0) -> tuple:
        line = list(self.skeletone.geoms)
        coords = np.asarray(list(line[geometry_index].coords))
        return [coords[0], coords[-1]]
    
    def return_mirrored(self, aroundaxis: None):
        """ Returns a mirrored aroundaxis copy of the Class

        Args:
            aroundaxis (None): 'x' or 'y'

        Returns:
            Structure: mirrored class
        """
        class_copy = self.copy()
        class_copy.mirror(aroundaxis)
        return class_copy