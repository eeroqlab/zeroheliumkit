import copy
import numpy as np

import shapely
from shapely import (Point, MultiPoint, LineString, MultiLineString,
                     Polygon, MultiPolygon, GeometryCollection)
from shapely import (affinity, unary_union,
                     difference, set_precision,
                     is_empty, crosses, intersection,
                     set_coordinates, get_coordinates)
from shapely.ops import linemerge

from phidl import Device

from ..helpers.plotting import plot_geometry
from ..importing import reader_dxf
from ..errors import TopologyError
from ..settings import GRID_SIZE, COLORS, PLG_CLASSES, LINE_CLASSES
from ..functions import *

from ..dev.anchors import Anchor, MultiAnchor
from ..dev.functions import (attach_line, save_geometries, convert_polygon_with_holes_into_muiltipolygon,
                             create_list_geoms, read_geometries)

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

        if (name[:1] != '_') and (name != "anchorsmod"):

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
                print("something went wrong with setting precision")
                print("the error is ", e)
        else:
            self.__dict__[name] = value

    def rename_layer(self, old_name: str, new_name: str) -> None:
        """ Changes the name of layer/attribute class

        Args:
            old_name (str): old name
            new_name (str): new name
        """
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
        self.anchorsmod = MultiAnchor([])

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
                if not callable(value) and (value is not None) and (attribute != "anchorsmod"):
                    name_list.append(attribute)

        if not geom_type:
            return name_list
        elif geom_type=="polygon":
            polygons_only = [l for l in name_list if l not in ("skeletone", "anchors")]
            return polygons_only
        else:
            raise NameError("Currently geom_type support only None or 'polygon'")


    ################################
    #### Geometrical operations ####
    ################################
    def rotate(self, angle: float=0, origin=(0,0)) -> None:
        """ rotates all objects in the class

        Args:
            angle (float): rotation angle
            origin (str, optional): rotations are made around this point. Defaults to (0,0).
        """

        attr_list = self.layer_names()
        for attr in attr_list:
            setattr(self, attr, affinity.rotate(getattr(self, attr), angle, origin))

        self.anchorsmod.rotate(angle, origin)

    def moveby(self, xy: tuple=(0,0)):
        attr_list = self.layer_names()
        for a in attr_list:
            translated_geometry = affinity.translate(getattr(self, a),
                                                     xoff = xy[0],
                                                     yoff = xy[1])
            setattr(self, a, translated_geometry)
        self.anchorsmod.move(xoff=xy[0], yoff=xy[1])

    def moveby_snap(self, anchor: str, to_point: tuple | str | Point):
        old_anchor_coord = self.anchorsmod.point(anchor).coords
        if isinstance(to_point, tuple):
            new_anchor_coord = to_point
        elif isinstance(to_point, str):
            new_anchor_coord = self.anchorsmod.point(to_point).coords
        elif isinstance(to_point, Point):
            new_anchor_coord = to_point.xy
        else:
            raise ValueError("not supported type for 'to_point'")
        dxdy = (new_anchor_coord[0] - old_anchor_coord[0],
                new_anchor_coord[1] - old_anchor_coord[1])

        self.moveby(dxdy)

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

        self.anchorsmod.scale(xfact=xfact, yfact=yfact, origin=origin)

    def mirror(self, aroundaxis: str, update_labels: bool=False, keep_original: bool=False):
        """ mirrors all objects in the class

        Args:
            axis (str): defines the mirror axis. 'x' or 'y' is only supported
        """

        if aroundaxis=='y':
            sign = (-1,1)
        elif aroundaxis=='x':
            sign = (1,-1)
        else:
            raise TypeError("choose x or y axis for mirroring")

        attr_list = self.layer_names()
        for attr in attr_list:
            mirrored = affinity.scale(getattr(self, attr), *sign, 1.0, origin=(0,0))
            if keep_original:
                original = getattr(self, attr)
                setattr(self, attr, unary_union([mirrored, original]))
            else:
                setattr(self, attr, mirrored)

        self.anchorsmod.mirror(aroundaxis=aroundaxis,
                               update_labels=update_labels,
                               keep_original=keep_original)


    ###############################
    #### Operations on anchors ####
    ###############################
    def add_anchor(self, points: list[Anchor] | Anchor):
        """ adding points to 'anchors' class attribute

        Args:
            points (list[tuple], optional): adds new anchor. Defaults to [].
        """
        self.anchorsmod.add(points)

    def get_anchor(self, label: str) -> Anchor:
        """ returns the Anchor class with given label

        Args:
            label (str): anchor label

        Returns:
            Anchor: class, contains label, direction and coordinates
        """
        return self.anchorsmod.point(label)

    def modify_anchor(self,
                      label: str,
                      new_name: str=None,
                      new_xy: tuple=None,
                      new_direction: float=None):
        """ Modifies the given anchor properties

        Args:
            label (str): the anchor to be modified
            new_name (str, optional): updates the name. Defaults to None.
            new_xy (tuple, optional): updates coordinates. Defaults to None.
            new_direction (float, optional): updates the direction. Defaults to None.
        """
        self.anchorsmod.modify(label=label,
                               new_name=new_name,
                               new_xy=new_xy,
                               new_direction=new_direction)

    def remove_anchor(self, labels: list | str):
        """ Delete anchor from the Entity

        Args:
            labels (list | str): provide list of labels or a label name
        """
        self.anchorsmod.remove(labels=labels)


    #############################
    #### Operations on lines ####
    #############################
    def add_line(self, line: LineString, direction: float=None, ignore_crossing=False) -> None:
        """ appending to skeletone a LineString """

        l1 = self.skeletone
        l2 = copy.deepcopy(line)

        if direction:
            l2 = affinity.rotate(l2, angle=direction, origin=(0,0))

        if not is_empty(l1):
            end = l1.boundary.geoms[-1]
            l2 = affinity.translate(l2, xoff = end.x, yoff = end.y)
            if not ignore_crossing:
                if crosses(l1, l2):
                    raise TopologyError("""Appending line crosses the skeletone.
                                        If crossing is intended use 'ignore_crossing=True'""")
            self.skeletone = attach_line(l1, l2)
        else:
            self.skeletone = l2

    def buffer_line(self, name: str, offset: float, **kwargs) -> None:
        """ create a class attribute with a Polygon
            Polygon is created by buffering skeletone 

        Args:
            name (str): attribute name
            offset (float): buffering skeletone by offset
        """

        setattr(self, name, self.skeletone.buffer(offset, **kwargs))

    def fix_line(self):
        try:
            self.skeletone = linemerge(self.skeletone)
        except Exception:
            print("there is nothing to fix in skeletone")

    ################################
    #### Operations on polygons ####
    ################################
    def add_polygon(self, lname: str, polygon: Polygon) -> None:
        """ appending to existing Polygon a new Polygon

        Args:
            body_name (str): attribute name with existing polygon
            polygon (Polygon): new polygon
        """

        body_list = [getattr(self, lname), polygon]
        setattr(self, lname, unary_union(body_list))

    def cut_polygon(self, lname: str, polygon: Polygon) -> None:
        """ cuts the polygon from main polygon in lname attribute

        Args:
            lname (str): name of the attribute, where main polygon is located
            polygon (Polygon): polygon used for cut
        """
        core_polygon = getattr(self, lname)
        setattr(self, lname, difference(core_polygon, polygon))

    def crop_all(self, bbox: Polygon):
        """ crop polygons in all layers

        Args:
            bbox (Polygon): cropping polygon
        """
        layer_names = self.layer_names(geom_type="polygon")
        for lname in layer_names:
            self.crop_layer(lname, bbox)

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


    ################################
    #### Operations on layers ####
    ################################
    def add_layer(self, lname: str, geometry: Polygon | MultiPolygon):
        setattr(self, lname, geometry)

    def remove_layer(self, lname: str):
        if hasattr(self, lname):
            delattr(self, lname)
        else:
            raise TypeError(f"layer '{lname}' doesn't exist")


    ###############################
    #### Additional operations ####
    ###############################
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

    def get_skeletone_boundary(self) -> list:
        """ finds first and last points of the skeletone

        Returns:
            list: list of two Points
        """
        coords = np.asarray(list(self.skeletone.coords))
        return (coords[0], coords[-1])


    ##############################
    #### Exporting operations ####
    ##############################
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
            add_text (tuple(text, layer, location, size), optional): 
                adds text to the "layer" at "location" [x,y]. Defaults to None.
        """

        if export_layers is None:
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
                    self._add_poly_to_device(device=D, polygon=p, gds_layer=gds_layer)
            else:
                self._add_poly_to_device(device=D, polygon=mp, gds_layer=gds_layer)

        D.write_gds(filename+'.gds')

    def _add_poly_to_device(self, device: Device, polygon: Polygon, gds_layer: str) -> Device:
        """ converts polygons with holes into simple polygons,
         and adds them to phidl.Device

        Args:
            device (Device): phidl.Device
            polygon (Polygon): shapely Polygon
            gds_layer (str): name of the gds_layer

        Returns:
            Device: phidl.Device
        """
        prepared_polygons = convert_polygon_with_holes_into_muiltipolygon(polygon)
        for pp in list(prepared_polygons.geoms):
            xpts, ypts = zip(*list(pp.exterior.coords))
            device.add_polygon( [xpts, ypts], layer = gds_layer)

        return device


    #############################
    #### Plotting operations ####
    #############################
    def plot(self,
             ax=None,
             layer: list=["all"],
             show_idx=False,
             color=None,
             alpha=1,
             draw_direction=True,
             **kwargs):
        """ plots the Entity

        Args:
            ax (_type_, optional): axis. Defaults to None.
            layer (list, optional): defines the layer to plot. Defaults to ["all"].
            show_idx (bool, optional): shows the id of the polygon. Defaults to False.
            color (str, optional): color. Defaults to None.
            alpha (int, optional): defines the transparency. Defaults to 1.
            draw_direction (bool, optional): draws arrows. Defaults to True.
        """

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
                if hasattr(self, l) and l != "anchorsmod":
                    geometry = getattr(self, l)
                    plot_geometry(geometry,
                                  ax=ax,
                                  show_idx=show_idx,
                                  color=c,
                                  alpha=alpha,
                                  **kwargs)
                elif l == "anchorsmod":
                    self.anchorsmod.plot(ax=ax, color=c, draw_direction=draw_direction)



class Structure(Entity):
    """ collection of geometries
        inherits internal structure(attributes) of appending Entities
    """

    def __init__(self):
        super().__init__()

    def append(self, structure: Entity, anchoring: tuple=None, direction_snap: bool=False) -> None:
        """ appends Entity or Structure to Structure

        Args:
            structure (Entity): Entity or Structure with collection of geometries 
                (Points, LineStrings, Polygons etc.)
            anchoring (list, optional): snaps appending object given by list of points. 
                                        [StructureObj Point, AppendingObj Point] 
                                        Defaults to None.
        """

        s = structure.copy()
        attr_list_device = self.layer_names()
        attr_list_structure = s.layer_names()
        attr_list = list(set(attr_list_device + attr_list_structure))

        if direction_snap:
            direction = - s.get_anchor(anchoring[1]).direction + self.get_anchor(anchoring[0]).direction
            s.rotate(direction, origin=(0, 0))

        if anchoring:
            c_point = self.get_anchor(anchoring[0])
            a_point = s.get_anchor(anchoring[1])
            offset = (c_point.x - a_point.x, c_point.y - a_point.y)
            s.moveby(offset)

        
        # appending anchors
        self.add_anchor(s.anchorsmod.multipoint) 

        # appending lines and polygons
        for a in attr_list:
            if not hasattr(self, a):
                value = self._combine_objects(None, getattr(s, a))
            elif not hasattr(s, a):
                value = self._combine_objects(getattr(self, a), None)
            else:
                value = self._combine_objects(getattr(self, a), getattr(s, a))

            setattr(self, a, value)
        
    
    def _combine_objects(self, 
                        obj1: LineString | Polygon| MultiLineString | MultiPolygon | None, 
                        obj2: LineString | Polygon| MultiLineString | MultiPolygon | None):
        """ combines two geometries with given connection type and connection point

        Args:
            obj1 (LineString | Polygon | MultiLineString | MultiPolygon | None): shapely geometry
            obj2 (LineString | Polygon | MultiLineString | MultiPolygon | None): shapely geometry

        Raises:
            TypeError: raise if appending shapely object is not [LineString | Polygon | MultiLineString | MultiPolygon]
            ValueError: something wrong went with combining geometries.
                        call _errors to inspect problematic core_obj
        """

        core_objs = self.empty_multiGeometry(obj1, obj2)
        if obj1:
            core_objs = self._append_geometry(core_objs, obj1)
        if obj2:
            core_objs = self._append_geometry(core_objs, obj2)
        
        return core_objs
    
    def empty_multiGeometry(self, obj1, obj2):
        """ creates an empty multi-geometry object based on the types of input geometries

        Raises:
            TypeError: if geometry types are not from supported list or do not match between each other

        Returns:
            empty Multi-Geometry
        """

        if (type(obj1) in LINE_CLASSES) or (type(obj2) in LINE_CLASSES):
            return MultiLineString()
        elif (type(obj1) in PLG_CLASSES) or (type(obj2) in PLG_CLASSES):
            return MultiPolygon()
        else:
            raise TypeError("incorrect shapely object types")
    
    def _append_geometry(self, core_objs, appending_objs):
        """ appends single or multi shapely geometries
            works with LineString, Polygon and multi-geometries
            if connection is provided - performs union on all geometries

        Returns:
            Multi-Geometry
        """
        geom_list = create_list_geoms(core_objs) + create_list_geoms(appending_objs)
        multi_geom = unary_union(geom_list)
        
        return multi_geom 
    
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
    

class GeomCollection(Entity):
    """ collection of geometries
        class attributes are created by layers dictionary 
        attr  |  dict
        name  <- key
        value <- item
    """
    def __init__(self, layers: dict=None, import_file: str=None):
        super().__init__()
        if layers:
            for k, item in layers.items():
                setattr(self, k, item)
        if import_file:
            if import_file[-3:]=="dxf":
                geoms_dict = reader_dxf(import_file)
                geoms = geoms_dict.geometries
            elif import_file[-6:]=="pickle":
                geoms = read_geometries(import_file)
                keys = list(geoms.keys())
                print(import_file + f": {keys}")
            else:
                raise ValueError("importing not supported format")
            
            for k, item in geoms.items():
                if not isinstance(k, str):
                    k = str(k)
                setattr(self, k, item)