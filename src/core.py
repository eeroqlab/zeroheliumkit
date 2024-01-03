import os
import copy
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from shapely import (Point, MultiPoint, LineString, MultiLineString,
                     Polygon, MultiPolygon, GeometryCollection)
from shapely import (affinity, unary_union,
                     difference, set_precision,
                     is_empty, crosses, intersection,
                     set_coordinates, get_coordinates)
from shapely.ops import linemerge

from phidl import Device

from .plotting import plot_geometry
from .importing import Exporter_DXF, Exporter_GDS, Exporter_Pickle
from .errors import RouteError
from .settings import GRID_SIZE, COLORS, PLG_CLASSES, LINE_CLASSES, SIZE_L

from .anchors import Anchor, MultiAnchor
from .functions import (flatten_lines, flatten_polygon, flatten_multipolygon,
                        create_list_geoms, polygonize_text)

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

        if (name[:1] != '_') and (name != "anchors"):

            try:
                geometry_on_grid = set_precision(value, grid_size=GRID_SIZE, mode="pointwise")
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
        self.anchors = MultiAnchor([])

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
                if not callable(value) and (value is not None) and (attribute != "anchors"):
                    name_list.append(attribute)

        if not geom_type:
            return name_list
        if geom_type=="polygon":
            polygons_only = [l for l in name_list if l not in ("skeletone", "anchors")]
            return polygons_only
        raise NameError("Currently geom_type support only None or 'polygon'")

    def clean(self) -> None:
        """ remove all atributes with empty polygons
        """
        for lname in self.layer_names("polygon"):
            geom = getattr(self, lname)
            if geom.is_empty:
                delattr(self, lname)

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

        self.anchors.rotate(angle, origin)

    def moveby(self, xy: tuple=(0,0)):
        attr_list = self.layer_names()
        for a in attr_list:
            translated_geometry = affinity.translate(getattr(self, a),
                                                     xoff = xy[0],
                                                     yoff = xy[1])
            setattr(self, a, translated_geometry)
        self.anchors.move(xoff=xy[0], yoff=xy[1])

    def moveby_snap(self, anchor: str, to_point: tuple | str | Point):
        old_anchor_coord = self.anchors.point(anchor).coords
        if isinstance(to_point, tuple):
            new_anchor_coord = to_point
        elif isinstance(to_point, str):
            new_anchor_coord = self.anchors.point(to_point).coords
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

        self.anchors.scale(xfact=xfact, yfact=yfact, origin=origin)

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

        self.anchors.mirror(aroundaxis=aroundaxis,
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
        self.anchors.add(points)

    def get_anchor(self, label: str) -> Anchor:
        """ returns the Anchor class with given label

        Args:
            label (str): anchor label

        Returns:
            Anchor: class, contains label, direction and coordinates
        """
        return self.anchors.point(label)

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
        self.anchors.modify(label=label,
                            new_name=new_name,
                            new_xy=new_xy,
                            new_direction=new_direction)

    
    def remove_anchor(self, *args):
        """ Delete anchors from the Entity

        Args:
            args: provide list of labels or a label names separated by comma
        """
        if args:
            if (len(args)==1 and isinstance(args[0], (list, tuple))):
                labels = args[0]
            else:
                labels = args
            self.anchors.remove(labels=labels)
        else:
            self.anchors = MultiAnchor([])


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
                    raise RouteError("""Appending line crosses the skeletone.
                                        If crossing is intended use 'ignore_crossing=True'""")
            self.skeletone = flatten_lines(l1, l2)
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
    
    def remove_skeletone(self):
        self.skeletone = LineString()

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
    
    def cut_all(self, polygon: Polygon) -> None:
        """ cuts the polygon from polygons in all layers

        Args:
            lname (str): name of the attribute, where main polygon is located
            polygon (Polygon): polygon used for cut
        """
        layer_names = self.layer_names(geom_type="polygon")
        for lname in layer_names:
            self.cut_polygon(lname, polygon)

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

    def remove_holes_from_polygons(self, lname: str):
        """ converting polygons with holes into a set of polygons without any holes

        Args:
            lname (str): name of the layer, where polygons with holes are located
        """

        polygons = getattr(self, lname)
        fl_multipolygon = flatten_multipolygon(polygons)
        setattr(self, lname, fl_multipolygon)


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
    def get_zhk_dict(self, flatten_poly: bool=False) -> dict:
        """ creates a dictionary with layer names and corresponding geometries in values

        Args:
            flatten_poly (bool, optional): remove holes or keep them. Defaults to False.

        Returns:
            dict: _description_
        """

        layer_names = self.layer_names() + ["anchors"]
        layer_cfg = dict.fromkeys(layer_names)
        for lname in layer_names:
            geometry = getattr(self, lname)
            if (flatten_poly and (lname not in ["anchors", "skeletone"])):
                layer_cfg[lname] = flatten_multipolygon(geometry)
            else:
                layer_cfg[lname] = geometry

        return layer_cfg
        

    def export_pickle(self, filename: str) -> None:
        zhk_layers = self.get_zhk_dict()
        exp = Exporter_Pickle(filename, zhk_layers)
        exp.save()
    
    def export_gds(self, filename: str, layer_cfg: dict) -> None:
        zhk_layers = self.get_zhk_dict(flatten_poly=True)
        exp = Exporter_GDS(filename, zhk_layers, layer_cfg)
        exp.save()

    def export_dxf(self, filename: str, layer_cfg: list) -> None:
        zhk_layers = self.get_zhk_dict(flatten_poly=True)
        exp = Exporter_DXF(filename, zhk_layers, layer_cfg)
        exp.save()

    #############################
    #### Plotting operations ####
    #############################
    def plot(self,
             ax=None,
             layer: list=None,
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
        if ax is None:
            fig = plt.figure(1, figsize=SIZE_L, dpi=90)
            ax = fig.add_subplot(111)

        for l, c in zip(layer, color):
            if hasattr(self, l) and l != "anchors":
                geometry = getattr(self, l)
                plot_geometry(geometry,
                                ax=ax,
                                show_idx=show_idx,
                                color=c,
                                alpha=alpha,
                                **kwargs)
            elif l == "anchors":
                self.anchors.plot(ax=ax, color=c, draw_direction=draw_direction)

    def quickplot(self, plot_config: dict, zoom: tuple=None, ax=None, show_idx: bool=False) -> None:
        """ provides a quick plot of the whole Entity

        Args:
            plot_config (dict): dictionary of ordered layers (keys) with predefined colors as dict values
            zoom (tuple, optional): ((x0, y0), zoom_scale, aspect_ratio). Defaults to None.
        """

        plot_layers = [k for k in plot_config.keys() if k in self.layer_names()]
        if self.anchors:
            plot_layers += ["anchors"]
        plot_colors = [plot_config[k] for k in plot_layers]
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=SIZE_L, dpi=90)
        self.plot(ax=ax, layer=plot_layers, color=plot_colors, show_idx=show_idx)

        if zoom is not None:

            xmin, xmax = plt.gca().get_xlim() 
            ymin, ymax = plt.gca().get_ylim()

            x0, y0 = zoom[0]
            dx = round((xmax - xmin)/zoom[1]/2)
            dy = round((ymax - ymin)/zoom[1]/2)
            if len(zoom) > 2:
                dy = dy/zoom[2]

            ax.set_xlim(x0 - dx, x0 + dx)
            ax.set_ylim(y0 - dy, y0 + dy)

        ax.set_aspect('equal')

        return ax


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
        self.add_anchor(s.anchors.multipoint)

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
            TypeError: raise if appending shapely object 
                       is not [LineString | Polygon | MultiLineString | MultiPolygon]
            ValueError: something wrong went with combining geometries.
                        call _errors to inspect problematic core_obj
        """

        core_objs = self._empty_multiGeometry(obj1, obj2)
        if obj1:
            core_objs = self._append_geometry(core_objs, obj1)
        if obj2:
            core_objs = self._append_geometry(core_objs, obj2)

        return core_objs

    def _empty_multiGeometry(self, obj1, obj2):
        """ creates an empty multi-geometry object based on the types of input geometries

        Raises:
            TypeError: if geometry types are not from supported list or
                       do not match between each other

        Returns:
            empty Multi-Geometry
        """

        if (type(obj1) in LINE_CLASSES) or (type(obj2) in LINE_CLASSES):
            return MultiLineString()
        if (type(obj1) in PLG_CLASSES) or (type(obj2) in PLG_CLASSES):
            return MultiPolygon()
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
        if hasattr(self.skeletone, "geoms"):
            line = list(self.skeletone.geoms)
            coords = list(line[geometry_index].coords)
        else:
            coords = self.skeletone.coords
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


class GeomCollection(Structure):
    """ collection of geometries
        class attributes are created by layers dictionary 
        attr  |  dict
        name  <- key
        value <- item
    """
    def __init__(self, layers: dict=None):
        super().__init__()
        if layers:
            for k, item in layers.items():
                setattr(self, k, item)
