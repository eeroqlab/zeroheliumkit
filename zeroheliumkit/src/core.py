"""
This file contains the core classes and methods for the ZeroHeliumKit library.
The classes include the base class `_Base` with default methods, and the `Entity` class
which represents a collection of shapely objects linked together.

The `Entity` class has various methods for geometrical operations such as rotation, translation,
scaling, and mirroring. It also provides functionality for managing anchor points and layers.

The file also imports modules for plotting, exporting, and importing geometries, as well as
defining settings and constants used throughout the library.
"""

import copy
import matplotlib.pyplot as plt

from shapely import (Point, MultiPoint, LineString, MultiLineString,
                     Polygon, MultiPolygon, GeometryCollection)
from shapely import (affinity, unary_union,
                     difference, set_precision, intersection,
                     set_coordinates, get_coordinates)
from shapely.ops import linemerge

from .plotting import plot_geometry, interactive_widget_handler
from .importing import Exporter_DXF, Exporter_GDS, Exporter_Pickle
from .settings import GRID_SIZE, PLG_CLASSES, LINE_CLASSES, SIZE_L

from .anchors import Anchor, MultiAnchor
from .functions import (flatten_multipolygon, append_line,
                        create_list_geoms, polygonize_text)



class _Base:
    """Base class with default methods.

    Attributes:
    ----------
        _errors: Holds any errors that occur during attribute assignment.

    Methods:
    -------
        copy(): Returns a deep copy of the class.
        __setattr__(name, value):
            Changes the behavior of attribute assignment:
            Whenever a new shapely object is created or an existing one is modified,
            it is set to the precision of the grid size.
        rename_layer(old_name, new_name):
            Changes the name of a layer/attribute in the class.
    """

    _errors = None

    def copy(self):
        """Returns a deepcopy of the class."""
        return copy.deepcopy(self)


    def __setattr__(self, name, value):
        """Changes the behavior of the __setattr__ method.
            Whenever a new shapely object is created or an existing one is modified,
            it is set to the precision of the grid size.

        Args:
        ----
        name (str): The class attribute name.
        value (obj): The shapely object.

        Note:
        ----
        Class attribute names with shapely geometries should not start with '_'.
        """

        if (name[:1] != '_') and (name != "anchors"):
            try:
                geometry_on_grid = set_precision(value, grid_size=GRID_SIZE, mode="pointwise")
                self.__dict__[name] = geometry_on_grid
            except Exception as e:
                self._errors = value
                print("Something went wrong with setting precision.")
                print("The error is:", e)
        else:
            self.__dict__[name] = value


    def rename_layer(self, old_name: str, new_name: str) -> None:
        """Changes the name of a layer/attribute in the class.

        Args:
        ----
        old_name (str): The old name.
        new_name (str): The new name.
        """
        self.__dict__[new_name] = self.__dict__.pop(old_name)


class Entity(_Base):
    """ Collection of shapely objects organized by layers and linked together

        Structure:
        ---------
        skeletone (Linestring): path of the Entity
        anchors (MultiPoint): anchor points of the Entity
        layers (Polygon or MultiPolygon): buffered skeletone, defined by user (multi) 
    
    """

    def __init__(self):
        self.skeletone = LineString()
        self.anchors = MultiAnchor([])
        # TODO add layers attribute


    def layer_names(self, geom_type: str=None) -> list:
        """ Returns a list of layer names.

        Args:
        ----
        geom_type (str, optional): Selects only layers with polygons.

        Returns:
        -------
        list: A list of layer names with geometries.

        Raises:
        ------
        NameError: If geom_type is not None or 'polygon'.
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
        raise NameError("Currently geom_type supports only None or 'polygon'")


    def clean(self) -> None:
        """ Removes all layers with empty polygons  """
        for lname in self.layer_names("polygon"):
            geom = getattr(self, lname)
            if geom.is_empty:
                delattr(self, lname)


    ################################
    #### Geometrical operations ####
    ################################
    def rotate(self, angle: float=0, origin=(0,0)) -> None:
        """ Rotates all objects in the class

        Args:
        ----
        angle (float): rotation angle
        origin (str, optional): rotations are made around this point.
        """

        attr_list = self.layer_names()
        for attr in attr_list:
            setattr(self, attr, affinity.rotate(getattr(self, attr), angle, origin))

        self.anchors.rotate(angle, origin)
        return self


    def moveby(self, xy: tuple=(0,0)):
        """ Move objects by the specified x and y offsets.

        Args:
        ----
        xy (tuple): The x and y offsets to move the geometry by.
        """
        attr_list = self.layer_names()
        for a in attr_list:
            translated_geometry = affinity.translate(getattr(self, a),
                                                     xoff = xy[0],
                                                     yoff = xy[1])
            setattr(self, a, translated_geometry)
        self.anchors.move(xoff=xy[0], yoff=xy[1])
        return self


    def moveby_snap(self, anchor: str, to_point: tuple | str | Point):
        """ Move objects by snapping it to a new anchor point or coordinates.

        Args:
        ----
        anchor (str): The name of the anchor point to move from.
        to_point (str | tuple | Point): The new anchor point or coordinates to snap to.

        Raises:
        ------
        ValueError: If the type of 'to_point' is not supported.
        """
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
        return self


    def scale(self, xfact=1.0, yfact=1.0, origin=(0,0)):
        """ Scales all objects

        Args:
        ----
        xfact (float, optional): scales along x-axis.
        yfact (float, optional): scales along y-axis.
        origin ((x,y), optional): scales with respect (x,y) point.
        """

        attr_list = self.layer_names()
        for attr in attr_list:
            setattr(self, attr, affinity.scale(getattr(self, attr), xfact, yfact, 1.0, origin))

        self.anchors.scale(xfact=xfact, yfact=yfact, origin=origin)
        return self


    def mirror(self, aroundaxis: str, update_labels: bool=False, keep_original: bool=False):
        """Mirror all objects.

        Args:
        ----
        aroundaxis (str): Defines the mirror axis. Only 'x' or 'y' are supported.
        update_labels (bool, optional): 
            Whether to update the labels after mirroring. Defaults to False.
        keep_original (bool, optional):
            Whether to keep the original objects after mirroring. Defaults to False.
        """
        if aroundaxis == 'y':
            sign = (-1, 1)
        elif aroundaxis == 'x':
            sign = (1, -1)
        else:
            raise TypeError("Choose 'x' or 'y' axis for mirroring")

        lnames = self.layer_names()
        for name in lnames:
            mirrored = affinity.scale(getattr(self, name), *sign, 1.0, origin=(0, 0))
            if keep_original:
                original = getattr(self, name)
                setattr(self, name, unary_union([mirrored, original]))
            else:
                setattr(self, name, mirrored)

        self.anchors.mirror(aroundaxis=aroundaxis,
                            update_labels=update_labels,
                            keep_original=keep_original)
        return self


    ###############################
    #### Operations on anchors ####
    ###############################
    def add_anchor(self, points: list[Anchor] | Anchor):
        """ Add anchors to the 'anchors' class attribute.

        Args:
        ----
        points (list[Anchor] | Anchor): The anchor(s) to be added.
        """
        self.anchors.add(points)
        return self


    def get_anchor(self, label: str) -> Anchor:
        """ Returns the Anchor class with the given label.

        Args:
        ----
        label (str): The label of the anchor.
        """
        return self.anchors.point(label)


    def modify_anchor(self,
                      label: str,
                      new_name: str=None,
                      new_xy: tuple=None,
                      new_direction: float=None):
        """ Modifies the properties of a given anchor

        Args:
        ----
        label (str): the anchor to be modified
        new_name (str, optional): updates the name. Defaults to None.
        new_xy (tuple, optional): updates coordinates. Defaults to None.
        new_direction (float, optional): updates the direction. Defaults to None.
        """
        self.anchors.modify(label=label,
                            new_name=new_name,
                            new_xy=new_xy,
                            new_direction=new_direction)

    
    def remove_anchor(self, *args: str) -> None:
        """Remove anchors from the Entity.

        Args:
        ----
            *args: A variable number of arguments representing the labels
            of the anchors to be removed.
            The labels can be provided as individual arguments or as a single list or tuple.
        """
        if args:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                labels = args[0]
            else:
                labels = args
            self.anchors.remove(labels=labels)
        else:
            self.anchors = MultiAnchor([])
        return self


    #############################
    #### Operations on lines ####
    #############################
    def add_line(self,
                 line: LineString,
                 direction: float=None,
                 ignore_crossing=False,
                 chaining=True) -> None:
        """ Appends a LineString to the skeleton.

        Args:
        ----
        line (LineString): The LineString to append.
        direction (float, optional): The direction of the LineString. Defaults to None.
        ignore_crossing (bool, optional): Whether to ignore crossing lines. Defaults to False.
        chaining (bool, optional): Whether to chain lines. 
            Defaults to True.
        """
        self.skeletone = append_line(self.skeletone, line, direction, ignore_crossing, chaining)
        return self


    def buffer_line(self, name: str, offset: float, **kwargs) -> None:
        """ Create a new layer(class attribute) by buffering the skeleton

        Args:
        ----
        name (str): new layer/attribute name
        offset (float): buffering skeleton by offset
        **kwargs: additional keyword arguments to be passed to the buffer method

        Note:
        ----
        - see Shapely 'buffer' function for additional keyword arguments
        """
        setattr(self, name, self.skeletone.buffer(offset, **kwargs))
        return self


    def fix_line(self):
        """ Fixes the skeletone by merging the lines """
        try:
            self.skeletone = linemerge(self.skeletone)
        except Exception:
            print("there is nothing to fix in skeletone")
        return self


    def remove_skeletone(self):
        """ Removes the skeletone from the object """
        self.skeletone = LineString()
        return self
    

    def remove_line(self, line_id: int | tuple | list):
        """ Remove a line from the skeletone

        Args:
        ----
        line_id (int | tuple | list): The index of the line to be removed.
        """
        if isinstance(line_id, int):
            line_id = [line_id]

        lines = list(self.skeletone.geoms)
        self.skeletone = MultiLineString([line for i, line in enumerate(lines) if i not in line_id])
        return self


    ################################
    #### Operations on polygons ####
    ################################
    def add_polygon(self, lname: str, polygon: Polygon) -> None:
        """ Add a new polygon to an existing layer

        Args:
        ----
        lname (str): The layer/attribute name.
        polygon (Polygon): The new polygon to be appended.
        """
        setattr(self, lname, unary_union([getattr(self, lname), polygon]))
        return self


    def cut_polygon(self, lname: str, polygon: Polygon) -> None:
        """ Cut the specified polygon from the layer

        Args:
        ----
        lname (str): The name of the attribute where the main polygon is located.
        polygon (Polygon): The polygon used for cutting.
        """
        setattr(self, lname, difference(getattr(self, lname), polygon))
        return self


    def cut_all(self, polygon: Polygon) -> None:
        """ Cut the given polygon from polygons in all layers

        Args:
        ----
        polygon (Polygon): The polygon used for cutting
        """
        lnames = self.layer_names(geom_type="polygon")
        for lname in lnames:
            self.cut_polygon(lname, polygon)
        return self


    def crop_all(self, polygon: Polygon):
        """ Crop polygons in all layers

        Args:
        ----
        polygon (Polygon): The cropping polygon.
        """
        lnames = self.layer_names(geom_type="polygon")
        for lname in lnames:
            self.crop_layer(lname, polygon)
        return self


    def crop_layer(self, lname: str, polygon: Polygon):
        """ Crop objects in a layer by a given polygon.

        Args:
        ----
        lname (str): The name of the layer.
        polygon (Polygon): The polygon used for cropping.
        """
        geoms = getattr(self, lname)

        # Crop geoms by polygon
        cropped_geoms = intersection(polygon, geoms)

        if isinstance(cropped_geoms, (Point, MultiPoint, LineString, MultiLineString)):
            cropped_geoms = MultiPolygon()
        elif isinstance(cropped_geoms, GeometryCollection):
            # Select only polygons
            polygon_list = [geom for geom in list(cropped_geoms.geoms) if isinstance(geom, Polygon)]
            cropped_geoms = MultiPolygon(polygon_list)
        setattr(self, lname, cropped_geoms)
        return self


    def modify_polygon_points(self, lname: str, obj_idx: int, points: dict):
        """ Update the point coordinates of an object in a layer

        Args:
        ----
        layer (str): layer name
        obj_idx (int): polygon index in multipolygon list
        points (dict): point index in a polygon. 
            Dictionary key corrresponds to the point idx in polygon exterior coord list.
            Dictionary value - list of new [x,y] coordinates
        """

        mpolygon = getattr(self, lname)
        if isinstance(mpolygon, Polygon):
            mpolygon = MultiPolygon([mpolygon])
        polygon_list = list(mpolygon.geoms)
        polygon = polygon_list[obj_idx]
        coords = get_coordinates(polygon)

        # updating coordinates of the polygon
        points_to_be_changed = list(points.keys())
        for point in points_to_be_changed:
            coords[point, 0] = coords[point, 0] + points[point]['x']
            coords[point, 1] = coords[point, 1] + points[point]['y']

        polygon_list[obj_idx] = set_coordinates(polygon, coords)
        setattr(self, lname, MultiPolygon(polygon_list))


    def remove_holes_from_polygons(self, lname: str):
        """ Converting polygons with holes into a set of polygons without any holes

        Args:
        ----
        lname (str): Name of the layer where polygons with holes are located.
        """
        polygons = getattr(self, lname)
        setattr(self, lname, flatten_multipolygon(polygons))
    

    def remove_polygon(self, lname: str, polygon_id: int | tuple | list):
        """ Remove a polygon from the layer

        Args:
        ----
        lname (str): The name of the layer.
        polygon_id (int | tuple | list): The index of the polygon to be removed.
        """
        polygons = getattr(self, lname)

        if isinstance(polygons, Polygon):
            raise ValueError("Cannot remove polygon from a single polygon object")

        if isinstance(polygon_id, int):
            polygon_id = [polygon_id]

        poly_list = list(polygons.geoms)
        setattr(self, lname, MultiPolygon([poly for i, poly in enumerate(poly_list) if i not in polygon_id]))


    ################################
    #### Operations on layers ####
    ################################
    def add_layer(self, lname: str, geometry: Polygon | MultiPolygon):
        """
        Add a layer to the object with the given name and geometry.

        Args:
        ----
        lname (str): The name of the layer.
        geometry (Polygon | MultiPolygon): The geometry of the layer.
        """
        setattr(self, lname, geometry)
        return self


    def remove_layer(self, *args: str):
        """ Remove one or more layers/attributes from the class.

        Args:
        ----
        *args: Variable number of layer names to be removed.
        """
        for layer_name in args:
            if hasattr(self, layer_name):
                delattr(self, layer_name)
        return self


    def simplify_layer(self, lname: str, tolerance: float=0.1):
        """ Simplify polygons in a layer

        Args:
        ----
        lname (str): The name of the layer.
        tolerance (float, optional): The tolerance value for simplification. Defaults to 0.1.
        """
        setattr(self, lname, getattr(self, lname).simplify(tolerance))
        return self


    ###############################
    #### Additional operations ####
    ###############################
    def add_text(self, text: str="abcdef", size: float=1000, loc: tuple=(0,0), layer: str=None):
        """ Converts text into polygons and adds them to the specified layer.

        Args:
        ----
        text (str, optional): The text to be converted into polygons. Defaults to "abcdef".
        size (float, optional): The size of the text. Defaults to 1000.
        loc (tuple, optional): The location where the text polygons will be placed.
        layer (str, optional): The name of the layer where the text polygons will be added.
        """
        ptext = polygonize_text(text, size)
        ptext = affinity.translate(ptext, *loc)
        self.add_polygon(layer, ptext)

    # deprecated
    # def get_skeletone_boundary(self) -> list:
    #     """ finds first and last points of the skeletone

    #     Returns:
    #         list: list of two Points
    #     """
    #     coords = np.asarray(list(self.skeletone.coords))
    #     return (coords[0], coords[-1])


    ##############################
    #### Exporting operations ####
    ##############################
    def get_zhk_dict(self, flatten_polygon: bool=False) -> dict:
        """ Returns a dictionary with layer names as keys
            and corresponding geometries as values

        Args:
        ----
        flatten_polygon (bool, optional): Flag to remove holes from polygons. Defaults to False.
        """
        lnames = self.layer_names() + ["anchors"]
        zhk_dict = dict.fromkeys(lnames)
        for lname in lnames:
            geometry = getattr(self, lname)
            if (flatten_polygon and (lname not in ["anchors", "skeletone"])):
                zhk_dict[lname] = flatten_multipolygon(geometry)
            else:
                zhk_dict[lname] = geometry
        return zhk_dict


    def export_pickle(self, filename: str) -> None:
        """ Export layers as a pickle file

        Args:
        ----
        filename (str): The name of the pickle file to be exported
        """
        zhkdict = self.get_zhk_dict()
        exp = Exporter_Pickle(filename, zhkdict)
        exp.save()


    def export_gds(self, filename: str, layer_cfg: dict) -> None:
        """ Export layers as a GDS file

        Args:
        ----
        filename (str): The name of the gds file to be exported
        layer_cfg (dict): A dictionary containing the layer configuration.

        Note:
        ----
        The layer configuration should be in the following format:
            layer_cfg = {"zhk_layer_name_1": {"gds_layer_name_1": int, "datatype": 0},
                         "zhk_layer_name_2": {"gds_layer_name_2": int, "datatype": 0}}
            For 'datatype' refer gdspy specification.
        """
        zhkdict = self.get_zhk_dict(flatten_polygon=True)
        exp = Exporter_GDS(filename, zhkdict, layer_cfg)
        exp.save()


    def export_dxf(self, filename: str, layer_cfg: list) -> None:
        """ Export layers as a DXF file

        Args:
        ----
        filename (str): The name of the dxf file to be exported
        layer_cfg (dict): A list of layer to be exported
        """
        zhkdict = self.get_zhk_dict(flatten_polygon=True)
        exp = Exporter_DXF(filename, zhkdict, layer_cfg)
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
        """ Plot the Entity.

        Args:
        ----
        ax (matplotlib.axes.Axes, optional): The axis to plot on. Defaults to None.
        layer (list, optional): The layer(s) to plot. Defaults to ["all"].
        show_idx (bool, optional): Whether to show the id of the polygon. Defaults to False.
        color (str or list, optional): The color(s) to use for plotting. Defaults to None.
        alpha (float, optional): The transparency of the plot. Defaults to 1.
        draw_direction (bool, optional): Whether to draw arrows. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the plot_geometry function.
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
            elif hasattr(self, l) and l == "anchors":
                self.anchors.plot(ax=ax, color=c, draw_direction=draw_direction)


    def quickplot(self, plot_config: dict, zoom: tuple=None,
                  ax=None, show_idx: bool=False, **kwargs) -> None:
        """ provides a quick plot of the whole Entity

        Args:
        ----
        plot_config (dict): dict of ordered layers (keys) with predefined colors as dict values
        zoom (tuple, optional): ((x0, y0), zoom_scale, aspect_ratio). Defaults to None.
        """
        plot_layers = [k for k in plot_config.keys() if k in self.layer_names()]
        if hasattr(self, "anchors"):
            plot_layers += ["anchors"]
        plot_colors = [plot_config[k] for k in plot_layers]

        if ax is None:
            interactive_widget_handler()
            _, ax = plt.subplots(1, 1, figsize=SIZE_L, dpi=90)
        self.plot(ax=ax, layer=plot_layers, color=plot_colors, show_idx=show_idx, **kwargs)

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
    """ Represents a structure that contains layers 
        with a collection of geometries (Points, LineStrings, Polygons, etc.).
        Inherits from the Entity class.

    Examples:
    --------
    Create a Structure object:
    >>> s = Structure()

    Append another Structure to the current Structure:
    >>> s1 = Structure()
    >>> s2 = Structure()
    >>> # ... add geometries to s1 and s2 ...
    >>> s1.append(s2, anchoring=["A", "B"])

    Return a mirrored copy of the Structure:
    >>> s = Structure()
    >>> mirrored = s.return_mirrored('x')
    """

    def __init__(self):
        super().__init__()

    def append(self,
               structure: Entity,
               anchoring: tuple=None,
               direction_snap: bool=False,
               remove_anchor: bool | str=False,
               upd_alabels: list[tuple]=None) -> None:
        """ Appends an Entity or Structure to the Structure.

        Args:
        ----
        structure (Entity): 
            The Entity or Structure with a collection of geometries 
            (Points, LineStrings, Polygons, etc.).
        anchoring (list, optional): 
            Snaps the appending object given by a list of points. 
            [StructureObj Point, AppendingObj Point]. 
            Defaults to None.
        direction_snap (bool, optional): 
            If True, aligns the direction of the appending object 
            with the direction of the anchor points. 
            Defaults to False.
        remove_anchor (bool or str, optional): 
            If True, removes the anchor points after appending. 
            If a string is provided, removes the specified anchor point. 
            Defaults to False.
        upd_alabels (list, optional):
            Renames anchor labels of the appending structure before appending.
            A list of tuples with the old and new anchor labels: (old_label, new_label) 
            Defaults to None.

        Example:
        -------
            >>> s1 = Structure()
            >>> s2 = Structure()
            >>> # ... add geometries to s1 and s2 ...
            >>> s1.append(s2, anchoring=["A", "B"])
        """

        s = structure.copy()
        attr_list_device = self.layer_names()
        attr_list_structure = s.layer_names()
        attr_list = list(set(attr_list_device + attr_list_structure))

        # snapping direction
        if direction_snap:
            angle = - s.get_anchor(anchoring[1]).direction + self.get_anchor(anchoring[0]).direction
            s.rotate(angle, origin=(0, 0))

        # snapping anchors
        if anchoring:
            c_point = self.get_anchor(anchoring[0])
            a_point = s.get_anchor(anchoring[1])
            offset = (c_point.x - a_point.x, c_point.y - a_point.y)
            s.moveby(offset)

        # appending anchors
        if upd_alabels:
            for label_old, label_new in upd_alabels:
                s.anchors.modify(label_old, new_name=label_new)
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

        # remove or not to remove anchor after appending
        if remove_anchor is True:
            self.remove_anchor(anchoring)
        elif isinstance(remove_anchor, str):
            self.remove_anchor(remove_anchor)

        return self


    def _combine_objects(self,
                         obj1: LineString | Polygon| MultiLineString | MultiPolygon | None,
                         obj2: LineString | Polygon| MultiLineString | MultiPolygon | None):
        """ Merge two geometries and return the result

        Args:
        ----
        obj1 (LineString | Polygon | MultiLineString | MultiPolygon | None): first geometry.
        obj2 (LineString | Polygon | MultiLineString | MultiPolygon | None): second geometry.

        Raises:
        ------
        TypeError: Raised if the appending object
                   is not [LineString, Polygon, MultiLineString, MultiPolygon].
        ValueError: Raised if error with merging the geometries.
                    Call _errors to inspect the problem.
        """
        merged = self._empty_multigeometry(obj1, obj2)
        if obj1:
            merged = self._append_geometry(merged, obj1)
        if obj2:
            merged = self._append_geometry(merged, obj2)
        return merged


    def _empty_multigeometry(self, obj1, obj2):
        """ Returns an empty multi-geometry object based on the types of input geometries

        Args:
        ----
        obj1: The first geometry object.
        obj2: The second geometry object.

        Raises:
        ------
        TypeError: If geometry types are not from the supported list
                   or do not match between each other
        """
        # TODO: add support for anchors
        if (type(obj1) in LINE_CLASSES) or (type(obj2) in LINE_CLASSES):
            return MultiLineString()
        if (type(obj1) in PLG_CLASSES) or (type(obj2) in PLG_CLASSES):
            return MultiPolygon()
        raise TypeError("Incorrect shapely object types")


    def _append_geometry(self, core_objs, appending_objs):
        """ Append single or multiple shapely geometries.

        Args:
        ----
        core_objs: shapely geometries to be appended.
        appending_objs: shapely geometries to append.

        Returns:
        -------
        Multi-Geometry: Union of all the geometries.

        Note:
        ----
        This method works with LineString, Polygon, and multi-geometries.
        """
        geom_list = create_list_geoms(core_objs) + create_list_geoms(appending_objs)
        return unary_union(geom_list)


    def return_mirrored(self, aroundaxis: str, **kwargs) -> 'Structure':
        """Returns a mirrored copy of the Structure class.

        Args:
        ----
        aroundaxis (str): The axis around which to mirror the class. Valid values are 'x' or 'y'.
        **kwargs: Additional keyword arguments.

        Returns:
        -------
        Structure: A mirrored copy of the Structure class.

        Example:
        -------
            >>> s = Structure()
            >>> mirrored = s.return_mirrored('x')
            >>> print(mirrored)
        """
        cc = self.copy()
        return cc.mirror(aroundaxis, **kwargs)


class GeomCollection(Structure):
    """ Collection of geometries.
        Class attributes are created by layers dictionary.
    
    Attributes:
    ----------
    layers (dict): Dictionary containing the layers and corresponding polygons/skeletone/anchors.
        
    Example:
    -------
        >>> layers = {
        >>>     'layer1': Rectangle(10, 13),
        >>>     'layer2': Circle(5)
        >>> }
        >>> collection = GeomCollection(layers)
    """
    def __init__(self, layers: dict=None):
        super().__init__()
        if layers:
            for k, item in layers.items():
                setattr(self, k, item)

        if not hasattr(self, "anchors"):
            self.anchors = MultiAnchor([])

        if not hasattr(self, "skeletone"):
            self.skeletone = LineString()
