"""
core.py

This file contains the core classes and methods for the ZeroHeliumKit library.

Classes:
-------
    `_Base`: Base class with default methods for managing shapely objects.
    `Entity`: A subclass of _Base which represents a collection of shapely objects linked together and provides methods for geometrical operations.
    `Structure`: A subclass of Entity that represents layers with collections of geometries (Points, LineStrings, Polygons, etc.).
    `GeomCollection`: A subclass of Structure that represents a collection of geometries.
"""

import copy, re
import matplotlib.pyplot as plt
from warnings import warn

from shapely import (Point, MultiPoint, LineString, MultiLineString,
                     Polygon, MultiPolygon, GeometryCollection)
from shapely import (affinity, unary_union,
                     difference, set_precision, intersection,
                     set_coordinates, get_coordinates, remove_repeated_points)

from .plotting import plot_geometry, interactive_widget_handler, tuplify_colors, draw_labels, ColorHandler
from .importing import Exporter_DXF, Exporter_GDS, Exporter_Pickle
from .settings import GRID_SIZE, SIZE_L, RED, DARKGRAY

from .anchors import Anchor, MultiAnchor, Skeletone
from .utils import flatten_multipolygon, create_list_geoms, polygonize_text, has_interior


class _Base:
    """
    Base class with default methods.

    Attributes:
    -----------
        - layers: List of layer names.

    Methods:
    --------
        - copy(): Returns a deep copy of the class.
        - add_layer(lname, geometry): Adds a new layer to the class.
        - remove_layer(lname): Removes a layer from the class.
        - rename_layer(old_name, new_name): Renames a layer in the class.
        - simplify_layer(lname, tolerance=0.1): Simplifies polygons in a layer.
        - has_layer(lname): Checks if a layer exists in the class.
    """
    layers = [] 
    """List of all layer names belonging to the object."""
    _errors = None

    def __init__(self):
        """
        Initializes a new Base class instance.
        """
        self.layers = []
        self.colors = ColorHandler({})
        self.errors = None

    def copy(self):
        """
        Creates a deep copy of the class instance.

        Returns:
        --------
            A deep copy of the class instance.
        """
        return copy.deepcopy(self)

    def __setattr__(self, name, value):
        """
        Changes the behavior of the __setattr__ method. 
        Whenever a new shapely object is created or an existing one is modified, it is set to the precision of the grid size.

        Args:
        -----
            - name (str): The class attribute name.
            - value (obj): The shapely object.
        """

        if name in self.layers:
            try:
                geometry_on_grid = set_precision(value, grid_size=GRID_SIZE, mode="pointwise")
                self.__dict__[name] = geometry_on_grid
            except Exception as e:
                self.errors = value
                print("Something went wrong with setting precision.")
                print("The error is:", e)
            
            value = remove_repeated_points(value, tolerance=0.0001)
        else:
            self.__dict__[name] = value


    ################################
    #### Operations on layers ####
    ################################

    def add_layer(self, lname: str, geometry: Polygon | MultiPolygon=Polygon(), color: str=None, alpha: int=1.0): 
        """
        Adds a layer to the class with the given name and geometry.

        Args:
        -----
            - lname (str): The name of the layer.
            - geometry (Polygon | MultiPolygon): The geometry of the layer.

        Returns:
        --------
            Updated instance (self) of the class with the new layer added.
        """
        self.layers.append(lname)
        
        self.colors.add_color(lname, color, alpha)

        setattr(self, lname, geometry)
        return self


    def remove_layer(self, lname: str):
        """
        Removes a layer from the class.

        Args:
        -----
            - lname (str): The name of the layer.

        Returns:
        --------
            Updated instance (self) of the class with the layer removed.
        """
        if lname in self.layers:
            self.layers.remove(lname)
            delattr(self, lname)
            self.colors.remove_color(lname)
        else:
            print(f"Layer '{lname}' not found in layers.")

        return self


    def rename_layer(self, old_name: str, new_name: str) -> None:
        """
        Changes the name of a layer/attribute in the class.

        Args:
        -----
            - old_name (str): The old name.
            - new_name (str): The new name.

        Returns:
        --------
            Updated instance (self) of the class with the layer renamed.
        """
        if old_name in self.layers:
            self.__dict__[new_name] = self.__dict__.pop(old_name)
            self.layers[self.layers.index(old_name)] = new_name
            self.colors.rename_color(old_name, new_name)
        else:
            print(f"Layer '{old_name}' not found in layers.")

        return self


    def simplify_layer(self, lname: str, tolerance: float=0.1):
        """ Simplify polygons in a layer

        Args:
        -----
            - lname (str): The name of the layer.
            - tolerance (float, optional): The tolerance value for simplification. Defaults to 0.1.

        Returns:
        --------
            Updated instance (self) of the class with the specified layer simplified.
        """
        if lname in self.layers:
            setattr(self, lname, getattr(self, lname).simplify(tolerance))
        else:
            print(f"Layer '{lname}' not found in layers.")
        return self


    def has_layer(self, lname: str) -> bool:
        """ Check if a layer exists in the class.

        Args:
        -----
            - lname (str): The name of the layer.

        Returns:
        --------
            bool: True if the layer exists, False otherwise.
        """
        return lname in self.layers
                


class Entity(_Base):
    """ 
    Represents collections of shapely objects organized by layers and linked together.
    Inherits from the _Base class.

    Attributes:
    -----------
        - skeletone (Skeletone):
            Represents a collection of lines linked to the Entity, which is an instance of the Skeletone class.
        - anchors (MultiAnchor):
            Represents the anchor points of the Entity, which are instances of MultiAnchor.

    Methods:
    --------
        - clean(): Removes all layers with empty polygons.
        - rotate(angle=0, origin=(0,0)): Rotates all objects in the class.
        - moveby(xy=(0,0)): Moves objects by the specified x and y offsets.
        - moveby_snap(anchor, to_point): Moves objects by snapping it to a new anchor point.
        - scale(xfact=1.0, yfact=1.0, origin=(0,0)): Scales all objects.
        - mirror(aroundaxis, keep_original=True, update_labels=False): Mirrors all objects.
        - add_anchor(points): Adds anchors to the 'anchors' class attribute.
        - get_anchor(label): Returns the Anchor class with the given label.
    """

    def __init__(self):
        """
        Initializes a new Entity class instance.
        """
        super().__init__()
        self.skeletone = Skeletone()
        self.anchors = MultiAnchor()


    def __repr__(self):
        class_name = self.__class__.__name__
        repr_name = f"{class_name} {tuple(self.layers)}"
        max_length = 75
        if len(repr_name) > max_length:
            return f"{repr_name[: max_length - 3]}..."
        return repr_name


    def clean(self):
        """
        Removes all layers with empty polygons  
        """
        for lname in self.layers:
            geom = getattr(self, lname)
            if geom.is_empty:
                delattr(self, lname)


    ################################
    #### Geometrical operations ####
    ################################

    def rotate(self, angle: float=0, origin=(0,0)):
        """ 
        Rotates all objects in the class

        Args:
        -----
            - angle (float, optional): rotation angle. Defaults to 0.
            - origin (str, optional): rotations are made around this point. Defaults to (0,0).
        
        Returns:
        --------
            Updated instance (self) of the class with all objects rotated.
        """

        for l in self.layers:
            setattr(self, l, affinity.rotate(getattr(self, l), angle, origin))

        self.skeletone.rotate(angle, origin)
        self.anchors.rotate(angle, origin)
        return self


    def moveby(self, xy: tuple=(0,0)):
        """
        Move objects by the specified x and y offsets.

        Args:
        -----
            xy (tuple, optional): The x and y offsets to move the geometry by. Defaults to (0, 0).
        
        Returns:
        --------
            Updated instance (self) of the class with all objects moved.
        """
        for l in self.layers:
            translated_geometry = affinity.translate(getattr(self, l),
                                                     xoff = xy[0],
                                                     yoff = xy[1])
            setattr(self, l, translated_geometry)

        self.skeletone.move(*xy)
        self.anchors.move(*xy)
        return self


    def moveby_snap(self, anchor: str, to_point: tuple | str | Point):
        """
        Move objects by snapping it to a new anchor point or coordinates.

        Args:
        -----
            - anchor (str): The name of the anchor point to move from.
            - to_point (str | tuple | Point): The new anchor point or coordinates to snap to.

        Raises:
        -------
            ValueError: If the type of 'to_point' is not supported.

        Returns:
        --------
            Updated instance (self) of the class with all objects moved to the new anchor point.
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
        """
        Scales all objects by the specified factors along the x and y axes.

        Args:
        -----
            - xfact (float, optional): scale along x-axis. Defaults to 1.0.
            - yfact (float, optional): scale along y-axis. Defaults to 1.0.
            - origin ((x,y), optional): scale with respect to an origin (x,y). Defaults to (0,0).

        Returns:
        --------
            Updated instance (self) of the class with all objects scaled.
        """

        for l in self.layers:
            setattr(self, l, affinity.scale(getattr(self, l), xfact, yfact, 1.0, origin))

        self.skeletone.scale(xfact, yfact, origin)
        self.anchors.scale(xfact, yfact, origin)
        return self


    def mirror(self, aroundaxis: str, keep_original: bool=True, update_labels: bool=True):
        """
        Mirror all objects around a specified axis.

        Args:
        -----
            - aroundaxis (str): Defines the mirror axis. Only 'x' or 'y' are supported.
            - update_labels (bool, optional): 
                Whether to update the labels after mirroring. Defaults to False.
            - keep_original (bool, optional):
                Whether to keep the original objects after mirroring. Defaults to False.

        Raises:
        -------
            TypeError: If the 'aroundaxis' is not 'x' or 'y'.

        Returns:
        --------
            Updated instance (self) of the class with all objects mirrored.
        """
        if aroundaxis == 'y':
            sign = (-1, 1)
        elif aroundaxis == 'x':
            sign = (1, -1)
        else:
            raise TypeError("Choose 'x' or 'y' axis for mirroring")

        for name in self.layers:
            mirrored = affinity.scale(getattr(self, name), *sign, 1.0, origin=(0, 0))
            if keep_original:
                original = getattr(self, name)
                setattr(self, name, unary_union([mirrored, original]))
            else:
                setattr(self, name, mirrored)

        self.skeletone.mirror(aroundaxis=aroundaxis,
                              keep_original=keep_original)
        self.anchors.mirror(aroundaxis=aroundaxis,
                            update_labels=update_labels,
                            keep_original=keep_original)
        return self


    ###############################
    #### Operations on anchors ####
    ###############################

    def add_anchor(self, points: list[Anchor] | Anchor):
        """
        Adds specified anchors to the 'anchors' attribute.

        Args:
        -----
            - points (list[Anchor] | Anchor): The anchor(s) to be added.
        
        Returns:
        --------
            Updated instance (self) of the class with the new anchors added.
        """
        self.anchors.add(points)
        return self


    def get_anchor(self, label: str) -> Anchor:
        """
        Returns the Anchor class with the given label.

        Args:
        -----
            - label (str): The label of the anchor.
        
        Returns:
        --------
            Anchor: The Anchor object with the specified label.
        """
        return self.anchors.point(label)


    def modify_anchor(self,
                      label: str,
                      new_name: str=None,
                      new_xy: tuple=None,
                      new_direction: float=None):
        """
        Modifies the properties of a given anchor

        Args:
        -----
            - label (str): the anchor to be modified
            - new_name (str, optional): updates the name. Defaults to None.
            - new_xy (tuple, optional): updates coordinates. Defaults to None.
            - new_direction (float, optional): updates the direction. Defaults to None.
        """
        self.anchors.modify(label=label,
                            new_name=new_name,
                            new_xy=new_xy,
                            new_direction=new_direction)

    
    def remove_anchor(self, *args: str) -> None:
        """
        Remove anchors from the Entity.

        Args:
        -----
            - *args: A variable number of arguments representing the labels of the anchors to be removed.
                The labels can be provided as individual arguments or as a single list or tuple.

        Returns:
        --------
            Updated instance (self) of the class with the specified anchors removed.
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
        """
        Appends a LineString to the skeleton.

        Args:
        -----
            - line (LineString): The LineString to append.
            - direction (float, optional): The direction of the LineString. Defaults to None.
            - ignore_crossing (bool, optional): Whether to ignore crossing lines. Defaults to False.
            - chaining (bool, optional): Whether to chain lines. Defaults to True.
        
        Returns:
        --------
            Updated instance (self) of the class with the new line added to the skeletone.
        """
        self.skeletone.add_line(line, direction, ignore_crossing, chaining)
        return self


    def buffer_line(self, name: str, offset: float, color: str=None, alpha: float=1.0, **kwargs) -> None:
        """
        Create a new layer (attribute) by buffering the skeleton.

        Args:
        -----
            - name (str): new layer/attribute name
            - offset (float): buffering skeleton by offset
            - **kwargs: additional keyword arguments to be passed to the buffer method
            See `Shapely buffer docs <https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html#shapely.buffer>`_ for additional keyword arguments.

        Returns:
        --------
            Updated instance (self) of the class with the new buffered line added as an attribute.
        """
        self.add_layer(name, self.skeletone.lines.buffer(offset, **kwargs), color, alpha)
        return self


    def fix_line(self):
        """ 
        Fixes the skeletone by merging the lines.

        Returns:
        --------
            Updated instance (self) of the class with the skeletone fixed.
        """
        self.skeletone.fix()
        return self


    def remove_skeletone(self):
        """
        Resets the skeletone attribute to a new Skeletone instance.
        
        Returns:
        --------
            Updated instance (self) of the class with a new Skeletone instance.
        """
        self.skeletone = Skeletone()
        return self
    

    def remove_line(self, line_id: int | tuple | list):
        """
        Removes a line from the skeletone.

        Args:
        -----
            - line_id (int | tuple | list): The index of the line to be removed.
        
        Returns:
        --------
            Updated instance (self) of the class with the specified line removed from the skeletone.
        """
        self.skeletone.remove_line(line_id)
        return self


    ################################
    #### Operations on polygons ####
    ################################

    def add_polygon(self, lname: str, polygon: Polygon):
        """
        Adds a new polygon to an existing layer.

        Args:
        -----
            - lname (str): The layer/attribute name.
            - polygon (Polygon): The new polygon to be appended.

        Returns:
        --------
            Updated instance (self) of the class with the new polygon added to the specified layer.
        """
        setattr(self, lname, unary_union([getattr(self, lname), polygon])) 

        return self
    

    def cut_polygon(self, lname: str, 
                        geom: Polygon | MultiPolygon,
                        loc: tuple[float, float]=(0,0)):
        """
        Cuts the given polygon from a layer with an optional location.
        
        Args:
        -----
            - lname (str): The name of the layer.
            - geom (Polygon | MultiPolygon): The polygon to be cut.
            - loc (tuple[float, float], optional): The location where the polygon will be cut.
                Defaults to (0, 0).

        Returns:
        --------
            Updated instance (self) of the class with the cut polygon.
        """
        cut_geom = affinity.translate(geom, xoff = loc[0], yoff = loc[1])
        
        setattr(self, lname, difference(getattr(self, lname), cut_geom))
        return self
    

    def cut_all(self, polygon: Polygon):
        """
        Cuts the specified polygon from polygons in all layers.

        Args:
        -----
            - polygon (Polygon): The polygon used for cutting
        
        Returns:
        --------
            Updated instance (self) of the class with the specified polygon cut from all layers.
        """
        for lname in self.layers:
            self.cut_polygon(lname, polygon)
        return self
    

    def crop_layer(self, lname: str, polygon: Polygon, 
                            loc: tuple[float, float]=(0,0)):
        """
        Crops objects in a layer by a given polygon.

        Args:
        -----
            - lname (str): The name of the layer.
            - polygon (Polygon): The polygon used for cropping.

        Returns:
        --------
            Updated instance (self) of the class with polygons in the specified layer cropped by the given polygon.
        """
        geoms = getattr(self, lname)

        polygon = affinity.translate(polygon, xoff = loc[0], yoff = loc[1]) 

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


    def crop_all(self, polygon: Polygon):
        """
        Crops polygons in all layers.

        Args:
        -----
            - polygon (Polygon): The cropping polygon.
        
        Returns:
        --------
            Updated instance (self) of the class with polygons in all layers cropped by the specified polygon.
        """
        for lname in self.layers:
            self.crop_layer(lname, polygon)
        return self


    def modify_polygon_points(self, lname: str, obj_idx: int, ext_points: dict, int_points: dict=None, int_idx: int=0):
        """
        Updates the point coordinates of an object in a layer.
        Can optionally modify a single interior's coordinates. Moves first and last points together.

        Args:
        -----
            - layer (str): layer name
            - obj_idx (int): polygon index in multipolygon list
            - next_points (dict): point indices to change in a polygon. 
                Keys: corrresponds to the point idx in polygon exterior coord list.
                Value: tuple of new [x,y] coordinates
            - int_points (dict, optional): point indices to change in a polygon. 
                Keys: corrresponds to the point idx in polygon interiors coord list.
                Value: tuple of new [x,y] coordinates
                Defaults to None. 
            - int_idx (int, optional): interior index in the polygon's interiors list. Defaults to 0.
        """
        mpolygon = getattr(self, lname)
        if isinstance(mpolygon, Polygon):
            mpolygon = MultiPolygon([mpolygon])
        polygon_list = list(mpolygon.geoms)
        polygon = polygon_list[obj_idx]
        coords = get_coordinates(polygon)

        #updating exterior coords of the polygon
        if ext_points:
            points_to_be_changed = list(ext_points.keys())
            for point in points_to_be_changed:
                coords[point, 0] = coords[point, 0] + ext_points[point][0] 
                coords[point, 1] = coords[point, 1] + ext_points[point][1]

                if point == (len(polygon.exterior.coords) - 1):
                    coords[0, 0] = coords[0, 0] + ext_points[point][0] 
                    coords[0, 1] = coords[0, 1] + ext_points[point][1]
                elif point == 0:
                    last_point = (len(polygon.exterior.coords) - 1)
                    coords[last_point, 0] = coords[last_point, 0] + ext_points[point][0] 
                    coords[last_point, 1] = coords[last_point, 1] + ext_points[point][1]


        #updating interior coordinates of the polygon
        if int_points and has_interior(polygon) and int_points:
            int_coords = polygon.interiors[int_idx].coords
            int_coords_start = len(polygon.exterior.coords) * (int_idx + 1)

            points_to_be_changed = list(int_points.keys())
            for point in points_to_be_changed:
                point_offset = int_coords_start + point
                
                coords[point_offset, 0] = coords[point_offset, 0] + int_points[point][0] 
                coords[point_offset, 1] = coords[point_offset, 1] + int_points[point][1]

                if point == len(int_coords) - 1:
                    coords[int_coords_start, 0] = coords[int_coords_start, 0] + int_points[point][0] 
                    coords[int_coords_start, 1] = coords[int_coords_start, 1] + int_points[point][1]

        polygon_list[obj_idx] = set_coordinates(polygon, coords)
        setattr(self, lname, MultiPolygon(polygon_list))


    ## just note that it has that length
    def remove_holes_from_polygons(self, lname: str):
        """
        Removes any holes from a multipolygon in a layer by vertically cutting along the centroid of each hole and piecing together the remaining Polygon geometries.
        Vertical cut is made with a default length of 1e6. 

        Args:
        -----
            - lname (str): Name of the layer where polygons with holes are located.
        """
        polygons = getattr(self, lname)
        setattr(self, lname, flatten_multipolygon(polygons))
    

    def remove_polygon(self, lname: str, polygon_id: int | tuple | list):
        """
        Removes a polygon from a layer.

        Args:
        -----
            - lname (str): The name of the layer.
            - polygon_id (int | tuple | list): The index of the polygon to be removed.
        """
        polygons = getattr(self, lname)

        if isinstance(polygons, Polygon):
            raise ValueError("Cannot remove polygon from a single polygon object")

        if isinstance(polygon_id, int):
            polygon_id = [polygon_id]

        poly_list = list(polygons.geoms)
        setattr(self, lname, MultiPolygon([poly for i, poly in enumerate(poly_list) if i not in polygon_id]))


    ###############################
    #### Additional operations ####
    ###############################

    def add_text(self, text: str="abcdef", size: float=1000, loc: tuple=(0,0), layer: str=None):
        """
        Converts text into polygons and adds them to the specified layer.

        Args:
        -----
            - text (str, optional): The text to be converted into polygons. Defaults to "abcdef".
            - size (float, optional): The size of the text. Defaults to 1000.
            - loc (tuple, optional): The location where the text polygons will be placed.
            - layer (str, optional): The name of the layer where the text polygons will be added.
        """
        ptext = polygonize_text(text, size)
        ptext = affinity.translate(ptext, *loc)
        self.add_polygon(layer, ptext)


    ##############################
    #### Exporting operations ####
    ##############################

    def get_zhk_dict(self, flatten_polygon: bool=False) -> dict:
        """ 
        Returns all layer names and their corresponding geometries in a Dictionary.

        Args:
        -----
            - flatten_polygon (bool, optional): Flag to remove holes from polygons. Defaults to False.

        Returns:
        --------
            zhk_dict(dict): A dictionary containing layer names as keys and their corresponding geometries as values.
        """
        lnames = self.layers + ["skeletone", "anchors"]
        zhk_dict = dict.fromkeys(lnames)
        for lname in lnames:
            geometry = getattr(self, lname)
            if (flatten_polygon and (lname not in ["anchors", "skeletone"])):
                zhk_dict[lname] = flatten_multipolygon(geometry)
            else:
                zhk_dict[lname] = geometry
        return zhk_dict


    def export_pickle(self, filename: str) -> None:
        """
        Exports all layers as a pickle file.

        Args:
        -----
            - filename (str): The name of the pickle file to be exported.
        """
        zhkdict = self.get_zhk_dict()
        exp = Exporter_Pickle(filename, zhkdict)
        exp.save()


    def export_gds(self, filename: str, layer_cfg: dict) -> None:
        """
        Exports all layers as a GDS file.

        Args:
        -----
            - filename (str): The name of the gds file to be exported.
            - layer_cfg (dict): A dictionary containing the layer configuration.
                See `gdspy docs <https://gdspy.readthedocs.io/en/stable/gettingstarted.html#layer-and-datatype>`_ for 'datatype' details.
        """
        zhkdict = self.get_zhk_dict(flatten_polygon=True)
        exp = Exporter_GDS(filename, zhkdict, layer_cfg)
        exp.save()


    def export_dxf(self, filename: str, layer_cfg: list) -> None:
        """
        Exports layers as a DXF file.

        Args:
        -----
            - filename (str): The name of the dxf file to be exported.
            - layer_cfg (dict): A list of layer to be exported.
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
            labels: bool=False,
            **kwargs):
        """
        Plots the Entity object on a given axis with specified layers and colors.

        Args:
        -----
            - ax (matplotlib.axes.Axes, optional): The axis to plot on. Defaults to None.
            - layer (list, optional): The layer(s) to plot. Defaults to ["all"].
            - show_idx (bool, optional): Whether to show the id of the polygon. Defaults to False.
            - color (str or list, optional): The color(s) to use for plotting. Defaults to None.
            - alpha (float, optional): The transparency of the plot. Defaults to 1.
            - draw_direction (bool, optional): Whether to draw arrows. Defaults to True.
            - draw_labels (bool, optional): Whether to draw labels on each Point. Defaults to False.
            - **kwargs: Additional keyword arguments to pass to the plot_geometry function.
        """
        if ax is None:
            fig = plt.figure(1, figsize=SIZE_L, dpi=90)
            ax = fig.add_subplot(111)

        for l, c in zip(layer, color):
            if l in self.layers:
                geometry = getattr(self, l)
                if isinstance(c, tuple):
                    (c, alpha) = c
                plot_geometry(geometry,
                              ax=ax,
                              show_idx=show_idx,
                              color=c,
                              alpha=alpha,
                              **kwargs)

        if labels:
            draw_labels(geometry, ax)

    def quickplot(self, color_config: dict=None, zoom: tuple=None,
                  ax=None, show_idx: bool=False, labels: bool=False, 
                  draw_anchor_dir: bool=True, **kwargs) -> None:
        """
        Plots the Entity object with predefined colors for each layer.

        Args:
        -----
            - plot_config (dict): dict of ordered layers (keys) with tuples of the color and alpha of the layer (values)
            - zoom (tuple, optional): ((x0, y0), zoom_scale, aspect_ratio). Defaults to None.

        Returns:
        --------
            ax (matplotlib.axes.Axes): The axis with the plotted Entity object.
        """
        plot_config = tuplify_colors(color_config) if color_config else self.colors.colors

        if "anchors" in plot_config:
            anchor_color = plot_config.pop("anchors")[0]
        else:
            anchor_color = RED

        if "skeletone" in plot_config:
            skeletone_color = plot_config.pop("skeletone")[0]
        else:
            skeletone_color = DARKGRAY

        if ax is None:
            interactive_widget_handler()
            _, ax = plt.subplots(1, 1, figsize=SIZE_L, dpi=90)

        #plot layers
        layer_colors = [plot_config[k][0] for k in plot_config]
        layers = list(plot_config.keys())
        self.plot(ax=ax, layer=layers, color=layer_colors, show_idx=show_idx, labels=labels, **kwargs)

        #plot skeletone
        self.skeletone.plot(ax=ax, color=skeletone_color)

        #plot anchors
        self.anchors.plot(ax=ax, color=anchor_color, draw_direction=draw_anchor_dir)

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
    """ 
    Represents a structure that contains layers with a collection of geometries (Points, LineStrings, Polygons, etc.).
    Inherits from the Entity class.
    
    Methods:
    --------
        - append(structure, anchoring=None, direction_snap=False, remove_anchor=False, upd_alabels=None):
            Appends an Entity or Structure to the Structure.
        - return_mirrored(aroundaxis, **kwargs): Returns a mirrored copy of the Structure class.
    """

    def __init__(self):
        super().__init__()

    def append(self,
               structure: Entity,
               anchoring: tuple=None,
               direction_snap: bool=False,
               remove_anchor: bool | str=False,
               upd_alabels: list[tuple]=None) -> None:
        """
        Appends an Entity or Structure to the Structure.

        Args:
        -----
            - structure (Entity): Entity or Structure with a collection of geometries 
            - anchoring (list, optional): 
                List of points to snap the appending object to the existing structure. 
                [StructureObj Point, AppendingObj Point]
                Defaults to None.
            - direction_snap (bool, optional): 
                If True, aligns the direction of the appending object with the direction of the anchor points. 
                Defaults to False.
            - remove_anchor (bool or str, optional): 
                If True, removes the anchor points after appending. 
                If a string is provided, removes the specified anchor point. 
                Defaults to False.
            - upd_alabels (list, optional):
                Renames anchor labels of the appending structure before appending.
                A list of tuples with the old and new anchor labels: (old_label, new_label) 
                Defaults to None.
        """
        s = structure.copy()
        attr_list_device = self.layers
        attr_list_structure = s.layers
        self.layers = list(set(attr_list_device + attr_list_structure))

        self.colors.colors = self.colors.colors | s.colors.colors

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

        # appending polygons
        for a in self.layers:
            if not hasattr(self, a):
                value = self._append_geometry(MultiPolygon(), getattr(s, a))
            elif not hasattr(s, a):
                value = self._append_geometry(getattr(self, a), MultiPolygon())
            else:
                value = self._append_geometry(getattr(self, a), getattr(s, a))
            setattr(self, a, value)

        # appending skeletones
        self.skeletone.add_line(s.skeletone.lines, ignore_crossing=True, chaining=False)

        # updating anchor labels in the appending structure
        if upd_alabels:
            for label_old, label_new in upd_alabels:
                s.anchors.modify(label_old, new_name=label_new)

        # remove or not to remove anchor after appending
        if remove_anchor is True:
            self.remove_anchor(anchoring[0])
            s.remove_anchor(anchoring[1])
        elif isinstance(remove_anchor, str):
            self.remove_anchor(remove_anchor)
            s.remove_anchor(remove_anchor)
        self.add_anchor(s.anchors.multipoint)

        return self


    # def _combine_objects(self,
    #                      obj1: Polygon| MultiPolygon | None,
    #                      obj2: Polygon| MultiPolygon | None):
    #     """ Merge two geometries and return the result

    #     Args:
    #     ----
    #     obj1 (Polygon | MultiPolygon | None): first geometry.
    #     obj2 (Polygon | MultiPolygon | None): second geometry.

    #     Raises:
    #     ------
    #     TypeError: Raised if the appending object
    #                is not [Polygon, MultiPolygon].
    #     ValueError: Raised if error with merging the geometries.
    #                 Call _errors to inspect the problem.
    #     """
    #     merged = MultiPolygon()
    #     if obj1:
    #         merged = self._append_geometry(merged, obj1)
    #     if obj2:
    #         merged = self._append_geometry(merged, obj2)
    #     return merged


    def _append_geometry(self, core_objs, appending_objs):
        """ 
        Appends single or multiple shapely geometries.

        Args:
        -----
            - core_objs: shapely geometries to be appended.
            - appending_objs: shapely geometries to append.

        Returns:
        --------
            Multi-Geometry: Union of all the geometries.

        Note:
        -----
            This method works with LineString, Polygon, and multi-geometries.
        """
        geom_list = create_list_geoms(core_objs) + create_list_geoms(appending_objs)
        return unary_union(geom_list)


    def return_mirrored(self, aroundaxis: str, **kwargs) -> 'Structure':
        """
        Returns a mirrored copy of the Structure class.

        Args:
        -----
            - aroundaxis (str): The axis around which to mirror the class. Valid values are 'x' or 'y'.
            - **kwargs: Additional keyword arguments.

        Returns:
        --------
            Structure: A mirrored copy of the Structure class.
        """
        cc = self.copy()
        return cc.mirror(aroundaxis, **kwargs)


class GeomCollection(Structure):
    """
    Represents a collection of geometries.
    Class attributes are created by layers dictionary.
    
    Attributes:
    -----------
        - layers (dict): Dictionary containing the layers and corresponding polygons/skeletone/anchors.
    """
    def __init__(self, layers: dict=None):
        super().__init__()
        if layers:
            for items in layers.items():
                match items:
                    case ("skeletone", LineString()) | ("skeletone", MultiLineString()):
                        self.skeletone.lines = items[1]
                    case ("skeletone", Skeletone()):
                        self.skeletone = items[1]
                    case ("skeletone", GeometryCollection()):
                        warn(message="imported skeletone contains GeometryCollection object. It will be ignored.")
                    case ("anchors", MultiAnchor()):
                        self.anchors = items[1]
                    case ("anchors", MultiPoint()):
                        for i, pt in enumerate(items[1].geoms):
                            self.anchors.add(Anchor(pt, 0, "anchor" + str(i)))
                    case _:
                        self.layers.append(items[0])
                        setattr(self, *items)

        if not hasattr(self, "anchors"):
            self.anchors = MultiAnchor()

        if not hasattr(self, "skeletone"):
            self.skeletone = Skeletone()

        self.colors.update_colors(self.layers)