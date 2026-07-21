"""
core.py

This file contains the core classes and methods for the ZeroHeliumKit library.

Classes:
    `Entity`: A class which represents a collection of shapely objects linked together and provides methods for geometrical operations.
    `Structure`: A subclass of Entity that represents layers with collections of geometries (Points, LineStrings, Polygons, etc.).
    `GeomCollection`: A subclass of Structure that represents a collection of geometries.
"""

import copy
import matplotlib.pyplot as plt
from warnings import warn

from shapely import Point, LineString, Polygon, MultiPolygon

from .plotting import interactive_widget_handler, listify_colors, ColorHandler
from .importing import Exporter_DXF, Exporter_GDS, Exporter_Pickle
from .settings import SIZE, SIZE_L, SIZE_S, RED, DARKGRAY
from .anchors import Anchor, MultiAnchor, Skeletone, Layer, get_dxdy
from .errors import hard_deprecated


class Entity():
    """ 
    Represents collections of shapely objects organized by layers and linked together.
    This class provides methods to add, remove, rename, and manipulate layers containing geometric objects
    (such as polygons and multipolygons), as well as to handle Anchors and Skeletone. It supports various
    geometric operations including union, difference, intersection, simplification, cropping, and modification
    of polygon points.

    Args:
        layers: List of layer names.
        colors: ColorHandler class, which handles info about colors of layers and their order.
        errors: Contains error messages. 
        skeletone (Skeletone): Represents a collection of lines linked to the Entity,
            which is an instance of the Skeletone class.
        anchors (MultiAnchor): Represents the anchor points of the Entity, which are instances of MultiAnchor.

    """
    layers = []
    skeletone = Skeletone()
    anchors = MultiAnchor()
    colors = ColorHandler({})
    errors = None

    def __init__(self):
        """
        Initializes a new Entity class instance.
        """
        self.layers = []
        self.skeletone = Skeletone()
        self.anchors = MultiAnchor()
        # self.colors = ColorHandler({})
        self.errors = None


    def __repr__(self):
        class_name = self.__class__.__name__
        repr_name = f"{class_name} {tuple(self.layers)}"
        max_length = 75
        if len(repr_name) > max_length:
            return f"{repr_name[: max_length - 3]}..."
        return repr_name
    

    def get(self, lname: str) -> Layer | None:
        """
        Retrieves a layer by its name.

        Args:
            lname (str): The name of the layer.

        Returns:
            Layer: The layer with the specified name, or None if not found.
        """
        if lname in self.layers:
            return getattr(self, lname, None)
        else:
            warn(f"Layer '{lname}' not found in layers.")
            return None


    def add(self, layer: Layer): 
        """
        Adds a layer to the class with the given name and geometry.

        Args:
            layer (Layer): The layer to be added.

        Returns:
            Updated instance (self) of the class with the new layer added.
        """
        self.layers.append(layer.name)
        # self.colors.add_color(layer.name, layer.color[0], layer.color[1])
        setattr(self, layer.name, layer)
        return self


    def remove(self, lname: str):
        """
        Removes a layer from the class.

        Args:
            lname (str): The name of the layer.

        Returns:
            Updated instance (self) of the class with the layer removed.
        """
        if lname in self.layers:
            self.layers.remove(lname)
            delattr(self, lname)
            # self.colors.remove_color(lname)
        else:
            print(f"Layer '{lname}' not found in layers.")

        return self


    def rename(self, old: str, new: str) -> None:
        """
        Changes the name of a layer/attribute in the class.

        Args:
            old (str): The old name.
            new (str): The new name.

        Returns:
            Updated instance (self) of the class with the layer renamed.
        """
        if old in self.layers:
            self.__dict__[new] = self.__dict__.pop(old)
            self.layers[self.layers.index(old)] = new
            self.__dict__[new].name = new
            # self.colors.rename_color(old, new)
        else:
            print(f"Layer '{old}' not found in layers.")

        return self


    def has_layer(self, name: str) -> bool:
        """ Check if a layer exists in the class.

        Args:
            name (str): The name of the layer.

        Returns:
            bool: True if the layer exists, False otherwise.
        """
        return name in self.layers
    

    def cut(self, geom: Polygon | MultiPolygon, loc: tuple[float, float]=None, ignore: list[str]=[]):
        """
        Cuts the specified polygon from polygons in all layers.

        Args:
            polygon (Polygon | MultiPolygon): The polygon used for cutting

        Returns:
            Updated instance (self) of the class with the specified polygon cut from all layers.
        """
        for lname in self.layers:
            if lname not in ignore:
                getattr(self, lname).cut(geom, loc)
        return self


    def crop(self, geom: Polygon | MultiPolygon, loc: tuple[float, float]=None, ignore: list[str]=[]):
        """
        Crops polygons in all layers.

        Args:
            geom (Polygon | MultiPolygon): The cropping polygon.

        Returns:
            Updated instance (self) of the class with polygons in all layers cropped by the specified polygon.
        """
        for lname in self.layers:
            if lname not in ignore:
                getattr(self, lname).crop(geom, loc)
        return self


    def slice(self, slice_line: LineString | list[LineString], ignore: list[str]=[]):
        """
        Slices polygons in a layer using a given line.

        Args:
            lname (str | list[str]): The name of the layer.
            slice_line (LineString): The line used for slicing.
        """
        for lname in self.layers:
            if lname not in ignore:
                getattr(self, lname).slice(slice_line)
        return self


    def copy(self, rename_anchors: bool=False, with_suffix: str = "_copy") -> 'Entity':
        """
        Creates a deep copy of the Entity instance.
        Optionally updates the labels of the anchors with a specified suffix.

        Args:
            rename_anchors (bool, optional): Whether to rename anchor labels in the copied instance. Defaults to False.
            with_suffix (str, optional): Suffix to append to anchor labels in the copied instance.

        Returns:
            Entity: A new instance of Entity with the same layers, skeletone, and anchors
        """
        new_instance = copy.deepcopy(self)
        if rename_anchors:
            new_instance.anchors = self.anchors.copy(upd_labels_with_suffix=with_suffix)
        return new_instance


    def clean(self):
        """
        Removes all layers with empty polygons  
        """
        empty_layers = []
        for lname in self.layers:
            if getattr(self, lname).is_empty:
                empty_layers.append(lname)
                delattr(self, lname)
        self.layers = [lname for lname in self.layers if lname not in empty_layers]


    ################################
    #### Geometrical operations ####
    ################################

    def rotate(self, angle: float=0, origin=(0,0)):
        """ 
        Rotates all objects in the class

        Args:
            angle (float, optional): rotation angle. Defaults to 0.
            origin (str, optional): rotations are made around this point. Defaults to (0,0).
        
        Returns:
            Updated instance (self) of the class with all objects rotated.
        """

        for l in self.layers:
            getattr(self, l).rotate(angle, origin)
        self.skeletone.rotate(angle, origin)
        self.anchors.rotate(angle, origin)
        return self


    def move(self, dx: float, dy: float):
        """
        Moves all objects in the class by the specified (dx, dy) offset.

        Args:
            dx (float): The x-offset to move by.
            dy (float): The y-offset to move by.

        Returns:
            Updated instance (self) of the class with all objects moved.
        """
        for lname in self.layers:
            getattr(self, lname).move(dx, dy)
        self.skeletone.move(dx, dy)
        self.anchors.move(dx, dy)
        return self


    def snap_to(
            self,
            point_from: tuple | Point | Anchor,
            point_to: tuple | Point | Anchor
            ):
        """
        Snaps all objects in the class so that point_from aligns with point_to.

        Args:
            point_from (tuple | Point | Anchor): The point to be moved.
            point_to (tuple | Point | Anchor): The target point to snap to.
        
        Returns:
            Updated instance (self) of the class with all objects snapped.
        """
        dxdy = get_dxdy(point_from, point_to)
        self.move(*dxdy)
        return self


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)):
        """
        Scales all objects by the specified factors along the x and y axes.

        Args:
            xfact (float, optional): scale along x-axis. Defaults to 1.0.
            yfact (float, optional): scale along y-axis. Defaults to 1.0.
            origin ((x,y), optional): scale with respect to an origin (x,y). Defaults to (0,0).

        Returns:
            Updated instance (self) of the class with all objects scaled.
        """

        for l in self.layers:
            getattr(self, l).scale(xfact, yfact, origin)

        self.skeletone.scale(xfact, yfact, origin)
        self.anchors.scale(xfact, yfact, origin)
        return self


    def mirror(self, aroundaxis: str, keep_original: bool=True, update_labels: bool=True):
        """
        Mirror all objects around a specified axis.

        Args:
            aroundaxis (str): Defines the mirror axis. Only 'x' or 'y' are supported.
            update_labels (bool, optional): 
                Whether to update the labels after mirroring. Defaults to False.
            keep_original (bool, optional):
                Whether to keep the original objects after mirroring. Defaults to False.

        Raises:
            TypeError: If the 'aroundaxis' is not 'x' or 'y'.

        Returns:
            Updated instance (self) of the class with all objects mirrored.
        """
        for lname in self.layers:
            getattr(self, lname).mirror(aroundaxis, keep_original)
        self.skeletone.mirror(aroundaxis, keep_original)
        self.anchors.mirror(aroundaxis, update_labels, keep_original)
        return self


    ##############################
    #### Exporting operations ####
    ##############################

    def as_dict(self, remove_holes: bool=False, include_anchors_skeletone: bool=True) -> dict:
        """ 
        Returns all layer names and their corresponding geometries in a Dictionary. Includes anchors and skeletone.

        Args:
            remove_holes (bool, optional): Flag to remove holes from polygons. Defaults to False.

        Returns:
            zhk_dict(dict): A dictionary containing layer names as keys and their corresponding geometries as values.
        """
        lnames = self.layers + ["skeletone", "anchors"] if include_anchors_skeletone else self.layers
        edict = dict.fromkeys(lnames)
        for lname in lnames:
            layer = getattr(self, lname)
            if (remove_holes and (lname not in ["anchors", "skeletone"])):
                layer.remove_holes()
            edict[lname] = layer
        return edict


    def export_pickle(self, filename: str) -> None:
        """
        Exports all layers as a pickle file.

        Args:
            filename (str): The name of the pickle file to be exported.
        """
        zhkdict = self.as_dict()
        zhkdict["colors"] = self.colors
        exp = Exporter_Pickle(filename, zhkdict)
        exp.save()


    def export_gds(self, filename: str, cellname: str="toplevel") -> None:
        """
        Exports all layers as a GDS file.

        Args:
            filename (str): The name of the gds file to be exported.
            layer_cfg (dict): A dictionary containing the layer configuration.
                See `gdspy docs <https://gdspy.readthedocs.io/en/stable/gettingstarted.html#layer-and-datatype>`_ for 'datatype' details.
        """
        zhkdict = self.as_dict(remove_holes=True, include_anchors_skeletone=False)
        exp = Exporter_GDS(filename, zhkdict, cellname)
        exp.save()


    def export_dxf(self, filename: str, layer_cfg: list) -> None:
        """
        Exports layers as a DXF file.

        Args:
            filename (str): The name of the dxf file to be exported.
            layer_cfg (dict): A list of layer to be exported.
        """
        zhkdict = self.as_dict(remove_holes=True, include_anchors_skeletone=False)
        exp = Exporter_DXF(filename, zhkdict, layer_cfg)
        exp.save()


    #############################
    #### Plotting operations ####
    #############################

    def quickplot(
            self,
            size="large",
            color_config: dict=None,
            zoom: tuple=None,
            show_idx: bool=False,
            off: list=[],
            labels: bool=False, 
            draw_anchor_dir: bool=True,
            ax=None,
            **kwargs
            ) -> None:
        """
        Plots the Entity object with predefined colors for each layer.

        Args:
            plot_config (dict): dict of ordered layers (keys) with tuples of the color and alpha of the layer (values)
            zoom (tuple, optional): ((x0, y0), zoom_scale, aspect_ratio). Defaults to None.

        Returns:
            ax (matplotlib.axes.Axes): The axis with the plotted Entity object.
        """
        plot_config = listify_colors(color_config) if color_config else self.colors.colors
        if "anchors" in plot_config:
            anchor_color = plot_config.pop("anchors")[0]
        else:
            anchor_color = RED

        if "skeletone" in plot_config:
            skeletone_color = plot_config.pop("skeletone")[0]
        else:
            skeletone_color = DARKGRAY
        
        match size:
            case "small":
                FIGSIZE = SIZE_S
            case "medium":
                FIGSIZE = SIZE
            case "large":
                FIGSIZE = SIZE_L
            case _:
                FIGSIZE = SIZE_L

        if ax is None:
            interactive_widget_handler()
            _, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=90)

        #plot layers
        plot_config = {k:v for k,v in plot_config.items() if k not in off}
        for lname, lcolor in plot_config.items():
            if self.has_layer(lname) and (not getattr(self, lname).is_empty):
                getattr(self, lname).color = lcolor
                getattr(self, lname).plot(ax=ax, show_idx=show_idx, labels=labels, **kwargs)

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
    The Structure class provides methods to append other entities or structures. 
    Inherits from the Entity class.
    """

    def __init__(self):
        super().__init__()

    def append(self,
               structure: Entity,
               anchoring: tuple=None,
               direction_snap: bool=False,
               remove_anchor: bool | str=False,
               upd_alabels: list[tuple]=None,
               move_s: tuple=None,
               rotate_s: float=None) -> None:
        """
        Appends an Entity or Structure to the Structure.

        Args:
            structure (Entity): Entity or Structure with a collection of geometries 
            anchoring (list, optional): 
                List of points to snap the appending object to the existing structure. 
                [StructureObj Point, AppendingObj Point]
                Defaults to None.
            direction_snap (bool, optional): 
                If True, aligns the direction of the appending object with the direction of the anchor points. 
                Defaults to False.
            remove_anchor (bool or str, optional): 
                If True, removes the anchor points after appending. 
                If a string is provided, removes the specified anchor point. 
                Defaults to False.
            upd_alabels (list, optional):
                Renames anchor labels of the appending structure before appending.
                A list of tuples with the old and new anchor labels: (old_label, new_label) 
                Defaults to None.
        """
        s = structure.copy()
        if rotate_s:
            s.rotate(rotate_s, origin=(0,0))
        if move_s:
            s.move(*move_s)
        attr_list_device = self.layers
        attr_list_structure = s.layers
        self.layers = list(set(attr_list_device + attr_list_structure))

        # self.colors.colors = self.colors.colors | s.colors.colors

        # snapping direction
        if direction_snap:
            angle = - s.anchors[anchoring[1]].direction + self.anchors[anchoring[0]].direction
            s.rotate(angle, origin=(0, 0))

        # snapping anchors
        if anchoring:
            c_point = self.anchors[anchoring[0]]
            a_point = s.anchors[anchoring[1]]
            dxdy = (c_point.x - a_point.x, c_point.y - a_point.y)
            s.move(*dxdy)

        # appending polygons
        for lname in self.layers:
            if not hasattr(self, lname):
                layer = getattr(s, lname)
            elif not hasattr(s, lname):
                layer = getattr(self, lname)
            else:
                layer = getattr(self, lname)
                layer.add(getattr(s, lname).polygons)
            setattr(self, lname, layer)

        # appending skeletones
        self.skeletone.add(s.skeletone.lines, ignore_crossing=True, chaining=False)

        # updating anchor labels in the appending structure
        if upd_alabels:
            for label_old, label_new in upd_alabels:
                s.anchors.modify(label_old, new_name=label_new)

        # remove or not to remove anchor after appending
        if remove_anchor is True:
            self.anchors.remove(anchoring[0])
            s.anchors.remove(anchoring[1])
        elif isinstance(remove_anchor, str):
            self.anchors.remove(remove_anchor)
            s.anchors.remove(remove_anchor)
        self.anchors.add(s.anchors.multipoint)
        del s

        return self


    def return_mirrored(self, aroundaxis: str, **kwargs) -> 'Structure':
        """
        Returns a mirrored copy of the Structure class.

        Args:
            aroundaxis (str): The axis around which to mirror the class. Valid values are 'x' or 'y'.
            **kwargs: Additional keyword arguments.

        Returns:
            Structure: A mirrored copy of the Structure class.
        """
        cc = self.copy()
        return cc.mirror(aroundaxis, **kwargs)
