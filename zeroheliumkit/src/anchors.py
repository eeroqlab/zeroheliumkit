"""
anchors.py

This file contains the Anchor, MultiAnchor, Skeletone, and Layer classes for ZeroHeliumKit.
Uses the Shapely library for geometric operations and the Tabulate library for tabular representation.

Classes:
-------
    `Anchor`: Represents a single anchor point with coordinates, orientation, and labeling.
        Contains methods for manipulating and visualizing an anchor point.
    `MultiAnchor`: Represents a collection of anchor points.
        Contains methods for manipulating and visualizing multiple anchor points.
    `Skeletone`: Represents a collection of connected lines for routing and wireframe geometry.
        Provides methods for creating and manipulating these paths.
    `Layer`: Represents a layer containing polygons with attributes such as name, color, and grid snapping.
"""
from __future__ import annotations

import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import Self
from tabulate import tabulate
from shapely import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely import (affinity, unary_union,
                     set_precision, distance,
                     set_coordinates, get_coordinates)
from shapely.plotting import plot_line, plot_polygon
from shapely.ops import linemerge
from matplotlib.axes import Axes as mpl_axes

from functools import wraps

from .utils import fmodnew, append_line, has_interior, flatten_multipolygon, split_polygon, polygonize_text
from .settings import GRID_SIZE, BLACK, RED, DARKGRAY
from .plotting import default_ax, plot_polygon_idx, plot_line_idx_in_polygon, draw_labels


def snap_on_grid(
    *,
    attr: str,
    grid_size = GRID_SIZE,
    mode = "pointwise",
    enabled_attr: str = "enable_grid_snap",
    return_self: bool = True
    ):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            updated = method(self, *args, **kwargs)

            # Allow in-place mutation methods
            if updated is None:
                updated = getattr(self, attr)

            if getattr(self, enabled_attr):
                updated = set_precision(updated, grid_size=grid_size, mode=mode)

            setattr(self, attr, updated)

            return self if return_self else getattr(self, attr)
        return wrapper
    return decorator


class Anchor():
    """
    Represents a single anchor point with attributes such as coordinates, orientation/direction, and label. 
    Provides methods for manipulating and visualizing the anchor point.

    Args:
        point (tuple | Point): The coordinates of the anchor point.
        direction (float): The orientation/direction of the anchor point.
        label (str): The label of the anchor point.

    Example:
        >>> from zeroheliumkit import Anchor
        >>> anchor = Anchor((0, 0), 45, "A")
        >>> print(anchor)
        <ANCHOR (POINT (0 0), 45.0, A)>
        >>> anchor.properties  # prints the properties of the anchor
        -----  ----------  ---------
        label  coords      direction
        A      (0.0, 0.0)  45.0
        -----  ----------  ---------
        >>> anchor.x
        0.0
        >>> anchor.y
        0.0
        >>> anchor.coords
        (0.0, 0.0)
        >>> anchor.direction
        45.0
        >>> anchor.label
        'A'
    """

    __slots__ = "point", "direction", "label"

    def __init__(self, point: Point | tuple=Point(), direction: float=0, label: str=None):
        if not isinstance(point, Point):
            coords = np.array(point).squeeze()
            self.point = set_precision(Point(coords), grid_size=GRID_SIZE)
        else:
            self.point = set_precision(point, grid_size=GRID_SIZE)
        self.direction = fmodnew(direction)
        self.label = label

    def __repr__(self):
        return f"<ANCHOR ({self.point}, {self.direction}, {self.label})>"

    @property
    def x(self):
        """The x-coordinate of the anchor point."""
        return self.point.x

    @property
    def y(self):
        """The y-coordinate of the anchor point."""
        return self.point.y

    @property
    def coords(self):
        """The (x, y) coordinates of the anchor point as a tuple."""
        return (self.point.x, self.point.y)

    @coords.setter
    def coords(self, xy: tuple):
        self.point = set_precision(Point(*xy), grid_size=GRID_SIZE)

    @property
    def properties(self):
        """The attributes of the anchor in a tabular format."""
        print(tabulate([["label", "coords", "direction"],
                        [self.label, self.coords, self.direction]]))


    def rename(self, newlabel: str) -> None:
        """
        Renames the anchor with a new label.

        Args:
            newlabel (str): The new label for the anchor.
        """
        self.label = newlabel


    def rotate(self, angle: float, origin: tuple=(0,0)) -> 'Anchor':
        """ 
        Rotates the point and arrow direction by the specified angle around the given origin.

        Args:
            angle (float): The angle of rotation in degrees.
            origin (tuple, optional): The origin point of rotation. Defaults to (0, 0).

        Returns:
            Updated instance (self) of the class with the rotated point and direction.
        """
        point_upd = affinity.rotate(self.point, angle, origin)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)
        if self.direction is not None:
            self.direction += angle

        self.direction = fmodnew(self.direction)

        return self


    def rotate_dir(self, angle: float) -> 'Anchor':
        """ 
        Rotates the direction of the only the anchor by the specified angle.

        Args:
            angle (float): The angle (in radians) by which to rotate the direction.

        Returns:
            Updated instance (self) of the class with the rotated direction.
        """
        self.direction = fmodnew(self.direction + angle)

        return self


    def move(self, xoff: float=0, yoff: float=0) -> 'Anchor':
        """ 
        Moves the anchor point by the specified offset in the x and y directions.

        Args:
            xoff (float): The offset in the x direction. Default is 0.
            yoff (float): The offset in the y direction. Default is 0.

        Returns:
            Updated instance (self) of the class with the moved point.
        """
        point_upd = affinity.translate(self.point, xoff=xoff, yoff=yoff)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)

        return self


    def offset(self, distance: float, angle: float) -> 'Anchor':
        """ 
        Moves the anchor point by the specified distance and angle.

        Args:
            distance (float): The distance to move the anchor point.
            angle (float): The angle in degrees at which to move the anchor point.

        Returns:
            Updated instance (self) of the class with the moved anchor point.
        """
        xoff = distance * np.cos(angle * np.pi / 180)
        yoff = distance * np.sin(angle * np.pi / 180)
        return self.move(xoff, yoff)


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)) -> 'Anchor':
        """ 
        Scales the anchor point by the given factors along the x and y axes.

        Args:
            xfact (float): The scaling factor along the x-axis. Default is 1.0.
            yfact (float): The scaling factor along the y-axis. Default is 1.0.
            origin (tuple): The origin point for scaling. Default is (0, 0).

        Returns:
            Updated instance (self) of the class with the scaled point and direction.
        """
        point_upd = affinity.scale(self.point, xfact=xfact, yfact=yfact, origin=origin)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)
        if self.direction is not None:
            if yfact < 0:
                #rotation around x-axis
                self.direction = (-1) * self.direction

            if xfact < 0:
                #rotation around y_axis
                self.direction = 180 - self.direction

        return self


    def mirror(self, aroundaxis: str=None, update_label: str=None) -> 'Anchor':
        """ 
        Mirrors the anchor object around the specified axis.

        Args:
            aroundaxis (str): The axis to mirror the anchor around.
            update_label (str): The updated label for the mirrored anchor.

        Raises:
            ValueError: If the `aroundaxis` parameter is not 'x', 'y', or None.

        Returns:
            Updated instance (self) of the class with the mirrored point and direction.
        """
        if aroundaxis=='x':
            self.scale(1, -1)
        elif aroundaxis=='y':
            self.scale(-1, 1)
        elif aroundaxis not in ['x', 'y', None]:
            raise ValueError("choose 'x', 'y or None for mirror arguments")

        if aroundaxis and update_label:
            self.label = update_label

        return self


    def plot(self, ax=None, color: str=None, draw_direction: bool=True) -> None:
        """ 
        Plots the anchor point on the given axes.

        Args:
            ax (optional): The axes on which to plot the anchor point.
                Defaults to the default axes.
            color (str, optional): The color of the anchor point and annotation box edge.
                Defaults to the default color.
            draw_direction (bool, optional): Whether to draw an arrow indicating the direction of the anchor.
                Defaults to True.
        """
        if ax is None:
            ax = default_ax()

        coords = (self.point.x, self.point.y)

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        if self.direction is not None and draw_direction:
            arrow_length = min(np.abs(x2-x1), np.abs(y2-y1)) * 0.08
            arrow_kwargs = {"arrowstyle": '->', "linewidth": 2, "color": RED}
            ax.annotate('',
                        xy=(coords[0] + arrow_length*np.cos(self.direction*np.pi/180),
                            coords[1] + arrow_length*np.sin(self.direction*np.pi/180)),
                        xytext=(coords[0], coords[1]),
                        arrowprops=arrow_kwargs)
        ax.annotate(self.label,
                    coords,
                    color=BLACK,
                    clip_on=True,
                    bbox={"facecolor": 'white',
                            "edgecolor": color,
                            "boxstyle": 'round',
                            "alpha": 0.7})
        ax.plot(coords[0], coords[1], linestyle="", marker=".", color=color, alpha=1, zorder=1e9)


    def distance_to(self, point: tuple | Point) -> float:
        """
        Calculates the unitless distance between this anchor and another point.

        Args:
            point (tuple | Point): The other point to calculate the distance to.

        Returns:
            float: The distance between this anchor and a given point.
        """
        if isinstance(point, tuple):
            point = Point(point)
        return distance(self.point, point)


class MultiAnchor():
    """ 
    Represents a collection of multiple anchor points.
    Provides methods for manipulating and visualizing multiple anchor points.

    Args:
        multipoint (list): A list of Anchor objects.

    Example:
        >>> from zeroheliumkit import MultiAnchor, Anchor
        >>> anchor1 = Anchor((0, 0), 0, "A")
        >>> anchor2 = Anchor((1, 1), 45, "B")
        >>> ma = MultiAnchor([anchor1, anchor2])
        >>> print(ma)
        <MULTIANCHOR ['A', 'B']>
        >>> ma.labels
        ['A', 'B']
        >>> ma.label_exist('A')
        True
        >>> ma.label_exist('C')
        False
        >>> ma["B]
        <ANCHOR (POINT (1 1), 45.0, B)>
        >>> ma.add(Anchor((2, 2), 90, "C"))
        >>> ma.remove("A")
        >>> print(ma)
        <MULTIANCHOR ['B', 'C']>
    """

    __slots__ = "multipoint"

    def __init__(self, multipoint: list=None):
        if multipoint is None:
            self.multipoint = []
            """The list of Anchor objects."""
        else:
            self.multipoint = multipoint
            """The list of Anchor objects."""

    def __repr__(self):
        name = f"<MULTIANCHOR {self.labels}>"
        max_length = 75
        if len(name) > max_length:
            return f"{name[: max_length - 3]}...>"

        return name


    @property
    def labels(self) -> list:
        """The list of labels of the anchors in the MultiAnchor."""
        return [p.label for p in self.multipoint]


    def __getitem__(self, label: str | int | list[str]) -> Anchor | list[Anchor]:
        """ 
        Returns an anchor or list of anchors by label or index.

        Args:
            label (str | int): The label or index of the anchor(s) to retrieve.

        Raises:
            TypeError: If the label is not a string or an integer.

        Returns:
            Anchor or list of Anchor objects.
        """
        if isinstance(label, int):
            return self.multipoint[label]
        elif isinstance(label, str):
            idx = self.labels.index(label)
            return self.multipoint[idx]
        elif isinstance(label, (list, tuple)):
            idx_list = [self.labels.index(l) for l in label if l in self.labels]
            return [self.multipoint[idx] for idx in idx_list]
        else:
            raise TypeError("label must be a string or an integer")
        

    def has_label(self, label: str | list) -> bool:
        """ 
        Checks if a label exists in the list of labels.

        Args:
            label (str): The label to check.

        Returns:
            bool: True if the label exists, False otherwise.
        """
        if isinstance(label, str):
            label = [label]
        label_set = set(label)

        return label_set.issubset(set(self.labels))


    def copy(self, upd_labels_with_suffix: str = None) -> 'MultiAnchor':
        """
        Creates a deep copy of the MultiAnchor instance.
        Optionally updates the labels of the anchors with a specified suffix.

        Args:
            upd_labels_with_suffix (str): The suffix to append to each label.

        Returns:
            MultiAnchor: A new instance of MultiAnchor with the same multipoint anchors
        """
        new_instance = copy.deepcopy(self)
        if upd_labels_with_suffix:
            for p in new_instance.multipoint:
                p.label = p.label + upd_labels_with_suffix
        return new_instance


    def rotate(self, angle: float, origin: tuple=(0,0)) -> 'MultiAnchor':
        """ 
        Rotates all anchors by a given angle around a specified origin point.

        Args:
            angle (float): The angle of rotation in degrees.
            origin (tuple, optional): The origin point of rotation. Defaults to (0, 0).

        Returns:
            MultiAnchor: Updated instance (self) of the class with the rotated anchors.
        """
        if self.multipoint:
            for p in self.multipoint:
                p.rotate(angle, origin)

        return self


    def move(self, xoff: float=0, yoff: float=0) -> 'MultiAnchor':
        """ 
        Moves the anchors by the specified offsets.

        Args:
            xoff (float): The horizontal offset to move the anchors by. Default is 0.
            yoff (float): The vertical offset to move the anchors by. Default is 0.

        Returns:
            MultiAnchor: Updated instance (self) of the class with the moved anchors.
        """
        for p in self.multipoint:
            p.move(xoff, yoff)

        return self


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)) -> 'MultiAnchor':
        """ 
        Scales the multipoint by the given factors along the x and y axes.

        Args:
            xfact (float): The scaling factor along the x-axis. Default is 1.0.
            yfact (float): The scaling factor along the y-axis. Default is 1.0.
            origin (tuple): The origin point for scaling. Default is (0, 0).

        Returns:
            MultiAnchor: Updated instance (self) of the class with the scaled anchors.
        """
        for p in self.multipoint:
            p.scale(xfact, yfact, origin)

        return self


    def mirror(self, aroundaxis: str=None, update_labels: bool=True, keep_original: bool=False) -> 'MultiAnchor':
        """ Mirrors the multipoint anchors around a specified axis.

        Args:
            aroundaxis (str): The axis around which to mirror the anchors.
            update_labels (bool): Whether to update the labels of the mirrored anchors.
            keep_original (bool): Whether to keep the original anchors.

        Returns:
            MultiAnchor: Updated instance (self) of the class with the mirrored anchors.
        """
        if not keep_original:
            if not update_labels:
                for p in self.multipoint:
                    p.mirror(aroundaxis)
            else:
                for p in self.multipoint:
                    p.mirror(aroundaxis, p.label + "_m")
        else:
            original = copy.deepcopy(self.multipoint)
            for p in self.multipoint:
                p.mirror(aroundaxis, p.label + "_m")
            self.multipoint = self.multipoint + original

        return self


    def remove(self, *args: str) -> 'MultiAnchor':
        """ 
        Removes the specified anchors from the multipoint.

        Args:
            args (str): The labels of the anchors to be removed. If no arguments are provided, all anchors will be removed.

        Returns:
            MultiAnchor: Updated instance (self) of the class with the specified anchors removed.
        """
        if not args:
            self.multipoint = []
            return self

        S1 = set(self[args])
        S2 = set(self.multipoint)
        self.multipoint = list(S2.difference(S1))

        return self


    def modify(self, label: str, new_name: str=None, new_xy: tuple=None, new_direction: float | int=None) -> 'MultiAnchor':
        """ 
        Modifies the properties of an anchor.

        Args:
            label (str): The anchor to modify.
            new_name (str, optional): The new name for the anchor. Defaults to None.
            new_xy (tuple, optional): The new coordinates (x, y) for the anchor. Defaults to None.
            new_direction (float | int, optional): The new direction for the anchor. Defaults to None.

        Returns:
            MultiAnchor: Updated instance (self) of the class with the modified anchor.
        """
        if new_xy:
            self[label].coords = new_xy
        if isinstance(new_direction, (float, int)):
            self[label].direction = new_direction
        if new_name:
            self[label].label = new_name

        return self


    def add(self, points: list[Anchor] | Anchor=[]) -> 'MultiAnchor':
        """ 
        Adds one or more Anchor objects to the MultiAnchor.

        Args:
            points (list[Anchor] | Anchor, optional): Anchor object(s) to be added. Defaults to an empty list.

        Returns:
            MultiAnchor: Updated instance (self) of the class with the added anchors.

        Raises:
            ValueError: If any of the Anchor objects being added have a label that already exists in the MultiAnchor.
        """
        if not isinstance(points, list):
            points = [points]
        for p in points:
            if self.has_label(p.label):
                raise ValueError(f"""point label {p} already exists in MultiAnchor.
                                 Choose different label name.""")
        self.multipoint += points

        return self


    def plot(self, ax=None, color: str=None, draw_direction: bool=True) -> None:
        """ 
        Plots the anchors on a given axis.

        Args:
            ax (matplotlib.axes.Axes, optional): The axis on which to plot the anchors. If not provided, a new axis will be created.
            color (str, optional): The color of the anchors.
            draw_direction (bool, optional): Whether to draw the direction of the anchors.
        """
        for p in self.multipoint:
            p.plot(ax=ax, color=color, draw_direction=draw_direction)


class Skeletone():
    """ 
        Represents a collection of connected lines for routing and wireframe geometry.
        Provides methods for creating and manipulating MultiLineString objects.

        Args:
            lines (MultiLineString): A collection of LineString objects representing the skeletone.

        Example:
            >>> from zeroheliumkit import Skeletone
            >>> from zeroheliumkit.settings import DARKGRAY
            >>> from shapely.geometry import MultiLineString, LineString
            >>> skeletone = Skeletone(MultiLineString([LineString([(0, 0), (1, 1)])]))
            <SKELETONE MULTILINESTRING ((0 0, 1 1))>
            >>> skeletone.add(LineString([(1, 1), (2, 2)]))
            >>> print(skeletone.length)
            2.8284271247461903
            >>> skeletone.rotate(90, origin=(0, 0))
            >>> skeletone.move(xoff=1, yoff=1)
            >>> skeletone.plot(color=DARKGRAY)
    """

    lines = MultiLineString()

    def __init__(self, lines: MultiLineString = MultiLineString()):
        self.lines = lines
    
    def __repr__(self):
        return f"<SKELETONE {self.lines}>"

    def __setattr__(self, name, value):
        """
        Changes the behavior of the __setattr__ method.
        Whenever a new LineString is created or an existing one is modified,
            it is set to the precision of the grid size.

        Args:
            name (str): The class attribute name.
            value (obj): The shapely object.
        """
        self.__dict__[name] = set_precision(value, grid_size=GRID_SIZE, mode="pointwise")

    @property
    def length(self) -> float:
        """The total length of all lines in the skeletone."""
        return self.lines.length
    
    @property
    def boundary(self) -> list:
        """The boundary points of the skeletone as a list of Point objects."""
        return list(self.lines.boundary.geoms)
    
    @property
    def numlines(self) -> int:
        """The number of lines in the skeletone."""
        return len(self.lines.geoms)


    def copy(self) -> 'Skeletone':
        """ 
        Creates a deep copy of the Skeletone instance.

        Returns:
            A new instance of Skeletone with the same lines.
        """
        return copy.deepcopy(self)


    def rotate(self, angle: float=0, origin=(0,0)) -> 'Skeletone':
        """ 
        Rotates the skeletone by the given angle around the origin.

        Args:
            angle (float): The angle of rotation in degrees.
            origin (tuple): The origin point of rotation.

        Returns:
            Updated instance (self) of the class with the rotated lines.
        """
        self.lines = affinity.rotate(self.lines, angle, origin)
        return self


    def move(self, dx: float=0, dy: float=0) -> 'Skeletone':
        """ 
        Moves the skeletone by the specified offsets.

        Args:
            dx (float): The horizontal offset to move the skeletone by.
            dy (float): The vertical offset to move the skeletone by.

        Returns:
            Updated instance (self) of the class with the moved lines.
        """
        self.lines = affinity.translate(self.lines, xoff=dx, yoff=dy)
        return self


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin=(0,0)) -> 'Skeletone':
        """ 
        Scales the skeletone by the given factors along the x and y axes.

        Args:
            xfact (float): The scaling factor along the x-axis.
            yfact (float): The scaling factor along the y-axis.
            origin (tuple): The origin point for scaling.

        Returns:
            Updated instance (self) of the class with the scaled lines.
        """
        self.lines = affinity.scale(self.lines, xfact=xfact, yfact=yfact, zfact=1.0, origin=origin)
        return self


    def mirror(self, aroundaxis: str=None, keep_original: bool=False) -> 'Skeletone':
        """ 
        Mirrors the skeletone around the specified axis.

        Args:
            aroundaxis (str, optional): The axis around which to mirror the skeletone. Defaults to None.
            keep_original (bool, optional): Whether to keep the original lines after mirroring. Defaults to False.

        Returns:
            Updated instance (self) of the class with the mirrored lines.
        """
        if aroundaxis == 'y':
            sign = (-1, 1)
        elif aroundaxis == 'x':
            sign = (1, -1)
        else:
            raise TypeError("Choose 'x' or 'y' axis for mirroring")
        
        mirrored = affinity.scale(self.lines, *sign, 1.0, origin=(0, 0))
        if keep_original:
            original = self.lines
            self.lines = unary_union([mirrored, original])
        else:
            self.lines = mirrored
        return self


    def add(self,
            line: LineString,
            direction: float=None,
            ignore_crossing=False,
            chaining=True) -> 'Skeletone':
        """ 
        Appends a LineString to the skeleton.

        Args:
            line (LineString): The LineString to append.
            direction (float, optional): The direction of the LineString. Defaults to None.
            ignore_crossing (bool, optional): Whether to ignore crossing lines. Defaults to False.
            chaining (bool, optional): Whether to chain lines. Defaults to True.

        Returns:
            Updated instance (self) of the class with the appended line.
        """
        if isinstance(self.lines, MultiLineString) or isinstance(line, MultiLineString):
            self.lines = append_line(self.lines, line, direction, ignore_crossing, chaining=False)
        else:
            self.lines = append_line(self.lines, line, direction, ignore_crossing, chaining)
        return self


    def remove(self, line_id: int | tuple | list = None) -> 'Skeletone':
        """ 
        Remove a line from the skeletone

        Args:
            line_id (int | tuple | list): The index of the line to be removed.

        Returns:
            Updated instance (self) of the class with the specified line removed.
        """
        if not line_id:
            self.lines = MultiLineString()
            return self

        if isinstance(line_id, int):
            line_id = [line_id]

        lines = list(self.lines.geoms)
        self.lines = MultiLineString([line for i, line in enumerate(lines) if i not in line_id])
        return self


    def fix(self) -> 'Skeletone':
        """ 
        Fixes the skeletone by merging lines that are connected end-to-end.

        Returns:
            Updated instance (self) of the class with the fixed lines.
        """
        try:
            self.lines = linemerge(self.lines)
        except Exception:
            print("there is nothing to fix in skeletone")
        return self


    def trim_line(self, polygon) -> 'Skeletone':
        """ 
        Cuts the skeletone with a polygon.

        Args:
            polygon (Polygon): The polygon to cut the skeletone with.

        Returns:
            Updated instance (self) of the class with the lines cut by the polygon.
        """
        self.lines = self.lines.difference(polygon)
        return self


    def buffer(self, offset: float, **kwargs) -> Polygon:
        """ 
        Creates a Polygon by buffering the skeleton

        Args:
            offset (float): Buffering skeleton by offset
            **kwargs: Additional keyword arguments to be passed to the buffer method
            See `Shapely buffer docs <https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html#shapely.buffer>`_ for additional keyword arguments.

        Returns:
            Polygon: A polygon representing the buffered skeleton.
        """
        return self.lines.buffer(offset, **kwargs)


    def plot(self, ax=None, color: str=DARKGRAY) -> mpl_axes:
        """ 
        Plots the skeleton on the given axes.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the skeleton. Defaults to None.
            color (str, optional): The color of the skeleton. Defaults to DARKGRAY.

        Returns:
            matplotlib.axes.Axes: The axes with the plotted skeleton.
        """
        if ax is None:
            ax = default_ax()
        
        if self.lines.is_empty:
            return ax
        plot_line(self.lines, ax=ax, color=color, add_points=False, ls="dashed", lw=1)
        return ax


def get_dxdy(point1: tuple | Point | Anchor, point2: tuple | Point | Anchor) -> list:
    """
    Returns the difference in x and y coordinates between two points.
    
    Args:
        point1 (tuple | Point | Anchor): The first point.
        point2 (tuple | Point | Anchor): The second point.

    Returns:
        list: A list containing the differences in x and y coordinates.
    """
    match point1:
        case tuple():
            p1_coords = point1
        case Point():
            p1_coords = point1.xy
        case Anchor():
            p1_coords = point1.coords

    match point2:
        case tuple():
            p2_coords = point2
        case Point():
            p2_coords = point2.xy
        case Anchor():
            p2_coords = point2.coords

    return [p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]]


class Layer():
    
    __slots__ = "name", "polygons", "color", "enable_grid_snap"

    def __init__(self,
                 name: str,
                 polygons: Polygon | MultiPolygon = MultiPolygon(),
                 color: tuple = (RED, 1),
                 enable_grid_snap: bool = True):
        self.name = name
        self.polygons = polygons
        self.color = color if isinstance(color, tuple) else (color, 1)
        self.enable_grid_snap = enable_grid_snap


    def __repr__(self):
        if self.is_empty:
            return f"<LAYER | {self.name} | is empty>"
        elif isinstance(self.polygons, Polygon):
            return f"<LAYER | {self.name} | with 1 polygon>"
        else:
            return f"<LAYER | {self.name} | with {len(self.polygons.geoms)} polygons>"

    @property
    def is_empty(self) -> bool:
        """
        Checks if the layer is empty (i.e., has no polygons).

        Returns:
            bool: True if the layer is empty, False otherwise.
        """
        return self.polygons.is_empty


    @property
    def area(self) -> float:
        """
        Calculates the total area of all polygons in the layer.

        Returns:
            float: The total area of the polygons.
        """
        return self.polygons.area


    def copy(self) -> 'Layer':
        """ 
        Creates a deep copy of the Layer instance.

        Returns:
            A new instance of Layer with the same polygons.
        """
        return copy.deepcopy(self)


    def clear(self) -> 'Layer':
        """ 
        Clears all polygons from the layer.

        Returns:
            Updated instance (self) of the class with no polygons.
        """
        self.polygons = MultiPolygon()
        return self


    @snap_on_grid(attr="polygons")
    def rotate(self, angle: float=0, origin=(0,0)) -> 'Layer':
        """ 
        Rotates the layer by the given angle around the origin.

        Args:
            angle (float, optional): rotation angle. Defaults to 0.
            origin (str, optional): rotations are made around this point. Defaults to (0,0).
        
        Returns:
            Updated instance (self) of the class with all objects rotated.
        """
        return affinity.rotate(self.polygons, angle, origin)


    @snap_on_grid(attr="polygons")
    def move(self, dx: float | int, dy: float | int):
        """
        Move objects either by an (x, y) offset OR by snapping one anchor to another point.

        Args:
            dx (float | int): The x offset to move by.
            dy (float | int): The y offset to move by.

        Returns:
            self: Updated instance with all polygons in layer moved.
        """
        return affinity.translate(self.polygons, xoff=dx, yoff=dy)


    @snap_on_grid(attr="polygons")
    def snap_to(self, point_from: tuple | Point, point_to: tuple | Point) -> 'Layer':
        """
        Snaps the layer from one point to another point by calculating the required offset.
        Args:
            point_from (tuple | Point): The point to snap from.
            point_to (tuple | Point): The point to snap to.
        Returns:
            Updated instance (self) of the class with all polygons moved.
        """
        dxdy = get_dxdy(point_from, point_to)
        return self.move(*dxdy)


    @snap_on_grid(attr="polygons")
    def scale(self, xfact=1.0, yfact=1.0, origin=(0,0)) -> 'Layer':
        """
        Scales all objects in layer by the specified factors along the x and y axes.

        Args:
            xfact (float, optional): scale along x-axis. Defaults to 1.0.
            yfact (float, optional): scale along y-axis. Defaults to 1.0.
            origin ((x,y), optional): scale with respect to an origin (x,y). Defaults to (0,0).

        Returns:
            Updated instance (self) of the class with all polygons scaled.
        """
        return affinity.scale(self.polygons, xfact, yfact, 1.0, origin)


    @snap_on_grid(attr="polygons")
    def mirror(
            self,
            aroundaxis: str,
            keep_original: bool=True
            ) -> 'Layer':
        """
        Mirror all objects in layer around a specified axis.

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
        if aroundaxis == 'y':
            sign = (-1, 1)
        elif aroundaxis == 'x':
            sign = (1, -1)
        else:
            raise TypeError("Choose 'x' or 'y' axis for mirroring")

        mirrored = affinity.scale(self.polygons, *sign, 1.0, origin=(0, 0))
        if keep_original:
            return unary_union([mirrored, self.polygons])
        else:
            return mirrored


    @snap_on_grid(attr="polygons")
    def add(self, geom: Polygon | MultiPolygon | Layer) -> Self:
        """ 
        Adds a polygon or multipolygon to the layer.

        Args:
            geom (Polygon | MultiPolygon | Layer): The polygon or multipolygon to add.

        Returns:
            Updated instance (self) of the class with the added polygon.
        """
        if isinstance(geom, Layer):
            geom = geom.polygons
        return unary_union([self.polygons, geom])


    @snap_on_grid(attr="polygons")
    def cut(self,
            geom: Polygon | MultiPolygon | Layer,
            loc: tuple[float, float]=None) -> Self:
        """ 
        Cuts the layer with a polygon or multipolygon.

        Args:
            geom (Polygon | MultiPolygon | Layer): The polygon to be cut.
            loc (tuple[float, float], optional): The location where the polygon will be cut.
                Defaults to None.

        Returns:
            Updated instance (self) of the class with the cut geometry.
        """
        if isinstance(geom, Layer):
            geom = geom.polygons
        cut_geom = affinity.translate(geom, xoff=loc[0], yoff=loc[1]) if loc else geom
        updated = self.polygons.difference(cut_geom)
        return updated


    @snap_on_grid(attr="polygons")
    def crop(self,
             geom: Polygon | MultiPolygon | Layer,
             loc: tuple[float, float] = None) -> Self:
        """ 
        Crops the layer with a polygon or multipolygon.

        Args:
            geom (Polygon | MultiPolygon | Layer): The polygon to be used for cropping.
            loc (tuple[float, float], optional): The location where the polygon will be applied.
                Defaults to None.

        Returns:
            Updated instance (self) of the class with the cropped geometry.
        """
        if isinstance(geom, Layer):
            geom = geom.polygons
        crop_geom = affinity.translate(geom, xoff = loc[0], yoff = loc[1]) if loc else geom
        updated = self.polygons.intersection(crop_geom)
        if isinstance(updated, (Point, MultiPoint, LineString, MultiLineString)):
            updated = MultiPolygon()
        elif isinstance(updated, GeometryCollection):
            # Select only polygons
            polygon_list = [geom for geom in list(updated.geoms) if isinstance(geom, Polygon)]
            updated = MultiPolygon(polygon_list)

        return updated


    @snap_on_grid(attr="polygons")
    def simplify(self, tolerance: float=0.1) -> 'Layer':
        """ Simplify polygons in a layer

        Args:
            tolerance (float, optional): The tolerance value for simplification. Defaults to 0.1.

        Returns:
            Updated instance (self) of the class with the specified layer simplified.
        """
        return self.polygons.simplify(tolerance)


    @snap_on_grid(attr="polygons")
    def modify_points(
            self,
            poly_id: int,
            ext_points: dict,
            int_points: dict=None,
            int_idx: int=0):
        """
        Updates the point coordinates of a polygon in a layer.
        Can optionally modify a single interior's coordinates. Moves first and last points together.

        Args:
            layer (str): layer name
            poly_id (int): polygon index in multipolygon list
            ext_points (dict): point indices to change in a polygon. 
                Keys: corrresponds to the point idx in polygon exterior coord list.
                Value: tuple of new [x,y] coordinates
            int_points (dict, optional): point indices to change in a polygon. 
                Keys: corrresponds to the point idx in polygon interiors coord list.
                Value: tuple of new [x,y] coordinates
                Defaults to None. 
            int_idx (int, optional): interior index in the polygon's interiors list. Defaults to 0.
        """
        p_list = list(self.polygons.geoms)
        p = p_list[poly_id]
        coords = get_coordinates(p)

        #updating exterior coords of the polygon
        if ext_points:
            points_to_be_changed = list(ext_points.keys())
            for point in points_to_be_changed:
                coords[point, 0] = coords[point, 0] + ext_points[point][0] 
                coords[point, 1] = coords[point, 1] + ext_points[point][1]

                if point == (len(p.exterior.coords) - 1):
                    coords[0, 0] = coords[0, 0] + ext_points[point][0] 
                    coords[0, 1] = coords[0, 1] + ext_points[point][1]
                elif point == 0:
                    last_point = (len(p.exterior.coords) - 1)
                    coords[last_point, 0] = coords[last_point, 0] + ext_points[point][0] 
                    coords[last_point, 1] = coords[last_point, 1] + ext_points[point][1]


        #updating interior coordinates of the polygon
        if int_points and has_interior(p) and int_points:
            int_coords = p.interiors[int_idx].coords
            int_coords_start = len(p.exterior.coords) * (int_idx + 1)

            points_to_be_changed = list(int_points.keys())
            for point in points_to_be_changed:
                point_offset = int_coords_start + point
                
                coords[point_offset, 0] = coords[point_offset, 0] + int_points[point][0] 
                coords[point_offset, 1] = coords[point_offset, 1] + int_points[point][1]

                if point == len(int_coords) - 1:
                    coords[int_coords_start, 0] = coords[int_coords_start, 0] + int_points[point][0] 
                    coords[int_coords_start, 1] = coords[int_coords_start, 1] + int_points[point][1]

        p_list[poly_id] = set_coordinates(p, coords)
        return MultiPolygon(p_list)


    @snap_on_grid(attr="polygons")
    def remove_holes(self, cut_position: float=None):
        """
        Removes any holes from a multipolygon in a layer by vertically cutting
        along the centroid of each hole and piecing together the remaining Polygon geometries.
        Vertical cut is made with a default length of 1e6. 

        Args:
            cut_position (float): The x-coordinate where the vertical cut is made.
                If None, the cut is made at the centroid of each hole.
        """
        return flatten_multipolygon(self.polygons, cut_position)


    @snap_on_grid(attr="polygons")
    def slice(self, slice_line: LineString | list[LineString]):
        """
        Slices polygons in a layer using a given line.

        Args:
            slice_line (LineString): The line used for slicing.
        """

        updated = self.polygons
        for slice in slice_line:
            updated = split_polygon(updated, slice)
        return updated


    def remove(self, polygon_id: int | tuple | list = None):
        """
        Removes a polygon from a layer.

        Args:
            polygon_id (int | tuple | list): The index of the polygon to be removed.
        """
        if not polygon_id:
            self.polygons = MultiPolygon()
            return self

        if isinstance(polygon_id, int):
            polygon_id = [polygon_id]

        poly_list = list(self.polygons.geoms)
        self.polygons = MultiPolygon([poly for i, poly in enumerate(poly_list) if i not in polygon_id])
        return self


    def add_text(self, text: str="abcdef", size: float=1000, loc: tuple=(0,0)):
        """
        Converts text into polygons and adds them to a layer.

        Args:
            text (str, optional): The text to be converted into polygons. Defaults to "abcdef".
            size (float, optional): The size of the text. Defaults to 1000.
            loc (tuple, optional): The location where the text polygons will be placed. Defaults to (0,0).
        """
        ptext = polygonize_text(text, size)
        ptext = affinity.translate(ptext, *loc)
        self.add(ptext)


    def buffer(self, offset: float, **kwargs) -> 'Layer':
        """
        Creates a new layer by buffering the current layer's polygons.

        Args:
            offset (float): The distance to buffer the polygons.
            **kwargs: Additional keyword arguments to be passed to the buffer method.

        Returns:
            Layer: A new Layer instance with the buffered polygons.
        """
        return self.polygons.buffer(offset, **kwargs)


    def plot(
            self,
            ax=None,
            size: tuple=(8,8),
            show_idx: bool=False,
            show_line_idx: bool=False,
            show_grid: bool=False,
            add_points: bool=False,
            labels: bool=False,
            edgecolor: str=BLACK,
            **kwargs
            ) -> mpl_axes:
        """
        Plots a layer polygons on the given axes.

        Args:
            ax (plt.Axes, optional): The axes object on which to plot the layer. If None, a default axes object will be used.
            show_idx (bool, optional): Whether to show the index of the layer object.
            show_line_idx (bool, optional): Whether to show the line index of the layer object.
            add_points (bool, optional): Whether to add points to the layer object.
            edgecolor (str, optional): The edge color of the layer object.
            **kwargs: Additional keyword arguments to be passed to the plotting functions.
        """
        if ax is None:
            plt.figure(1, figsize=size, dpi=90)
            ax = default_ax()

        plot_polygon(self.polygons, 
                     ax=ax, 
                     color=self.color[0],
                     alpha=self.color[1],
                     add_points=add_points, 
                     edgecolor=edgecolor,
                     **kwargs)
        if show_idx:
            plot_polygon_idx(self.polygons, ax=ax, color=self.color[0])
        if self.color[1] != 1:
            plot_line(self.polygons.boundary, ax=ax, color=BLACK, add_points=False, lw=1.5)
        if show_line_idx:
            plot_line_idx_in_polygon(self.polygons, ax=ax, color=self.color[0])
        if not show_grid:
            ax.grid(False)
        
        if labels:
            draw_labels(self.polygons, ax=ax)
