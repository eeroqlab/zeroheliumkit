
"""
This module defines the Anchor and MultiAnchor classes
for handling anchor points in a coordinate system.

The Anchor class represents a single anchor point with attributes
such as coordinates, orientation/direction, and label. It provides methods
for manipulating and visualizing the anchor point.

The MultiAnchor class represents a collection of multiple anchor points.
It provides methods for manipulating and visualizing multiple anchor points simultaneously.

Both classes utilize the Shapely library for geometric operations
and the Tabulate library for tabular representation.

Note: This module assumes the existence of other modules
such as `settings` and `plotting` for importing necessary dependencies.
"""

import copy
import numpy as np
from math import fmod
from tabulate import tabulate
from shapely import Point, affinity, set_precision, distance

from .settings import GRID_SIZE, BLACK, RED
from .plotting import default_ax


def fmodnew(angle: float | int) -> float:
    """ Returns a Modified modulo calculations for angles in degrees.
        The lower branch always has a negative sign.

    Args:
    -----
    angle (float | int): The angle in degrees.
    """
    if np.abs(angle) % 360 == 180:
        return 180
    if np.abs(angle) % 360 < 180:
        return fmod(angle, 360)
    if np.sign(angle) > 0:
        return angle % 360 - 360
    return angle % 360


class Anchor():
    """
    Anchor class, which represents a single anchor point with attributes
    such as coordinates, orientation/direction, and label. It provides methods
    for manipulating and visualizing the anchor point.

    Attributes:
    ----------
    point (tuple | Point): The coordinates of the anchor point.
    direction (float): The orientation/direction of the anchor point.
    label (str): The label of the anchor point.

    Example:
    --------
        >>> anchor = Anchor((0, 0), 45, "A")
        >>> anchor.properties  # prints the properties of the anchor
        >>> distance = anchor.distance_to((3, 4))  # calculates the distance to another point
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

    @property
    def x(self):
        return self.point.x

    @property
    def y(self):
        return self.point.y

    @property
    def coords(self):
        return (self.point.x, self.point.y)

    @coords.setter
    def coords(self, xy: tuple):
        self.point = set_precision(Point(*xy), grid_size=GRID_SIZE)

    @property
    def properties(self):
        print(tabulate([["label", "coords", "direction"],
                        [self.label, self.coords, self.direction]]))


    def distance_to(self, point: tuple | Point) -> float:
        """ Returns the unitless distance between this anchor and another point.

        Args:
        ____
        point (tuple | Point): The other point to calculate the distance to.
        """
        if isinstance(point, tuple):
            point = Point(point)
        return distance(self.point, point)


    def rename(self, newlabel: str) -> None:
        """ Renames the anchor with a new label.

        Args:
        ____
        newlabel (str): The new label for the anchor.
        """
        self.label = newlabel


    def rotate(self, angle: float, origin: tuple=(0,0)):
        """ Rotates the point by the specified angle around the given origin.
            Updates arrow direction too.

        Args:
        ____
        angle (float): The angle of rotation in degrees.
        origin (tuple, optional): The origin point of rotation. Defaults to (0, 0).
        """
        point_upd = affinity.rotate(self.point, angle, origin)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)
        if self.direction is not None:
            self.direction += angle

        self.direction = fmodnew(self.direction)

        return self


    def rotate_dir(self, angle: float):
        """ Rotates the direction of the anchor by the specified angle. Doesn't update the point.

        Args:
        ____
        angle (float): The angle (in radians) by which to rotate the direction.
        """
        self.direction = fmodnew(self.direction + angle)

        return self


    def move(self, xoff: float=0, yoff: float=0):
        """ Moves the anchor point by the specified offset in the x and y directions.

        Args:
        ----
        xoff (float): The offset in the x direction. Default is 0.
        yoff (float): The offset in the y direction. Default is 0.
        """
        point_upd = affinity.translate(self.point, xoff=xoff, yoff=yoff)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)

        return self


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)):
        """ Scales the anchor point by the given factors along the x and y axes.

        Args:
        ----
        xfact (float): The scaling factor along the x-axis. Default is 1.0.
        yfact (float): The scaling factor along the y-axis. Default is 1.0.
        origin (tuple): The origin point for scaling. Default is (0, 0).
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


    def mirror(self, aroundaxis: str=None, update_label: str=None):
        """ Mirrors the anchor object around the specified axis.

        Args:
        ----
        aroundaxis (str): The axis to mirror the anchor around.
                              Valid values are 'x', 'y', or None.
        update_label (str): The updated label for the mirrored anchor.

        Raises:
        ------
            ValueError: If the `aroundaxis` parameter is not 'x', 'y', or None.
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


    def plot(self, ax=None, color: str=None, draw_direction: bool=True):
        """ Plots the anchor point on the given axes.

        Args:
        ----
        ax (optional): The axes on which to plot the anchor point.
                        If not provided, a default axes will be used.
        color (str, optional): The color of the anchor point and annotation box edge.
                                If not provided, the default color will be used.
        draw_direction (bool, optional): Whether to draw an arrow indicating
                                            the direction of the anchor.
                                            Default is True.
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


class MultiAnchor():
    """ 
        MultiAnchor class, which has one attribute:
        1. multipoint - a list of Anchors

        Example:
        -------
            >>> anchor1 = Anchor(Point(0, 0), 0, "A")
            >>> anchor2 = Anchor(Point(1, 1), 45, "B")
            >>> multi_anchor = MultiAnchor([anchor1, anchor2])
            >>> multi_anchor.labels
            ['A', 'B']
            >>> multi_anchor.label_exist('A')
            True
            >>> multi_anchor.label_exist('C')
            False
            >>> multi_anchor.rotate(90, (0, 0))
    """

    __slots__ = "multipoint"

    def __init__(self, multipoint: list=[]):
        self.multipoint = multipoint

    @property
    def labels(self) -> list:
        return [p.label for p in self.multipoint]


    def label_exist(self, label: str) -> bool:
        """ Checks if a label exists in the list of labels.

        Args:
        ----
        label (str): The label to check.
        """
        if label in self.labels:
            return True
        return False


    def copy(self):
        """ Returns a deep copy of the class """
        return copy.deepcopy(self)


    def rotate(self, angle: float, origin: tuple=(0,0)):
        """ Rotates all anchors by a given angle around a specified origin point.
        
        Args:
        ----
        angle (float): The angle of rotation in degrees.
        origin (tuple, optional): The origin point of rotation. Defaults to (0, 0).
        """
        if self.multipoint:
            for p in self.multipoint:
                p.rotate(angle, origin)

        return self


    def move(self, xoff: float=0, yoff: float=0):
        """ Moves the anchors by the specified offsets.

        Args:
        ----
        xoff (float): The horizontal offset to move the anchors by. Default is 0.
        yoff (float): The vertical offset to move the anchors by. Default is 0.
        """
        for p in self.multipoint:
            p.move(xoff, yoff)

        return self


    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)):
        """ Scales the multipoint by the given factors along the x and y axes.

        Args:
        ----
        xfact (float): The scaling factor along the x-axis. Default is 1.0.
        yfact (float): The scaling factor along the y-axis. Default is 1.0.
        origin (tuple): The origin point for scaling. Default is (0, 0).
        """
        for p in self.multipoint:
            p.scale(xfact, yfact, origin)

        return self


    def mirror(self, aroundaxis: str=None, update_labels: bool=False, keep_original: bool=False):
        """ Mirrors the multipoint anchors around a specified axis.

        Args:
        ----
        aroundaxis (str): The axis around which to mirror the anchors.
        update_labels (bool): Whether to update the labels of the mirrored anchors.
        keep_original (bool): Whether to keep the original anchors.
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


    def __point(self, label: str):
        idx = self.labels.index(label)
        return self.multipoint[idx]


    def point(self, labels: list[str]):
        """ Returns an anchor or list of anchors of the given labels.

        Args:
        ----
        labels (list[str]): A list of labels for which to retrieve the anchors.
        """
        if isinstance(labels, (list, tuple)):
            return [self.__point(l) for l in labels]
        return self.__point(labels)


    def remove(self, labels: list | str):
        """ Removes the specified anchors from the multipoint.

        Args:
        ----
        labels (list or str): The anchors to be removed.
        """
        if isinstance(labels, str):
            labels = [labels]
        S1 = set(self.point(labels))
        S2 = set(self.multipoint)
        self.multipoint = list(S2.difference(S1))

        return self


    def modify(self, label: str, new_name: str=None, new_xy: tuple=None, new_direction: float=None):
        """ Modifies the properties of an anchor.

        Args:
        ----
        label (str): The anchor to modify.
        new_name (str, optional): The new name for the anchor. Defaults to None.
        new_xy (tuple, optional): The new coordinates (x, y) for the anchor. Defaults to None.
        new_direction (float, optional): The new direction for the anchor. Defaults to None.
        """
        if new_xy:
            self.__point(label).coords = new_xy
        if new_direction:
            self.__point(label).direction = new_direction
        if new_name:
            self.__point(label).label = new_name

        return self


    def add(self, points: list[Anchor] | Anchor=[]):
        """ Adds one or more Anchor objects to the MultiAnchor.

        Args:
        ----
        points (list[Anchor] | Anchor, optional): 
            A single Anchor object or a list of Anchor objects to be added.
            Defaults to an empty list.

        Raises:
        ------
        ValueError: If any of the Anchor objects being added
        have a label that already exists in the MultiAnchor.
        """
        if not isinstance(points, list):
            points = [points]
        for p in points:
            if self.label_exist(p.label):
                raise ValueError(f"""point label {p} already exists in MultiAnchor.
                                 Choose different label name.""")
        self.multipoint += points

        return self


    def plot(self, ax=None, color: str=None, draw_direction: bool=True):
        """ Plots the anchors on a given axis.

        Args:
        ----
        ax (matplotlib.axes.Axes, optional): The axis on which to plot the anchors.
            If not provided, a new axis will be created.
        color (str, optional): The color of the anchors.
        draw_direction (bool, optional): Whether to draw the direction of the anchors.
        """
        for p in self.multipoint:
            p.plot(ax=ax, color=color, draw_direction=draw_direction)
