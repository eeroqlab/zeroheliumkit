import copy
from math import fmod
import numpy as np

from tabulate import tabulate
from shapely import Point, affinity, set_precision

from .settings import GRID_SIZE, BLACK, RED
from .plotting import default_ax


def modFMOD(angle: float | int) ->float:
    """ modified modulo calculations for angles in degrees
        lower branch always has negative sign

    Args:
        angle (float):

    Returns:
        angle: angle modulo 360
    """

    if np.abs(angle) % 360 == 180:
        return 180

    if np.abs(angle) % 360 < 180:
        return fmod(angle, 360)

    if np.sign(angle) > 0:
        return angle % 360 - 360

    return angle % 360


class Anchor():
    """ Anchor class, which has three attributes:
        1. coordinates (tuple | Point)
        2. orientation/direction
        3. label
    """

    __slots__ = "point", "direction", "label"

    def __init__(self, point: Point | tuple=Point(), direction: float=0, label: str=None):
        if not isinstance(point, Point):
            coords = np.array(point).squeeze()
            self.point = set_precision(Point(coords), grid_size=GRID_SIZE)
        else:
            self.point = set_precision(point, grid_size=GRID_SIZE)
        self.direction = modFMOD(direction)
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

    def rename(self, newlabel: str) -> None:
        self.label = newlabel

    def rotate(self, angle: float, origin: tuple=(0,0)):
        """ Rotates the point by 'angle' around 'origin'"""
        point_upd = affinity.rotate(self.point, angle, origin)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)
        if self.direction is not None:
            self.direction += angle

        self.direction = modFMOD(self.direction)
    
    def rotate_dir(self, angle: float):
        self.direction = modFMOD(self.direction + angle)

    def move(self, xoff: float=0, yoff: float=0):
        point_upd = affinity.translate(self.point, xoff=xoff, yoff=yoff)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)

    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)):
        point_upd = affinity.scale(self.point, xfact=xfact, yfact=yfact, origin=origin)
        self.point = set_precision(point_upd, grid_size=GRID_SIZE)
        if self.direction is not None:
            if yfact < 0:
                #rotation around x-axis
                self.direction = (-1) * self.direction

            if xfact < 0:
                #rotation around y_axis
                self.direction = 180 - self.direction

    def mirror(self, aroundaxis: str=None, update_label: str=None):
        if aroundaxis=='x':
            self.scale(1, -1)
        elif aroundaxis=='y':
            self.scale(-1, 1)
        elif aroundaxis not in ['x', 'y', None]:
            raise ValueError("choose 'x', 'y or None for mirror arguments")

        if aroundaxis and update_label:
            self.label = update_label

    def plot(self, ax=None, color: str=None, draw_direction: bool=True):
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
    """ MultiAnchor class, which has one attribute:
        1. multipoint - a list of Anchors
    """

    __slots__ = "multipoint"

    def __init__(self, multipoint: list=[]):
        self.multipoint = multipoint

    @property
    def labels(self) -> list:
        return [p.label for p in self.multipoint]

    def label_exist(self, label: str) -> bool:
        if label in self.labels:
            return True
        return False

    def copy(self):
        """ Returns  deepcopy of the class """
        return copy.deepcopy(self.multipoint)

    def rotate(self, angle: float, origin: tuple=(0,0)):
        if self.multipoint:
            for p in self.multipoint:
                p.rotate(angle, origin)

    def move(self, xoff: float=0, yoff: float=0):
        for p in self.multipoint:
            p.move(xoff, yoff)

    def scale(self, xfact: float=1.0, yfact: float=1.0, origin: tuple=(0,0)):
        for p in self.multipoint:
            p.scale(xfact, yfact, origin)

    def mirror(self, aroundaxis: str=None, update_labels: bool=False, keep_original: bool=False):
        if not keep_original:
            if not update_labels:
                for p in self.multipoint:
                    p.mirror(aroundaxis)
            else:
                for p in self.multipoint:
                    p.mirror(aroundaxis, p.label + "_m")
        else:
            original = self.copy()
            for p in self.multipoint:
                p.mirror(aroundaxis, p.label + "_m")
            self.multipoint = self.multipoint + original

    def __point(self, label: str):
        idx = self.labels.index(label)
        return self.multipoint[idx]

    def point(self, labels: list[str]):
        if isinstance(labels, list):
            return [self.__point(l) for l in labels]
        return self.__point(labels)

    def remove(self, labels: list | str):
        if isinstance(labels, str):
            labels = [labels]
        S1 = set(self.point(labels))
        S2 = set(self.multipoint)
        self.multipoint = list(S2.difference(S1))

    def modify(self, label: str, new_name: str=None, new_xy: tuple=None, new_direction: float=None):
        if new_xy:
            self.__point(label).coords = new_xy
        if new_direction:
            self.__point(label).direction = new_direction
        if new_name:
            self.__point(label).label = new_name

    def add(self, points: list[Anchor] | Anchor=[]):
        if not isinstance(points, list):
            points = [points]
        for p in points:
            if self.label_exist(p.label):
                raise ValueError(f"""point label {p} already exists in MultiAnchor.
                                 Choose different label name.""")
        self.multipoint += points

    def plot(self, ax=None, color: str=None, draw_direction: bool=True):
        for p in self.multipoint:
            p.plot(ax=ax, color=color, draw_direction=draw_direction)
