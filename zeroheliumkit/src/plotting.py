"""
plotting.py

This file contains functions for plotting geometries using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.axes
from tabulate import tabulate
import colorsys

from shapely import get_coordinates, Point
from shapely.plotting import plot_line, plot_polygon

from .settings import *
from .utils import to_geometry_list, calculate_label_pos, has_interior

def interactive_widget_handler() -> None:
    """
    Closes the current matplotlib figure if it exists.
    """
    try:
        plt.close()
    except:
        pass


def default_ax() -> plt.Axes:
    """ 
    Gets the default axis object (matplotlib.axes.Axes) with grid enabled and equal aspect ratio.

    Returns:
        plt.Axes: The default axes object.
    """
    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def adjust_lightness(color: str, amount: float=0.5) -> tuple:
    """ 
    Adjusts the lightness of a given color and converts it to hexidecimal format.

    Args:
        color (str): The color to adjust. Can be a named color or a hexadecimal color code.
        amount (float, optional): The amount by which to adjust the lightness. Defaults to 0.5.

    Returns:
        tuple: The adjusted color in RGB format.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def segments(curve) -> list:
    """
    Returns a list of segments from a given curve.
    Each segment is represented as a LineString object.
    
    Args:
        curve: A LineString or MultiLineString object representing the curve.
        
    Returns:
        list: A list of LineString objects representing the segments of the curve.
    """
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))[::-1]


def plot_geometry(geometry, ax=None, show_idx=False, color=None, 
                    edgecolor=BLACK, alpha=1, show_line_idx=False, add_points=False, **kwargs) -> None:
    """
    Plots a geometry object on the given axes.

    Args:
        geometry (Polygon | MultiPolygon): The geometry object to be plotted.
        ax (plt.Axes, optional): The axes object on which to plot the geometry. If None, a default axes object will be used.
        show_idx (bool, optional): Whether to show the index of the geometry object.
        color (str, optional): The color of the geometry object.
        edgecolor (str, optional): The edge color of the geometry object.
        alpha: The transparency of the geometry object.
        **kwargs: Additional keyword arguments to be passed to the plotting functions.
    """
    if ax is None:
        ax = default_ax()
    
    if type(geometry) in PLG_CLASSES:
        plot_polygon(geometry, 
                     ax=ax, 
                     color=color, 
                     add_points=add_points, 
                     alpha=alpha, 
                     edgecolor=edgecolor)
        if show_idx:
            plot_polygon_idx(geometry, ax=ax, color=color)
        if alpha!=1:
            plot_line(geometry.boundary, ax=ax, color=BLACK, add_points=False, lw=1.5)
        if show_line_idx:
            plot_line_idx_in_polygon(geometry, ax=ax, color=color)
    

def plot_polygon_idx(geometry: Polygon | MultiPolygon, ax=None, color=None) -> None:
    """ 
    Plots the index of each polygon in the given geometry on the specified axes.

    Args:
        geometry (Polygon | MultiPolygon): A Polygon or MultiPolygon object representing the geometry.
        ax: The axes on which to plot the index. If None, the current axes will be used.
        color: The color of the index annotation. If None, a default color will be used.
    """
    if hasattr(geometry, "geoms"):
        for idx, polygon in enumerate(list(geometry.geoms)):
            center_xy = list(polygon.centroid.coords)
            ax.annotate(idx,
                        center_xy[0],
                        color=adjust_lightness(color, 0.5),
                        clip_on=True,
                        bbox=dict(facecolor='white', edgecolor=BLACK, alpha=1))
    else:
        center_xy = list(geometry.centroid.coords)
        ax.annotate("0",
                    center_xy[0],
                    color=adjust_lightness(color, 0.5),
                    clip_on=True,
                    bbox=dict(facecolor='white', edgecolor=BLACK, alpha=1))


def plot_line_idx_in_polygon(poly: Polygon | MultiPolygon, ax=None, color=None) -> None:
    """ 
    Plots the index of each line in the given polygon on the specified axes.

    Args:
        poly (Polygon | MultiPolygon): A Polygon object representing the geometry.
        ax: The axes on which to plot the index. If None, the current axes will be used.
        color: The color of the index annotation. If None, a default color will be used.
    """
    for idx, segment in enumerate(segments(poly.boundary)):
        ax.annotate(idx,
                    list(segment.centroid.coords)[0],
                    color=color,
                    clip_on=True,
                    bbox=dict(facecolor='white', edgecolor=BLACK, alpha=1))


def plot_points_withlabel(geometry, ax=None, color=None, marker=".") -> None:
    """ 
    Plot points with labels on a given axis.

    Args:
        geometry: The geometry object containing the coordinates of the points.
        ax: The axis on which to plot the points. If None, a default axis will be used.
        color: The color of the points and labels.
        marker: The marker style for the points.
    """
    if ax is None:
        ax = default_ax()
    coords = get_coordinates(geometry)
    for idx in range(len(coords)):
        ax.annotate(idx,
                    (coords[idx,0], coords[idx,1]),
                    color=BLACK,
                    clip_on=True, 
                    bbox=dict(facecolor='white', edgecolor=color, boxstyle='round', alpha=0.7))
    ax.plot(coords[:, 0], coords[:, 1], linestyle="", marker=marker, color=color, alpha=1, zorder=1e9)


def set_limits(ax, coor: list | Point, dxdy: list) -> None:
    """ 
    Sets the limits of the given axes object.

    Args:
        ax: The axes object to set the limits for.
        coor (list | Point): The coordinates of the center point as a list or a Point object.
        dxdy (list): The width and height of the axes as a list.
    """
    dx, dy = dxdy
    if type(coor) is Point:
        x0 = coor.x
        y0 = coor.y
    else:
        x0, y0 = coor
    ax.set_xlim(x0 - dx/2, x0 + dx/2)
    #ax.set_xticks(range(x0, xN+1))
    ax.set_ylim(y0 - dy/2, y0 + dy/2)
    #ax.set_yticks(range(y0, yN+1))
    ax.set_aspect("equal")


def listify_colors(layer_colors: dict) -> dict:
    """
    Converts a dict of layer color specifications into a dict
    where all values are [color, transparency] lists.

    Args:
        layer_colors (dict): Keys are layer names. Values are either:
            - a color value (e.g. string like 'red' or '#ff0000')
            - or a tuple or a list: (color_value, transparency)

    Returns:
        dict: Same keys, with values as [color_value, transparency] lists.
    """
    standardized = {}
    for layer, value in layer_colors.items():
        if isinstance(value, tuple):
            standardized[layer] = list(value)
        elif isinstance(value, list):
            pass  # skip lists:
        else:
            standardized[layer] = [value, 1.0]
    return standardized


def draw_labels(geometry, ax: matplotlib.axes.Axes) -> None:
    """
    Draws labels on the given axis for each point in the geometry.

    Args:
        geometry: The geometry object containing the points to label.
        ax (matplotlib.axes.Axes): The axes on which to draw the labels.
    """
    label_distance = 0.5
    geoms_list = to_geometry_list(geometry)
    for polygon in geoms_list:
        label = 1
        for x, y in polygon.exterior.coords:
            label_x, label_y = calculate_label_pos(x, y, polygon.centroid, label_distance)
            if label != len(polygon.exterior.coords):
                ax.plot(x, y, 'ro')
                ax.text(label_x, label_y, str(label), color='red')
                label += 1
        if has_interior(polygon):
            for int in polygon.interiors:
                label = 1
                for x, y in int.coords:
                    label_x, label_y = calculate_label_pos(x, y, polygon.centroid, label_distance)

                    if label != len(int.coords):
                        ax.plot(x, y, 'ro')
                        ax.text(label_x, label_y, str(label), ha='left', va='bottom', color='red')
                        label += 1

class ColorHandler():
    """
    A class to handle color/layer adjustments for plotting. 
    The 'colors' attribute of the Base class is set to an instance of ColorHandler(),
    which holds and modifies the colors/layers mapping and order for plotting.

    Args:
        colors (dict): dictionary mapping of layer names to (color, transparancy) tuples.
        color_cycle (list): list of colors to cycle through when no color is provided.
    """
    __slots__ = "colors", "color_cycle"

    def __init__(self, colors: dict):
        self.colors = listify_colors(colors)
        """Dictionary mapping of layer names to [color, transparancy] tuples."""
        self.color_cycle = COLORS
        """List of colors to cycle through when no color is provided."""


    def __repr__(self):
        name = f"<COLORHANDLER ({self.colors})>"
        max_length = 75
        if len(name) > max_length:
            return f"{name[: max_length - 3]}...>"

        return name
    
    def print(self):
        col_names = ["i", "layer", "color/alpha"]
        layers = self.colors.keys()
        table = []
        for i, lname in enumerate(layers):
            table.append([i, lname, self.colors[lname]])
        print(tabulate(table, headers=col_names))


    @property
    def is_empty(self) -> bool:
        """
        Checks if the colors attribute is empty.

        Returns:
            bool: True if the colors attribute is empty, False otherwise.
        """
        return not bool(self.colors)


    def change_color(self, lname: str, new_color: tuple | list) -> 'ColorHandler':
        """
        Updates the color of a layer in the colors attribute.

        Args: 
            lname (str): The name of the layer to update.
            new_color (str): Color code, alpha value, or a tuple of both to update the layer with.

        Returns:
-
            Updated instance (self) of the class with the specified layer's color changed. 

        Raises:

            ValueError: If the new_color parameter is not a tuple, string, or float.
            ValueError: If the given color is anot a valid color code.
        """
        if isinstance(new_color, (tuple, list)):
            if not bool(mc.is_color_like(new_color[0])):
                raise ValueError("Input color is not a valid color.")
            if lname in self.colors:
                self.colors[lname] = list(new_color)
        return self


    def update_colors(self, layers: list) -> 'ColorHandler':
        """
        Updates colors when layers are imported from external files.

        Args:
            layers (list): list of layers to update the colors list in accordance with.

        Returns:
-
            Updated instance (self) with the udpated colors attribute. 
        """
        for i, l in enumerate(layers):
            if l not in self.colors:
                color = self.color_cycle[i]
                self.colors[l] = [color, 1.0]

        return self


    def add_color(self, layer: str, color: str | None, alpha: float | None) -> 'ColorHandler':
        """
        Adds a color to the colors list when a layer is added.

        Args:
            layer (str): layer name to add to the list.
            color_info (tuple(str, int)): color and transparancy to map to the layer.
        """
        if color is None:
            color = self.color_cycle[0]

        if alpha is None:
            alpha = 1.0

        self.colors[layer] = [color, alpha]
        return self


    def rename_color(self, old_color: str, new_color: str) -> 'ColorHandler':
        """
        Renames a color when a layer is renamed.

        Args:
            old_color (str): old color name
            new_color (str): new color name
        """

        if old_color in self.colors:
            self.colors[new_color] = self.colors.pop(old_color)
        else:
            print(f"Layer '{old_color}' not found in colors.")

        return self


    def remove_color(self, color: str) -> 'ColorHandler':
        """
        Removes a color when a layer is removed.

        Args:
            color (str): color to remove in the colors attribute.
        """

        if color in self.colors:
            del self.colors[color]
        else:
            print(f"Layer '{color}' not found in colors.")

        return self


    def move_layer(self, layer: str, move_by: int=1):
        """
        Moves a layer by a given number of indices in the color dictionary.

        Args:
            layer (str): the name of the layer to move.
            move_by (int): the number of indices to move the color by.
        """
        lst = list(self.colors.items())
        curr_index = lst.index((layer, self.colors[layer]))
        
        new_index = curr_index + move_by
        elem = lst.pop(curr_index)

        # Clamp the index within valid list boundaries
        new_index = max(0, min(new_index, len(lst)))
        lst.insert(new_index, elem)
        self.colors = dict(lst)
    

    def bring_to_front(self, layer: str):
        self.move_layer(layer, move_by=len(self.colors))
    
    def send_to_back(self, layer: str):
        self.move_layer(layer, move_by=-len(self.colors))
    
    def bring_forward(self, layer: str):
        self.move_layer(layer, move_by=1)
    
    def send_backward(self, layer: str):
        self.move_layer(layer, move_by=-1)
