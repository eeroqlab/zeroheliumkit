"""
plotting.py

This file contains functions for plotting geometries using matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

from shapely import get_coordinates, Point
from shapely.plotting import plot_line, plot_polygon

from .settings import *

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
    Gets the default axis object (matplotlib.axes.Axes)
        with grid enabled and equal aspect ratio.

    Returns:
    -------
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
    ----
        color (str): The color to adjust. Can be a named color or a hexadecimal color code.
        amount (float, optional): The amount by which to adjust the lightness. Defaults to 0.5.

    Returns:
    -------
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
    ----
        curve: A LineString or MultiLineString object representing the curve.
        
    Returns:
    -------
        list: A list of LineString objects representing the segments of the curve.
    """
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))[::-1]


def plot_geometry(geometry, ax=None, show_idx=False, color=None, 
                    edgecolor=BLACK, alpha=1, show_line_idx=False, add_points=False, **kwargs) -> None:
    """
    Plots a geometry object on the given axes.

    Args:
    ----
        geometry: The geometry object to be plotted.
        ax: The axes object on which to plot the geometry. If None, a default axes object will be used.
        show_idx: Whether to show the index of the geometry object.
        color: The color of the geometry object.
        edgecolor: The edge color of the geometry object.
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

    elif type(geometry) in LINE_CLASSES:
        plot_line(geometry, ax=ax, color=color, add_points=False, ls="dashed", lw=1)

    elif type(geometry) in PTS_CLASSES:
        plot_points_withlabel(geometry, ax=ax, color=color, marker=".")
    

def plot_polygon_idx(geometry: Polygon | MultiPolygon, ax=None, color=None) -> None:
    """ 
    Plots the index of each polygon in the given geometry on the specified axes.

    Args:
    ----
        geometry: A Polygon or MultiPolygon object representing the geometry.
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
    ----
        poly: A Polygon object representing the geometry.
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
    ----
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
    ____
        ax: The axes object to set the limits for.
        coor: The coordinates of the center point as a list or a Point object.
        dxdy: The width and height of the axes as a list.
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
