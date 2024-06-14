""" This module contains functions and a class for analyzing
    and plotting field data extracted from FEM simulations.

Functions:
- flatten(l: list) -> list:
    Flattens a nested list into a single list.
- find_max_index(arr: list) -> int:
    Finds the index of the maximum value in a 2D array.
- find_nearest(array: list, value: float | int) -> int: 
    Finds the index of the element in an array that is closest to a given value.
- center_within_area(x0: float, y0: float, xlist: list, ylist: list, tol=0.25) -> bool:
    Checks if a point (x0, y0) is within the specified area defined by xlist and ylist.
- inside_trap(geom: Polygon, x: float, y: float) -> bool:
    Checks if a point (x, y) is inside a polygon.
- set_limits(ax, xminmax, yminmax, aspect='equal'):
    Sets the limits and aspect ratio of a matplotlib axes object.
- _default_ax():
    Returns the default matplotlib axes object.
- fmt(x):
    Formats a number as a string with one decimal place.
- init_data_collecting(line: str, dtype: str) -> tuple:
    Initializes data collection based on the specified data type.
- read_ff_output(filename: str, ff_type: str) -> dict:
    Reads the output file and extracts the data based on the specified ff_type.

Classes:
- FieldAnalyzer: A class for analyzing and plotting field data extracted from FEM simulations.
"""


import numpy as np
import matplotlib.pyplot as plt

from numpy import ma
from shapely import Polygon, Point
from ..src.settings import GRAY


ff_types = ['2Dmap', '2Dslices']

def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def find_max_index(arr: list) -> int:
    max_index = np.unravel_index(arr.argmax(), arr.shape)
    return max_index


def find_nearest(array: list, value: float | int) -> int:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def center_within_area(x0: float, y0: float, xlist: list, ylist: list, tol=0.25) -> bool:
    if ((x0 + tol > xlist[-1]) or (x0 - tol < xlist[0])
        or (y0 + tol > ylist[-1]) or (y0 - tol < ylist[0])):
        return False
    return True


def inside_trap(geom: Polygon, x: float, y: float) -> bool:
    """ checks if (x,y) is inside polygon """
    return Point(x, y).within(geom)


def set_limits(ax, xminmax, yminmax, aspect='equal'):
    ax.set_xlim(*xminmax)
    ax.set_ylim(*yminmax)
    ax.set_aspect(aspect)


def _default_ax():
    ax = plt.gca()
    ax.set_aspect("equal")
    return ax


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s} "


def init_data_collecting(line: str, dtype: str) -> tuple:
    if dtype == "2Dmap":
        empty_array = []
    elif dtype == "2Dslices":
        empty_array = {}
    elif dtype == "xy":
        empty_array = []
    else:
        raise TypeError(f"your {dtype} is not from a list ['2Dmap', '2Dslices', 'xy']")
    t = line.split()
    dict_key = t[1]
    return dict_key, empty_array, dtype


def read_ff_output(filename: str, ff_type: str) -> dict:
    """ Reads the output file and extract the data based on the specified ff_type.
        Returns a dictionary with coupling constant matricies and 'x' and 'y' lists.

    Args:
    ____
    filename (str): The path to the input file.
    ff_type (str): The type of the extracted data.
                   Choose from '2Dmap' (single) or '2Dslices' (multiple).

    Raises:
    ----
    Exception: If the specified ff_type is incorrect.
    """

    if ff_type not in ff_types:
        raise TypeError(f"Incorrect {ff_type}, choose from {ff_types}")

    if ff_type == '2Dmap':
        # handles a single 2Dmap which has multiple coupling constants
        data = {}
        with open(filename) as file:
            for line in file:
                if line[:9] == 'startDATA':
                    t = line.split()
                    key, array, dtype = (t[1], [], 'data')
                elif line[:7] == 'startXY':
                    t = line.split()
                    key, array, dtype = (t[1], [], 'xy')
                elif line[:3] == 'END':
                    if dtype == 'data':
                        data.update({key: np.asarray(array, dtype=float)})
                    elif dtype == 'xy':
                        data.update({key: np.asarray(flatten(array), dtype=float)})
                    else: pass
                elif line.split() == []:
                    pass
                else:
                    array.append([float(item) for item in line.split()])

    elif ff_type == '2Dslices':
        # handles a multiple 2Dmaps which contains multiple coupling constants
        data = {}
        with open(filename) as file:
            for line in file:
                if line[:9] == 'startDATA':
                    t = line.split()
                    key, array, dtype = (t[1], {}, 'data')
                elif line[:12] == 'start2DSLICE':
                    t = line.split()
                    s_key, s_array = (t[1], [])
                elif line[:7] == 'startXY':
                    t = line.split()
                    key, s_array, dtype = (t[1], [], 'xy')
                elif line[:3] == 'end':
                    array.update({s_key: np.asarray(s_array, dtype=float)})
                elif line[:3] == 'END':
                    if dtype == 'data':
                        data.update({key: array})
                    elif dtype == 'xy':
                        data.update({key: np.asarray(flatten(s_array), dtype=float)})
                    else: pass
                elif line.split() == []:
                    pass
                else:
                    s_array.append([float(item) for item in line.split()])
    else: pass
    return data


###########################
#### Main Reader Class ####
###########################
class FieldAnalyzer():
    """ A class for analyzing and plotting field data extracted from FEM."""

    def __init__(self, *filename_args: tuple[str, str, str]):
        for fname, attrname, dtype in filename_args:
            data = read_ff_output(fname, dtype)
            setattr(self, attrname, data)

    def potential(self, couplingConst: dict, voltages: dict, zlevel_key=None) -> tuple:
        """ Calculates the potential distribution based on the coupling constants and voltages.

        Args:
        _____
        couplingConst (dict): A dictionary containing the coupling constants.
        voltages (dict): A dictionary containing the voltages.
        zlevel_key (optional): The key for the z-level. Defaults to None.

        Returns:
        _______
        tuple: A tuple containing the x-coordinates, y-coordinates, and the potential data.
        """
        nx, ny = len(couplingConst['xlist']), len(couplingConst['ylist'])
        data = np.zeros((nx, ny), dtype=np.float64)
        for (k, v) in couplingConst.items():
            if k in ('xlist', 'ylist'):
                pass
            else:
                if not zlevel_key:
                    data = data + voltages.get(k) * v
                else:
                    data = data + voltages.get(k) * v.get(zlevel_key)

        return couplingConst['xlist'], couplingConst['ylist'], data


    def plot_coupling_const(self, couplingConst: list, gate: str, ax=None) -> None:
        """ Plots the coupling constants for a specific gate.

        Args:
        _____
        couplingConst (list): The coupling constants.
        gate (str): The gate for which the coupling constants are plotted.
        ax (optional): The matplotlib axes object to plot on.
                       If not provided, a new figure and axes will be created.
        """
        if ax is None:
            ax = _default_ax()
        ax.contourf(couplingConst['xlist'], couplingConst['ylist'], couplingConst[gate], 17,
                    cmap='RdYlBu_r', vmin=-0.03)
        set_limits(ax, (couplingConst['xlist'][0], couplingConst['xlist'][-1]),
                   (couplingConst['ylist'][0], couplingConst['ylist'][-1]))


    def plot_potential2D(self,
                         couplingConst: list,
                         voltage_list: list,
                         ax=None,
                         zero_line=None,
                         **kwargs):
        """ Plots the 2D potential distribution based on the coupling constants and voltages.

        Args:
        _____
        couplingConst (list): The coupling constants.
        voltage_list (list): The voltages.
        ax (optional): The matplotlib axes object to plot on.
                        If not provided, a new figure and axes will be created.
        zero_line (optional): The value at which to draw a dashed line.
                              If True, the zero line will be drawn at 0.
                              If None, no zero line will be drawn.
        **kwargs: Additional keyword arguments to pass to the `contourf` function.
        """
        if ax is None:
            ax = _default_ax()
        data = self.potential(couplingConst, voltage_list)
        im = ax.contourf(data[0], data[1], np.transpose(data[2]), 17, **kwargs)
        if zero_line:
            if isinstance(zero_line,bool):
                zero_line = 0
            ax.contour(data[0], data[1], np.transpose(data[2]), [zero_line],
                       linestyles='dashed', colors=GRAY)


    def get_potential_1D(self,
                         couplingConst: dict,
                         voltages: dict,
                         xy_cut: str,
                         loc: float,
                         zlevel_key=None) -> np.ndarray:
        """ Returns (tuple(xlist, data)) the 1D potential distribution
            along a specified cut in the XY plane.

        Args:
        _____
        couplingConst (dict): A dictionary containing the coupling constants.
        voltages (dict): A dictionary containing the voltages.
        xy_cut (str): The cut direction. Can be either 'x' or 'y'.
        loc (float): The location along the cut.
        zlevel_key (optional): The key for the z-level. Defaults to None.
        """
        X, Y, Phi = self.potential(couplingConst, voltages, zlevel_key)
        if xy_cut == 'x':
            idx = find_nearest(Y, loc)
            return X, -Phi[:, idx]
        elif xy_cut == 'y':
            idy = find_nearest(X, loc)
            return Y, -Phi[idy, :]
        else:
            raise ValueError("xy_cut must be either 'x' or 'y'.")


    def get_field_1D(self,
                     couplingConst: dict,
                     voltages: dict,
                     xy_cut: str,
                     loc: float,
                     zlevel_key=None) -> np.ndarray:
        """ Returns (tuple(xlist, data)) the 1D electric field distribution
            along a specified cut in the XY plane.

        Args:
        _____
        couplingConst (dict): A dictionary containing the coupling constants.
        voltages (dict): A dictionary containing the voltages.
        xy_cut (str): The cut direction. Can be either 'x' or 'y'.
        loc (float): The location along the cut.
        zlevel_key (optional): The key for the z-level. Defaults to None.
        """
        X, Y, Phi = self.potential(couplingConst, voltages, zlevel_key)
        if xy_cut == 'x':
            idx = find_nearest(Y, loc)
            return X, np.gradient(Phi[:, idx], X)
        if xy_cut == 'y':
            idy = find_nearest(X, loc)
            return Y, np.gradient(Phi[idy, :], Y)
        else:
            raise ValueError("xy_cut must be either 'x' or 'y'.")


    def plot_potential_1D(self,
                          couplingConst: dict,
                          voltages: dict,
                          xy_cut: str,
                          loc: float,
                          ax=None,
                          zlevel_key=None,
                          **kwargs):
        """ Plots the 1D potential distribution along a specified cut in the XY plane.
            Returns ax: The matplotlib axes object.

        Args:
        _____
        couplingConst (dict): A dictionary containing the coupling constants.
        voltages (dict): A dictionary containing the voltages.
        xy_cut (str): The cut direction. Can be either 'x' or 'y'.
        loc (float): The location along the cut.
        ax (optional): The matplotlib axes object to plot on. If not provided,
                        a new figure and axes will be created.
        zlevel_key (optional): The key for the z-level. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `plot` function.
        """
        if ax is None:
            ax = _default_ax()

        x, y = self.get_potential_1D(couplingConst, voltages, xy_cut, loc, zlevel_key)
        ax.plot(x, y*1e3, **kwargs)
        ax.set_xlabel(r'$x$ or $y$ (um)')
        ax.set_ylabel(r'potential $-\phi(x)$ (mV)')
        return ax


    def plot_field_1D(self,
                      couplingConst: dict,
                      voltages: dict,
                      xy_cut: str,
                      loc: float,
                      ax=None,
                      zlevel_key=None,
                      **kwargs):
        """ Plots the 1D electric field distribution along a specified cut in the XY plane.
            Returns ax: The matplotlib axes object.

        Args:
        _____
        couplingConst (dict): A dictionary containing the coupling constants.
        voltages (dict): A dictionary containing the voltages.
        xy_cut (str): The cut direction. Can be either 'x' or 'y'.
        loc (float): The location along the cut.
        ax (optional): The matplotlib axes object to plot on. If not provided,
                        a new figure and axes will be created.
        zlevel_key (optional): The key for the z-level. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the `plot` function.
        """
        if ax is None:
            ax = _default_ax()

        x, y = self.get_field_1D(couplingConst, voltages, xy_cut, loc, zlevel_key)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(r'$x$ or $y$ (um)')
        ax.set_ylabel(r'field $E(x)$ (V/um)')
        return ax


    def crop_data(self, new_attr_name: str, attr_name: str, crop_area: Polygon) -> None:
        """ Crops the data based on the specified crop area and stores it as a new attribute.

        Args:
        _____
        new_attr_name (str): The name of the new attribute to store the cropped data.
        attr_name (str): The name of the attribute containing the original data.
        crop_area (Polygon): The polygon representing the crop area.
        """
        data = getattr(self, attr_name)
        y, x = np.meshgrid(data.get('xlist'), data.get('ylist'))
        mask = np.vectorize(inside_trap, excluded=["geom"])(crop_area, y, x)
        mask = np.invert(mask)
        mask = np.transpose(mask)

        cropped_data ={}
        for (k, v) in data.items():
            if k in ('xlist', 'ylist'):
                cropped_data[k] = v
            else:
                cropped_data[k] = ma.masked_array(v, mask=mask)

        setattr(self, new_attr_name, cropped_data)
