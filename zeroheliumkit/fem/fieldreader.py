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
- action_on_data(data_item: dict | np.ndarray, action: Callable, **kwargs) -> dict | np.ndarray:
    Applies a function to each item in a dictionary or to a numpy array.
- make_masked(couplings: CouplingConstants, mask_area: Polygon) -> CouplingConstants:
    Masks the coupling constants based on a specified polygon area.
- make_cropped(couplings: CouplingConstants, xrange: tuple=(-1,1), yrange: tuple=(-1,1)) -> CouplingConstants:
    Crops the coupling constants based on specified x and y ranges.
- make_symmetric(couplings: CouplingConstants, symmetry_axis: str, symmetric_electrodes: list, mirror_electrodes: list[tuple]) -> CouplingConstants:
    Makes the coupling constants symmetric based on a specified symmetry axis and electrode pairs.
- make_smooth(couplings: CouplingConstants, gaussian_power: int, **kwargs) -> CouplingConstants:
    Smooths the coupling constants using a Gaussian filter.

Classes:
- FreeFemResultParser: A class for parsing and loading results from FreeFem simulations.
- FieldAnalyzer: A class for analyzing and plotting field data extracted from FEM simulations.
"""

import yaml
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from typing import Callable
from numpy import ma
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from shapely import Polygon, Point
from pathlib import Path
from ..src.settings import GRAY


@dataclass(slots=True)
class CouplingConstants:
    x: np.ndarray
    y: np.ndarray
    data: dict


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def find_max_index(arr: list) -> int:
    max_index = np.unravel_index(arr.argmax(), arr.shape)
    return max_index


def find_nearest(array: list | ArrayLike, value: float | int) -> int:
    """
    Finds the nearest value in array. Returns index of array for which this is true.
    """
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

def generate_mask(xlist: list | np.ndarray, ylist: list | np.ndarray, mask_area: Polygon) -> np.ndarray:
    X, Y = np.meshgrid(xlist, ylist)
    mask = np.vectorize(inside_trap, excluded=["geom"])(mask_area, X, Y)
    mask = np.invert(mask)
    return mask

def set_limits(ax, xminmax, yminmax, aspect='equal'):
    ax.set_xlim(*xminmax)
    ax.set_ylim(*yminmax)
    ax.set_aspect(aspect)


def _default_ax():
    ax = plt.gca()
    #ax.set_aspect("equal")
    return ax


def crop_vector(x: ArrayLike, xrange: tuple) -> tuple:
    """Crops the vector to the boundaries specified by xrange."""
    xmin_idx, xmax_idx = find_nearest(x, xrange[0]), find_nearest(x, xrange[1])
    return x[xmin_idx:xmax_idx + 1], (xmin_idx, xmax_idx)


def crop_matrix(U: ArrayLike, x_idxs: tuple, y_idxs: tuple) -> ArrayLike:
    """Crops the matrix to the boundaries specified by xrange and yrange.

    Args:
    ----
        U (ArrayLike): two dimensional array to be cropped.
        x_idxs (tuple): tuple of two ints that indicate the min and max range for the x-coordinate.
        y_idxs (tuple): tuple of two int that indicate the min and max range for the y-coordinate.

    Returns:
        ArrayLike: cropped array
    """
    (x1,x2) = x_idxs
    (y1,y2) = y_idxs

    return U[y1 : y2 + 1, x1 : x2 + 1]


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} " if plt.rcParams["text.usetex"] else f"{s} "


def action_on_data(data_item: dict | np.ndarray, action: Callable, **kwargs) -> dict | np.ndarray:
    """ Applies a function to each item in the data dictionary.

    Args:
        data_item (dict | np.ndarray): The data to apply the function to.
        action (Callable): The function to apply.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns
    dict | np.ndarray: The data with the function applied to each item.
    """
    if isinstance(data_item, dict):
        new_dict = {}
        for (k, v) in data_item.items():
            new_dict[k] = action(v, **kwargs)
        return new_dict
    elif isinstance(data_item, np.ndarray):
        return action(data_item, **kwargs)


def make_masked(couplings: CouplingConstants, mask_area: Polygon) -> CouplingConstants:
    """
    Masks the data based on the specified mask area.

    Args:
        couplings (CouplingConstants): The coupling constants to be masked.
        mask_area (Polygon): The polygon representing the crop area.

    Returns:
        CouplingConstants: The masked coupling constants.
    """
    mask = generate_mask(couplings.x, couplings.y, mask_area)

    maskedcouplings = CouplingConstants(
        x = couplings.x,
        y = couplings.y,
        data = action_on_data(couplings.data, ma.masked_array, mask=mask)
    )
    return maskedcouplings


def make_cropped(couplings: CouplingConstants, xrange: tuple=(-1,1), yrange: tuple=(-1,1)) -> CouplingConstants:
    """
    Crop the data based on the specified x and y ranges.

    Args:
        couplings (CouplingConstants): The coupling constants to be cropped.
        xrange (tuple, optional): The range of x-values to crop the data. Defaults to (-1, 1).
        yrange (tuple, optional): The range of y-values to crop the data. Defaults to (-1, 1).

    Returns:
        CouplingConstants: The cropped coupling constants.
    """
    xnew, (i1, i2) = crop_vector(couplings.x, xrange)
    ynew, (j1, j2) = crop_vector(couplings.y, yrange)
    croppedcouplings = CouplingConstants(
        x = xnew,
        y = ynew,
        data = action_on_data(couplings.data, crop_matrix, x_idxs=(i1, i2), y_idxs=(j1, j2))
    )
    return croppedcouplings


def make_symmetric(
        couplings: CouplingConstants,
        axis: str,
        symmetric_electrodes: list=None,
        mirror_electrodes: list[tuple]=None
        ) -> CouplingConstants:
    """
    Makes the coupling constants symmetric based on the specified symmetry axis and electrode pairs.

    Args:
        couplings (CouplingConstants): The coupling constants to be made symmetric.
        axis (str): The axis of symmetry ('x' or 'y').
        symmetric_electrodes (list): The list of electrodes to make symmetric.
        mirror_electrodes (list[tuple]): The list of electrode pairs which are mirror to each other.

    Returns:
        CouplingConstants: The symmetric coupling constants.
    """

    data = couplings.data
    nx = len(couplings.x)
    ny = len(couplings.y)
    mid_idx_x = int(nx/2)
    mid_idx_y = int(ny/2)

    new_dict = {}

    if symmetric_electrodes:
        for k in symmetric_electrodes:
            if axis == 'x':
                v = data[k]
                idx = mid_idx_y
            elif axis == 'y':
                v = data[k].T
                idx = mid_idx_x
            else:
                raise ValueError("axis must be either 'x' or 'y'.")
            symm_array = np.zeros(v.shape)
            # averaging over positive and negative y
            averaged_half = (v[idx+1:,:] + v[:idx,:][::-1,:])/2
            symm_array[idx+1:,:] = averaged_half
            symm_array[:idx,:] = averaged_half[::-1,:]
            symm_array[idx,:] = v[idx,:]
            new_dict[k] = symm_array if axis == 'x' else symm_array.T

    if mirror_electrodes:
        for k1, k2 in mirror_electrodes:
            if axis == 'x':
                v1 = data[k1]
                v2 = data[k2]
            elif axis == 'y':
                v1 = data[k1].T
                v2 = data[k2].T
            else:
                raise ValueError("axis must be either 'x' or 'y.")
            v1 = data[k1]
            v2 = data[k2]
            mirror_array = (v1 + v2[::-1,:])/2
            new_dict[k1] = mirror_array if axis == 'x' else mirror_array.T
            new_dict[k2] = mirror_array[::-1,:] if axis == 'x' else mirror_array[::-1,:].T

    symmetriccouplings = CouplingConstants(
        x = couplings.x,
        y = couplings.y,
        data = new_dict
    )
    return symmetriccouplings


def make_smooth(couplings: CouplingConstants, sigma: int, **kwargs) -> CouplingConstants:
    """
    Smooths the coupling constants of the fieldreader object using a Gaussian filter.

    Args:
        couplings (CouplingConstants): The coupling constants to be smoothed.
        gaussian_power (int): The power of the Gaussian filter.
        **kwargs: Additional keyword arguments to pass to the Gaussian filter.
            See documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_.

    Returns:
        CouplingConstants: The smoothed coupling constants.
    """
    smoothedcouplings = CouplingConstants(
        x = couplings.x,
        y = couplings.y,
        data = action_on_data(couplings.data, gaussian_filter, sigma=sigma, **kwargs)
    )

    return smoothedcouplings



class FreeFemResultParser():
    """
    Parses and loads results from FreeFem simulations using metadata and data files.

    Args:
        metadata_file (str): Path to the YAML metadata file containing simulation information.
    """

    __slots__ = ['metadata', 'data']

    def __init__(self, metadata_file: str, show: bool=True):
        with open(metadata_file, 'r') as file:
            self.metadata = yaml.load(file, Loader=yaml.FullLoader)
        if show:
            self.print_table()
            print("Control Electrodes: " + str(self.metadata["Control Electrodes"]))


    def print_table(self):
        """
        Prints a formatted table of selected metadata attributes using the tabulate library.
        """

        exclude_list = ["Capacitance Matrix", "Control Electrodes"]
        meta = {k: v for k, v in self.metadata.items() if k not in exclude_list}
        col_names = list(meta.keys())
        row_names = meta[col_names[0]].keys()
        table = []
        for rname in row_names:
            row = [rname]
            for _, v in meta.items():
                row.append(v[rname])
            table.append(row)
        print(tabulate(table, headers=col_names))


    def load_data(self, savedir: str, fname: str):
        """
        Loads electrode field data from a Parquet file and organizes it into a structured dictionary.

        Args:
            savedir (str): Directory path where the Parquet file is stored.
            fname (str): Base filename (without extension) of the Parquet file to load.
        """
        fullpath = Path(savedir) / Path(fname + ".parquet")
        df = pl.read_parquet(str(fullpath))

        self.data = {}

        for electrode in df.columns:
            self.data[electrode] = {}
            electrode_res = df[electrode].to_numpy()

            schema = self.metadata[fname]["Schema"]
            num_strings = schema[1:-1].split(", ")
            shape = tuple(int(num) for num in num_strings)

            array = np.reshape(electrode_res, shape)
            for i, slice in enumerate(array):
                slice_value = self.metadata[fname]["Slice Values"][i]
                self.data[electrode][slice_value] = slice

        self.data['xlist'] = np.linspace(self.metadata[fname]["X Min"], self.metadata[fname]["X Max"], self.metadata[fname]["X Num"], endpoint=True)
        self.data['ylist'] = np.linspace(self.metadata[fname]["Y Min"], self.metadata[fname]["Y Max"], self.metadata[fname]["Y Num"], endpoint=True)


    def get_coupling_constants(self, slice_value: float, round_with_decimals: int=6) -> CouplingConstants:
        """ Returns the coupling constants at a specific slice value and rounds the data to the specified number of decimal places."""
        cc = CouplingConstants(
            x=self.data['xlist'],
            y=self.data['ylist'],
            data={k: np.round(v[slice_value], decimals=round_with_decimals) for k, v in self.data.items() if k not in ['xlist', 'ylist']}
        )
        return cc


    def get_capacitance_matrix(self):
        """ Returns the capacitance matrix from the metadata. Rows and columns order follows electrodes order in metadata. Units are multiples of 8.854 aF (eps0 * um= 8.8..x 10^-18 F) """
        return self.metadata["Capacitance Matrix"]


class FieldAnalyzer():
    """
    A class for analyzing and plotting field data extracted from FEM simulations.
    This class provides methods to:
    - Load and store field data as attributes.
    - Calculate potential distributions from coupling constants and voltages.
    - Plot 2D and 1D potential and electric field distributions.

    Args:
        coupling_constants (CouplingConstants): The coupling constants used for field analysis.

    """

    __slots__ = ['couplings', 'names', 'voltages', 'potential']

    def __init__(self, coupling_constants: CouplingConstants):
        # shorter attribute name; keep original as alias for backward compatibility
        self.couplings = coupling_constants
        self.names = list(coupling_constants.data.keys())


    def set_voltages(self, voltages: dict) -> None:
        """
        Sets the voltages for the electrodes.

        Args:
            voltages (dict): A dictionary containing the voltages.
        """
        self.voltages = {k: voltages[k] if k in voltages else 0 for k in self.names}
        self.update()


    def update(self) -> None:
        """
        Updates the potential distribution based on the current voltages.
        """
        self.accumulate_arrays()


    def accumulate_arrays(self) -> None:
        """
        Calculates the potential distribution based on the coupling constants and voltages.
        """
        nx, ny = len(self.couplings.x), len(self.couplings.y)
        self.potential = np.zeros((ny, nx), dtype=np.float64)
        for (k, v) in self.couplings.data.items():
            self.potential = self.potential + self.voltages.get(k) * v


    def get_data(self, slice: tuple=None) -> np.ndarray:
        """
        Returns the potential distribution. If a slice is provided, returns the potential along that slice.

        Args:
            slice (tuple, optional): A tuple specifying the slice of data to use. Defaults to None -> In this case, the full 2D array is returned.
        """
        if slice is None:
            return self.potential
        else:
            assert len(slice) == 2, "slice must be a tuple of length 2"
            assert slice[0] in ["x", "y"], "slice[0] must be either 'x' or 'y'"
            assert isinstance(slice[1], (int, float)), "slice[1] must be a number"
            if slice[0] == "x":
                idx = find_nearest(self.couplings.y, slice[1])
                return self.potential[idx, :]
            elif slice[0] == "y":
                idx = find_nearest(self.couplings.x, slice[1])
                return self.potential[:, idx]
            else:
                raise ValueError("Invalid slice[0]. Must be either 'x' or 'y'.")


    def get_gradient(self, grad_axis: str, slice: tuple=None) -> np.ndarray:
        """Returns the gradient of the potential data.

        Args:
            grad_axis (str): The axis along which to compute the gradient. Must be either 'x' or 'y'.
            slice (tuple, optional): A tuple specifying the slice of data to use. Defaults to None -> In this case, the full 2D array is used.

        Raises:
            ValueError: If grad_axis is not 'x' or 'y'.

        Returns:
            np.ndarray: The gradient of the potential data.
        """
        assert grad_axis in ["x", "y"], "grad_axis must be either 'x' or 'y'"
        if slice is None:
            if grad_axis == "x":
                return np.gradient(self.potential, self.couplings.x, axis=1)
            elif grad_axis == "y":
                return np.gradient(self.potential, self.couplings.y, axis=0)
        else:
            data = self.get_data(slice)
            if grad_axis == "x":
                return np.gradient(data, self.couplings.x)
            elif grad_axis == "y":
                return np.gradient(data, self.couplings.y)
            else:
                raise ValueError("Invalid grad_axis. Must be either 'x' or 'y'.")


    def plot2D_data(self, ax=None, num_levels: int=17, zero_line :float | int=None, **kwargs):
        """
        Plots the 2D potential distribution based on the coupling constants and voltages.

        Args:
            ax (optional): The matplotlib axes object to plot on.
                If not provided, a new figure and axes will be created.
            num_levels (int, optional): The number of contour levels to use. Defaults to 17.
            zero_line (optional): The value at which to draw a dashed line.
                If True, the zero line will be drawn at 0.
                If None, no zero line will be drawn.
            **kwargs: Additional keyword arguments to pass to the `contourf` function.
                See documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html>`_.
        """
        if ax is None:
            ax = _default_ax()
        ax.contourf(self.couplings.x, self.couplings.y, self.potential, num_levels, **kwargs)
        if zero_line:
            if isinstance(zero_line, bool):
                zero_line = 0
            ax.contour(self.couplings.x, self.couplings.y, self.potential, [zero_line],
                       linestyles='dashed', colors=GRAY)


    def plot2D_gradient(self, grad_axis: str, ax=None, num_levels: int=17, **kwargs):
        """Plots the 2D gradient distribution based on the coupling constants and voltages.

        Args:
            grad_axis (str): The axis along which to compute the gradient. Must be either 'x' or 'y'.
            ax (optional): The matplotlib axes object to plot on. Defaults to None.
            num_levels (int, optional): The number of contour levels to use. Defaults to 17.
            **kwargs: Additional keyword arguments to pass to the `contourf` function.
                See documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html>`_.

        Raises:
            ValueError: If grad_axis is not 'x' or 'y'.
        """
        if ax is None:
            ax = _default_ax()
        if grad_axis == "x":
            gradient = self.get_gradient("x")
        elif grad_axis == "y":
            gradient = self.get_gradient("y")
        else:
            raise ValueError("Invalid grad_axis. Must be either 'x' or 'y'.")
        ax.contourf(self.couplings.x, self.couplings.y, gradient, num_levels, **kwargs)


    def plot2D_vectorfield(self, ax=None, step: int=5, **kwargs):
        """
        Plots the 2D electric field distribution as a vector field.

        Args:
            ax (optional): The matplotlib axes object to plot on. Defaults to None.
                If not provided, a new figure and axes will be created.
            **kwargs: Additional keyword arguments to pass to the matplotlib `quiver` function.
                See documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html>`_.
        """
        if ax is None:
            ax = _default_ax()
        Ex = -self.get_gradient("x")[::step, ::step]
        Ey = -self.get_gradient("y")[::step, ::step]
        mag = np.sqrt(Ex**2 + Ey**2)
        X, Y = np.meshgrid(self.couplings.x[::step], self.couplings.y[::step])
        ax.quiver(
            X,
            Y,
            Ex,
            Ey,
            mag,
            **kwargs
            )
        ax.set_xlabel(r'$x$ (um)')
        ax.set_ylabel(r'$y$ (um)')


    def plot1D_data(self, slice: tuple, ax=None, scale=1, add_offset=0, **kwargs):
        """
        Plots the 1D potential distribution along a specified slice in the XY plane.

        Args:
            slice (tuple): A tuple specifying the slice of data to use. Must be of the form ('x' or 'y', value).
            scale (float, optional): A scaling factor to apply to the potential data. Defaults to 1e3.
            add_offset (float, optional): An offset to add to the potential data after scaling. Defaults to 0.
            ax (optional): The matplotlib axes object to plot on. If not provided,
                a new figure and axes will be created.
            **kwargs: Additional keyword arguments to pass to the `plot` function.
        """
        if ax is None:
            ax = _default_ax()
        if slice[0] == "x":
            x = self.couplings.x
            xlabel = r'$x$ (um)'
        elif slice[0] == "y":
            x = self.couplings.y
            xlabel = r'$y$ (um)'
        else:
            raise ValueError("Invalid slice[0]. Must be either 'x' or 'y'.")
        y = self.get_data(slice)

        ax.plot(x, y * scale + add_offset, **kwargs)
        ax.set_xlabel(xlabel)


    def plot1D_gradient(self, grad_axis: str, slice: tuple, ax=None, scale=1, add_offset=0, **kwargs):
        """
        Plots the 1D gradient of the potential distribution along a specified slice in the XY plane.

        Args:
            grad_axis (str): The axis along which to compute the gradient. Must be either 'x' or 'y'.
            slice (tuple): A tuple specifying the slice of data to use. Must be of the form ('x' or 'y', value).
            scale (float, optional): A scaling factor to apply to the gradient data. Defaults to 1e3.
            add_offset (float, optional): An offset to add to the gradient data after scaling. Defaults to 0.
            ax (optional): The matplotlib axes object to plot on. If not provided,
                a new figure and axes will be created.
            **kwargs: Additional keyword arguments to pass to the `plot` function.
        """
        if ax is None:
            ax = _default_ax()
        if slice[0] == "x":
            x = self.couplings.x
            xlabel = r'$x$ (um)'
        elif slice[0] == "y":
            x = self.couplings.y
            xlabel = r'$y$ (um)'
        else:
            raise ValueError("Invalid slice[0]. Must be either 'x' or 'y'.")
        y = self.get_gradient(grad_axis, slice)

        ax.plot(x, y * scale + add_offset, **kwargs)
        ax.set_xlabel(xlabel)


    def print_voltages(self) -> None:
        print(tabulate(prepare_to_tabulate(self.voltages), tablefmt='fancy_grid', floatfmt=".2f"))
