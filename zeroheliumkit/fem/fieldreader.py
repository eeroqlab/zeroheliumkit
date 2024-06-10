import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from shapely import Polygon
from ..src.settings import *


ff_types = ['2Dmap', '2Dslices']

def flatten(l):
    return [item for sublist in l for item in sublist]

def find_max_index(arr):
    max_index = np.unravel_index(arr.argmax(), arr.shape)
    return max_index

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def center_within_area(x0: float, y0: float, xlist: list, ylist: list, tol=0.25) -> bool:
    if (x0 + tol > xlist[-1]) or (x0 - tol < xlist[0]) or (y0 + tol > ylist[-1]) or (y0 - tol < ylist[0]):
        return False
    else:
        return True

def inside_trap(geom: Polygon, x: float, y: float) -> bool:
    """ checks if (x,y) is inside polygon """
    return Point(x, y).within(geom)

def set_limits(ax, x0, xN, y0, yN, aspect='equal'):
    ax.set_xlim(x0, xN)
    ax.set_ylim(y0, yN)
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
    
    if ff_type not in ff_types:
        raise Exception(f"Incorrect {ff_type}, choose from {ff_types}")
    
    if ff_type == '2Dmap':
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
        data = {}
        #s_array = []
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


# Main Class
class FieldAnalyzer():
    
    def __init__(self, *filename_args: tuple[str, str, str]):
        for fname, attrname, dtype in filename_args:
            data = read_ff_output(fname, dtype)
            setattr(self, attrname, data)

    def potential(self, couplingConst: dict, voltages: dict, zlevel_key=None) -> tuple:
        num = len(couplingConst.keys()) - 2
        nx, ny = len(couplingConst['xlist']), len(couplingConst['ylist'])
        data = np.zeros((nx, ny), dtype=np.float64)
        for (k, v) in couplingConst.items():
            if k=='xlist' or k=='ylist':
                pass
            else:
                if not zlevel_key:
                    data = data + voltages.get(k) * v
                else:
                    data = data + voltages.get(k) * v.get(zlevel_key)

        return couplingConst['xlist'], couplingConst['ylist'], data
    
    def plot_coupling_const(self, couplingConst: list, gate: str, ax=None) -> None:
        if ax is None:
            ax = _default_ax()
        ax.contourf(couplingConst['xlist'], couplingConst['ylist'], couplingConst[gate], 17, cmap='RdYlBu_r', vmin=-0.03)
        set_limits(ax, couplingConst['xlist'][0], couplingConst['xlist'][-1], couplingConst['ylist'][0], couplingConst['ylist'][-1])
    
    def plot_potential2D(self, couplingConst: list, voltage_list: list, ax=None, zero_line = None, **kwargs):
        if ax is None:
            ax = _default_ax()
        data = self.potential(couplingConst, voltage_list)
        im = ax.contourf(data[0], data[1], np.transpose(data[2]), 17, **kwargs)
        if zero_line:
            if isinstance(zero_line,bool):
                zero_line = 0
            ax.contour(data[0], data[1], np.transpose(data[2]), [zero_line], linestyles='dashed', colors=GRAY)


    def get_potential_1D(self, couplingConst: dict, voltages: dict, xy_cut: str, loc: float, zlevel_key=None) -> np.ndarray:
        """ Return the 1D potential distribution along a specified cut in the XY plane.

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
    
    def get_field_1D(self, couplingConst: dict, voltages: dict, xy_cut: str, loc: float, zlevel_key=None) -> np.ndarray:
        """ Return the 1D electric field distribution along a specified cut in the XY plane.

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
        elif xy_cut == 'y':
            idy = find_nearest(X, loc)
            return Y, np.gradient(Phi[idy, :], Y)
        else:
            raise ValueError("xy_cut must be either 'x' or 'y'.")


    def plot_potential_1D(self, couplingConst: dict, voltages: dict, xy_cut: str, loc: float, ax=None, zlevel_key=None, **kwargs):
        if ax is None:
            ax = _default_ax()

        x, y = self.get_potential_1D(couplingConst, voltages, xy_cut, loc, zlevel_key)
        ax.plot(x, y*1e3, **kwargs)
        ax.set_xlabel('$x$ or $y$ (um)')
        ax.set_ylabel('potential $-\phi(x)$ (mV)')
        return ax
    
    def plot_field_1D(self, couplingConst: dict, voltages: dict, xy_cut: str, loc: float, ax=None, zlevel_key=None, **kwargs):
        if ax is None:
            ax = _default_ax()

        x, y = self.get_field_1D(couplingConst, voltages, xy_cut, loc, zlevel_key)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('$x$ or $y$ (um)')
        ax.set_ylabel('field $E(x)$ (V/um)')
        return ax
    
    def crop_data(self, new_attr_name: str, attr_name: str, crop_area: Polygon) -> None:
        data = getattr(self, attr_name)
        yy, xx = np.meshgrid(data.get('xlist'), data.get('ylist'))
        mask = np.vectorize(inside_trap, excluded=["geom"])(crop_area, yy, xx)
        mask = np.invert(mask)
        mask = np.transpose(mask)

        cropped_data ={}
        for (k, v) in data.items():
            if k=='xlist' or k=='ylist':
                cropped_data[k] = v
            else:
                cropped_data[k] = ma.masked_array(v, mask=mask)

        setattr(self, new_attr_name, cropped_data)