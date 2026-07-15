""" 
freefemer.py

This module contains functions and a class for creating freefem files. Created by Niyaz B / January 10th, 2023.
"""

import os
import yaml
import sys
import asyncio
import psutil
import shutil
import subprocess
import logging
import polars as pl
import numpy as np
import ipywidgets as widgets

from pathlib import Path
from platform import system
from dataclasses import dataclass, asdict
from typing import List, Optional
from IPython.display import display

from ..src.errors import *
from ..helpers.constants import rho, g, alpha

# from logging.handlers import FileHandler


axis_ordering = {'xy':   'ax1,ax2,ax3',
                 'xz':   'ax1,ax3,ax2',
                 'yz':   'ax3,ax1,ax2'
}
config_quantity = {'phi': 'u',
                   'Ex': 'dx(u)',
                   'Ey': 'dy(u)',
                   'Ez': 'dz(u)'}


def scaling_size(bulk_helium_distance: float=1e-1):
    """
    Calculates the scaling size for the helium curvature displacement based on the bulk helium distance.

    Args
        bulk_helium_distance (float): The distance between the bulk helium atoms in meters. Default is 1e-1 m.

    Returns
        lengthscale (float): The scaling size in micrometers.
    """
    lengthscale = (rho * g * bulk_helium_distance)/alpha * 1e-6      
    return lengthscale


def headerFrame(header: str) -> str:
    """
    Creates a header frame for the FreeFEM script.

    Args
        header (str): The header text to be included in the frame.

    Returns
        edp (str): A formatted string containing the header frame.
    """
    edp = '\n//////////////////////////////////////////////////////////\n'
    edp += '//\n'
    edp += '//    ' + header + '\n'
    edp += '//\n'
    edp += '//////////////////////////////////////////////////////////\n\n'
    return edp


def add_spaces(num: int) -> str:
    """
    Adds a specified number of spaces for indentation in the FreeFEM script.

    Args
        num (int): The number of spaces to add.

    Returns
        str: A string containing the specified number of spaces.
    """
    return ' ' * num


def get_dict_by_name(items: list[dict], name: str) -> dict | None:
    """
    Return the first dictionary from the list where dict['name'] == name.
    If not found, return None.
    """
    return next((item for item in items if item.get("name") == name), None)


def format_freefem_path(*parts: str) -> str:
    """
    Join path components using OS-correct separators, while ensuring FreeFEM-compatible escaping on Windows.

    Args:
        *parts (str): Components of the path.

    Returns:
        str: A fully joined, platform-correct path.
    """
    platform = system()
    # Use os.path.join to get a correct native path
    path = os.path.join(*parts)

    # FreeFEM requires escaped backslashes on Windows for string literals
    if platform == "Windows":
        path = path.replace("\\", "\\\\")  # escape backslashes for FreeFEM
    return path



class FreeFemError(Exception):
    pass

@dataclass
class MeshAdaptationConfig():
    """
    Dataclass for storing mesh adaptation configuration parameters.

    Args:
        mesh_adaptation (bool): Whether to use mesh adaptation. Default is False.
        anisotropic_adaptation (bool): Whether to use anisotropic adaptation. Default is False.
        n_adapt (int): Number of adaptation iterations. Default is 3.
        err_target (float): Interpolation error target for mshmet. Default is 0.01.
        hmin_scale (float): Divisor for computing hmin from domain size. Default is 500.0.
        hmax_scale (float): Divisor for computing hmax from domain size. Default is 5.0.
        save_adapted_mesh (bool): Whether to save the adapted mesh to disk. Default is False.
    """
    mesh_adaptation: bool = False
    anisotropic_adaptation: bool = False
    n_adapt: int = 3
    err_target: float = 0.01
    hmin_scale: float = 500.0 # fine near features scaling
    hmax_scale: float = 5.0 #// coarse features scaling
    save_adapted_mesh: bool = False

    def __post_init__(self):
        if not isinstance(self.mesh_adaptation, bool):
            raise TypeError("'mesh_adaptation' must be a boolean")
        if not isinstance(self.anisotropic_adaptation, bool):
            raise TypeError("'anisotropic_adaptation' must be a boolean")
        if not isinstance(self.n_adapt, int) or self.n_adapt < 1:
            raise ValueError("'n_adapt' must be a positive integer")
        if not isinstance(self.err_target, float) or self.err_target <= 0:
            raise ValueError("'err_target' must be a positive float")
        if not isinstance(self.hmin_scale, float) or self.hmin_scale <= 0:
            raise ValueError("'hmin_scale' must be a positive float")
        if not isinstance(self.hmax_scale, float) or self.hmax_scale <= 0:
            raise ValueError("'hmax_scale' must be a positive float")
        if not isinstance(self.save_adapted_mesh, bool):
            raise TypeError("'save_adapted_mesh' must be a boolean")
        if self.anisotropic_adaptation and not self.mesh_adaptation:
            raise ValueError("'anisotropic_adaptation' can only be True if 'mesh_adaptation' is also True")


@dataclass
class ExtractConfig():
    """
    Dataclass for storing extraction configuration parameters.

    Args:
        name (str): The name of the extract config; acts as a label for the data configuration.
        quantity (str): The quantity to be extracted (e.g., 'phi', 'Ex', 'Ey', 'Ez', 'Cm').
        plane (str): The plane for extraction (e.g., 'xy', 'yz', 'xz', 'xyZ').
        coordinate1 (tuple): A tuple containing the start and end coordinates and the number of points for the first coordinate.
        coordinate2 (tuple): A tuple containing the start and end coordinates and the number of points for the second coordinate.
        coordinate3 (float | list | dict): User can input a float, list, or dictionary depending on their desired configuration. All inputs are converted to a list.
    """ 
    name: str 
    quantity: str
    plane: str
    coordinate1: tuple
    coordinate2: tuple
    coordinate3: int | float | list | dict
    curvature_config: dict = None

    def __post_init__(self):
        if self.quantity not in config_quantity.keys():
            raise KeyError(f'unsupported extract quantity. Supported quantity types are {config_quantity}')
        if self.plane not in axis_ordering.keys():
            raise KeyError(f'Wrong plane! choose from {axis_ordering.keys()}')
        if not isinstance(self.coordinate1, tuple):
            raise TypeError("'coordinate1' parameter must be a tuple (x1, x2, num)")
        if not isinstance(self.coordinate2, tuple):
            raise TypeError("'coordinate2' parameter must be a tuple (y1, y2, num)")

        if isinstance(self.coordinate3, (float, int)):
            self.coordinate3 = [self.coordinate3]
        elif isinstance(self.coordinate3, dict):
            self.curvature_config = self.coordinate3
            self.coordinate3 = self.curvature_config['bulk_helium_distances']
        elif not isinstance(self.coordinate3, list):
            raise TypeError("'coordinate3' parameter must be a list, float, or helium curvature config dict")
        

@dataclass
class FFconfigurator():
    """
    Dataclass to create the FreeFEM config yaml file.

    Args:
        config_file (str): Path to the FreeFEM config yaml file.
        dielectric_constants (dict): Dictionary containing the dielectric constants for different physical volumes.
        ff_polynomial (int): Polynomial order for the FreeFEM script.
        extract_opt (list[ExtractConfig] | dict): List of ExtractConfig objects or a dictionary containing extraction options.
        msh_refinements (int): The number of iterations over which to refine the GMSH meshfile using TetGen.
            If no number is provided, will default to None and not iterate at all.
        adaptation_config (MeshAdaptationConfig): Configuration object for mesh adaptation settings.
            If None, no mesh adaptation is used. Default is None.
    """
    config_file: str
    dielectric_constants: dict
    ff_polynomial: int
    extract_opt: list[ExtractConfig, dict] | dict | ExtractConfig
    msh_refinements: int = None
    adaptation_config: MeshAdaptationConfig = None

    def __post_init__(self):
        with open(self.config_file, 'r') as file:
            gmsh_config = yaml.safe_load(file)
        with open(self.config_file, 'w') as file:
            mergeddict = gmsh_config | self.__dict__
            if isinstance(self.extract_opt, list):
                mergeddict["extract_opt"] = [asdict(e) if isinstance(e, ExtractConfig) else e for e in self.extract_opt]
            elif isinstance(self.extract_opt, ExtractConfig):
                mergeddict["extract_opt"] = [asdict(self.extract_opt)]
            elif isinstance(self.extract_opt, dict):
                mergeddict["extract_opt"] = [self.extract_opt]

            del mergeddict["config_file"]

            if isinstance(self.adaptation_config, MeshAdaptationConfig):
                mergeddict['adaptation_config'] = asdict(self.adaptation_config)
            yaml.safe_dump(mergeddict, file, sort_keys=False, indent=3)



# ============================================================
# Automatic Discovery of FreeFem Installation
# ============================================================

def detect_freefem() -> Optional[str]:
    """
    Returns the directory containing FreeFem++ or None.
    Works on Windows, macOS, Linux.
    """

    candidates = []

    if sys.platform.startswith("win"):
        # standard Windows installs
        candidates += [
            r"C:\Program Files\FreeFem++",
            r"C:\Program Files (x86)\FreeFem++",
        ]
    elif sys.platform == "darwin":
        # macOS standard locations
        candidates += [
            "/Applications/FreeFem++.app/Contents/MacOS",
            "/usr/local/bin",
            "/opt/homebrew/bin",
        ]
    else:
        # Linux
        candidates += [
            "/usr/bin",
            "/usr/local/bin",
            "/snap/bin",
        ]

    exe_names = ["FreeFem++", "FreeFem++.exe"]

    for d in candidates:
        d = Path(d)
        for exe in exe_names:
            if (d / exe).exists():
                return str(d)

    # Try PATH
    for exe in exe_names:
        if shutil.which(exe):
            return str(Path(shutil.which(exe)).parent)

    return None




def get_edp_logger(edp_file: str) -> logging.Logger:
    """
    Create or return a logger dedicated to a specific EDP file.
    Thread-safe and can be called multiple times.
    """

    file_path =  Path(edp_file)
    log_dir = file_path.parent.parent / Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"freefem.{file_path.stem}"
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers twice
    if logger.handlers:
        for h in logger.handlers:
            h.close()
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # Log file path
    log_path = log_dir / f"{file_path.stem}.log"

    # Rotating file handler (2MB per file, keep 3 backups)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))

    # Console handler (INFO-level)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(logging.Formatter(
    #     f"[{edp_file}] %(levelname)s: %(message)s"
    # ))

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    logger.propagate = False  # keep logs separate

    return logger




class EDPpreparer():
    """
    Class for creating and running FreeFEM scripts.

    Args:
        config (str): filepath containing FreeFEM config yaml file.
    """
    
    def __init__(self, config_file: str):

        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.savedir = Path(self.config["savedir"]) / Path("edp")
        self.savedir.mkdir(exist_ok=True)

        self.physicalVols = self.config.get('physicalVolumes')
        self.physicalSurfs = self.config.get('physicalSurfaces')

        self.edp_files = []
        self.result_files = {config["name"]: [] for config in self.config["extract_opt"]}
        self.result_files["cm"] = []
        self.num_electrodes = len(self.physicalSurfs)

        if isinstance(self.config.get('extract_opt'), dict):
            self.config['extract_opt'] = [self.config['extract_opt']]

        self.write_edpScript()


    def add_helium_curvature_edp(self, extract_cfg: ExtractConfig) -> str:
        """
        Adds the helium curvature script to the FreeFEM script if the curvature configuration is provided.

        Args:
            extract_cfg (dict): Dictionary containing the extraction configuration.

        Returns:
            code (str): code containing the helium curvature script.
        """
        code  = headerFrame("HELIUM CURVATURE")
        code += extract_cfg.get('curvature_config')["script"]
        code += headerFrame("HELIUM CURVATURE")
        return code
    

    def write_edpScript(self):
        """
        Creates the main FreeFEM script based on the configuration and physical surfaces.
        """
        for electrode in self.physicalSurfs.keys():
            code = self.make_edp_content(electrode)
            path = format_freefem_path(str(self.savedir), "ff_" + electrode + ".edp")
            self.edp_files.append(path)
            with open(path, 'w') as file:
                file.write(code)


    def make_edp_content(self, electrode_name: int) -> str:
        """
        Returns the contents of an electrode_k.edp file with the desired electrode name in the place of 'k'.

        Args:
            electrode_name (str): Name of the electrode to generate .edp file content for.

        Returns:
            code (str): code containing the entire edp content written for electrode_name.
        """
        code = ''
        for c in self.config.get('extract_opt'):
            if c.get('curvature_config'):
                code += self.add_helium_curvature_edp(c)
                break
        code += self.script_create_savefiles(electrode_name)
        code += self.script_load_packages_and_mesh()
        adapt_cfg = self.config.get('adaptation_config') or {}
        mesh_adaptation = adapt_cfg.get('mesh_adaptation', False)
        if mesh_adaptation:
            code += 'cout << "=== Initial mesh: " << Th.nv << " vertices, " << Th.nt << " tetrahedra ===" << endl;\n'
        code += self.script_declare_variables()
        code += self.script_create_coupling_const_matrix()
        aniso = adapt_cfg.get('anisotropic_adaptation', False)
        n_adapt = adapt_cfg.get('n_adapt', 3)
        err_target = adapt_cfg.get('err_target', 0.01)
        hmin_scale = adapt_cfg.get('hmin_scale', 500.0)
        hmax_scale = adapt_cfg.get('hmax_scale', 5.0)
        save_adapted_mesh = adapt_cfg.get('save_adapted_mesh', False)
        code += self.script_problem_definition(electrode_name, mesh_adaptation=mesh_adaptation)
        if mesh_adaptation:
            code += self.script_mesh_adaptation(electrode_name, aniso=aniso, n_adapt=n_adapt, err_target=err_target, hmin_scale=hmin_scale, hmax_scale=hmax_scale, save_adapted_mesh=save_adapted_mesh)
        elif self.config.get('msh_refinements'):
            code += self.script_refine_mesh(self.config.get('msh_refinements'))
        code += self.script_save_cmatrix(electrode_name)

        for extract_config in self.config.get('extract_opt'):
            code += self.script_save_data(extract_config)

        return code


    def script_create_savefiles(self, electrode_name: str):
        """
        Creates the necessary files for saving results based on the configuration and electrode index.

        Args:
            electrode_name (str): Name of the electrode for which the files are being created.

        Returns:
            code (str): string containing the necessary lines of code to save the data.
        """
        code = "\n"

        for econfig in self.config.get('extract_opt'):
            name = econfig['name']
            code += f"ofstream {name}"
            name += f"_{electrode_name}"

            path = format_freefem_path(str(self.savedir), name + ".npy")
            self.result_files[econfig['name']].append((electrode_name, path))
            
            code += f"""("{path}", binary);\n"""
        
        return code


    def script_load_packages_and_mesh(self) -> str:
        """
        Loads the necessary FreeFEM packages and the mesh file into the script.

        Returns:
            code (str): code containing the necessary FreeFEM packages and mesh file declarations.
        """
        code = """load "msh3"\n"""
        code += """load "gmsh"\n"""
        code += """load "medit"\n"""
        code += """load "mshmet"\n"""
        code += """load "tetgen"\n"""
        code += "\n"
        # path = format_freefem_path(str(self.savedir), self.config["meshfile"])
        path = Path(self.config["meshfile"])
        if sys.platform.startswith("win"):
            path = str(path).replace("\\", "\\\\")
        else:
            path = str(path)

        code += f"""mesh3 Th = gmshload3("{path}");\n"""
        return code


    def script_declare_variables(self) -> str:
        """
        Declares the necessary variables for the FreeFEM script, including physical surfaces and volumes.

        Returns:
            code (str): code containing the necessary variable declarations.
        """

        code = "\n"
        code += "int n1, n2, n3;\n"
        code += "real xmin, xmax, ymin, ymax, ax3;\n"

        return code


    def script_create_coupling_const_matrix(self) -> str:
        """
        Creates the coupling constant matrix for the FreeFEM script, which is used to define the interaction between electrodes.

        Returns:
            code (str): code containing the coupling constant matrix.
        """
        number_of_electrodes = len(list(self.physicalSurfs.keys()))
        electrode_list = list(self.physicalSurfs.values())

        code = "\n"

        code += f"int numV = {number_of_electrodes};\n"
        code += "\n"
        code += f"real[int] electrodeid = {electrode_list};"
        code += "\n"

        return code


    def _script_dielectric(self, var_name: str, space_name: str = "FunctionRegion") -> str:
        """
        DOCSTRING HERE!
        Helper function

        Returns:
            code(str): code containing the dielectric constant
        """
        epsilon = self.config["dielectric_constants"]
        code = f"{space_name} {var_name} =\n"
        for k, v in self.physicalVols.items():
            code += add_spaces(26) + f"+ {epsilon[k]} * (region == {v})\n"
        code += add_spaces(26) + ";\n"
        return code


    def _script_electrode_bcs(self, electrode_name: str, var_name: str) -> str:
        """
        DOCSTRING HERE!

        Helper function

        Returns:
            code(str): code contsining the electrode names, and grounded ones
        """
        main_electrode = self.physicalSurfs.get(electrode_name)
        ground_electrodes = [v for v in self.physicalSurfs.values() if v != main_electrode]
        code = add_spaces(16) + f"+ on({main_electrode}, {var_name} = 1.0)\n"
        for v in ground_electrodes:
            code += add_spaces(16) + f"+ on({v}, {var_name} = 0.0)\n"
        return code
    
    def _script_macros(self) -> str:
        """
        Defines the FreeFEM macros used in the problem formulation.

        Returns:
            code (str): code containing the norm, Grad, and field macro definitions.
        """
        code = "macro norm [N.x,N.y,N.z] //\n"
        code += "macro Grad(u) [dx(u),dy(u),dz(u)] //\n"
        code += "macro field(u,x,y,z) [dx(u)(x,y,z),dy(u)(x,y,z),dz(u)(x,y,z)] //\n \n"
        return code


    def script_problem_definition(self, electrode_name: str, mesh_adaptation: bool = False, declare_globals: bool = True) -> str:
        """
        Defines the problem for the electrostatic potential in FreeFEM, including the finite element space and the dielectric constants.

        Args:
            electrode_name (str): Name of the electrode for which the problem is being defined.
            mesh_adaptation (bool): If True, skips the final Electro; call so that
                script_mesh_adaptation() can call it on the adapted mesh instead. Default is False.
            declare_globals (bool): If True, emits the eps constant and macro definitions.
                Set to False when these have already been declared earlier in the script
                (e.g. during the final solve of mesh adaptation). Default is True.

        Returns:
            code (str): code containing the problem definition.
        """
        polynomial = self.config["ff_polynomial"]

        code = "\n"

        if polynomial == 1:
            femSpace = 'P13d'
        elif polynomial == 2:
            femSpace = 'P23d'
        else:
            raise Exception("Wrong polynomial order! Choose between 1 or 2")

        if declare_globals:
            code += "real eps = 1e-6;\n"
            code += self._script_macros()

        if not mesh_adaptation:
            if 'periodic_BC' in self.config:
                code += f"""fespace Vh(Th,{femSpace}, periodic=[[{self.config.get('periodic_BC')[0]}, x, y], [{self.config.get('periodic_BC')[1]}, x, y]]);\n"""
            else:
                code += f"""fespace Vh(Th,{femSpace});\n"""

            code += "fespace FunctionRegion(Th,P03d);\n"
            code += "Vh u,v;\n"
            code += self._script_dielectric("dielectric")

            code += "problem Electro(u,v,solver=CG) =\n"
            code += add_spaces(16) + "int3d(Th)(dielectric * Grad(u)' * Grad(v))\n"
            code += self._script_electrode_bcs(electrode_name, "u")
            code += add_spaces(16) + ";\n"
            code += "Electro;\n"

        return code
    

    

    def script_save_data(self, config: dict) -> str:
        """
        Generates a code block for extracting 2D slice data based on the provided configuration.

        Returns:
            str: A string containing the generated code block for 2D slice data extraction.
        """


        xyz = axis_ordering[config.get('plane')]

        code  = headerFrame("2D SLICES DATA EXTRACTION BLOCK START")
        code += "{\n"
        
        code += f"n1 = {config['coordinate1'][2]};\n"
        code += f"n2 = {config['coordinate2'][2]};\n"
        code += f"xmin = {config['coordinate1'][0]};\n"
        code += f"xmax = {config['coordinate1'][1]};\n"
        code += f"ymin = {config['coordinate2'][0]};\n"
        code += f"ymax = {config['coordinate2'][1]};\n"
        code += f"n3 = {len(config['coordinate3'])};\n"

        if config.get('curvature_config'):
            bulkHelevels = np.asarray(config.get('curvature_config')["bulk_helium_distances"])
            scaling = scaling_size(bulkHelevels)
            surfaceHelevel = config.get('curvature_config')["surface_helium_level"]
            code += f"real[int] bulkHeliumLevels = {np.array2string(bulkHelevels, separator=', ')};\n"
            code += f"real[int] bulkHeliumLevelDispScales = {np.array2string(scaling, separator=', ')};\n"
        else:
            code += f"real[int] zcoords = {config['coordinate3']};\n"

        # first for loop, going over the slices
        code += "for(int m = 0; m < n3; m++){\n"
        if not config.get('curvature_config'):
            code += add_spaces(4) + "real ax3 = zcoords[m];\n"
        # second for loop
        code += add_spaces(4) + "for(int j = 0; j < n2; j++){\n"
        code += add_spaces(8) + "real ax2 = ymin + j*(ymax-ymin)/(n2-1);\n"
        # third for loop
        code += add_spaces(8) + "for(int i = 0; i < n1; i++){\n"
        code += add_spaces(12) + "real ax1 = xmin + i*(xmax-xmin)/(n1-1);\n"
        if config.get('curvature_config'):
            code += add_spaces(12) + f"real ax3 = {surfaceHelevel} - bulkHeliumLevelDispScales[m] * {config.get('curvature_config')['displacement']}(ax1,ax2);\n"
        
        quantity = config_quantity.get(config['quantity'])

        code += add_spaces(12) + f"""{config['name']} << {quantity}({xyz}) << endl;\n"""
        code += add_spaces(12) + """}\n"""
        code += add_spaces(8) + """}\n"""
        code += add_spaces(4) + "}\n"
        code += "}\n"

        code += headerFrame("2D SLICES DATA EXTRACTION BLOCK END")

        return code


    def script_save_cmatrix(self, electrode_name: str) -> str:
        """
        Saves the capacitance matrix based on the provided parameters and the FreeFEM object name.
        
        Args:
            electrode_name (str): Name of the electrode for the capacitance matrix extraction.

        Returns:
            code (str): code containing the Capacitance Matrix.
        """
        path = format_freefem_path(str(self.savedir), 'cm_' + electrode_name + ".txt")
        self.result_files["cm"].append((electrode_name, path))


        code = headerFrame("START / Calculate Capacitance Matrix")        
        code += f"""ofstream cmextract("{path}");\n"""
        code += "\n"
        code += "for(int i = 0; i < numV; i++){\n"
        code += add_spaces(4) + f"real charge = int2d(Th,electrodeid[i])((dielectric(x + eps*N.x, y + eps*N.y, z + eps*N.z) * field(u, x + eps*N.x, y + eps*N.y, z + eps*N.z)' * norm\n"
        code += add_spaces(42) + f"- dielectric(x - eps*N.x, y - eps*N.y, z - eps*N.z) * field(u, x - eps*N.x, y - eps*N.y, z - eps*N.z)' * norm));\n"
        code += add_spaces(4) + f"cmextract << charge << endl;\n"
        code += "}\n"
        code += headerFrame("END / Calculate Capacitance Matrix")

        return code
    

    def script_refine_mesh(self, iterations: int=3) -> str:
        """
        Refines the mesh using TetGen and mshmet for a specified number of iterations.
        
        Args:
            iterations (int): Number of iterations to refine the mesh. Default is 3.
        
        Returns:
            code (str): code containing the mesh refinement process.
        """
        code = "\n"
        code += """real errm=1e-2;\n"""
        code += "\n"
        code += f"""for(int i=0; i<{iterations}; i++)\n"""
        code += """{\n"""
        code += "Electro;\n"
        code += """cout <<" u min, max = " <<  u[].min << " "<< u[].max << endl;\n"""
        code += """fespace VhMetric(Th, P23d);\n"""
        code += """real[int] metric = mshmet(Th, u, hmin=1e-2,hmax=0.3,err=errm);\n"""
        code += "\n"
        code += """cout <<" h min, max = " <<  metric.min << " "<< metric.max << " " << metric.n << " " << Th.nv << endl;\n"""
        code += "\n"
        code += """fespace Ph(Th, P1);\n"""
        code += """Ph vol;\n"""
        code += """vol[] = metric;"""
        code += "\n"
        code += """errm*= 0.8;\n"""
        code += """cout << " Th" << Th.nv << " " << Th.nt << endl;\n"""
        code += """Th=tetgreconstruction(Th,switch="raAQ",sizeofvolume=vol);\n"""
        code += "\n"
        code += """}\n"""

        return code


    def script_mesh_adaptation(self, electrode_name: str, n_adapt: int = 3, err_target: float = 0.01, aniso: bool = False, hmin_scale: float = 500.0, hmax_scale: float = 5.0, save_adapted_mesh: bool = False) -> str:
        """

        Generates mesh adaptation loop code using mshmet and tetgen.
        Based on the adaptation logic provided as reference in ff_a.edp.
        Solves iteratively on progressively refined meshes, concentrating
        elements in regions of high field gradient, then performs a final
        solve on the adapted mesh.

        Args:
            electrode_name (str): Name of the active electrode.
            n_adapt (int): Number of adaptation iterations. Default is 3.
            err_target (float): Interpolation error target for mshmet. Default is 0.01.
            aniso (bool): Whether to use anisotropic adaptation. If True, mshmet computes
                a full tensor metric per vertex allowing elements to stretch directionally.
                Default is False (isotropic).
            hmin_scale (float): Divisor for computing hmin from domain size. Default is 500.0.
            hmax_scale (float): Divisor for computing hmax from domain size. Default is 5.0.
            save_adapted_mesh (bool): Whether to save the adapted mesh to disk. Default is False.

        Returns:
            code (str): FreeFEM code containing the mesh adaptation loop.
        """
        if aniso:
            aniso_val = 1
        else:
            aniso_val = 0

        polynomial = self.config["ff_polynomial"]
        femSpace = 'P23d' if polynomial == 2 else 'P13d'

        code = headerFrame("MESH ADAPTATION PARAMETERS")

        # Parameters
        code += f"int nAdapt = {n_adapt};        // number of adaptation iterations\n"
        code += f"real errTarget = {err_target}; // interpolation error target\n\n"

        # Estimate hmin/hmax from mesh bounding box
        code += "// Estimate hmin/hmax from mesh bounding box\n"
        code += "real bbxmin = Th(0).x, bbxmax = Th(0).x;\n"
        code += "real bbymin = Th(0).y, bbymax = Th(0).y;\n"
        code += "real bbzmin = Th(0).z, bbzmax = Th(0).z;\n"
        code += "for (int i = 1; i < Th.nv; i++) {\n"
        code += add_spaces(4) + "bbxmin = min(bbxmin, Th(i).x); bbxmax = max(bbxmax, Th(i).x);\n"
        code += add_spaces(4) + "bbymin = min(bbymin, Th(i).y); bbymax = max(bbymax, Th(i).y);\n"
        code += add_spaces(4) + "bbzmin = min(bbzmin, Th(i).z); bbzmax = max(bbzmax, Th(i).z);\n"
        code += "}\n"
        code += "real domainSize = max(bbxmax - bbxmin, max(bbymax - bbymin, bbzmax - bbzmin));\n"
        code += f"real hmin = domainSize / {hmin_scale};  // fine near features\n"
        code += f"real hmax = domainSize / {hmax_scale};    // coarse far from features\n\n"

        code += 'cout << "Domain bounding box: ["\n'
        code += '     << bbxmin << ", " << bbxmax << "] x ["\n'
        code += '     << bbymin << ", " << bbymax << "] x ["\n'
        code += '     << bbzmin << ", " << bbzmax << "]" << endl;\n'
        code += 'cout << "hmin = " << hmin << ", hmax = " << hmax << endl;\n\n'

        code += headerFrame("MESH ADAPTATION LOOP")

        # Track energy convergence
        code += "// Track energy convergence\n"
        code += f"real[int] energyHistory(nAdapt);\n"
        code += "real energyPrev = 0.0;\n\n"

        # Adaptation iterations: solve, compute metric, remesh
        code += "// Adaptation iterations: solve, compute metric, remesh\n"
        code += "for (int iter = 0; iter < nAdapt - 1; iter++) {\n"
        code += add_spaces(4) + 'cout << "=== Adaptation iteration " << iter+1 << " / " << nAdapt << " ===" << endl;\n'
        code += add_spaces(4) + 'cout << "  Mesh: " << Th.nv << " vertices, " << Th.nt << " tetrahedra" << endl;\n\n'

# Check if you can put this part in problem definition: 
        code += add_spaces(4) + "fespace VhLoop(Th, P23d);\n"
        code += add_spaces(4) + "fespace FRLoop(Th, P03d);\n\n"
        code += add_spaces(4) + "VhLoop uLoop, vLoop;\n"
        code += add_spaces(4) + self._script_dielectric("dielectricLoop", "FRLoop")

        code += add_spaces(4) + "problem ElectroLoop(uLoop, vLoop, solver=CG) =\n"
        code += add_spaces(8) + "int3d(Th)(dielectricLoop * Grad(uLoop)' * Grad(vLoop))\n"
        code += self._script_electrode_bcs(electrode_name, "uLoop")
        code += add_spaces(8) + ";\n"
        code += add_spaces(4) + "ElectroLoop;\n\n"

# until here

        # Track electrostatic energy
        code += add_spaces(4) + "// Track electrostatic energy\n"
        code += add_spaces(4) + "real energy = int3d(Th)(dielectricLoop * Grad(uLoop)' * Grad(uLoop));\n"
        code += add_spaces(4) + "energyHistory[iter] = energy;\n"
        code += add_spaces(4) + "real relChange = (iter > 0) ? abs(energy - energyPrev) / abs(energyPrev) * 100.0 : 100.0;\n"
        code += add_spaces(4) + 'cout << "  Energy = " << energy << "  (change = " << relChange << "%)" << endl;\n'
        code += add_spaces(4) + "energyPrev = energy;\n\n"

        # Compute metric (edge-length field) based on solution Hessian
        code += add_spaces(4) + "// Compute metric (edge-length field) based on solution Hessian\n"
        code += add_spaces(4) + "fespace Vh1(Th, P13d);\n"
        code += add_spaces(4) + "Vh1 h;\n"
        code += add_spaces(4) + "h[] = mshmet(Th, uLoop,\n"
        code += add_spaces(8) + "normalization = 1,\n"
        code += add_spaces(8) + f"aniso = {aniso_val},\n"
        code += add_spaces(8) + "nbregul = 1,\n"
        code += add_spaces(8) + "hmin = hmin,\n"
        code += add_spaces(8) + "hmax = hmax,\n"
        code += add_spaces(8) + "err = errTarget\n"
        code += add_spaces(4) + ");\n\n"

        # Convert edge-length metric to volume constraint and remesh
        code += add_spaces(4) + "// Convert edge-length metric to volume constraint and remesh\n"
        code += add_spaces(4) + 'Th = tetgreconstruction(Th, switch="raAQ",\n'
        code += add_spaces(8) + "sizeofvolume = h * h * h / 6.0);\n"
        code += "}\n\n"

        # Final solve on adapted mesh
        code += "// Final solve on adapted mesh (u stays in scope for data extraction)\n"
        code += 'cout << "=== Final solve (iteration " << nAdapt << " / " << nAdapt << ") ===" << endl;\n'
        code += 'cout << "  Mesh: " << Th.nv << " vertices, " << Th.nt << " tetrahedra" << endl;\n'
        code += self.script_problem_definition(electrode_name, declare_globals=False)
        code += "\n"

        # Final energy
        code += "// Final energy\n"
        code += "real energyFinal = int3d(Th)(dielectric * Grad(u)' * Grad(u));\n"
        code += "energyHistory[nAdapt - 1] = energyFinal;\n"
        code += "real relChangeFinal = abs(energyFinal - energyPrev) / abs(energyPrev) * 100.0;\n\n"
        code += 'cout << "=== Final mesh: " << Th.nv << " vertices, " << Th.nt << " tetrahedra ===" << endl;\n'
        code += 'cout << "  Energy = " << energyFinal << "  (change = " << relChangeFinal << "%)" << endl;\n\n'

        # Print convergence summary
        code += "// Print convergence summary\n"
        code += 'cout << endl << "=== ENERGY CONVERGENCE SUMMARY ===" << endl;\n'
        code += "for (int i = 0; i < nAdapt; i++) {\n"
        code += add_spaces(4) + "real relCh = (i > 0) ? abs(energyHistory[i] - energyHistory[i-1]) / abs(energyHistory[i-1]) * 100.0 : 0.0;\n"
        code += add_spaces(4) + 'cout << "  Iteration " << i + 1 << ": energy = " << energyHistory[i]\n'
        code += add_spaces(9) + '<< "  change = " << relCh << "%" << endl;\n'
        code += "}\n"
        code += "if (relChangeFinal < 1.0)\n"
        code += add_spaces(4) + 'cout << "  >> Mesh adaptation CONVERGED (final change < 1%)" << endl;\n'
        code += "else\n"
        code += add_spaces(4) + 'cout << "  >> Mesh adaptation NOT converged - consider increasing nAdapt" << endl;\n'
        code += 'cout << "===================================" << endl;\n\n'

        if save_adapted_mesh:
            savedir = self.config.get('savedir', 'dump')
            mesh_path = format_freefem_path(savedir, 'geo', f'{electrode_name}_adapted.mesh')
            code += f'savemesh(Th, "{mesh_path}");\n'
            code += f'cout << "Saved adapted mesh to {mesh_path}" << endl;\n'

        return code




# ============================================================
# FreeFEM Runner
# ============================================================

class FreeFEMrunner:
    def __init__(self, edp_files: List[str]):
        self.edp_files = edp_files
        
    
    # ---------------------------------------------------------
    # Build correct FreeFem executable path
    # ---------------------------------------------------------

    def _resolve_freefem_exe(self, freefem_path: str) -> str:
        """Return full path to FreeFem executable based on OS."""

        d = Path(freefem_path)

        if d.is_file():
            return str(d)

        if sys.platform.startswith("win"):
            exe = d / "FreeFem++.exe"
            if exe.exists():
                return str(exe)
        else:
            exe = d / "FreeFem++"
            if exe.exists():
                return str(exe)

        raise FileNotFoundError(
            f"FreeFem++ not found in {freefem_path}. "
            "Provide the directory OR full path to executable."
        )

    # ---------------------------------------------------------
    # Run a FreeFEM script in a thread (Jupyter-safe)
    # ---------------------------------------------------------

    async def edp_exec(
        self,
        edp_file: str,
        freefem_path: str,
        print_log: bool = False,
        timeout: Optional[int] = None,
        retry: int = 0,
    ):
        """
        Runs one FreeFEM job using threads (Jupyter-safe).
        """
        logger = get_edp_logger(edp_file)
        logger.info(f"Starting EDP: {edp_file}")

        progress = widgets.HTML(
            f"<b>⏳ Running:</b> {edp_file}",
            layout=widgets.Layout(margin="4px 0")
        )
        display(progress)

        exe = self._resolve_freefem_exe(freefem_path)

        cmd = [exe, "-ns", edp_file]

        # Threaded subprocess
        def run_subprocess():
            for attempt in range(retry + 1):
                try:
                    return subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        # env=env,
                        shell=(os.name == "nt"),
                        timeout=timeout
                    )
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout on attempt {attempt+1}/{retry+1}")
                    if attempt == retry:
                        raise
            return None

        # Await result
        try:
            result = await asyncio.to_thread(run_subprocess)
        except subprocess.TimeoutExpired:
            progress.value = f"<b>❌ Timeout:</b> {edp_file}"
            logger.error("Execution timed out")
            raise FreeFemError(f"Timeout running FreeFEM: {edp_file}")

        # Parse output
        output = result.stdout or ""
        logger.debug("Full output captured")  

        for line in output.splitlines():
            logger.info(line)

            if print_log:
                print(line)

        if output.splitlines()[-1] == "Ok: Normal End":
            progress.value = f"<b>✅ Done:</b> {edp_file}"
            logger.info(f"Completed EDP: {edp_file}")
        else:
            progress.value = f"<b>❌ Error:</b> {edp_file}"
            logger.error(f"FreeFEM error")
            #raise FreeFemError(error)

    # ---------------------------------------------------------
    # Parallel execution (async semaphore)
    # ---------------------------------------------------------

    async def limited_exec(self, semaphore, *args, **kwargs):
        async with semaphore:
            await self.edp_exec(*args, **kwargs)

    # ---------------------------------------------------------
    # Master run function
    # ---------------------------------------------------------

    async def run(
        self,
        cores: int = 4,
        print_log: bool = False,
        freefem_path: Optional[str] = None,
        timeout: Optional[int] = None,
        retry: int = 0
    ):
        """
        Runs all FreeFEM EDP files in parallel.
        """

        # Auto-detect FreeFem++ if not provided
        if freefem_path is None:
            freefem_path = detect_freefem()
            if not freefem_path:
                raise FileNotFoundError(
                    "FreeFem++ not found automatically. "
                    "Specify freefem_path manually."
                )

        sys_cores = psutil.cpu_count(logical=False)
        if cores > sys_cores:
            raise ValueError(f"Input core count is greater than the available cores on this system.")
        semaphore = asyncio.Semaphore(cores)

        tasks = [
            self.limited_exec(
                semaphore,
                file,
                freefem_path,
                print_log,
                timeout,
                retry
            )
            for file in self.edp_files
        ]

        await asyncio.gather(*tasks)



class ResultGatherer():

    def __init__(
            self,
            savedir: str | Path,
            result_files: dict,
            extract_opt: list[dict],
            remove_files: bool=False
            ):
        self.savedir = savedir
        self.result_files = result_files   
        self.extract_opt = extract_opt             
        self.gather_results(remove_files)


    def __make_header(self, result_name: str) -> dict:
        opt = get_dict_by_name(self.extract_opt, result_name)
        data = {}
        data['Quantity'] = opt['quantity']
        data['Plane'] = opt['plane']
        data['X Min'] = opt['coordinate1'][0]
        data['X Max'] = opt['coordinate1'][1]
        data['X Num'] = opt['coordinate1'][2]
        data['Y Min'] = opt['coordinate2'][0]
        data['Y Max'] = opt['coordinate2'][1]
        data['Y Num'] = opt['coordinate2'][2]
        data['Slices'] = len(opt['coordinate3'])
        data['Slice Values'] = opt['coordinate3']
        data['Curved Surface'] = bool(opt['curvature_config'])
        data['Schema'] = str((len(opt['coordinate3']), opt['coordinate2'][2], opt['coordinate1'][2]))
        return data


    def __create_polarsdf(
            self,
            filename: str,
            result_files: list,
            remove: bool=False
        ):
        dataframe = pl.DataFrame({})
        for elname, fname in result_files:
            data = pl.read_csv(source=fname,
                               has_header=False,
                               new_columns=[elname],
                               schema_overrides={elname: pl.Float64})

            dataframe = pl.concat([dataframe, data], how="horizontal")
            if remove:
                os.remove(fname)
        path = self.savedir / Path(filename)
        dataframe.write_parquet(path.with_suffix(".parquet"), compression="zstd")


    def gather_results(self, remove: bool=False):
        """
        Gathers results for all electrodes into one polars DataFrame for each extract config. Redirected to .parquet files for easy parsing.

        Args:
            remove (bool): Whether or not to remove the .npy files in the user's file system. Defaults to True.
        """
        yaml_data = {}

        for result_name, files in self.result_files.items():
            if result_name != "cm":
                self.__create_polarsdf(result_name, files, remove)
                yaml_data[result_name] = self.__make_header(result_name)

        yaml_data['Capacitance Matrix'] = self.gather_cm_results(remove)
        yaml_data['Control Electrodes'] = [item[0] for item in files]

        with open(self.savedir.parent / "metadata.yaml", 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    

    def gather_cm_results(self, remove: bool=True) -> list:
        """
        Gathers the capacitance matrix results from the saved text files into a 2D list.
        
        Returns:
            capacitance_matrix (list): 2D list containing the capacitance matrix values.
        """

        capacitance_matrix = []

        for ename, file in self.result_files["cm"]:
            row = np.loadtxt(file)
            row = row.reshape(len(self.result_files["cm"])).tolist()
            capacitance_matrix.append(row)
            if remove:
                os.remove(file)

        return capacitance_matrix



class FreeFEM():
    def __init__(self, config_file: str):
    
        edps = EDPpreparer(config_file)
        self.savedir = Path(edps.config['savedir']) / Path("results")
        self.savedir.mkdir(exist_ok=True)
        self.result_files = edps.result_files
        self.extract_opt = edps.config['extract_opt']
        self.ffrunner = FreeFEMrunner(edps.edp_files)

    
    async def run(
        self,
        cores: int = 4,
        print_log: bool = False,
        freefem_path: Optional[str] = None,
        timeout: Optional[int] = None,
        retry: int = 0,
        remove: bool=True
        ):
        
        await self.ffrunner.run(cores, print_log, freefem_path, timeout, retry)
        rg = ResultGatherer(self.savedir, self.result_files, self.extract_opt, remove_files=remove)
        logging.shutdown()
        self.convergence_matrix()

    #new convergence matrix function:
    def convergence_matrix(self):
        """
        Parses the log files for each electrode and extracts energy convergence data.
        Saves a summary to convergence_matrix.log in the logs directory.
        """
        log_dir = self.savedir.parent / "logs"
        output_path = log_dir / "convergence_matrix.log"
        
        rows = []
        
        for electrode in self.result_files["cm"]:
            electrode_name = electrode[0]
            log_file = log_dir / f"ff_{electrode_name}.log"
            
            if not log_file.exists():
                continue
            
            iteration = 0
            with open(log_file, "r") as f:
                for line in f:
                    if "Energy =" in line:
                        iteration += 1
                        # line looks like: Energy = 758.732  (change = 100%)
                        parts = line.strip().split("Energy =")[1]
                        energy_part = parts.split("(change =")[0].strip()
                        change_part = parts.split("(change =")[1].replace("%)", "").strip()
                        rows.append({
                            "electrode": electrode_name,
                            "iteration": iteration,
                            "energy": float(energy_part),
                            "change": float(change_part)
                        })
        
        with open(output_path, "w") as f:
            f.write(f"{'Electrode':<12} {'Iteration':<12} {'Energy':<20} {'Change (%)':<12}\n")
            f.write("-" * 56 + "\n")
            for row in rows:
                f.write(f"{row['electrode']:<12} {row['iteration']:<12} {row['energy']:<20} {row['change']:<12}\n")
        
        print(f"Convergence matrix saved to {output_path}")