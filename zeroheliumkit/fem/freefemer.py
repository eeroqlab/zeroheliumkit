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
    """
    config_file: str
    dielectric_constants: dict
    ff_polynomial: int
    extract_opt: list[ExtractConfig, dict] | dict | ExtractConfig
    msh_refinements: int = None

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
        code += self.script_declare_variables()
        code += self.script_create_coupling_const_matrix()
        code += self.script_problem_definition(electrode_name)
        if self.config.get('msh_refinements'):
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


    def script_problem_definition(self, electrode_name: str) -> str:
        """
        Defines the problem for the electrostatic potential in FreeFEM, including the finite element space and the dielectric constants.

        Args:
            electrode_name (str): Name of the electrode for which the problem is being defined.

        Returns:
            code (str): code containing the problem definition.
        """
        polynomial = self.config["ff_polynomial"]
        epsilon = self.config["dielectric_constants"]

        code = "\n"

        if polynomial == 1:
            femSpace = 'P13d'
        elif polynomial == 2:
            femSpace = 'P23d'
        else:
            raise Exception("Wrong polynomial order! Choose between 1 or 2")
        
        if 'periodic_BC' in self.config:
            code += f"""fespace Vh(Th,{femSpace}, periodic=[[{self.config.get('periodic_BC')[0]}, x, y], [{self.config.get('periodic_BC')[1]}, x, y]]);\n"""
        else:
            code += f"""fespace Vh(Th,{femSpace});\n"""

        code += "fespace FunctionRegion(Th,P03d);\n"
        code += "real eps = 1e-6;\n"
        code += "macro norm [N.x,N.y,N.z] //\n"
        code += "macro Grad(u) [dx(u),dy(u),dz(u)] //\n"
        code += "macro field(u,x,y,z) [dx(u)(x,y,z),dy(u)(x,y,z),dz(u)(x,y,z)] //\n \n"
        code += "Vh u,v;\n"
        code += "FunctionRegion dielectric =\n"

        for k, v in self.physicalVols.items():
            code += add_spaces(26) + f"""+ {epsilon[k]} * (region == {v})\n"""
        code += add_spaces(26) + ";\n"

        code += "problem Electro(u,v,solver=CG) =\n"
        code += add_spaces(16) + "int3d(Th)(dielectric * Grad(u)' * Grad(v))\n"

        main_electrode = self.physicalSurfs.get(electrode_name)
        ground_electrodes = [item for item in self.physicalSurfs.values() if item != main_electrode]
        code += add_spaces(16) + f"+ on({main_electrode},u = 1.0)\n"
        for v in ground_electrodes:
            code += add_spaces(16) + f"+ on({v},u = 0.0)\n"
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
