""" 
freefemer.py

This module contains functions and a class for creating freefem files. Created by Niyaz B / January 10th, 2023.
"""

import os, yaml, re, time
import asyncio, psutil
import numpy as np
import pandas as pd
import warnings
import ipywidgets as widgets
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from IPython.display import display
from ..src.errors import *
from ..helpers.constants import rho, g, alpha


config_planes_2D = ['xy', 'yz', 'xz']
config_planes_3D = ['xyZ']
config_quantity = {'phi': 'u',
                   'Ex': 'dx(u)',
                   'Ey': 'dy(u)',
                   'Ez': 'dz(u)',
                   'Cm': None}


def scaling_size(bulk_helium_distance: float=1e-1):
    """
    Calculates the scaling size for the helium curvature displacement based on the bulk helium distance.

    Args:
    -----
        bulk_helium_distance (float): The distance between the bulk helium atoms in meters. Default is 1e-1 m.

    Returns:
    --------
        lengthscale (float): The scaling size in micrometers.
    """
    lengthscale = (rho * g * bulk_helium_distance)/alpha * 1e-6      # in um
    return lengthscale


def headerFrame(header: str) -> str:
    """
    Creates a header frame for the FreeFEM script.

    Args:
    -----
        header (str): The header text to be included in the frame.

    Returns:
    --------
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

    Args:
    -----
        num (int): The number of spaces to add.

    Returns:
    --------
        str: A string containing the specified number of spaces.
    """
    return ' ' * num


@dataclass
class ExtractConfig():
    """
    Data class for storing extraction configuration parameters.

    Attributes:
    -----------
        - quantity (str): The quantity to be extracted (e.g., 'phi', 'Ex', 'Ey', 'Ez', 'Cm').
        - plane (str): The plane for extraction (e.g., 'xy', 'yz', 'xz', 'xyZ').
        - coordinate1 (tuple): A tuple containing the start and end coordinates and the number of points for the first coordinate.
        - coordinate2 (tuple): A tuple containing the start and end coordinates and the number of points for the second coordinate.
        - coordinate3 (float | list): A float or list of floats for the third coordinate.
        - additional_name (str, optional): An additional name to append to the result file name.
    """ 
    quantity: str
    plane: str
    coordinate1: tuple
    coordinate2: tuple
    coordinate3: float | list
    additional_name: str = None

    def __post_init__(self):
        if self.quantity not in config_quantity.keys():
            raise KeyError(f'unsupported extract quantity. Supported quantity types are {config_quantity}')
        if self.plane not in config_planes_2D + config_planes_3D:
            raise KeyError(f'Wrong plane! choose from {config_planes_2D} or {config_planes_3D}')
        if not isinstance(self.coordinate1, tuple):
            raise TypeError("'coordinate1' parameter must be a tuple (x1, x2, num)")
        if not isinstance(self.coordinate2, tuple):
            raise TypeError("'coordinate2' parameter must be a tuple (y1, y2, num)")
        if not isinstance(self.coordinate3, float | list | None):
            raise TypeError("'coordinate3' parameter must be a list or float")


@dataclass
class FFconfigurator():
    """
    Dataclass to create the FreeFEM config yaml file.

    Attributes:
    -----------
        - config_file (str): Path to the FreeFEM config yaml file.
        - dielectric_constants (dict): Dictionary containing the dielectric constants for different physical volumes.
        - ff_polynomial (int): Polynomial order for the FreeFEM script.
        - extract_opt (list[ExtractConfig] | dict): List of ExtractConfig objects or a dictionary containing extraction options.
        - include_helium_curvature (dict): Optional dictionary containing helium curvature configuration. 
    """
    config_file: str
    dielectric_constants: dict
    ff_polynomial: int
    extract_opt: list[ExtractConfig, dict] | dict | ExtractConfig
    include_helium_curvature: dict=None

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

            if self.include_helium_curvature:
                for opt in mergeddict["extract_opt"]:
                    if opt["coordinate3"]:
                        warnings.warn("Both a curvature config and coordinate3 has been provided." \
                        "The value passed in for coordinate3 will be ignored," \
                        "please use None for coordinate3 with Helium Curvature enabled.") 
                        opt["coordinate3"] = None
            yaml.safe_dump(mergeddict, file, sort_keys=False, indent=3)


class FreeFEM():
    """
    Class for creating and running FreeFEM scripts.

    Attributes:
    -----------
        config (str): filepath containing FreeFEM config yaml file.
        savedir (str): Directory name where the FreeFEM files will be saved.
        run_from_notebook (bool): Flag indicating if the script is run from a Jupyter notebook.
        electrode_files (list): List of coupling constant files.
        result_files (list): 2D list of existing electrode result files based on extract configs.
        extract_names (str): List of names for the cumulative result files based on extract configs.
        logs (str): Log messages from the FreeFEM execution.
    """
    
    def __init__(self,
                 config_file: str,
                 run_from_notebook: bool=False):

        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.savedir = self.config.get("savedir") + "/"
        self.run_from_notebook = run_from_notebook
        self.electrode_files = []
        self.result_files = [[] for _ in range(len(self.config.get('extract_opt')))]
        self.extract_names = [None for _ in range(len(self.config.get('extract_opt')))]
        self.logs = ""

        self.curvature_config = self.config.get("include_helium_curvature")
        self.physicalVols = self.config.get('physicalVolumes')
        self.physicalSurfs = self.config.get('physicalSurfaces')
        self.num_electrodes = len(list(self.physicalSurfs.keys()))
        if isinstance(self.config.get('extract_opt'), dict):
            self.config['extract_opt'] = [self.config['extract_opt']]
                
        self.write_edpScript()


    def add_helium_curvature_edp(self) -> str:
        """
        Adds the helium curvature script to the FreeFEM script if the curvature configuration is provided.

        Returns:-
        """
        code  = headerFrame("HELIUM CURVATURE")
        code += self.curvature_config["script"]
        code += headerFrame("HELIUM CURVATURE")
        return code
    

    def write_edpContent(self, electrode_name: int) -> str:
        """
        Returns the contents of an electrode_k.edp file with the desired electrode name in the place of 'k'.

        Args:
        -----
            - electrode_name (str): Name of the electrode to generate .edp file content for.

        Returns:
        --------
            code (str): code containing the entire edp content written for electrode_name.
        """
        code = ''
        if self.curvature_config:
            code += self.add_helium_curvature_edp()
        code += self.script_create_savefiles(electrode_name)
        code += self.script_load_packages_and_mesh()
        code += self.script_declare_variables()
        code += self.script_create_coupling_const_matrix()
        code += self.script_problem_definition(electrode_name)

        for i, extract_config in enumerate(self.config.get('extract_opt')):
            if isinstance(extract_config, ExtractConfig):
                extract_config = asdict(extract_config)
            fem_object_name = self.__extract_opt[i] + electrode_name
            # check supported parameters for extraction
            if extract_config.get("quantity") not in config_quantity.keys():
                raise KeyError(f'unsupported extract quantity. Supported quantity types are {config_quantity}')

            if extract_config.get("quantity") == 'Cm':
                # extract capacitance matrix
                code += self.script_save_cmatrix(extract_config, fem_object_name)
            else:
                # extract electrostatic field solutions
                plane = extract_config.get('plane')
                if plane in config_planes_2D:
                    code += self.script_save_data_2D(extract_config, fem_object_name, electrode_name)
                elif plane in config_planes_3D:
                    code += self.script_save_data_3D(extract_config, fem_object_name, electrode_name)
                else:
                    raise KeyError(f'Wrong plane! choose from {config_planes_2D} or {config_planes_3D}')
        
        return code
    

    def write_edpScript(self):
        """
        Creates the main FreeFEM script based on the configuration and physical surfaces.
        """
        #create new files for each electrode, only inputting the lines of code needed
        for electrode in self.physicalSurfs.keys():
            code = self.write_edpContent(electrode)
            filename = "electrode_" + electrode
            self.electrode_files.append(filename)
                    
            with open(self.savedir + filename + '.edp', 'w') as file:
                file.write(code)


    def script_load_packages_and_mesh(self) -> str:
        """
        Loads the necessary FreeFEM packages and the mesh file into the script.

        Returns:
        --------
            code (str): code containing the necessary FreeFEM packages and mesh file declarations.
        """
        code = """load "msh3"\n"""
        code += """load "gmsh"\n"""
        code += """load "medit"\n"""
        #code += f"""system("mkdir -p {self.savedir}");\n"""
        code += "\n"
        if self.run_from_notebook:
            path_meshfile = self.savedir + self.config["meshfile"]
        else:
            path_meshfile = self.config["meshfile"] + '.msh2'
        code += f"""mesh3 Th = gmshload3("{path_meshfile}");\n"""
        return code


    def script_declare_variables(self) -> str:
        """
        Declares the necessary variables for the FreeFEM script, including physical surfaces and volumes.

        Returns:
        --------
            code (str): code containing the necessary variable declarations.
        """
        electrode_names = list(self.physicalSurfs.keys())
        electrode_id = list(self.physicalSurfs.values())

        code = "\n"
        code += "int n1, n2, n3;\n"
        code += "real xmin, xmax, ymin, ymax, ax3, zmin, zmax;\n"

        # electrodestring_name = '"' + '","'.join(electrode_names) + '"'
        # code += f"string[int] electrodenames = [{electrodestring_name}];\n"

        # code += f"int[int] electrodeid = {electrode_id};\n"
        return code


    def script_create_coupling_const_matrix(self) -> str:
        """
        Creates the coupling constant matrix for the FreeFEM script, which is used to define the interaction between electrodes.

        Returns:
        --------
            code (str): code containing the coupling constant matrix

        Returns:
        --------
            code (str): code containing the coupling constant matrix.
        """
        number_of_electrodes = len(list(self.physicalSurfs.keys()))

        code = "\n"
        code += f"int numV = {number_of_electrodes};\n"

        code += "\n"
        code += "real[int, int] CapacitanceMatrix(numV, numV);\n"
        code += "for (int i = 0; i < numV; i+= 1){\n"
        code += add_spaces(4) + "for (int j = 0; j < numV; j+= 1){\n"
        code += add_spaces(8) + "CapacitanceMatrix(i, j) = 0.0;}}\n"

        return code


    def script_create_savefiles(self, electrode_name: str):
        """
        Creates the necessary files for saving results based on the configuration and electrode index.

        Args:
        -----
            - electrode_name (str): Name of the electrode for which the files are being created.

        Returns:
        --------
            code (str): string containing the necessary lines of code to save the data.
        """
        code = "\n"
        self.__extract_opt = {}

        for idx, extract_cfg in enumerate(self.config.get('extract_opt')):
            addname = extract_cfg['additional_name']
            qty = extract_cfg['quantity']
            pln = extract_cfg['plane']
            name = (addname + "_" if addname else "") + qty + ("_" + pln if pln else "")

            if self.run_from_notebook:
                name = self.savedir + name
            self.extract_names[idx] = name
            name += f'_{electrode_name}'
            self.result_files[idx].append(name)
            code += f"""ofstream extract{idx}{qty}{electrode_name}("{name}.btxt", binary);\n"""
            self.__extract_opt[idx] = f"extract{idx}{qty}"
        
        return code

    def script_problem_definition(self, electrode_name: str) -> str:
        """
        Defines the problem for the electrostatic potential in FreeFEM, including the finite element space and the dielectric constants.

        Args:
        -----
            - electrode_name (str): Name of the electrode for which the problem is being defined.

        Returns:
        --------
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
        code += """cout << "calculations are finished, saving data" << endl;\n \n"""

        return code


    def script_save_data_2D(self, params: dict, fem_object_name: str, electrode_name: str) -> str:
        """
        Saves 2D data extraction based on the provided parameters and the FreeFEM object name.
        
        Args:
        -----
            - params (dict): Dictionary containing the parameters for the 2D data extraction.
            - fem_object_name (str): Name of the FreeFEM object to save the data to.
            - electrode_name (str): of the electrode for which the data is being saved.

        Returns:
        --------
            code (str): code containing the 2D data saving code.
        """
        if params.get('plane')=='xy':
            xyz = "ax1,ax2,ax3"
        elif params.get('plane')=='yz':
            xyz = "ax3,ax1,ax2"
        elif params.get('plane')=='xz':
            xyz = "ax1,ax3,ax2"
        else:
            raise KeyError(f'Wrong plane! choose from {config_planes_2D}')

        # working along the helium curvature lines if the option is enabled
        if self.curvature_config:
            scaling = scaling_size(self.curvature_config["bulk_helium_distances"][0])
            zcoord_code = "\n"
        else:
            zcoord_code = f"ax3  = {params['coordinate3']};\n"
        
        name = fem_object_name  #params['quantity']

 
        code  = headerFrame("2D DATA EXTRACTION BLOCK START")
        code += "{\n"
        code += f"n1 = {params['coordinate1'][2]};\n"
        code += f"n2 = {params['coordinate2'][2]};\n"
        code += f"xmin = {params['coordinate1'][0]};\n"
        code += f"xmax = {params['coordinate1'][1]};\n"
        code += f"ymin = {params['coordinate2'][0]};\n"
        code += f"ymax = {params['coordinate2'][1]};\n"
        code += zcoord_code
        code += "real[int,int] quantity(n1,n2);\n"
        code += "real[int] xList(n1), yList(n2);\n \n"

        code += "for(int i = 0; i < n1; i++){\n"
        code += add_spaces(4) + "real ax1 = xmin + i*(xmax-xmin)/(n1-1);\n"
        code += add_spaces(4) + "xList[i] = ax1;\n"
        code += add_spaces(4) + "for(int j = 0; j < n2; j++){\n"
        
        quantity = config_quantity.get(params['quantity'])
        code += add_spaces(8) + "real ax2 = ymin + j*(ymax-ymin)/(n2-1);\n"
        code += add_spaces(8) + "yList[j] = ax2;\n"

        # working along the helium curvature lines if the option is enabled
        if self.curvature_config:
            code += add_spaces(8) + f"ax3 = {scaling} * {self.curvature_config['displacement']}(ax1,ax2);\n"

        code += add_spaces(8) + f"""{name} << {quantity}({xyz}) << endl;\n"""
        code += add_spaces(8) + """}\n"""
        code += add_spaces(4) + """}\n"""
        code += "}\n"

        code += headerFrame("2D DATA EXTRACTION BLOCK END")

        return code  


    def script_save_data_3D(self, params: dict, fem_object_name: str, electrode_name: str) -> str:
        """
        Saves 3D data extraction based on the provided parameters and the FreeFEM object name.

        Args:
        -----
            - params (dict): Dictionary containing the parameters for the 3D data extraction.
            - fem_object_name (str): Name of the FreeFEM object to save the data to.
            - electrode_name (str): Name of the electrode for which the data is being saved.

        Returns:
        --------
            code (str): code containing the 2D slicing code.
        """
        if params.get('plane')=='xyZ':
            if self.curvature_config:
                xyz = "ax1, ax2, bulkHeliumLevelDispScales[m]"
            else:
                xyz = "ax1,ax2,zcoords[m]"
        else:
            raise KeyError(f'Wrong plane! choose from {config_planes_3D}')

        name = fem_object_name  # params['quantity']

        code  = headerFrame("2D SLICES DATA EXTRACTION BLOCK START")
        code += "{\n"

        # working along the helium curvature lines if the option is enabled
        if self.curvature_config:
            bulkHelevels = np.asarray(self.curvature_config["bulk_helium_distances"])
            scaling = scaling_size(bulkHelevels)
            surfaceHelevel = self.curvature_config["surface_helium_level"]
            code += f"n3  = {len(scaling)};\n"
            code += f"real[int] bulkHeliumLevels = {np.array2string(bulkHelevels, separator=', ')};\n"
            code += f"real[int] bulkHeliumLevelDispScales = {np.array2string(scaling, separator=', ')};\n"
        else:
            code += f"n3  = {params['coordinate3'][2]};\n"
            code += f"zmin  = {params['coordinate3'][0]};\n"
            code += f"zmax  = {params['coordinate3'][1]};\n"
            code += f"real[int] zcoords = {list(params['coordinate3'])};\n"

        code += f"n1 = {params['coordinate1'][2]};\n"
        code += f"xmin = {params['coordinate1'][0]};\n"
        code += f"xmax = {params['coordinate1'][1]};\n"
        code += f"n2 = {params['coordinate2'][2]};\n"
        code += f"ymin = {params['coordinate2'][0]};\n"
        code += f"ymax = {params['coordinate2'][1]};\n"

        code += "real[int,int] quantity(n1,n2);\n"
        code += "real[int] xList(n1), yList(n2), zList(n3);\n \n"

        code += "for(int m = 0; m < n3; m++){\n"
        if self.curvature_config:
            code += add_spaces(4) + "zList[m] = bulkHeliumLevels[m];\n"
        
        code += add_spaces(4) + "for(int i = 0; i < n1; i++){\n"
        code += add_spaces(8) + "real ax1 = xmin + i*(xmax-xmin)/(n1-1);\n"
        code += add_spaces(8) + "xList[i] = ax1;\n"
        code += add_spaces(8) + "for(int j = 0; j < n2; j++){\n"
        code += add_spaces(12) + "real ax2 = ymin + j*(ymax-ymin)/(n2-1);\n"
        code += add_spaces(12) + "yList[j] = ax2;\n"
        if self.curvature_config:
            code += add_spaces(12) + f"real ax3 = {surfaceHelevel} - bulkHeliumLevelDispScales[m] * {self.curvature_config['displacement']}(ax1,ax2);\n"
        
        quantity = config_quantity.get(params['quantity'])

        code += add_spaces(12) + f"""{name} << {quantity}({xyz}) << endl;\n"""
        code += add_spaces(12) + """}\n"""
        code += add_spaces(8) + """}\n"""
        code += add_spaces(4) + "}\n"
        code += "}\n"
        code += headerFrame("2D SLICES DATA EXTRACTION BLOCK END")

        return code


    def script_save_cmatrix(self, params: dict, fem_object_name: str) -> str:
        """
        Saves the capacitance matrix based on the provided parameters and the FreeFEM object name.
        
        Args:
        -----
            - params (dict): Dictionary containing the parameters for the capacitance matrix extraction.
            - fem_object_name (str): Name of the FreeFEM object to save the capacitance matrix to.

        Returns:
        --------
            code (str): code containing the Capacitance Matrix.
        """
        name = fem_object_name  # params['quantity']

        code = add_spaces(4) + "{\n"
        code += add_spaces(4) + "for(int i = k; i < numV; i++){\n"
        code += add_spaces(8) + f"real charge = int2d(Th,electrodeid[i])((dielectric(x + eps*N.x, y + eps*N.y, z + eps*N.z) * field(u, x + eps*N.x, y + eps*N.y, z + eps*N.z)' * norm\n"
        code += add_spaces(46) + f"- dielectric(x - eps*N.x, y - eps*N.y, z - eps*N.z) * field(u, x - eps*N.x, y - eps*N.y, z - eps*N.z)' * norm));\n"
        code += add_spaces(8) + "CapacitanceMatrix(k,i) = charge;\n"
        code += add_spaces(8) + "CapacitanceMatrix(i,k) = charge;}\n"
        code += add_spaces(4) + """if (k == numV - 1){\n"""
        code += add_spaces(8) + f"{name} << electrodenames << endl;\n"
        code += add_spaces(8) + f"{name} << CapacitanceMatrix << endl;" + "}\n"
        code += add_spaces(4) + "}\n"

        return code


    async def edp_exec(self, edp_file: str, filepath: str, print_log: bool=False):
        """
        Executes the FreeFEM script asynchronously and captures the output.
        
        Args:
        -----
            - edp_file (str): Name of the FreeFEM script file to execute.
            - filepath (str): Path to the FreeFEM executable.
            - print_log (bool): Flag to indicate whether to print the output log.
        """
        progress = widgets.Label(f"⏳ Running calculations for {edp_file}")
        display(progress)

        bashCommand = ['freefem++', self.savedir + edp_file + '.edp']
        env = os.environ.copy()
        env['PATH'] += filepath
        process = await asyncio.create_subprocess_exec(
            *bashCommand,
            stdout=asyncio.subprocess.PIPE,
            env=env
        )

        async for line in process.stdout:
            output_log = line.decode()
            self.logs += edp_file + ':' + output_log
            if output_log[1:6] == "Error":
                raise FreefemError(output_log)
            elif print_log:
                print(output_log)

        await process.wait()
        progress.value = f"✅ {edp_file} complete"


    def write_res_header(self, header_data):
        lines = []
        lines.append("---") 
        if header_data.get('additional_name'):
            lines.append(f"CONFIG - {header_data['additional_name']}")
        else:
            lines.append("CONFIG")
        lines.append(f"quantity,{header_data['quantity']}")
        lines.append(f"plane,{header_data['plane']}")
        lines.append(f"coordinate1,{tuple(header_data['coordinate1'])}")
        lines.append(f"coordinate2,{tuple(header_data['coordinate2'])}")
        if header_data.get('coordinate3'):
            lines.append(f"coordinate3,{header_data['coordinate3']}")
        if not self.curvature_config:
            lines.append(f"helium_curvature,{bool(self.curvature_config)}")
        else:
            lines.append(f"helium_curvature, {self.curvature_config['bulk_helium_distances']}")
        lines.append("---")
        return "\n".join(lines) + "\n"


    def gather_results(self, single_data_file: bool=False):
        """
        Gathers the results from the individual FreeFEM result files into a single file.

        Args:
        -----
            - res_num (int): Index of the extract config to gather results for.
        """
        filename = "ff_data_" + self.config['meshfile'].split('.')[0]
        
        filepath = self.savedir + filename + ".csv"

        open(filepath, "w+").close()

        for i, _ in enumerate(self.config.get('extract_opt')):
            config = self.config.get('extract_opt')[i]

            if not single_data_file and config['additional_name']:
                filename = f"ff_data_{self.config['meshfile'].split('.')[0]}_{config['additional_name']}"
                filepath = self.savedir + filename + ".csv"

            pd.DataFrame([[self.write_res_header(config)]]).to_csv(filepath, index=False, header=False, mode='a')
            pd.DataFrame([[]]).to_csv(filepath, index=False, header=False, mode='a')
            
            for file in self.result_files[i]:
                electrode_name = file.split('_')[-1]

                array = np.loadtxt(file + ".btxt")

                n1 = config['coordinate1'][2]
                n2 = config['coordinate2'][2]
                if self.curvature_config:
                    n3 = len(self.curvature_config['bulk_helium_distances'])
                elif isinstance(config['coordinate3'], list):
                    n3 = len(config['coordinate3'])
                else:
                    n3 = 1

                if config['plane'] in config_planes_3D:
                    pd.DataFrame([[f"[START SLICE - {electrode_name}]"]]).to_csv(filepath, index=False, header=False, mode='a')
                    new_arr = array.reshape((n3, n1, n2))
                    for slice in new_arr:
                        frame = pd.DataFrame(slice)
                        frame.to_csv(filepath, index=False, header=False, mode='a')
                        pd.DataFrame([[]]).to_csv(filepath, index=False, header=False, mode='a')
                elif config['plane'] in config_planes_2D:
                    pd.DataFrame([[f"[START DATA - {electrode_name}]"]]).to_csv(filepath, index=False, header=False, mode='a')
                    new_arr = array.reshape((n1, n2))
                    frame = pd.DataFrame(new_arr)
                    frame.to_csv(filepath, index=False, header=False, mode='a')
                    pd.DataFrame([[]]).to_csv(filepath, index=False, header=False, mode='a')

                os.remove(f"{file}.btxt")
            self.result_files[i].clear()


    def log_history(self, edp_code: str, total_time: float):
        """
        Logs the current run and edp code to an existing, running history file (ff_history.md)

        Args:
        -----
            - edp_code (str): Skeleton edp code to place in the body of the history entry.
            - total_time (float): Time elapsed from the latest FreeFem calculation to input in the header.
        """
        curr_date = datetime.now()
        try:
            with open(self.savedir + 'ff_history.md', 'r+', encoding='utf-8') as hist:
                contents = hist.read()
                start_header = contents.find("\n")
                iteration = int(contents[0:start_header]) + 1 if contents else 1
                hist.seek(0)
                hist.write(str(iteration) + "\n")
                hist.seek(0, 2) 
                hist.write(f"\n## [{iteration}] - {curr_date} - Run in {total_time} seconds\n")
                config = self.config.get('extract_opt')
                for c in config:
                    hist.write(f"## QUANTITY: {c['quantity']} - PLANE: {c['plane']} - COORD1: {c['coordinate1']} - COORD2: {c['coordinate2']} - COORD3: {c['coordinate3']}\n")
                hist.write("```freefem\n")
                hist.write(edp_code)
                hist.write("```\n")
        except FileNotFoundError:
            with open(self.savedir + 'ff_history.md', 'w', encoding='utf-8') as hist:
                hist.seek(0)
                hist.write('1')
                hist.write(f"\n## [1] - {curr_date} - Run in {total_time} seconds\n")
                hist.write("```freefem\n")
                hist.write(edp_code)
                hist.write("```\n")
        except Exception as e:
            print(e)


    async def limited_exec(self, semaphore, *args, **kwargs):
        """
        Executes the FreeFEM script with a semaphore to limit the number of concurrent executions.

        Args:
        -----
            - semaphore (asyncio.Semaphore): Semaphore to run the edp_exec method with.
        """
        async with semaphore:
            await self.edp_exec(*args, **kwargs)


    async def run_from_history(self, iteration: int, electrode_name: int | list, cores: int, 
                               print_log: bool=False, freefem_path: str=":/Applications/FreeFem++.app/Contents/ff-4.15/bin"):
        """
        Runs the edp file from a given iteration of the ff_history.md file.

        Args:
        -----
            - iteration (int): Iteration number to grab the edp code from.
            - electrode name (int | list): Name(s) of the electrode(s) to run again on this edp file.
            - print_log (bool): Whether or not to print the logs to stdout. Defaults to False.
            - freefem_path (str): Path to the FreeFem package. Has a default path. 
        """
        names = []
        if electrode_name is None:
            names = list(self.physicalSurfs.keys())
        elif isinstance(electrode_name, str):
            names = [electrode_name]

        pattern = rf'^##\s+\[({iteration})]'
        with open(self.savedir + "ff_history.md", 'r+') as hist:
            history_content = hist.read()
            match = re.search(pattern, history_content, re.MULTILINE)

            if match is None:
                raise ValueError("No iteration of that kind can be found")
            
            code_start = history_content.find("```freefem", match.end())
            if code_start == -1:
                raise ValueError("Given index does not have freefem code logged in history.")
            code_end = history_content.find("```", code_start + 4)
            if code_end == -1:
                raise ValueError("Given index does not have freefem code logged in history.")

            hist.seek(0)
            history_content = hist.read()
            code = history_content[code_start + 12:code_end]
            self.electrode_files.clear()
            self.result_files = [[] for _ in range(len(self.config.get('extract_opt')))]
            self.logs = ""

            problem_start = code.find("fespace")
            if problem_start != -1:
                problem_end = code.find("endl;", problem_start)
            else:
                raise ValueError("Given edp file from History does not contain a problem definition block.")

            for idx, _ in enumerate(self.config.get('extract_opt')):
                for name in names:
                    temp_code = self.script_problem_definition(name)
                    new_code = code[0:problem_start] + temp_code + code[problem_end + 5::]
                    new_code = new_code.replace('k', name)
                    
                    with open(self.savedir + f"electrode_{name}.edp", 'w+') as edp:
                        edp.write(new_code)
                    self.electrode_files.append(f"electrode_{name}")
                    extract_names = self.extract_names[idx]
                    extract_names += f'_{name}'
                    self.result_files[idx].append(extract_names)
                    new_code = ""
                    temp_code = ""
            await self.run(cores, print_log, freefem_path)


    async def run(self,
            cores,
            print_log=False,
            single_data_file=True,
            freefem_path=":/Applications/FreeFem++.app/Contents/ff-4.15/bin"):
        """
        Runs the FreeFEM calculations asynchronously with a specified number of cores.
        
        Args:
        -----
            cores (int): Number of cores to use for the calculations.
            print_log (bool): Flag to indicate whether to print the output log.
            freefem_path (str): Path to the FreeFEM executable.
        """
        
        sys_cores = psutil.cpu_count(logical=False)
        if cores > sys_cores:
            raise ValueError(f"Input core count is greater than the available cores on this system.")
        semaphore = asyncio.Semaphore(cores)

        start_time = time.perf_counter()
        
        try:
            asynch_in = [
                self.limited_exec(semaphore, edp_name, freefem_path, print_log)
                for edp_name in self.electrode_files
            ]

            await asyncio.gather(*asynch_in)

        except KeyboardInterrupt:
            message = 'Interrupted by user'
            print(message)
            self.logs += message

        finally:
            with open(os.path.join(self.savedir, 'ff_logs.txt'), 'w') as outfile:
                outfile.write(self.logs)
                
            self.gather_results(single_data_file)
            filename = self.electrode_files[0]
            with open(self.savedir + filename + ".edp", 'r') as file:
                skel_name = self.electrode_files[0].split('_')[-1]
                edp_skel = file.read()
            for file in self.electrode_files:
                os.remove(self.savedir + file + ".edp")
            self.electrode_files.clear()

            end_time = time.perf_counter()
            total_time = end_time - start_time

            edp_skel = edp_skel.replace(skel_name, 'k')
            self.log_history(edp_skel, total_time)

            print(f'Freefem calculations are complete. Ran in {total_time:.2f} seconds.')


    def clean_directory(self, keep_files: list=None):
        for file in os.listdir(f'{self.savedir}'):
            if file.startswith("ff_data") and file not in keep_files:
                print(f'Cleaning {file} from directory')
                os.remove(self.savedir + file)
            
            if ".btxt" in file and file not in keep_files:
                print(f'Cleaning {file} from directory')
                os.remove(self.savedir + file)


if __name__=="__main__":

    pyff = FreeFEM(config_file="config/dot.yaml")
    pyff.run(print_log=True)
