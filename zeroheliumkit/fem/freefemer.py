""" This module contains functions and a class for creating freefem files
    created by Niyaz B / January 10th, 2023
"""

import os
import yaml
import subprocess
import numpy as np
from alive_progress import alive_it
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
    lengthscale = (rho * g * bulk_helium_distance)/alpha * 1e-6      # in um
    return lengthscale


def headerFrame(header: str) -> str:
    edp = '\n//////////////////////////////////////////////////////////\n'
    edp += '//\n'
    edp += '//    ' + header + '\n'
    edp += '//\n'
    edp += '//////////////////////////////////////////////////////////\n\n'
    return edp

def extract_results(quantity: str,
                    plane: str=None,
                    axis1_params: tuple=None,
                    axis2_params: tuple=None,
                    axis3_params: float | tuple=None,
                    additional_name: str=None) -> dict:
    return {
        'quantity': quantity,
        'plane': plane,
        'coordinate1': axis1_params,
        'coordinate2': axis2_params,
        'coordinate3': axis3_params,
        'additional_name': additional_name
    }


class FreeFEM():
    """ creates .edp files to run FreeFem++ calculations """
    
    def __init__(self,
                config: dict,
                dirname: str,
                run_from_notebook: bool=False):

        self.config = config
        self.dirname = dirname
        self.run_from_notebook = run_from_notebook
        self.cc_files = []

        self.curvature_config = config.get("include_helium_curvature")
        self.physicalVols = config.get('physicalVolumes')
        self.physicalSurfs = config.get('physicalSurfaces')
        self.num_electrodes = len(list(self.physicalSurfs.keys()))
        self.write_edpScript()


    def write_edpScript(self):
        script = self.create_edpScripts()
        with open(self.dirname + self.config["meshfile"] + '.edp', 'w') as file:
            file.write(script)


    def add_spaces(self, num: int) -> str:
        return ' ' * num


    def add_helium_curvature_edp(self) -> str:
        code  = headerFrame("HELIUM CURVATURE")
        code += self.curvature_config["script"]
        code += headerFrame("HELIUM CURVATURE")
        return code
    

    def create_edpScripts(self) -> str:
        if self.curvature_config:
            main_code = self.add_helium_curvature_edp()
        else:
            main_code = "\n"
        main_code += headerFrame("ELECTROSTATIC POTENTIAL")
        main_code += self.script_load_packages_and_mesh()
        main_code += self.script_declare_variables()
        main_code += self.script_create_coupling_const_matrix()

        #create new files for each electrode, only inputting the lines of code needed
        for j in range(self.num_electrodes):
            code = ''

            filename = "electrode_" + str(j)
            self.cc_files.append(filename)
            code += self.script_create_savefiles(j)
            code += self.script_problem_definition(j)

            for i, extract_config in enumerate(self.config.get('extract_opt')):
                fem_object_name = self.__extract_opt[i] + f'_{j}'
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
                        code += self.script_save_data_2D(extract_config, fem_object_name)
                    elif plane in config_planes_3D:
                        code += self.script_save_data_3D(extract_config, fem_object_name)
                    else:
                        raise KeyError(f'Wrong plane! choose from {config_planes_2D} or {config_planes_3D}')
                    
            with open(self.dirname + filename + '.edp', 'w') as file:
                file.write(code)
        
        return main_code


    def create_edpScript(self) -> str:
        """ Creates main .edp script by combining different script components."""

        if self.curvature_config:
            code = self.add_helium_curvature_edp()
        else:
            code = "\n"
        code += headerFrame("ELECTROSTATIC POTENTIAL")
        code += self.script_load_packages_and_mesh()
        code += self.script_declare_variables()
        code += self.script_create_coupling_const_matrix()
        code += self.script_create_savefiles()

        code += self.script_problem_definition()

        for i, extract_config in enumerate(self.config.get('extract_opt')):
            fem_object_name = self.__extract_opt[i]
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
                    code += self.script_save_data_2D(extract_config, fem_object_name)
                elif plane in config_planes_3D:
                    code += self.script_save_data_3D(extract_config, fem_object_name)
                else:
                    raise KeyError(f'Wrong plane! choose from {config_planes_2D} or {config_planes_3D}')

        code += "}\n"
        return code


    def script_load_packages_and_mesh(self):
        code = """load "msh3"\n"""
        code += """load "gmsh"\n"""
        code += """load "medit"\n"""
        #code += f"""system("mkdir -p {self.dirname}");\n"""
        code += "\n"
        if self.run_from_notebook:
            path_meshfile = self.dirname + self.config["meshfile"] + '.msh2'
        else:
            path_meshfile = self.config["meshfile"] + '.msh2'
        code += f"""mesh3 Th = gmshload3("{path_meshfile}");\n"""
        return code


    def script_declare_variables(self) -> str:
        electrode_names = list(self.physicalSurfs.keys())
        electrode_id = list(self.physicalSurfs.values())

        code = "\n"
        code += "int n1, n2, n3;\n"
        code += "real xmin, xmax, ymin, ymax, ax3, zmin, zmax;\n"

        electrodestring_name = '"' + '","'.join(electrode_names) + '"'
        code += f"string[int] electrodenames = [{electrodestring_name}];\n"

        code += f"int[int] electrodeid = {electrode_id};\n"
        return code


    def script_create_coupling_const_matrix(self) -> str:
        number_of_electrodes = len(list(self.physicalSurfs.keys()))

        code = "\n"
        code += f"int numV = {number_of_electrodes};\n"
        code += "real[int, int] V(numV, numV);\n"
        code += "for (int i = 0; i < numV; i+= 1){\n"
        code += self.add_spaces(4) + "for (int j = 0; j < numV; j+= 1){\n"
        code += self.add_spaces(8) + "if (i == j) {V(i, j) = 1.0;}\n"
        code += self.add_spaces(8) + "else {V(i, j) = 1e-5;}}}\n"

        code += "\n"
        code += "real[int, int] CapacitanceMatrix(numV, numV);\n"
        code += "for (int i = 0; i < numV; i+= 1){\n"
        code += self.add_spaces(4) + "for (int j = 0; j < numV; j+= 1){\n"
        code += self.add_spaces(8) + "CapacitanceMatrix(i, j) = 0.0;}}\n"

        return code


    def script_create_savefiles(self, j: int):
        code = "\n"
        self.__extract_opt = {}
        for idx, extract_cfg in enumerate(self.config.get('extract_opt')):
            addname = extract_cfg['additional_name']
            qty = extract_cfg['quantity']
            pln = extract_cfg['plane']
            name = (addname + "_" if addname else "") + qty + ("_" + pln if pln else "") + f'_{j}'

            if self.run_from_notebook:
                name = self.dirname + name
            code += f"""ofstream extract{idx}{qty}("{name}.txt");\n"""
            self.__extract_opt[idx] = f"extract{idx}{qty}"

        return code


    def script_problem_definition(self, electrode_num: int) -> str:
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
            code += self.add_spaces(26) + f"""+ {epsilon[k]} * (region == {v})\n"""
        code += self.add_spaces(26) + ";\n"

        code += self.add_spaces(4) + "problem Electro(u,v,solver=CG) =\n"
        code += self.add_spaces(20) + "int3d(Th)(dielectric * Grad(u)' * Grad(v))\n"

        for i, v in enumerate(self.physicalSurfs.values()):
            code += self.add_spaces(20) + f"+ on({v},u = V({electrode_num},{i}))\n"
        code += self.add_spaces(20) + ";\n"

        code += self.add_spaces(4) + "Electro;\n"
        code += self.add_spaces(4) + """cout << "calculations are finished, saving data" << endl;\n \n"""

        return code


    def script_save_data_2D(self, params: dict, fem_object_name: str) -> str:
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
        code += self.add_spaces(4) + "{\n"
        code += self.add_spaces(4) + f"n1 = {params['coordinate1'][2]};\n"
        code += self.add_spaces(4) + f"n2 = {params['coordinate2'][2]};\n"
        code += self.add_spaces(4) + f"xmin = {params['coordinate1'][0]};\n"
        code += self.add_spaces(4) + f"xmax = {params['coordinate1'][1]};\n"
        code += self.add_spaces(4) + f"ymin = {params['coordinate2'][0]};\n"
        code += self.add_spaces(4) + f"ymax = {params['coordinate2'][1]};\n"
        code += self.add_spaces(4) + zcoord_code
        code += self.add_spaces(4) + "real[int,int] quantity(n1,n2);\n"
        code += self.add_spaces(4) + "real[int] xList(n1), yList(n2);\n \n"

        code += self.add_spaces(4) + "for(int i = 0; i < n1; i++){\n"
        code += self.add_spaces(8) + "real ax1 = xmin + i*(xmax-xmin)/(n1-1);\n"
        code += self.add_spaces(8) + "xList[i] = ax1;\n"
        code += self.add_spaces(8) + "for(int j = 0; j < n2; j++){\n"
        
        quantity = config_quantity.get(params['quantity'])
        code += self.add_spaces(12) + "real ax2 = ymin + j*(ymax-ymin)/(n2-1);\n"
        code += self.add_spaces(12) + "yList[j] = ax2;\n"

        # working along the helium curvature lines if the option is enabled
        if self.curvature_config:
            code += self.add_spaces(12) + f"ax3 = {scaling} * {self.curvature_config['displacement']}(ax1,ax2);\n"

        code += self.add_spaces(12) + f"quantity(i,j) = {quantity}({xyz});" + "}}\n \n"
        code += self.add_spaces(4) + f"""{name} << "startDATA " + electrodenames[k] + " ";\n"""
        code += self.add_spaces(4) + f"""{name} << quantity << endl;\n"""
        code += self.add_spaces(4) + f"""{name} << "END" << endl;\n"""

        code += self.add_spaces(4) + "if (k == numV - 1){\n"
        code += self.add_spaces(8) + f"""{name} << "startXY xlist ";\n"""
        code += self.add_spaces(8) + f"""{name} << xList << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "END" << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "startXY ylist ";\n"""
        code += self.add_spaces(8) + f"""{name} << yList << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "END" << endl;""" + "}\n"
        code += self.add_spaces(4) + "}\n"

        code += headerFrame("2D DATA EXTRACTION BLOCK END")

        return code  


    def script_save_data_3D(self, params: dict, fem_object_name: str) -> str:
        if params.get('plane')=='xyZ':
            xyz = "ax1,ax2,ax3"
        else:
            raise KeyError(f'Wrong plane! choose from {config_planes_3D}')

        name = fem_object_name  # params['quantity']

        code  = headerFrame("2D SLICES DATA EXTRACTION BLOCK START")
        code += self.add_spaces(4) + "{\n"

        # working along the helium curvature lines if the option is enabled
        if self.curvature_config:
            bulkHelevels = np.asarray(self.curvature_config["bulk_helium_distances"])
            scaling = scaling_size(bulkHelevels)
            surfaceHelevel = self.curvature_config["surface_helium_level"]
            code += self.add_spaces(4) + f"n3  = {len(scaling)};\n"
            code += self.add_spaces(4) + f"real[int] bulkHeliumLevels = {np.array2string(bulkHelevels, separator=', ')};\n"
            code += self.add_spaces(4) + f"real[int] bulkHeliumLevelDispScales = {np.array2string(scaling, separator=', ')};\n"
        else:
            code += self.add_spaces(4) + f"n3  = {params['coordinate3'][2]};\n"
            code += self.add_spaces(4) + f"zmin  = {params['coordinate3'][0]};\n"
            code += self.add_spaces(4) + f"zmax  = {params['coordinate3'][1]};\n"

        code += self.add_spaces(4) + f"n1 = {params['coordinate1'][2]};\n"
        code += self.add_spaces(4) + f"xmin = {params['coordinate1'][0]};\n"
        code += self.add_spaces(4) + f"xmax = {params['coordinate1'][1]};\n"
        code += self.add_spaces(4) + f"n2 = {params['coordinate2'][2]};\n"
        code += self.add_spaces(4) + f"ymin = {params['coordinate2'][0]};\n"
        code += self.add_spaces(4) + f"ymax = {params['coordinate2'][1]};\n"

        code += self.add_spaces(4) + "real[int,int] quantity(n1,n2);\n"
        code += self.add_spaces(4) + "real[int] xList(n1), yList(n2), zList(n3);\n \n"
        code += self.add_spaces(4) + f"""{name} << "startDATA " + electrodenames[k] << endl;\n"""

        code += self.add_spaces(4) + "for(int m = 0; m < n3; m++){\n"
        if self.curvature_config:
            code += self.add_spaces(8) + "zList[m] = bulkHeliumLevels[m];\n"
        else:
            code += self.add_spaces(8) + "real ax3 = zmin + m*(zmax-zmin)/(n3-1);\n"
            code += self.add_spaces(8) + "zList[m] = ax3;\n"

        code += self.add_spaces(8) + f"""{name} << "start2DSLICE " << zList[m] + " ";\n"""
        code += self.add_spaces(8) + "for(int i = 0; i < n1; i++){\n"
        code += self.add_spaces(12) + "real ax1 = xmin + i*(xmax-xmin)/(n1-1);\n"
        code += self.add_spaces(12) + "xList[i] = ax1;\n"
        code += self.add_spaces(12) + "for(int j = 0; j < n2; j++){\n"
        code += self.add_spaces(16) + "real ax2 = ymin + j*(ymax-ymin)/(n2-1);\n"
        code += self.add_spaces(16) + "yList[j] = ax2;\n"
        if self.curvature_config:
            code += self.add_spaces(16) + f"real ax3 = {surfaceHelevel} - bulkHeliumLevelDispScales[m] * {self.curvature_config['displacement']}(ax1,ax2);\n"
        
        quantity = config_quantity.get(params['quantity'])
        code += self.add_spaces(16) + f"""quantity(i,j) = {quantity}({xyz});""" + "}}\n"

        code += self.add_spaces(8) + f"""{name} << quantity << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "end" << endl;""" + "}\n"
        code += self.add_spaces(4) + f"""{name} << "END" << endl;\n \n"""

        code += self.add_spaces(4) + """if (k == numV - 1){\n"""
        code += self.add_spaces(8) + f"""{name} << "startXY xlist ";\n"""
        code += self.add_spaces(8) + f"""{name} << xList << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "END" << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "startXY ylist ";\n"""
        code += self.add_spaces(8) + f"""{name} << yList << endl;\n"""
        code += self.add_spaces(8) + f"""{name} << "END" << endl;""" + "}\n"
        code += self.add_spaces(4) + "}\n"

        code += headerFrame("2D SLICES DATA EXTRACTION BLOCK END")

        return code


    def script_save_cmatrix(self, params: dict, fem_object_name: str) -> str:
        name = fem_object_name  # params['quantity']

        code = self.add_spaces(4) + "{\n"
        code += self.add_spaces(4) + "for(int i = k; i < numV; i++){\n"
        code += self.add_spaces(8) + f"real charge = int2d(Th,electrodeid[i])((dielectric(x + eps*N.x, y + eps*N.y, z + eps*N.z) * field(u, x + eps*N.x, y + eps*N.y, z + eps*N.z)' * norm\n"
        code += self.add_spaces(46) + f"- dielectric(x - eps*N.x, y - eps*N.y, z - eps*N.z) * field(u, x - eps*N.x, y - eps*N.y, z - eps*N.z)' * norm));\n"
        code += self.add_spaces(8) + "CapacitanceMatrix(k,i) = charge;\n"
        code += self.add_spaces(8) + "CapacitanceMatrix(i,k) = charge;}\n"
        code += self.add_spaces(4) + """if (k == numV - 1){\n"""
        code += self.add_spaces(8) + f"{name} << electrodenames << endl;\n"
        code += self.add_spaces(8) + f"{name} << CapacitanceMatrix << endl;" + "}\n"
        code += self.add_spaces(4) + "}\n"

        return code


    def run(self,
            print_log=False,
            freefem_path=":/Applications/FreeFem++.app/Contents/ff-4.15/bin"):
        
        print("hello")
        
        try:
            edp_name = self.config["meshfile"]

            for edp_name in self.cc_files:
                print(f'processing file {edp_name}')
                bashCommand_ff = ['freefem++', self.dirname + edp_name + '.edp']
                env = os.environ.copy()
                env['PATH'] += freefem_path

                process = subprocess.Popen(bashCommand_ff, stdout=subprocess.PIPE, env=env)
                print(f'opened subprocess for {edp_name}')

                logs = ""
                items = iter(process.stdout.readline, b'')
                bar = alive_it(items, title='Freefem running ', force_tty=True, refresh_secs=1/35)

                for line in bar:
                    output_log = line.decode()
                    logs += edp_name + ':' + output_log
                    if output_log[1:6] == "Error":
                        raise FreefemError(output_log)
                    elif print_log:
                        print(output_log)
                    else:
                        pass

                process.stdout.close()
                process.wait()

        except KeyboardInterrupt:
            message = 'Interrupted by user'
            print(message)
            logs += message

        finally:
            with open(os.path.join(self.dirname, 'ff_logs.txt'), 'w') as outfile:
                outfile.write(logs)
            print('Freefem calculations are complete')


if __name__=="__main__":
    
    with open(r'freefem_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    pyff = FreeFEM(config=config, dirname='')
    pyff.run(print_log=True)
