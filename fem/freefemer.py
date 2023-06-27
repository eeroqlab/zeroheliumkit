'''
    created by Niyaz B / January 10th, 2023
'''

import yaml
import subprocess
from alive_progress import alive_it
import os

from ..errors import *

config_types_2D = ['xy', 'yz', 'xz']
config_types_3D = ['xy_z']

def extract_results(name: str, type: str, axis1_params: tuple, axis2_params: tuple, axis3_value: float or tuple) -> dict:
    return {
        'name': name,
        'type': type,
        'n1': axis1_params[2],
        'n2': axis2_params[2],
        'xmin': axis1_params[0],
        'xmax': axis1_params[1],
        'ymin': axis2_params[0],
        'ymax': axis2_params[1],
        'planeLoc': axis3_value
    }

class FreeFEM():
    
    def __init__(self, 
                edp_name: str, 
                mesh_name: str, 
                config: dict,
                dirname: str):

        self.edp_name = edp_name
        self.mesh_name = mesh_name
        self.config = config
        self.dirname = dirname
        self.filename = edp_name

        self.physicalVols = config.get('physicalVolumes')
        self.physicalSurfs = config.get('physicalSurfaces')
        self.num_electrodes = len(list(self.physicalSurfs.keys()))
        self.write_edpScript()

    def write_edpScript(self):
        script = self.create_edpScript()
        with open(self.dirname + self.filename + '.edp', 'w') as file:
            file.write(script)

    def create_edpScript(self):
        code  = self.script_load_packages_and_mesh()
        code += self.script_declare_variables()
        code += self.script_create_coupling_const_matrix(self.num_electrodes)
        code += self.script_create_savefiles()
        code += self.script_problem_definition(self.config.get('ff_polynomial'))
        for extract_config in self.config.get('extract_opt'):
            config_type = extract_config.get('type')
            if config_type in config_types_2D:
                code += self.script_save_data_2D(extract_config)
            elif config_type in config_types_3D:
                code += self.script_save_data_3D(extract_config)
            else:
                raise KeyError(f'Wrong type! choose from {config_types_2D} or {config_types_3D}')
        code += "\n        }"
        return code

    def script_load_packages_and_mesh(self):
        code = f"""
        load "msh3"
        load "gmsh"
        load "medit"
        //system("mkdir -p {self.dirname}");
        mesh3 Th = gmshload3("{self.dirname}{self.mesh_name}.msh2");
        """
        return code
    
    def script_declare_variables(self) -> str:
               
        electrode_names = list(self.physicalSurfs.keys())
        code = f"""
        int n1, n2, n3;
        real xmin, xmax, ymin, ymax, ax3, zmin, zmax;
        """
        electrodestring = '"'+'","'.join(electrode_names)+'"'
        code += f'string[int] electrodenames = [{electrodestring}];'
        
        return code
    
    def script_create_coupling_const_matrix(self, number_of_electrodes: int) -> str:
        code = f"""
        int numV = {number_of_electrodes};
        """

        code += """
        real[int, int] V(numV, numV);
        for (int i = 0; i < numV; i+= 1){
            for (int j = 0; j < numV; j+= 1){
                if (i == j) {V(i, j) = 1.0;}
                else {V(i, j) = 1e-5;}
            }
        }
        """
        return code

    def script_create_savefiles(self):
        code = ""
        for extract_config in self.config.get('extract_opt'):
            name = extract_config.get('name')
            code += f"""
        ofstream {name}("{self.dirname}{self.filename}_{name}.txt");"""
        return code
    
    def script_problem_definition(self, polynomial=1) -> str:
        if polynomial == 1:
            femSpace = 'P13d'
        elif polynomial == 2:
            femSpace = 'P23d'
        else:
            raise Exception("Wrong polynomial order! Choose between 1 or 2")
        
        code = """

        for(int k = 0; k < numV; k++){"""
        
        if 'periodic_BC' in self.config:
            code += f"""
        fespace Vh(Th,{femSpace}, periodic=[[{self.config.get('periodic_BC')[0]}, x, y], [{self.config.get('periodic_BC')[1]}, x, y]]);
            """
        else:
            code += f"""
        fespace Vh(Th,{femSpace});
            """
        
        code += """
        Vh u,v;
        macro Grad(u) [dx(u),dy(u),dz(u)] //
        problem Electro(u,v,solver=CG) =
        """

        eps_consts = self.config.get('dielectric_constants')
        
        code += ""
        for i, (k,v) in enumerate(self.physicalVols.items()):
            if i==0:
                code += self.script_varf(eps_consts[k], v, first=True)
            else:
                code += "      " + self.script_varf(eps_consts[k], v)
        
        code += ""
        for i, (k,v) in enumerate(self.physicalSurfs.items()):
            code += self.script_boundary(v, i)
        code += "              ;\n"

        code += """
        Electro;
        func real phi(real X, real Y, real Z){
        if (abs(u(X, Y, Z)) < 1e-6) {return 0.0;} 
        else {return u(X, Y, Z);}}
        """
        return code
    
    def script_varf(self, eD: float, physVol: int, first=False) -> str:
        plus_symbol = "" if first else "+ "
        code = "        " + plus_symbol + f"int3d(Th,{physVol})({eD}*Grad(u)' * Grad(v))\n"
        return code
    
    def script_boundary(self, physSurf: int, index: int) -> str:
        code = f"              + on({physSurf},u = V(k,{index}))\n"
        return code

    def script_save_data_2D(self, params: dict) -> str:
        
        if params.get('type')=='xy':
            xyz = "ax1,ax2,ax3"
        elif params.get('type')=='yz':
            xyz = "ax3,ax1,ax2"
        elif params.get('type')=='xz':
            xyz = "ax1,ax3,ax2"
        else:
            raise KeyError(f'Wrong type! choose from {config_types_2D}')
 
        code = """
        {"""
        code += f"""
        n1 = {params['n1']};
        n2 = {params['n2']};
        xmin = {params['xmin']};
        xmax = {params['xmax']};
        ymin = {params['ymin']};
        ymax = {params['ymax']};
        ax3  = {params['planeLoc']};
        real[int,int] potential(n1,n2);
        real[int] xList(n1), yList(n2);
        """

        code += """
        for(int i = 0; i < n1; i++){
        """
        code += f"""    real ax1 = xmin + i*(xmax-xmin)/(n1-1);
            xList[i] = ax1;
        """
        code += """    for(int j = 0; j < n2; j++){
        """
        code += f"""        real ax2 = ymin + j*(ymax-ymin)/(n2-1);
                yList[j] = ax2;
                potential(i,j) = phi({xyz});"""+"}}\n"
        
        name = params['name']
        code += f"""
        {name} << "startDATA " + electrodenames[k] + " ";
        {name} << potential << endl;
        {name} << "END" << endl;
        """

        code += """
        if (k == numV - 1){"""
        code += f"""
            {name} << "startXY xlist ";
            {name} << xList << endl;
            {name} << "END" << endl;
            {name} << "startXY ylist ";
            {name} << yList << endl;
            {name} << "END" << endl;""" + "}"

        code += """
        }\n"""
        return code  
    
    def script_save_data_3D(self, params: dict) -> str:
        
        if params.get('type')=='xy_z':
            xyz = "ax1,ax2,ax3"
        else:
            raise KeyError(f'Wrong type! choose from {config_types_3D}')
        
        name = params['name']

        code = """
        {"""
        code += f"""
        n1 = {params['n1']};
        n2 = {params['n2']};
        n3  = {params['planeLoc'][2]};
        xmin = {params['xmin']};
        xmax = {params['xmax']};
        ymin = {params['ymin']};
        ymax = {params['ymax']};
        zmin  = {params['planeLoc'][0]};
        zmax  = {params['planeLoc'][1]};
        real[int,int] potential(n1,n2);
        real[int] xList(n1), yList(n2), zList(n3);
        {name} << "startDATA " + electrodenames[k] << endl;
        """

        code += """
        for(int m = 0; m < n3; m++){
            real ax3 = zmin + m*(zmax-zmin)/(n3-1);
            zList[m] = ax3;
        """
        code += f"""
            {name} << "start2DSLICE " << ax3 + " ";
        """
        code += """
            for(int i = 0; i < n1; i++){
                real ax1 = xmin + i*(xmax-xmin)/(n1-1);
                xList[i] = ax1;
        """
        code += """     
                for(int j = 0; j < n2; j++){
                    real ax2 = ymin + j*(ymax-ymin)/(n2-1);
                    yList[j] = ax2;
        """
        code += f"""            
                    potential(i,j) = phi({xyz});"""+"}}\n"
        
        
        code += f"""
            {name} << potential << endl;
            {name} << "end" << endl;
        """
        code += """}\n"""
        code += f"""        {name} << "END" << endl;
        """

        code += """
        if (k == numV - 1){"""
        code += f"""
            {name} << "startXY xlist ";
            {name} << xList << endl;
            {name} << "END" << endl;
            {name} << "startXY ylist ";
            {name} << yList << endl;
            {name} << "END" << endl;""" + "}"

        code += """
        }\n"""
        return code
        

    def run(self, print_log=False):

        try:
            bashCommand_ff = ['freefem++', self.dirname + self.edp_name + '.edp']
            #env = os.environ.copy()
            #env['PATH'] += ":/Applications/FreeFem++.app/Contents/ff-4.12/bin"
            process = subprocess.Popen(bashCommand_ff, stdout=subprocess.PIPE)#, env=env)

            logs = ""
            items = iter(process.stdout.readline, b'')
            bar = alive_it(items, title='Freefem running ', force_tty=True, refresh_secs=1/35) 
            
            for line in bar:
                output_log = line.decode()
                logs += output_log 
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
            with open(self.dirname + '/ff_logs.txt', 'w') as outfile:
                outfile.write(logs)
            print('Freefem calculations are complete')


class CreateFreeFEMscript_InducedCharge(FreeFEM):
    def __init__(self, 
                edp_name: str, 
                mesh_name: str, 
                config: dict,
                dirname: str,
                filename: str):
        super().__init__(edp_name, mesh_name, config, dirname, filename)

    def create_edpScript(self):
        code  = self.script_load_packages_and_mesh()

        code += self.script_problem_definition(self.config.get('ff_polynomial'))
        for extract_config in self.config.get('extract_opt'):
            config_type = extract_config.get('type')
            if config_type in config_types_2D:
                code += self.script_save_data_2D(extract_config)
            elif config_type in config_types_3D:
                code += self.script_save_data_3D(extract_config)
            else:
                raise KeyError(f'Wrong type! choose from {config_types_2D} or {config_types_3D}')
        code += "\n        }"
        return code
    
    def script_charge_distribution(self):
        pass

if __name__=="__main__":

    directory = 'data/trap/'
    with open(directory + r'/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    pyff = FreeFEM(
                edp_name='qchip', 
                mesh_name='qchip', 
                config=config,
                dirname=directory)

    pyff.run(print_log=True)