import gmsh
import sys
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_it
from shapely import Polygon, MultiPolygon, get_coordinates
from scipy.interpolate import LinearNDInterpolator
from matplotlib.ticker import MaxNLocator


sys.path.insert(1, "/Volumes/EeroQ/lib/zeroheliumkit-dev/")
from zeroheliumkit import Structure, Entity
from zeroheliumkit.helpers.constants import alpha, rho, g

def flatten(l):
    return [item for sublist in l for item in sublist]

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def add_spaces(num: int) -> str:
    return ' ' * num

def scaling_size(bulk_helium_distance: float=1e-1):
    lengthscale = (rho * g * bulk_helium_distance)/alpha * 1e-6      # in um
    return lengthscale


class GMSHmaker2D():
    def __init__(self,
                 layout: Structure | Entity,
                 electode_config: dict,
                 mesh_config: dict | list[dict],
                 filename: str,
                 savedir :str = ''):
        self.layout = layout
        self.electrode_config = electode_config
        self.mesh_config = mesh_config
        self.filename = filename
        self.savedir = savedir

        gmsh.initialize()
        gmsh.model.add("DFG 3D")

        surfaces = self.create_gmsh_surfaces()
        self.fragmentation(flatten(list(surfaces.values())))
        self.physsurfaces = self.create_physical_surfaces(surfaces)
        self.physlines = self.create_physical_lines_from_polygons(self.electrode_config)
        self.define_mesh(self.mesh_config)


    def create_gmsh_surface(self, polygon: Polygon, zcoord: float) -> int:
        """
        Creates a Gmsh surface based on the given polygon and z-coordinate.

        Args:
            polygon (Polygon): Shapely polygon defining the shape of the surface.
            zcoord (float): The z-coordinate of the surface.

        Returns:
            int: The ID of the created Gmsh surface.
        """
        coords = get_coordinates(polygon)

        # creating gmsh Points
        points = []
        xy = coords[:-1]
        for coord in xy[::-1]:
            p = gmsh.model.occ.addPoint(coord[0], coord[1], zcoord)
            points.append(p)

        # creating gmsh Lines
        lines = []
        for i in range(len(points)-1):
            l = gmsh.model.occ.addLine(points[i], points[i+1])
            lines.append(l)
        closing_line = gmsh.model.occ.addLine(points[-1], points[0])
        lines.append(closing_line)

        curvedLoop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([curvedLoop])

        return surface


    def create_gmsh_surfaces(self) -> list:
        """
        Creates Gmsh surfaces based on the polygons in the layout.

        Returns:
            list: A list of the IDs of the created Gmsh surfaces.
        """
        surfaces = {}
        layers = self.layout.get_zhk_dict()
        for i, (k, v) in enumerate(layers.items()):
            if isinstance(v, MultiPolygon):
                surf_list = []
                for polygon in list(v.geoms):
                    gmsh_surface = self.create_gmsh_surface(polygon, 0)
                    surf_list.append(gmsh_surface)
                surfaces[k] = surf_list
            elif isinstance(v, Polygon):
                gmsh_surface = self.create_gmsh_surface(v, 0)
                surfaces[k] = [gmsh_surface]
        
        gmsh.model.occ.synchronize()
        
        return surfaces
    

    def fragmentation(self, surfaces: list) -> list:
        """ Gluing all surfaces together
            Handles correctly the shared surfaces between surfaces

        Args:
            surfaces (list): list of all surfaces

        Returns:
            list: list of reconfigured surfaces
        """
        item_base = [(2, surfaces[0])]
        item_rest = []
        for v in surfaces[1:]:
            item_rest.append((2, v))
        new_volumes = gmsh.model.occ.fragment(item_base, item_rest)
        
        gmsh.model.occ.synchronize()

        return new_volumes
    

    def create_physical_surfaces(self, surfaces: dict):
        """ Create physical surfaces from the surfaces

        Args:
            surfaces (dict): dictionary of surfaces
        """
        upd_dict = {}
        for group_id, (k, v) in enumerate(surfaces.items()):
            gmsh.model.addPhysicalGroup(2, v, group_id+1, name=k)
            upd_dict[k] = {"entities": v, "id": group_id+1}
        gmsh.model.occ.synchronize()

        return upd_dict
    

    def create_physical_lines_from_polygons(self, electrode_config: dict):
        """
        Create physical lines in the Gmsh model based on the electrode configuration.

        Args:
            electrode_config (dict): A dictionary containing the electrode configuration.
                The keys represent the names of the electrode groups, and the values
                contain information about the layer and polygons associated with each group.

        Example:
            electrode_config = {
                "ElectrodeGroup1": {
                    "layer": ("Layer1", [0, 1]),
                },
                "ElectrodeGroup2": {
                    "layer": ("Layer2", [2]),
                },
            }
            create_physical_lines(electrode_config)

        """
        electrode_config = removekey(electrode_config, "type")
        boundary_config = {}
        n0 = len(self.physsurfaces)
        for group_id, (k, v) in enumerate(electrode_config.items()):
            v["gmsh_id"] = []
            surfaces = self.physsurfaces[v["layer"][0]]['entities']
            sh_multipolygon = getattr(self.layout, v["layer"][0])
            if hasattr(sh_multipolygon, "geoms"):
                sh_polygons = list(sh_multipolygon.geoms)
            else:
                sh_polygons = [sh_multipolygon]
            sh_polygons = [sh_polygons[i] for i in v["layer"][1]]
            for surf in surfaces:
                for sh_poly in sh_polygons:
                    if self.gmsh_to_polygon_matches(surf, sh_poly):
                        _, down = gmsh.model.getAdjacencies(2, surf)  # 'down' contains all line tags the boundary of the surface is made of, 'up' is empty
                        v["gmsh_id"].extend(down)
            v["id"] = group_id + n0 + 1
            v["gmsh_id"] = [e for e in v["gmsh_id"] if e not in v["exclude"]]
            boundary_config[k] = {"group_id": v["id"], "entities": v["gmsh_id"], "value": v["value"]}
            gmsh.model.addPhysicalGroup(1, v["gmsh_id"], v["id"], name=k)

        self.electrode_config = electrode_config

        return boundary_config


    def create_physical_lines_from_lines(self, electrode_config: dict):
        electrode_config = removekey(electrode_config, "type")
        n0 = len(self.physsurfaces)
        for group_id, (k, v) in enumerate(electrode_config.items()):
            v["gmsh_id"] = []

    def gmsh_to_polygon_matches(self, gmsh_surface: int, shapely_polygon: Polygon) -> bool:
        """ compares gmsh Volumes with extruded shapely Polygon

        Args:
            Volume (int): gmsh Volume
            polygon (Polygon): shapely Polygon
            gmsh_layer (str): gmsh layer name, which contains info about z-coordinates

        Returns:
            bool: True, if Volume equals to extruded Polygon within tolerance
        """
        tol = 1e-4
        com_gmsh = gmsh.model.occ.getCenterOfMass(2, gmsh_surface)
        com_shapely_2D = list(shapely_polygon.centroid.coords)
        com_shapely = (com_shapely_2D[0][0], com_shapely_2D[0][1], 0)

        return np.allclose(np.asarray(com_gmsh), np.asarray(com_shapely), atol=tol)
    

    def define_mesh(self, mesh_config: dict | list[dict]):
        """ Define the mesh based on the given mesh configuration.

        Args:
        ----
            mesh_config (dict | list[dict]): The mesh configuration. It can be a single dictionary or a list of dictionaries.
                Each dictionary should contain the following keys:
                - "Thickness" (float): The thickness of the box.
                - "VIn" (float): The inner value of the box.
                - "VOut" (float): The outer value of the box.
                - "box" (list[float]): The coordinates of the box in the format [XMin, YMin, XMax, YMax].

        Returns:
        -------
            None
        """

        if isinstance(mesh_config, dict):
            mesh_config = [mesh_config]
        boxes = []
        for cfg in mesh_config:
            box = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(box, "Thickness", cfg["Thickness"])
            gmsh.model.mesh.field.setNumber(box, "VIn", cfg["VIn"])
            gmsh.model.mesh.field.setNumber(box, "VOut", cfg["VOut"])
            gmsh.model.mesh.field.setNumber(box, "XMin", cfg["box"][0])
            gmsh.model.mesh.field.setNumber(box, "XMax", cfg["box"][2])
            gmsh.model.mesh.field.setNumber(box, "YMin", cfg["box"][1])
            gmsh.model.mesh.field.setNumber(box, "YMax", cfg["box"][3])
            gmsh.model.mesh.field.setNumber(box, "ZMin", 0)
            gmsh.model.mesh.field.setNumber(box, "ZMax", 0)
            boxes.append(box)

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", boxes)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.occ.synchronize()
    

    def create_geo(self):
        gmsh.write(self.savedir + self.filename + ".geo_unrolled")
    
    def create_mesh(self, dim=2, print_progress=True):
        if print_progress:
            bar = alive_it([0], title='Gmsh generation ', length=3, spinner='elements', force_tty=True)
        else:
            bar = [0]
        try:
            for item in bar:
                gmsh.model.mesh.generate(dim)
                gmsh.write(self.savedir + self.filename + ".msh2")
        except KeyboardInterrupt:
            print('interrupted by user')
        
    def open_gmsh(self):
        gmsh.option.setColor("Geometry.Points", 255, 165, 0)
        gmsh.option.setColor("General.Text", 255, 255, 255)
        gmsh.option.setColor("Mesh.Points", 255, 0, 0)

        r, g, b, a = gmsh.option.getColor("Geometry.Points")
        gmsh.option.setColor("Geometry.Surfaces", r, g, b, a)

        if "-nopopup" not in sys.argv:
            gmsh.fltk.initialize()
            while gmsh.fltk.isAvailable():
                gmsh.fltk.wait()

    def finalize(self):
        gmsh.finalize()
    
    def disable_consoleOutput(self):
        gmsh.option.setNumber("General.Terminal", 0)

    def export_physical(self):
        gmsh_physical_config ={
            'physicalLines': [{"name": k,
                               "id": v.get('group_id'),
                               "value": v.get("value")
                               } for (k, v) in self.physlines.items()],
            "meshname": self.filename,
            "savedir": self.savedir
        }
        return gmsh_physical_config


class HeliumSurfaceFreeFEM():
    def __init__(self,
                fem_config: dict,
                save_edp: bool = False):
        self.fem_config = fem_config
        if save_edp:
            self.save_edp(fem_config['meshname'])


    def create_edp(self, meshfile_path: str=None):
        code = """load "gmsh"\n"""
        if meshfile_path:
            code += f"""mesh heliumsurfTh = gmshload("{meshfile_path}{self.fem_config['meshname']}.msh2");\n"""
        else:
            code += f"""mesh heliumsurfTh = gmshload("{self.fem_config['meshname']}.msh2");\n"""

        code += """cout << "Area: " << int2d(heliumsurfTh)(1.0) << endl;\n"""
        code += """\n"""

        code += """fespace heliumsurfVh(heliumsurfTh,P2);\n"""
        code += """heliumsurfVh<real> disp,vdisp;\n"""
        code += """problem HeliumSurfaceCalculate(disp,vdisp,solver=CG) =\n"""
        code += add_spaces(4) + """int2d(heliumsurfTh)( (dx(disp)*dx(vdisp) + dy(disp)*dy(vdisp)) )\n"""
        code += add_spaces(4) + """- int2d(heliumsurfTh)( 1*vdisp )\n"""
        for boundary_condition in self.fem_config['physicalLines']:
            id = boundary_condition['id']
            value = boundary_condition['value']
            code += add_spaces(4) + f"""+ on ({id},disp={value});\n"""

        code += """\n"""
        code += """HeliumSurfaceCalculate;\n"""
        code += """heliumsurfTh = adaptmesh( heliumsurfTh, disp, hmax = .5, hmin = 0.02, iso = 1, nbvx = 10000 );\n"""
        code += """HeliumSurfaceCalculate;\n"""
        code += """cout << "Helium Surface Calculations are finished" << endl;\n"""

        return code


    def save_edp(self, filename: str):
        with open(filename + ".edp", 'w') as f:
            f.write(self.create_edp(self.fem_config['savedir']))


    def get_code_config(self, bulk_helium_distances: float|list=0, surface_helium_level: float=0):
        config = {
            "script": self.create_edp(self.fem_config['savedir']),
            "displacement": "disp",
            "bulk_helium_distances": bulk_helium_distances if isinstance(bulk_helium_distances, list) else [bulk_helium_distances],
            "surface_helium_level": surface_helium_level
        }
        return config


    def run_pyfreefem(self):
        try:
            import pyFreeFem as pyff
        except ImportError:
            print("pyfreefem not installed. See https://github.com/neoh54/pyFreeFem for instructions.")

        script = pyff.edpScript(self.create_edp(self.fem_config['savedir']))
        script += """fespace Vh2(heliumsurfTh,P1);\n"""
        script += """Vh2 dispOutput = disp;\n"""
        script += pyff.OutputScript( heliumsurfTh = 'mesh' )
        script += pyff.OutputScript( dispOutput = 'vector' )
        ff_output = script.get_output()
        return ff_output

    def plot_mesh(self,
                  ff_output: dict,
                  ax: plt.Axes = None,
                  color: str = 'black',
                  plot_boundaries: bool = True,
                  boundary_color: str = None):
        Th = ff_output['heliumsurfTh']
        if ax is None:
            fig = plt.figure(figsize=(8,5))
            ax = plt.subplot(111)
        ax.set_aspect('equal')
        Th.plot_triangles(ax=ax, color=color, lw =.5, alpha =.3 )
        if plot_boundaries:
            Th.plot_boundaries(color=boundary_color)

    def plot_results(self,
                     ff_output: dict,
                     bulk_helium_distance: float = 1e-1,
                     ax: plt.Axes = None,
                     cmap: str = "RdYlBu", levels: int = 7,
                     plot_mesh: bool = True):
        Th = ff_output['heliumsurfTh']
        u = ff_output['dispOutput']
        scaling_factor = scaling_size(bulk_helium_distance)

        if ax is None:
            fig = plt.figure(figsize=(12,6))
            ax = plt.subplot(111)
        ax.set_aspect('equal')

        cs = ax.tricontourf(Th, u*scaling_factor*1e3, cmap=cmap, levels=levels)

        if plot_mesh:
            self.plot_mesh(ff_output, ax=ax, color='white', plot_boundaries=True, boundary_color='black')

        cbar = plt.colorbar(cs, ax=ax, extend='both', location='top', orientation='horizontal', shrink=0.6)
        cbar.set_label(r"Helium Surface Displacement (nm)", fontsize=10)
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()

        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")

    def plot_1D(self,
                ff_output: dict,
                bulk_helium_distance: float = 1e-1,
                ax: plt.Axes = None,
                marker: str = None,
                color: str = 'black',
                cut_axis: str = 'x',
                cut_value: float = 0,
                xlist: list = None,
                **kwargs):
        Th = ff_output['heliumsurfTh']
        u = ff_output['dispOutput']
        scaling_factor = scaling_size(bulk_helium_distance)
        interp = LinearNDInterpolator(list(zip(Th.x,Th.y)), u)

        if ax is None:
            fig = plt.figure(figsize=(8,5))
            ax = plt.subplot(111)

        if cut_axis == 'x':
            displacement = interp(xlist, cut_value) * scaling_factor * 1e3
        elif cut_axis == 'y':
            displacement = interp(cut_value, xlist) * scaling_factor * 1e3
        else:
            raise ValueError("cut_axis must be either 'x' or 'y'.")
        ax.plot(xlist, displacement, marker=marker, color=color, **kwargs)
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Helium Surface Displacement (nm)")
    
    def get_displacement(self, ff_output: dict, bulk_helium_distance: float = 1e-1, location: tuple=(0,0)):
        """ Returns the displacement at a given location on the helium surface (units are 'xy' coordinate units).
        Args:
        - ff_output (dict): The output dictionary from the freefem calculation.
        - bulk_helium_distance (float): The distance from the bulk helium surface.
        - location (tuple): The location coordinates (x, y) where the displacement is calculated.
        """

        Th = ff_output['heliumsurfTh']
        u = ff_output['dispOutput']
        scaling_factor = scaling_size(bulk_helium_distance)
        interp = LinearNDInterpolator(list(zip(Th.x,Th.y)), u)
        displacement = interp(*location) * scaling_factor
        return displacement