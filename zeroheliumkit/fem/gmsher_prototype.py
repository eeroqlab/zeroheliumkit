'''
    created by Niyaz B / October 10th, 2022
    for additional functionality look here:
    gmsh python API https://gitlab.onelab.info/gmsh/gmsh/-/tree/master
    some other examples http://jsdokken.com/src/tutorial_gmsh.html
    requirements:
    pip install gmsh
'''

import gmsh
import sys, os, yaml
import numpy as np

from shapely import Polygon, MultiPolygon, get_coordinates, point_on_surface
from alive_progress import alive_it
from pathlib import Path
from dataclasses import dataclass, field

from ..src.core import Structure, Entity


#---------------------------------------------
# some useful functions

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_group(i: int):
    return {'group_id': i, 'entities': []}

def ensure_polygon_list(geometry: Polygon | MultiPolygon):
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    else:
        return [geometry]


#---------------------------------------------
# gmsh configuration dataclasses

@dataclass
class ExtrudeSettings:
    """
    Settings for the extrusion process.

    Args:
        geometry (Polygon | MultiPolygon): Geometry to be extruded.
        z (float): Z-coordinate or position of the layer.
        d (float): Thickness of the layer.
        physical_name (str): Physical name associated with the layer.
        cut (tuple, optional): Tuple specifying the volumes which will be cut from the main volume.
            Defaults to None.
        forConstruction (bool, optional): Flag indicating if the layer is for construction purposes. Defaults to False.
    """
    geometry: Polygon | MultiPolygon
    z_base: float = 0.0
    height: float = 1.0
    physical_name: str = "DIELECTRIC"
    cut: tuple[str] = None
    forConstruction: bool = False


@dataclass
class SurfaceSettings:
    """
    Settings for defining additional surfaces on the volumes.

    Args:
        geometry (Polygon | MultiPolygon): Geometry of the surface.
        z (float): Z-coordinate of the surface.
        index (list[int], optional): List of polygon indices associated with the surface. Defaults to an empty list.
    """
    geometry: Polygon | MultiPolygon
    z: float
    index: list[int] = field(default_factory=list)


@dataclass
class PECSettings:
    """
    Settings for defining Perfect Electric Conductor (PEC) boundaries.

    Args:
        geometry (Polygon | MultiPolygon): Geometry of the PEC boundary.
        indices (list[int]): List of polygon indices associated with the PEC boundary.
        volume (str, optional): Volume identifier associated with the PEC boundary. Defaults to None.
        z (float, optional): Z-coordinate of the PEC boundary. Defaults to None.
        linked_to (str, optional): Identifier of another entity this PEC boundary is linked to. Defaults to None.
    """
    geometry: Polygon | MultiPolygon
    indices: list[int]
    volume: str = None
    z: float = None
    linked_to: str = None

    def __post_init__(self):
        if (self.z is not None) and (self.volume is not None):
            raise ValueError("PECSettings: Only one of 'z' or 'volume' should be provided.")


@dataclass
class BuildPlan:
    build_1: list[str] = field(default_factory=list)
    build_2: list[str] = field(default_factory=list)
    build_3: list[tuple[str, list]] = field(default_factory=list)
    build_4: list[tuple[str, list]] = field(default_factory=list)


#---------------------------------------------
# MAIN class, which constructs 3D geometry and mesh

class GMSHmaker2():
    """
    GMSHmaker class constructs 3D geometry and mesh using GMSH Python API.
    
    Args:
        layout (Structure | Entity): geometry design
        extrude_config (dict): configuration for extruding 2D polygons into 3D volumes
        electrodes_config (dict): configuration for defining physical surfaces for electrodes
        mesh_params (tuple): parameters for mesh generation
        additional_surfaces (dict, optional): additional surfaces to be included in the geometry. Defaults to None.
        savedir (str, optional): directory to save the mesh file. Defaults to "dump".
        configdir (str, optional): directory to save the configuration file. Defaults to "config".
        filename (str, optional): name of the mesh file. Defaults to "device".
    """

    def __init__(self,
                 extrude: dict,
                 surfaces: dict=None,
                 pecs: dict=None,
                 mesh: dict=None,
                 save: dict={"dir": "dump/", "filename": "device"},
                 open_gmsh: bool=False,
                 debug_mode: bool=False):
        
        self.extrude = extrude
        self.surfaces = surfaces
        self.pecs = pecs
        self.mesh = mesh
        self.save = save
        self.open_gmsh = open_gmsh
        self.debug_mode = debug_mode

        self.make_mesh()


    def make_mesh(self):
        gmsh.initialize()
        gmsh.model.add("DFG 3D")

        if self.debug_mode:
            self.disable_consoleOutput()
        
        try:
            vols = self.create_gmsh_objects()
            
            self.fragmentation(flatten(list(vols.values())))
            
            self.physicalVolumes = self.create_PhysicalVolumes(vols)
            self.physicalSurfaces = self.create_PhysicalSurfaces()

            self.export_config()
            self.define_mesh()
            self.create_geo()
            self.create_mesh()
            if self.open_gmsh:
                self.launch_gmsh_gui()
        
        except Exception as e:
            print(f'Error during gmsh mesh creation: {e}')
            raise e
        finally:
            gmsh.finalize()
        
    def get_polygon(self, lname: str, idx: int) -> Polygon:
        """ Returns a shapely Polygon from a specific layer and idx in layout dict"""
        return self.layout.get(lname)[idx]
    
    def build_gmsh_surface(self, base_polygon: Polygon, base_z: float) -> int:
        """
        Creates a Gmsh surface based on the given base_polygon and z-coordinate.

        Args:
            base_polygon (Polygon): Shapely polygon defining the shape of the surface.
            base_z (float): The z-coordinate of the surface.

        Returns:
            int: The ID of the created Gmsh surface.
        """
        coords = get_coordinates(base_polygon)

        # creating gmsh Points
        points = []
        for coord in coords[:-1]:
            p = gmsh.model.occ.addPoint(coord[0], coord[1], base_z)
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

    def build_gmsh_volume(self, base_polygon: Polygon, base_z: float, height: float):
        """
        Creates 3D gmsh Volume by extruding shapely Polygon.

        Args:
            base_polygon (Polygon): shapely Polygon
            base_z (float): staring z-coordinate
            height (float): height of the Volume

        Returns:
            gmsh Volume tag
        """

        surface_id = self.build_gmsh_surface(base_polygon, base_z)
        shell = gmsh.model.occ.extrude([(2, surface_id)], 0, 0, height)
        
        surfaces = []
        for item in shell:
            dimTag, objTag = item
            if dimTag == 2:
                surfaces.append(objTag)
            elif dimTag == 3: 
                volume = objTag
            else:
                print('could not create extruded Polygon')
        
        return volume

    def register_gmsh_volumes(self, volume_names: list[str], volume_dict: dict) -> dict:
        """
        Build multiple Gmsh volumes and register them in the volume dictionary.
        Populates predefined 'volumes' dict with gmsh 3D objects.
        3D object is created by extruding shapely Polygons.

        Args:
            volume_names (list): list of gmsh volume names
            volume_dict (dict): volumes database

        Returns:
            dict: volumes database
        """
        
        for name in volume_names:
            config = self.extrude.get(name)
            polygons = ensure_polygon_list(config['geometry'])

            for poly in polygons:
                gmsh_volume = self.build_gmsh_volume(poly, config['z_base'], config['height'])
                volume_dict[name].append(gmsh_volume)

        return volume_dict
    
    def gmshObjs_forCutting(self, cutting_layer_names: tuple, volumes: dict) -> list:
        """
        Creates a list of gmsh Volumes from the list of cutting_layer_names.

        Args:
            cutting_layer_names (tuple): contains a list of gmsh layer names will be used for cutting
            volumes (dict): volumes database, from which gmsh Volumes will be selected

        Returns:
            list: list of Volumes, which will be used for cutting
        """
        list_gmsh_objects = []
        for lname in cutting_layer_names:
            if lname in volumes.keys():
                for gmsh_obj in volumes.get(lname):    
                    list_gmsh_objects.append((3, gmsh_obj))
        
        return list_gmsh_objects
    
    def sort_gmshLayer_names(self) -> BuildPlan:
        """
        Prepares four lists of gmsh layer names defining the order of 3D gmsh construction

        Construction logic
        1. build_1 - contains a list of gmsh layer names, which will be constructed first by extruding polygons
        2. build_2 - contains a list of gmsh layer names, 
                    which will be temporarely constructed and but not present in the end geometry
        3. build_3 - temporarely constructed, not present in the end geometry
                    contains cut_info, which is the list of gmsh layers to be cutted
        4. build_4 - list of gmsh layer to be constructed the last
                    contains cut_info, which could be gmsh layer names from 'build_1', 'build_2' and 'build_3' lists

        Returns:
            BuildPlan: 4 lists with gmsh layer names with or without cut_info
        """

        plan = BuildPlan()
        for vol_name, config in self.extrude.items():
            match (config.forConstruction, isinstance(config.cut, tuple)):
                case (False, False):
                    plan.build_1.append(vol_name)
                case (True, False):
                    plan.build_2.append(vol_name)
                case (True, True):
                    plan.build_3.append((vol_name, config.cut))
                case (False, True):
                    plan.build_4.append((vol_name, config.cut))
        
        return plan
    
    
    def create_gmsh_objects(self) -> dict:
        """
        Main 3D gmsh constructor function.

        Returns:
            dict: configuration about gmsh layers and constructed Volumes contained in these layers.
        """
        config = self.extrude_config
        gmsh_layers = list(config.keys())

        volumes = {key:[] for key in gmsh_layers if not config[key].get('forConstruction')}
        volumes_forConstruction = {key:[] for key in gmsh_layers if config[key].get('forConstruction')}

        # Prepare the list of gmsh layers and define the order of the operation
        first, final, first_forConstruction, final_forConstruction = self.sort_gmshLayer_names()

        # first we create gmsh objects by extruding shPolygons
        # which doesn't have forConstruction tag and 'cut' argument
        # returns updated volumes dict
        volumes = self.populate_volumes(first, volumes)

        # next we create gmsh objects forConstruction
        # which doesn't have 'cut' argument and have forConstruction tag
        # returns updated volumes_forConstruction dict
        volumes_forConstruction = self.populate_volumes(first_forConstruction, volumes_forConstruction)

        # next we create gmsh objects forConstruction
        # which have 'cut' argument and have forConstruction tag
        # 'cut' tuple should contain only gmsh layers with tag forConstruction
        # returns updated volumes_forConstruction dict
        for lname, cutting_layers in final_forConstruction:
            params = self.extrude_config.get(lname)
            polygons = self.layout.get(params['reference'])

            gmshObjs_forCutting = self.gmshObjs_forCutting(cutting_layers, volumes_forConstruction)

            for p in polygons:
                base = self.create_gmsh_volume_by_extrusion(p, params['z'], params['thickness'])
                base = gmsh.model.occ.cut([(3, base)], gmshObjs_forCutting, removeTool=True)
                for item in base[0]:
                    volumes_forConstruction[lname].append(item[1])

        # finally we create gmsh objects, which have 'cut' argument. 
        # 'cut' tuple can contain gmsh layers with and without forConstruction tag
        # returns updated volumes dict
        for lname, cutting_layers in final:
            params = self.extrude_config.get(lname)
            polygons = self.layout.get(params['reference'])
            
            gmshObjs_forCutting = self.gmshObjs_forCutting(cutting_layers, volumes)
            gmshObjs_forCutting_andRemove = self.gmshObjs_forCutting(cutting_layers, volumes_forConstruction)

            for p in polygons:
                base = self.create_gmsh_volume_by_extrusion(p, params['z'], params['thickness'])
                base_gmshObj = gmsh.model.occ.cut([(3, base)], gmshObjs_forCutting, removeTool=False)
                if gmshObjs_forCutting_andRemove:
                    base_gmshObj = gmsh.model.occ.cut(base_gmshObj[0], gmshObjs_forCutting_andRemove, removeTool=True)
                for item in base_gmshObj[0]:
                    volumes[lname].append(item[1])

        # adding additional surfaces
        if self.additional_surfaces:
            for surface in self.additional_surfaces.values():
                surface['gmshID'] = []
                if isinstance(surface['geometry'], MultiPolygon):
                    for poly in surface['geometry'].geoms:
                        gmshID = self.create_gmsh_surface(poly, surface['z0'])
                        surface['gmshID'].append(gmshID)
                else:
                    gmshID = self.create_gmsh_surface(surface['geometry'], surface['z0'])
                    surface['gmshID'].append(gmshID)

        gmsh.model.occ.synchronize()
        
        return volumes
    
    def fragmentation(self, volumes: list) -> list:
        """
        Gluing all Volumes together. Handles correctly the shared surfaces between Volumes.

        Args:
            volumes (list): list of all Volumes

        Returns:
            list: list of reconfigured Volumes
        """
        item_base = [(3, volumes[0])]
        item_rest = []
        for v in volumes[1:]:
            item_rest.append((3, v))
        if self.additional_surfaces:
            for value in self.additional_surfaces.values():
                for gID in value['gmshID']:
                    item_rest.append((2, gID))
        new_volumes = gmsh.model.occ.fragment(item_base, item_rest)
        
        gmsh.model.occ.synchronize()
        
        return new_volumes

    def create_PhysicalVolumes(self, volumes: dict) -> dict:
        """
        Tags Volumes to Physical Volumes.

        Args:
            volumes (dict): dict with gmsh layer and corresponding volumes.

        Returns:
            dict: key - physical volume names; value - list of Volumes.
        """
        config = self.extrude_config
        
        layers = [key for key in config.keys() if not config[key].get('forConstruction')]
        volume_names = [item['physical'] for key, item in list(config.items()) if not config[key].get('forConstruction')]
        unique_names = list(set(volume_names))
        physVolumes_groups = {key: group_dict(i + 1) for i, key in enumerate(unique_names)}
        # assigning gmshEntities to 'layer' in paramteres
        for l in layers:
            physical_name = config[l]['physical']
            physVolumes_groups[physical_name]['entities'].extend(volumes[l])
        
        # assigning volumes to group_ids
        for name in unique_names:
            group_id = physVolumes_groups[name]['group_id']
            gmsh.model.addPhysicalGroup(3, physVolumes_groups[name]['entities'], group_id, name=name)
        
        gmsh.model.occ.synchronize()
        
        return physVolumes_groups
    
    def gmsh_to_polygon_matches(self, Volume: int, polygon: Polygon, gmsh_layer: str) -> bool:
        """
        Compares gmsh Volumes with extruded shapely Polygon.

        Args:
            Volume (int): gmsh Volume
            polygon (Polygon): shapely Polygon
            gmsh_layer (str): gmsh layer name, which contains info about z-coordinates

        Returns:
            bool: True, if Volume equals to extruded Polygon within tolerance
        """
        tol = 1e-4
        com = gmsh.model.occ.getCenterOfMass(3, Volume)
        centroid_xy = list(polygon.centroid.coords)
        centroid_z = self.extrude_config[gmsh_layer]['z'] + self.extrude_config[gmsh_layer]['thickness']/2
        centroid = (centroid_xy[0][0], centroid_xy[0][1], centroid_z)

        return np.allclose(np.asarray(com), np.asarray(centroid), atol=tol)
    
    def create_PhysicalSurfaces(self) -> dict:
        """
        Defines the physical Surfaces, where voltages will be applied.

        Args:
            electrodes (dict): electrodes config

        Returns:
            dict: populated electrodes config dict
        """
        init_index = len(self.physicalVolumes.keys())
        if 'METAL' in self.physicalVolumes:
            gmsh_entities_in_Metal = self.physicalVolumes['METAL']['entities']
        else:
            gmsh_entities_in_Metal = None
        electrodes = self.electrodes_config

        # populating electrodes with gmshEntities
        for i, (k, v) in enumerate(electrodes.items()):
            if gmsh_entities_in_Metal is not None:
                for gmsh_entity in gmsh_entities_in_Metal:
                    for polygon_id in v['polygons']:
                        polygon = self.layout[v['ref_layer']][polygon_id]
                        if self.gmsh_to_polygon_matches(gmsh_entity, polygon, v['gmsh_layer']):
                            _, down = gmsh.model.getAdjacencies(3, gmsh_entity)  # 'down' contains all surface tags the boundary of the volume is made of, 'up' is empty
                            v['entities'].extend(down)
            else:
                allSurfaces = gmsh.model.occ.getEntities(dim=2)
                for polygon_id in v['polygons']:
                    polygon = self.layout[v['ref_layer']][polygon_id]
                    point_inside = point_on_surface(polygon)
                    outDimTags, _, _ = gmsh.model.occ.getClosestEntities(point_inside.x, point_inside.y, v['gmsh_layer'], allSurfaces, n=1)
                    v['entities'].append(outDimTags[0][1])
        
        # assigning electrodes to group_ids
        unique_electrodes = {}
        for i, (k, v) in enumerate(electrodes.items()):
            group_id = i + init_index + 1    
            if v['linked_to']:
                physSurfName = v['linked_to']
                new_surfaces = v['entities']
                uniques = unique_electrodes[physSurfName]['entities']
                combined_without_duplicates = uniques + list(set(new_surfaces) - set(uniques))
                unique_electrodes[physSurfName]['entities'] = combined_without_duplicates
            else:
                unique_electrodes[k] = {'group_id': group_id, 'entities': v['entities']}

        for k, v in unique_electrodes.items():
            gmsh.model.addPhysicalGroup(2, v['entities'], v['group_id'], name=k)

        # assigning additional surfaces to group_ids
        # group_id += 1
        # if self.additional_surfaces:
        #     for i, (k, v) in enumerate(self.additional_surfaces.items()):
        #         gmsh.model.addPhysicalGroup(2, [v['gmshID']], group_id + i, name=k)
        #         unique_electrodes[k] = {'group_id': group_id + i, 'entities': [v['gmshID']]}

        gmsh.model.occ.synchronize()

        return unique_electrodes
    
    def get_surfaces_onEdges(self, Btype: str):
        allowed_types = ['x', 'y', 'z']
        if Btype not in allowed_types:
            raise TypeError(f'Btype error: only {allowed_types} is allowed')
        
        #gmsh_ent_points = gmsh.model.occ.getEntities(dim=0)
        #gmsh_ent_onSurf1 = gmsh.model.occ.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim=-1)
    
    def define_mesh(self, mesh_config: dict | list[dict]):
        """
        The mesh setup config. It can be a single dictionary or a list of dictionaries.

        Args:
            mesh_config (dict | list[dict]):
                Each dictionary should contain the following keys:
                - "Thickness" (float): The thickness of the box.
                - "VIn" (float): The inner value of the box.
                - "VOut" (float): The outer value of the box.
                - "box" (list[float]): The coordinates of the box in the format [XMin, XMax, YMin, YMax, ZMin, ZMax].
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
            gmsh.model.mesh.field.setNumber(box, "XMax", cfg["box"][1])
            gmsh.model.mesh.field.setNumber(box, "YMin", cfg["box"][2])
            gmsh.model.mesh.field.setNumber(box, "YMax", cfg["box"][3])
            gmsh.model.mesh.field.setNumber(box, "ZMin", cfg["box"][4])
            gmsh.model.mesh.field.setNumber(box, "ZMax", cfg["box"][5])
            boxes.append(box)

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", boxes)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.occ.synchronize()
    
    def create_geo(self):
        """
        Generates and writes the geometry definition to a file in GMSH format.
        """
        gmsh.write(str(self.savedir / self.filename) + ".geo_unrolled")
    
    def create_mesh(self, dim='2'):
        """
        Generates a mesh using Gmsh and saves it to the specified directory.

        Args:
            dim (str, optional): The dimension of the mesh to generate ('2' or '3'). Defaults to '2'.

        Raises:
            KeyboardInterrupt: If the mesh generation is interrupted by the user.
        """

        os.makedirs(self.savedir, exist_ok=True)
        bar = alive_it([0], title='Gmsh generation ', length=3, spinner='elements', force_tty=True) 
        try:
            for item in bar:
                gmsh.model.mesh.generate(dim)
                print("mesh is constructed")
                gmsh.model.mesh.setOrder(1)
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.option.setNumber("Mesh.Binary", 0)
                gmsh.write(str(self.savedir / self.filename) + ".msh2")
                print("mesh saved")
        except KeyboardInterrupt:
            print('interrupted by user')
        
    def launch_gmsh_gui(self):
        """
        Opens the Gmsh graphical user interface with customized color options for geometry points and text.
        """

        gmsh.option.setColor("Geometry.Points", 255, 165, 0)
        gmsh.option.setColor("General.Text", 255, 255, 255)
        #gmsh.option.setColor("Mesh.Points", 255, 0, 0)

        r, g, b, a = gmsh.option.getColor("Geometry.Points")
        gmsh.option.setColor("Geometry.Surfaces", r, g, b, a)

        if "-nopopup" not in sys.argv:
            gmsh.fltk.initialize()
            while gmsh.fltk.isAvailable():
                gmsh.fltk.wait()

    
    def disable_consoleOutput(self):
        """
        Disables console output in GMSH by setting the 'General.Terminal' option to 0.
        This method suppresses terminal messages from GMSH, which can be useful for 
        running scripts in environments where console output is not desired.
        """
        gmsh.option.setNumber("General.Terminal", 0)

    def export_config(self):
        """
        Exports the current GMSH configuration to a YAML file.
        The configuration includes the save directory, mesh file name, extrusion settings,
        and mappings of physical surfaces and volumes to their group IDs.
        """

        gmsh_config ={
            'savedir': str(self.savedir),
            'meshfile': self.filename,
            'extrude': self.extrude_config,
            'physicalSurfaces': {k: v.get('group_id') for (k, v) in self.physicalSurfaces.items()},
            'physicalVolumes': {k: v.get('group_id') for (k, v) in self.physicalVolumes.items()}
        }
        os.makedirs(self.configdir, exist_ok=True)
        config_filename = self.filename.replace(".msh2","") + ".yaml"
        with open(self.configdir / config_filename, 'w') as file:
            yaml.safe_dump(gmsh_config, file, sort_keys=False, indent=3)
