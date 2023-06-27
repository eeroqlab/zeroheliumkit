'''
    created by Niyaz B / October 10th, 2022
    for additional functionality look here:
    gmsh python API https://gitlab.onelab.info/gmsh/gmsh/-/tree/master
    some other examples http://jsdokken.com/src/tutorial_gmsh.html
    requirements:
    pip install gmsh
'''

import gmsh
import sys
import numpy as np

from shapely import Polygon, MultiPolygon, get_coordinates
from tabulate import tabulate
from alive_progress import alive_it

from ..core import Structure, Entity


#---------------------------------------------
# some useful functions

def gmshLayer_info(ref_layer: str, d: float, z: float, physical_name: str, cut: tuple=None, forConstruction: bool=False) -> dict:
    return {
        'reference':        ref_layer,
        'thickness':        d,
        'z':                z,
        'physical':         physical_name,
        'cut'     :         cut,
        'forConstruction':  forConstruction
        }

def physSurface_info(ref_layer: str, polygon_idx: list, gmsh_layer: str):
    return {
        'ref_layer':    ref_layer,
        'polygons':     polygon_idx,
        'gmsh_layer':   gmsh_layer,
        'group_id':     int(),
        'entities':     []
    }

def flatten(l):
    return [item for sublist in l for item in sublist]

def group_dict(i: int):
    return {'group_id': i, 'entities': []}

def convert_polygons_to_list(geometry: Polygon | MultiPolygon):
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    else:
        return [geometry]
    
def convert_layout_to_dict(layout: Structure | Entity) -> dict:
    """ Converts 'layout' class into dictionary.

    Args:
        layout (Structure | Entity): geometry design

    Returns:
        dict: dictionary wih layout info
                dict keys - layout layer names (only shapely Polygons are allowed)
                dict items - list of shapely Polygons
    """

    layer_names = layout.layer_names(geom_type="polygon")
    layer_geoms = [getattr(layout, k) for k in layer_names]     # list of Polygons and MultiPolygons
    geoms_list  = [convert_polygons_to_list(geoms) for geoms in layer_geoms]
    
    return dict(zip(layer_names, geoms_list))


#---------------------------------------------
# MAIN class, which constructs 3D geometry and mesh

class GMSHmaker():

    def __init__(self, 
                 layout: Structure | Entity,
                 extrude_config: dict, 
                 electrodes_config: dict, 
                 mesh_params: tuple, 
                 log: bool=False):
        
        self.layout = convert_layout_to_dict(layout)
        self.extrude_config = extrude_config
        #self.electrodes_config = electrodes_config

        gmsh.initialize()
        gmsh.model.add("DFG 3D")
        
        vols = self.create_gmsh_objects()
        
        self.fragmentation(flatten(list(vols.values())))
        
        self.physicalVolumes = self.create_PhysicalVolumes(vols)
        self.physicalSurfaces = self.create_PhysicalSurfaces(electrodes_config)
        
        self.define_mesh(*mesh_params)
        
    def get_polygon(self, lname: str, idx: int) -> Polygon:
        """ Returns a shapely Polygon from a specific layer and idx in layout dict"""
        return self.layout.get(lname)[idx]

    def polygon_extruded(self, polygon: Polygon, zcoord: float, thickness: float):
        """ Create  3D gmsh Volume by extruding shapely Polygon

        Args:
            polygon (Polygon): shapely Polygon
            zcoord (float): staring z-coordinate
            thickness (float): thickness of the Volume

        Returns:
            gmsh Volume tag
        """
        coords = get_coordinates(polygon)

        # creating gmsh Points
        points = []
        for coord in coords[:-1]:
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
        shell = gmsh.model.occ.extrude([(2, surface)], 0, 0, thickness)
        
        surfaces = []
        for item in shell:
            dimTag, objTag = item
            if dimTag == 2:
                surfaces.append(objTag)
            elif dimTag ==3: 
                volume = objTag
            else:
                print('could not create extruded Polygon')
        
        return volume

    def populate_volumes(self, names_list: list, volumes: dict) -> dict:
        """ Populates predefined 'volumes' dict with gmsh 3D objects
            3D object is created by extruding shapely Polygons 

        Args:
            names_list (list): list of gmsh volume names
            volumes (dict): volumes database

        Returns:
            dict: volumes database
        """
        
        for name in names_list:
            params = self.extrude_config.get(name)
            polygons = self.layout.get(params['reference'])

            for p in polygons:
                volumes[name].append(self.polygon_extruded(p, params['z'], params['thickness']))
            
        return volumes
    
    def gmshObjs_forCutting(self, cutting_layer_names: tuple, volumes: dict) -> list:
        """ Creates a list of gmsh Volumes from the list of cutting_layer_names

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
    
    def sort_gmshLayer_names(self) -> tuple:
        """ Prepares four lists of gmsh layer names 
            defining the order of 3D gmsh construction

            Construction logic
            1. first - contains a list of gmsh layer names, which will be constructed first by extruding polygons
            2. first_forConstruction - contains a list of gmsh layer names, 
                                       which will be temporarely constructed and but not present in the end geometry
            3. final_forConstruction - temporarely constructed, not present in the end geometry
                                       contains cut_info, which is the list of gmsh layers to be cutted
            4. final - list of gmsh layer to be constructed the last
                       contains cut_info, which could be gmsh layer names from 'first', 'first_foConstruction' and 'final_forConstruction' lists

        Returns:
            tuple: 4 lists with gmsh layer names with or without cut_info
        """
        config = self.extrude_config
        gmsh_layers = list(config.keys())

        first = []
        final = []
        first_forConstruction = []
        final_forConstruction = []
        for l in gmsh_layers:
            cut_info = config[l].get('cut')
            if not config[l].get('forConstruction'):
                if cut_info!=None:
                    final.append((l, cut_info))
                else:
                    first.append(l)
            else:
                if cut_info!=None:
                    final_forConstruction.append((l, cut_info))
                else:
                    first_forConstruction.append(l)
        
        return first, final, first_forConstruction, final_forConstruction
    
    
    def create_gmsh_objects(self) -> dict:
        """ Main 3D gmsh constructor function.

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
                base = self.polygon_extruded(p, params['z'], params['thickness'])
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
                base = self.polygon_extruded(p, params['z'], params['thickness'])
                base_gmshObj = gmsh.model.occ.cut([(3, base)], gmshObjs_forCutting, removeTool=False)
                if gmshObjs_forCutting_andRemove:
                    base_gmshObj = gmsh.model.occ.cut(base_gmshObj[0], gmshObjs_forCutting_andRemove, removeTool=True)
                for item in base_gmshObj[0]:
                    volumes[lname].append(item[1])
                
        gmsh.model.occ.synchronize()
        
        return volumes
    
    def fragmentation(self, volumes: list) -> list:
        """ Gluing all Volumes together
            Handles correctly the shared surfaces between Volumes

        Args:
            volumes (list): list of all Volumes

        Returns:
            list: list of reconfigured Volumes
        """
        item_base = [(3, volumes[0])]
        item_rest = []
        for v in volumes[1:]:
            item_rest.append((3, v))
        new_volumes = gmsh.model.occ.fragment(item_base, item_rest)
        
        gmsh.model.occ.synchronize()
        
        return new_volumes

    def create_PhysicalVolumes(self, volumes: dict) -> dict:
        """ Tags Volumes to Physical Volumes

        Args:
            volumes (dict): dict with gmsh layer and corresponding volumes

        Returns:
            dict: key - physical volume names; value - list of Volumes
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
        """ compares gmsh Volumes with extruded shapely Polygon

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
    
    def create_PhysicalSurfaces(self, electrodes: dict) -> dict:
        """ Defines the physical Surfaces, where voltages will be applied

        Args:
            electrodes (dict): electrodes config

        Returns:
            dict: populated electrodes config dict
        """
        init_index = len(self.physicalVolumes.keys())
        gmsh_entities_in_Metal = self.physicalVolumes['METAL']['entities']
        
        # populating electrodes with gmshEntities
        for i, (k, v) in enumerate(electrodes.items()):
            for gmsh_entity in gmsh_entities_in_Metal:
                for polygon_id in v['polygons']:
                    polygon = self.layout[v['ref_layer']][polygon_id]
                    if self.gmsh_to_polygon_matches(gmsh_entity, polygon, v['gmsh_layer']):
                        up, down = gmsh.model.getAdjacencies(3, gmsh_entity)  # 'down' contains all surface tags the boundary of the volume is made of, 'up' is empty
                        v['entities'].extend(down)
        
        # assigning electrodes to group_ids
        for i, (k, v) in enumerate(electrodes.items()):
            group_id = i + init_index + 1
            v['group_id'] = group_id
            gmsh.model.addPhysicalGroup(2, v['entities'], group_id, name=k)
        
        gmsh.model.occ.synchronize()
        
        return electrodes
    
    def get_surfaces_onEdges(self, Btype: str):
        allowed_types = ['x', 'y', 'z']
        if Btype not in allowed_types:
            raise TypeError(f'Btype error: only {allowed_types} is allowed')
        
        #gmsh_ent_points = gmsh.model.occ.getEntities(dim=0)
        #gmsh_ent_onSurf1 = gmsh.model.occ.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim=-1)
    
    def define_mesh(self, m1: float, m2: float, m3: float, fineregion_box: list, mediumregion_box: list):
        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "Thickness", 4)
        gmsh.model.mesh.field.setNumber(box, "VIn", m2)
        gmsh.model.mesh.field.setNumber(box, "VOut", m1)
        gmsh.model.mesh.field.setNumber(box, "XMin", mediumregion_box[0])
        gmsh.model.mesh.field.setNumber(box, "XMax", mediumregion_box[1])
        gmsh.model.mesh.field.setNumber(box, "YMin", mediumregion_box[2])
        gmsh.model.mesh.field.setNumber(box, "YMax", mediumregion_box[3])
        gmsh.model.mesh.field.setNumber(box, "ZMin", mediumregion_box[4])
        gmsh.model.mesh.field.setNumber(box, "ZMax", mediumregion_box[5])

        box2 = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box2, "Thickness", 2)
        gmsh.model.mesh.field.setNumber(box2, "VIn", m3)
        gmsh.model.mesh.field.setNumber(box2, "VOut", m1)
        gmsh.model.mesh.field.setNumber(box2, "XMin", fineregion_box[0])
        gmsh.model.mesh.field.setNumber(box2, "XMax", fineregion_box[1])
        gmsh.model.mesh.field.setNumber(box2, "YMin", fineregion_box[2])
        gmsh.model.mesh.field.setNumber(box2, "YMax", fineregion_box[3])
        gmsh.model.mesh.field.setNumber(box2, "ZMin", fineregion_box[4])
        gmsh.model.mesh.field.setNumber(box2, "ZMax", fineregion_box[5])

        """
        cyl = gmsh.model.mesh.field.add("Cylinder")
        gmsh.model.mesh.field.setNumber(cyl, "Radius", fineregion_R)
        gmsh.model.mesh.field.setNumber(cyl, "VIn", m3)
        gmsh.model.mesh.field.setNumber(cyl, "VOut", m1)
        gmsh.model.mesh.field.setNumber(cyl, "XAxis", mediumregion_box[1] - mediumregion_box[0])
        gmsh.model.mesh.field.setNumber(cyl, "XCenter", (mediumregion_box[1] + mediumregion_box[0])/2)
        gmsh.model.mesh.field.setNumber(cyl, "YAxis", 0)
        gmsh.model.mesh.field.setNumber(cyl, "YCenter", 0)
        gmsh.model.mesh.field.setNumber(cyl, "ZAxis", 0)
        gmsh.model.mesh.field.setNumber(cyl, "ZCenter", 0.5)
        """
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", [box, box2])
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.occ.synchronize()
    
    def create_geo(self, filename: str):
        gmsh.write(filename + ".geo_unrolled")
    
    def create_mesh(self, filename: str, dim='2'):
        bar = alive_it([0], title='Gmsh generation ', length=3, spinner='elements', force_tty=True) 
        try:
            for item in bar:
                gmsh.model.mesh.generate(dim)
                print("mesh is constructed")
                gmsh.write(filename + ".msh2")
                print("mesh saved")
        except KeyboardInterrupt:
            print('interrupted by user')
        
    def open_gmsh(self):
        gmsh.option.setColor("Geometry.Points", 255, 165, 0)
        gmsh.option.setColor("General.Text", 255, 255, 255)
        #gmsh.option.setColor("Mesh.Points", 255, 0, 0)

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