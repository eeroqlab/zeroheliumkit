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

from shapely import Polygon, MultiPolygon, LineString, MultiLineString, get_coordinates, point_on_surface
from alive_progress import alive_it
from dataclasses import dataclass, field, asdict
from tabulate import tabulate
from pathlib import Path


#---------------------------------------------
# some useful functions

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_group(i: int):
    return {'group_id': i, 'tags': []}

def ensure_polygon_list(geometry: Polygon | MultiPolygon):
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    else:
        return [geometry]

def ensure_linestring_list(geometry: LineString | MultiLineString):
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    else:
        return [geometry]

def custom_dict_factory(data):
    """
    A custom dict_factory to exclude 'geometry' attributes.
    """
    return {k: v for (k, v) in data if k != "geometry"}


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
    volume: ExtrudeSettings = None
    z: float = None
    group_id: int = None
    tags: list[int] = field(default_factory=list)
    linked_to: str = None
    prepared_polygons: list[Polygon] = field(init=False, default_factory=list)

    def __post_init__(self):
        if (self.z is not None) and (self.volume is not None):
            raise ValueError("PECSettings: Only one of 'z' or 'volume' should be provided initially.")
        for id in self.indices:
            if isinstance(self.geometry, MultiPolygon):
                self.prepared_polygons.append(self.geometry.geoms[id])
            elif isinstance(self.geometry, Polygon):
                self.prepared_polygons.append(self.geometry)
            else:
                raise TypeError("PECSettings: 'geometry' must be a Polygon or MultiPolygon.")
        if self.volume:
            self.z = self.volume.z_base + self.volume.height/2  # center z-coordinate of the volume


@dataclass
class MeshSettings:
    dim: int = 3
    fields: dict = field(default_factory=dict)


@dataclass
class BoxFieldMeshSettings:
    Thickness: float
    VIn: float
    VOut: float
    box: list[float]  # [XMin, XMax, YMin, YMax, ZMin, ZMax]

@dataclass
class DistanceFieldMeshSettings:
    geometry: LineString | MultiLineString
    base_z: float
    sampling: int
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float

    def __post_init__(self):
        self.lines = ensure_linestring_list(self.geometry)

@dataclass
class FixedFieldMeshSettings:
    pass


@dataclass
class BuildPlan:
    build_1: list[str] = field(default_factory=list)
    build_2: list[str] = field(default_factory=list)
    build_3: list[tuple[str, list]] = field(default_factory=list)
    build_4: list[tuple[str, list]] = field(default_factory=list)


#---------------------------------------------
# MAIN class, which constructs 3D geometry and mesh

class GMSHmaker():
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
                 extrude: ExtrudeSettings,
                 surfaces: SurfaceSettings=None,
                 pecs: PECSettings=None,
                 mesh: MeshSettings=None,
                 save: dict={"dir": "dump", "filename": "device"},
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
        """
        Creates the mesh using GMSH.

        Raises:
            e: Exception raised during mesh creation.
        """
        gmsh.initialize()
        gmsh.model.add("DFG 3D")

        if not self.debug_mode:
            self.disable_consoleOutput()
        
        try:
            vols = self.create_gmsh_objects()
            self.fragmentation(vols)
            self.physicalVolumes = self.create_PhysicalVolumes(vols)
            self.physicalSurfaces = self.create_PhysicalSurfaces()
            self.export_config()

            self.setup_mesh_fields()
            self.create_geo()
            self.create_mesh(dim=self.mesh.dim)
            if self.open_gmsh:
                self.launch_gmsh_gui()
        
        except Exception as e:
            print(f'Error during gmsh mesh creation: {e}')
            raise e
        finally:
            gmsh.finalize()


    def build_gmsh_points(self, coordinates: list[tuple[float, float, float]], meshSize: float=0.0) -> list[int]:
        """
        Creates Gmsh points based on the given coordinates.

        Args:
            coordinates (list[tuple[float, float, float]]): List of (x, y, z) tuples defining the points.
            meshSize (float, optional): Mesh size at the points. Defaults to 0.0.

        Returns:
            list[int]: List of IDs of the created Gmsh points.
        """
        points = []
        for xyz in coordinates:
            p = gmsh.model.occ.addPoint(xyz[0], xyz[1], xyz[2], meshSize=meshSize)
            points.append(p)
        return points


    def build_gmsh_lines(self, point_ids: list[int], closed: bool) -> list[int]:
        """
        Creates Gmsh lines connecting the given point IDs.

        Args:
            point_ids (list[int]): List of Gmsh point IDs.
            closed (bool): Whether to close the loop (connect last point to first).

        Returns:
            list[int]: List of IDs of the created Gmsh lines.
        """
        lines = []
        for i in range(len(point_ids)-1):
            l = gmsh.model.occ.addLine(point_ids[i], point_ids[i+1])
            lines.append(l)
        if closed:
            closing_line = gmsh.model.occ.addLine(point_ids[-1], point_ids[0])
            lines.append(closing_line)
        return lines


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
        points =  self.build_gmsh_points([(x, y, base_z) for x, y in coords[:-1]])

        # creating gmsh Lines
        lines =  self.build_gmsh_lines(points, closed=True)

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
        volume_dimTags = gmsh.model.occ.extrude([(2, surface_id)], 0, 0, height)
        
        surfaces = []
        for dim, tag in volume_dimTags:
            if dim == 2:
                surfaces.append(tag)
            elif dim == 3: 
                volume = (3, tag)
            else:
                print('could not create extruded Polygon')
        
        return volume


    def build_gmsh_volumes_from_plan(
            self,
            plan: list[tuple],
            volume_registry: dict,
            plan_id: int,
            tool_registry: dict=None) -> dict:
        """
        Build multiple Gmsh volumes and register them in the volume registry.
        Populates predefined 'volumes' dict with gmsh 3D objects.
        3D object is created by extruding shapely Polygons.

        Args:
            plan (list): list of tuples with (gmsh layer name, cut_info)
            volume_registry (dict): volumes database
            plan_id (int): defines the construction logic
            tool_registry (dict, optional): volumes database for cutting tools. Defaults to None.

        Returns:
            dict: updated volume registry containing built Gmsh volume IDs.
        """
        
        for layer_name, cut_info in plan:
            config = self.extrude.get(layer_name)
            polygons = ensure_polygon_list(config.geometry)

            # Gather cutting entities (as (dim, tag) pairs) from registries if requested
            if cut_info:
                cut_entities = self.get_gmsh_cut_entities(cut_info, volume_registry)
            if tool_registry:
                # used in the final step of build, if forConstruction volumes are also used for cutting
                temp_cut_entities = self.get_gmsh_cut_entities(cut_info, tool_registry)

            for poly in polygons:
                volume_dimTag = self.build_gmsh_volume(poly, config.z_base, config.height)
                
                match plan_id:
                    case 1 | 2:
                        out_entities = [volume_dimTag]
                    case 3:
                        # used to prepare forConstruction volumes, which will be cut
                        out_entities, _ = gmsh.model.occ.cut([volume_dimTag], cut_entities, removeTool=True)
                    case 4:
                        out_entities, _ = gmsh.model.occ.cut([volume_dimTag], cut_entities, removeTool=False)
                        if tool_registry:
                            # used in the final step of build, if forConstruction volumes are also used for cutting
                            out_entities, _ = gmsh.model.occ.cut(out_entities, temp_cut_entities, removeTool=True)

                for dim, tag in out_entities:
                    if dim == 3:
                        volume_registry[layer_name].append(tag)
        return volume_registry


    def get_gmsh_cut_entities(self, cut_layer_names: tuple, volume_registry: dict) -> list:
        """
        Collects Gmsh 3D entities (dim=3) from specified layers to be used as cutting tools.

        Args:
            cut_layer_names (tuple): Names of layers whose volumes will be used for cutting.
            volume_registry (dict): volumes database, from which gmsh Volumes will be selected

        Returns:
            list: List of (dimension, tag) pairs representing Gmsh volumes to cut with.
        """
        cut_dimTags = [
            (3, volume_id)
            for name in cut_layer_names
            for volume_id in volume_registry.get(name, [])
        ]
        return cut_dimTags


    def build_additional_gmsh_surfaces(self):
        """
        Adds additional surfaces to the Gmsh model based on the provided surface configurations.

        Each entry in `self.surfaces` must define:
            - geometry (Polygon or MultiPolygon)
            - z (float): the z-coordinate of the surface

        The generated Gmsh surface IDs are appended to `config.index`.
        """
        for _, config in self.surfaces.items():
            polygons = ensure_polygon_list(config.geometry)
            for poly in polygons:
                gmsh_id = self.build_gmsh_surface(poly, config.z)
                config.index.append(gmsh_id)

    
    def make_plan(self) -> BuildPlan:
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
                    plan.build_1.append((vol_name, None))
                case (True, False):
                    plan.build_2.append((vol_name, None))
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

        vol_registry, vol_registry_forConstruction = {}, {}
        for name, config in self.extrude.items():
            if config.forConstruction:
                vol_registry_forConstruction[name] = []
            else:
                vol_registry[name] = []

        # Prepare the list of gmsh layers and define the order of the operation
        plan = self.make_plan()

        # first we create gmsh objects by extruding shPolygons
        # which doesn't have forConstruction tag and 'cut' argument
        # returns updated vol_registry dict
        vol_registry = self.build_gmsh_volumes_from_plan(plan.build_1, vol_registry, plan_id=1)

        # next we create gmsh objects forConstruction
        # which have forConstruction tag and doesn't have 'cut' argument
        # returns updated vol_registry_forConstruction dict
        vol_registry_forConstruction = self.build_gmsh_volumes_from_plan(plan.build_2, vol_registry_forConstruction, plan_id=2)

        # next we create gmsh objects forConstruction
        # which have have forConstruction tag and 'cut' argument 
        # 'cut' tuple should contain only gmsh layers with tag forConstruction
        # returns updated volumes_forConstruction dict
        vol_registry_forConstruction = self.build_gmsh_volumes_from_plan(plan.build_3, vol_registry_forConstruction, plan_id=3)

        # finally we create gmsh objects, which have 'cut' only argument.
        # 'cut' tuple can contain gmsh layers with and without forConstruction tag
        # returns updated vol_registry dict
        vol_registry = self.build_gmsh_volumes_from_plan(plan.build_4, vol_registry, tool_registry=vol_registry_forConstruction, plan_id=4)

        # adding additional surfaces
        if self.surfaces:
            self.build_additional_gmsh_surfaces()

        gmsh.model.occ.synchronize()

        if self.debug_mode:
            self.launch_gmsh_gui()

        return vol_registry


    def fragmentation(self, volume_registry: dict) -> list:
        """
        Gluing all Volumes together. Handles correctly the shared surfaces between Volumes.

        Args:
            volume_registry (dict): dict with gmsh layer and corresponding volumes.

        Returns:
            list: list of reconfigured Volumes
        """
        volumes = flatten(list(volume_registry.values()))
        item_base = [(3, volumes[0])]
        item_rest = []
        for volume_tag in volumes[1:]:
            item_rest.append((3, volume_tag))
        if self.surfaces:
            indicies = []
            for config in self.surfaces.values():
                indicies.append(config.index)
            for surface_tag in flatten(indicies):
                item_rest.append((2, surface_tag))

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
        config = self.extrude

        layer_names = []
        volume_names = []
        for name, config in self.extrude.items():
            if not config.forConstruction:
                layer_names.append(name)
                volume_names.append(config.physical_name)
        unique_names = list(set(volume_names))
        physVolumes_groups = {key: make_group(i + 1) for i, key in enumerate(unique_names)}

        # assigning gmshEntities to 'layer' in paramteres
        for name in layer_names:
            physical_name = self.extrude[name].physical_name
            physVolumes_groups[physical_name]['tags'].extend(volumes[name])

        # assigning volumes to group_ids
        for name in unique_names:
            group_id = physVolumes_groups[name]['group_id']
            gmsh.model.addPhysicalGroup(3, physVolumes_groups[name]['tags'], group_id, name=name)
        
        gmsh.model.occ.synchronize()
        
        return physVolumes_groups

    
    def create_PhysicalSurfaces(self) -> dict:
        """
        Defines the physical Surfaces, where voltages will be applied.

        Returns:
            dict: populated electrodes config dict
        """
        
        metal_group = self.physicalVolumes.get("METAL")
        metal_volume_tags = metal_group['tags'] if metal_group else []

        allSurfaces = gmsh.model.occ.getEntities(dim=2)

        # populating electrodes with gmshEntities
        for pec_name, config in self.pecs.items():
            for poly in config.prepared_polygons:
                point_inside = point_on_surface(poly)
                for volumetag in metal_volume_tags:
                    if gmsh.model.isInside(3, volumetag, [point_inside.x, point_inside.y, config.z], parametric=False):
                        _, down = gmsh.model.getAdjacencies(3, volumetag)  # 'down' contains all surface tags the boundary of the volume is made of, 'up' is empty
                        config.tags.extend(down)
                if config.volume is None:
                    outDimTags, _, _ = gmsh.model.occ.getClosestEntities(point_inside.x, point_inside.y, config.z, allSurfaces, n=1)
                    config.tags.append(outDimTags[0][1])
        
        # assigning self.pecs to group_ids
        offset = len(self.physicalVolumes) # to avoid overlapping group_ids between volumes and surfaces
        unique_electrodes = {}
        for i, (pec_name, config) in enumerate(self.pecs.items()):
            group_id = i + offset + 1    
            if not config.linked_to:
                unique_electrodes[pec_name] = {'group_id': group_id, 'tags': config.tags}
            else:
                new_surfaces = config.tags
                uniques = unique_electrodes[config.linked_to]['tags']
                combined_without_duplicates = uniques + list(set(new_surfaces) - set(uniques))
                unique_electrodes[config.linked_to]['tags'] = combined_without_duplicates

        for k, v in unique_electrodes.items():
            gmsh.model.addPhysicalGroup(2, v['tags'], v['group_id'], name=k)
        gmsh.model.occ.synchronize()

        return unique_electrodes


    def get_surfaces_onEdges(self, Btype: str):
        allowed_types = ['x', 'y', 'z']
        if Btype not in allowed_types:
            raise TypeError(f'Btype error: only {allowed_types} is allowed')
        
        #gmsh_ent_points = gmsh.model.occ.getEntities(dim=0)
        #gmsh_ent_onSurf1 = gmsh.model.occ.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dim=-1)


    def make_box_field_mesh(self, boxes: list[BoxFieldMeshSettings]) -> list[int]:
        """
        Creates box fields for mesh refinement in Gmsh.
        Args:
            boxes (list[BoxFieldMeshSettings]): List of box field configurations.
        Returns:
            list: List of box field IDs created in Gmsh.
        """
        box_field_ids = []
        for config in boxes:
            box_id = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(box_id, "Thickness", config.Thickness)
            gmsh.model.mesh.field.setNumber(box_id, "VIn", config.VIn)
            gmsh.model.mesh.field.setNumber(box_id, "VOut", config.VOut)
            gmsh.model.mesh.field.setNumber(box_id, "XMin", config.box[0])
            gmsh.model.mesh.field.setNumber(box_id, "XMax", config.box[1])
            gmsh.model.mesh.field.setNumber(box_id, "YMin", config.box[2])
            gmsh.model.mesh.field.setNumber(box_id, "YMax", config.box[3])
            gmsh.model.mesh.field.setNumber(box_id, "ZMin", config.box[4])
            gmsh.model.mesh.field.setNumber(box_id, "ZMax", config.box[5])
            box_field_ids.append(box_id)

        return box_field_ids
    

    def setup_distance_field_mesh(self, lines: list[LineString], base_z: float, sampling: int=300) -> int:
        """ Sets up a distance field mesh in GMSH.

        Args:
            lines (list[LineString]): List of shapely LineStrings defining the lines for the distance field.
            base_z (float): The base Z-coordinate for the distance field.
            sampling (int, optional): Number of sampling points along the lines. Defaults to 300.

        Returns:
            int: The ID of the created distance field.
        """
        wires = []
        for l in lines:
            coords = get_coordinates(l)
            points = self.build_gmsh_points([(x, y, base_z) for x, y in coords])
            wire_ids = self.build_gmsh_lines(points, closed=False)
            wires.append(wire_ids)
        wires = flatten(wires)

        field_id = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_id, "CurvesList", wires)
        gmsh.model.mesh.field.setNumber(field_id, "Sampling", sampling)

        return field_id


    def setup_threshold_field_mesh(self, distance_field_id: int, SizeMin: float, SizeMax: float, DistMin: float, DistMax: float) -> int:
        """ Sets up a threshold field mesh in GMSH.

        Args:
            distance_field_id (int): The ID of the distance field to base the threshold on.
            SizeMin (float): Minimum mesh size.
            SizeMax (float): Maximum mesh size.
            DistMin (float): Minimum distance for mesh size transition.
            DistMax (float): Maximum distance for mesh size transition.

        Returns:
            int: The ID of the created threshold field.
        """
        field_id = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_id, "InField", distance_field_id)
        gmsh.model.mesh.field.setNumber(field_id, "SizeMin", SizeMin)
        gmsh.model.mesh.field.setNumber(field_id, "SizeMax", SizeMax)
        gmsh.model.mesh.field.setNumber(field_id, "DistMin", DistMin)
        gmsh.model.mesh.field.setNumber(field_id, "DistMax", DistMax)
        return field_id


    def make_distance_threshold_field_mesh(self, distances: list[DistanceFieldMeshSettings]) -> list[int]:
        """
        Creates distance and threshold fields for mesh refinement in Gmsh.

        Args:
            distances (list[DistanceFieldMeshSettings]): List of distance field configurations.

        Returns:
            list (int): List of threshold field IDs created in Gmsh.
        """
        field_ids = []
        for config in distances:
            distance_field_id = self.setup_distance_field_mesh(config.lines, config.base_z, config.sampling)
            threshold_field_id = self.setup_threshold_field_mesh(distance_field_id, config.SizeMin, config.SizeMax, config.DistMin, config.DistMax)
            field_ids.append(threshold_field_id)
        return field_ids


    def setup_mesh_fields(self):
        """
        Build and set the background mesh field.

        - Iterates over self.mesh.fields (name -> config)
        - For each recognized field name, calls the corresponding builder
        (e.g., self.make_box_field_mesh).
        - If multiple fields are created, combines them with a "Min" field.
        - Sets the resulting field as the background mesh.
        """
        field_ids = []
        for mesh_field_name, configs in self.mesh.fields.items():
            match mesh_field_name:
                case "Box":
                    field_ids.extend(self.make_box_field_mesh(configs))
                case "Distance":
                    field_ids.extend(self.make_distance_threshold_field_mesh(configs))
                case _:
                    print(f"Mesh field '{mesh_field_name}' is not recognized.")

        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.model.occ.synchronize()


    def create_geo(self):
        fullpath = Path(self.save["dir"]) / Path(self.save["filename"] + ".geo_unrolled")
        gmsh.write(str(fullpath))

    def create_mesh(self, dim=2):
        """
        Generates a mesh using Gmsh and saves it to the specified directory.

        Args:
            dim (str, optional): The dimension of the mesh to generate ('2' or '3'). Defaults to '2'.

        Raises:
            KeyboardInterrupt: If the mesh generation is interrupted by the user.
        """

        os.makedirs(self.save["dir"], exist_ok=True)
        bar = alive_it([0], title='Gmsh generation ', length=3, spinner='elements', force_tty=True) 
        try:
            for _ in bar:
                gmsh.model.mesh.generate(dim)
                print("mesh is constructed")
                gmsh.model.mesh.setOrder(1)
                gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                gmsh.option.setNumber("Mesh.Binary", 0)
                fullpath = Path(self.save["dir"]) / Path(self.save["filename"] + ".msh")
                gmsh.write(str(fullpath))
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
            'savedir': self.save["dir"],
            'meshfile': self.save["filename"] + ".msh2",
            'extrude': {k: asdict(v, dict_factory=custom_dict_factory) for (k, v) in self.extrude.items()},
            'physicalSurfaces': {k: v.get('group_id') for (k, v) in self.physicalSurfaces.items()},
            'physicalVolumes': {k: v.get('group_id') for (k, v) in self.physicalVolumes.items()}
        }
        os.makedirs(self.save["dir"], exist_ok=True)
        fullpath = Path(self.save["dir"]) / Path(self.save["filename"] + ".yaml")
        with open(str(fullpath), 'w') as file:
            yaml.safe_dump(gmsh_config, file, sort_keys=False, indent=3)
    

    def print_physical(self):
        """
        Prints tables of physical volumes and surfaces with their corresponding group IDs.
        """

        table = []
        for k, v in self.physicalVolumes.items():
            row = [k, v["group_id"]]
            table.append(row)
        print(tabulate(table, headers=["Volume", "ID"]))

        print("\n #-----------------------------------\n")

        table = []
        for k, v in self.physicalSurfaces.items():
            row = [k, v["group_id"]]
            table.append(row)
        print(tabulate(table, headers=["Surface", "ID"]))
