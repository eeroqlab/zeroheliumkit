import numpy as np
import gdstk
import math

from .anchors import GDSSpec
from .core import Structure, Entity
from .supercore import GeomCollection 
from .functions import write_layers_to_cell, read_layers_from_cell

def structure_to_cell(
        structure: Entity | Structure,
        cell_name: str,
        library: gdstk.Library = None
    ) -> gdstk.Cell:
    """
    Flattens a zhk Entity/Structure's layers into a gdstk.Cell. No file I/O.

    Args:
        structure (Entity): the zhk Entity/Structure to convert.
        cell_name (str): name of the gdstk.Cell to create (or reuse) within `library`.
        library (gdstk.Library, optional): library to add the cell to. If None, a fresh
            library is created. Pass the same library across calls (e.g. via
            GDSAssembly.add_structure) to accumulate cells for later referencing.

    Returns:
        gdstk.Cell: the created (or reused) cell, containing the structure's polygons.
    """
    dict_of_layers = structure.as_dict(remove_holes=True, include_anchors_skeletone=False)

    if library is None:
        library = gdstk.Library()

    existing_names = {cell.name: cell for cell in library.cells}
    if cell_name in existing_names:
        cell = existing_names[cell_name]
    else:
        cell = gdstk.Cell(cell_name)
        library.add(cell)

    write_layers_to_cell(cell, dict_of_layers)
    return cell


def cell_to_structure(cell: gdstk.Cell, name_mapping: dict) -> GeomCollection:
    """
    Resolves a gdstk.Cell's polygons back into a flat zhk GeomCollection.

    Nested references inside `cell` are resolved recursively (see _read_layers_from_cell),
    but the result is flat -- hierarchy information itself is not preserved. See
    GDSAssembly.from_gds for hierarchy-preserving import.

    Args:
        cell (gdstk.Cell): the cell to convert.
        name_mapping (dict): layer configuration, e.g. {1: "metal", 2: "ground"}.
            Used to map GDS layer numbers back to zhk layer names.

    Returns:
        GeomCollection: flat structure with one Layer per layer number found in `name_mapping`.
    """
    gds_layer_dict = read_layers_from_cell(cell)

    dict_of_layers = {}
    dict_of_gdsspec = {}
    for key, multipolygon in gds_layer_dict.items():
        name = name_mapping.get(key[0], None)
        if name is None:
            print(message=f"cell '{cell.name}' has layer {key[0]}, datatype {key[1]} "
                         "not present in name_mapping; skipping.")
            continue
        dict_of_layers[name] = multipolygon
        dict_of_gdsspec[name] = GDSSpec(**key)

    return GeomCollection(dict_of_layers, dict_of_gdsspec)



class GDSAssembly():
    """
    Holds a gdstk.Library and a top cell, and composes zhk Structures and/or raw
    gdstk cells into a hierarchical GDS design via cell references (gdstk.Reference).

    Unlike Structure/Entity, GDSAssembly has no flat layers of its own -- it exists
    purely to manage the gdstk.Library/cell-reference hierarchy. To include local
    geometry alongside references, build it as a normal Structure and add it via
    `add_structure` like any other referenced cell.
    """

    def __init__(self, top_cell_name: str = 'toplevel'):
        self.library = gdstk.Library()
        self.top_cell = self.library.new_cell(top_cell_name)

    @property
    def cell_names(self) -> list:
        return [cell.name for cell in self.library.cells]

    ##############################
    #### reference operations ####
    ##############################

    def add_structure(self, structure: Entity, layer_cfg: dict, cell_name: str,
                       position: tuple = (0, 0), rotation: float = 0) -> gdstk.Cell:
        """
        Converts `structure` to a gdstk.Cell in this assembly's library (reusing the
        cell if `cell_name` was already added) and places one reference to it.

        Args:
            structure (Entity): the zhk Entity/Structure to place.
            layer_cfg (dict): layer configuration passed to `structure_to_cell`.
            cell_name (str): name for the converted cell.
            position (tuple): (x, y) placement of the reference.
            rotation (float): rotation of the reference, in degrees.

        Returns:
            gdstk.Cell: the converted cell, reusable for further references
            (via `add_reference`/`add_reference_array`) without reconverting.
        """
        cell = structure_to_cell(structure, layer_cfg, cell_name, library=self.library)
        self.add_reference(cell, position, rotation)
        return cell

    def add_reference(self, cell: gdstk.Cell, position: tuple, rotation: float = 0) -> None:
        """
        Places a single reference to `cell` in the top cell.

        Args:
            cell (gdstk.Cell): the cell to reference (e.g. from `add_structure`,
                or loaded directly via gdstk).
            position (tuple): (x, y) placement of the reference.
            rotation (float): rotation of the reference, in degrees.
        """
        if cell.name not in self.cell_names:
            self.library.add(cell)
        self.top_cell.add(gdstk.Reference(cell, position, rotation=math.radians(rotation)))

    def add_reference_array(self, cell: gdstk.Cell, position: tuple = (0, 0),
                             columns: int = 1, rows: int = 1, spacing: tuple = (0, 0),
                             rotation: float = 0) -> None:
        """
        Places a 2D array of references to `cell` in the top cell.

        Args:
            cell (gdstk.Cell): the cell to reference and array.
            position (tuple): (x, y) placement of the first instance of the array.
            columns (int): number of columns in the array.
            rows (int): number of rows in the array.
            spacing (tuple): (dx, dy) spacing between column/row centerpoints.
            rotation (float): rotation of each reference, in degrees.
        """
        if cell.name not in self.cell_names:
            self.library.add(cell)
        self.top_cell.add(gdstk.Reference(cell, position, columns=columns, rows=rows,
                                           spacing=spacing, rotation=math.radians(rotation)))

    ##############################
    #### Exporting operations ####
    ##############################

    def export_gds(self, filename: str) -> None:
        """
        Writes this assembly's library (top cell + all referenced cells) to a GDS file.

        Args:
            filename (str): the name of the gds file to be exported (without extension).
        """
        self.library.write_gds(filename + '.gds')

    ##############################
    #### Importing operations ####
    ##############################

    @classmethod
    def from_gds(cls, filepath: str, top_cell_name: str = None) -> "GDSAssembly":
        """
        Loads a GDS file, keeping its full cell hierarchy intact (no flattening).

        Args:
            filepath (str): path to the .gds file.
            top_cell_name (str, optional): name of the top cell. Defaults to the
                library's own top-level cell.

        Returns:
            GDSAssembly: assembly wrapping the loaded library.
        """
        assembly = cls.__new__(cls)
        assembly.library = gdstk.read_gds(filepath)
        assembly.top_cell = (assembly.library[top_cell_name] if top_cell_name
                              else assembly.library.top_level()[0])
        return assembly

    def flatten_to_structure(self, layer_cfg: dict, cell_name: str = None) -> GeomCollection:
        """
        Resolves one cell's polygons (default: the top cell) into a flat GeomCollection.
        Deliberately lossy -- does not preserve hierarchy. See `cell_to_structure`.

        Args:
            layer_cfg (dict): layer configuration used to map GDS layer numbers to names.
            cell_name (str, optional): name of the cell to flatten. Defaults to the top cell.

        Returns:
            GeomCollection: flat structure with the cell's polygons.
        """
        cell = self.library[cell_name] if cell_name else self.top_cell
        return cell_to_structure(cell, layer_cfg)