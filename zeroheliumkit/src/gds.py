import numpy as np
import gdstk

from .anchors import GDSSpec
from .core import Structure, Entity
from .supercore import GeomCollection 
from .functions import write_layers_to_cell, read_layers_from_cell

def structure_to_cell(
        structure: Entity | Structure,
        cell_name: str
    ) -> gdstk.Cell:
    """
    Flattens a zhk Entity/Structure's layers into a gdstk.Cell. No file I/O.

    Args:
        structure (Entity): the zhk Entity/Structure to convert.
        cell_name (str): name of the gdstk.Cell to create (or reuse) within `library`.

    Returns:
        gdstk.Cell: the created (or reused) cell, containing the structure's polygons.
    """
    dict_of_layers = structure.as_dict(remove_holes=True, include_anchors_skeletone=False)
    
    return write_layers_to_cell(cell_name, dict_of_layers)


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
