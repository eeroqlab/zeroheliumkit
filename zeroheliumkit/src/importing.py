import warnings
import numbers
import gdspy
import ezdxf
import pickle

from ezdxf.colors import BYLAYER
from shapely import Polygon, MultiPolygon, union

from .errors import *


class Exporter_GDS():
    """ A class for exporting geometries to GDSII format.

    Attributes:
    ----------
    name (str): The name of the GDSII file.
    zhk_layers (dict): A dictionary containing the geometries for each layer.
    gdsii (gdspy.GdsLibrary): The GDSII library object.
    layer_cfg (dict): A dictionary containing the layer configuration.
    """

    __slots__ = "name", "zhk_layers", "gdsii", "layer_cfg"

    def __init__(self, name: str, zhk_layers: dict, layer_cfg: dict) -> None:
        self.name = name
        self.zhk_layers = zhk_layers
        self.layer_cfg = layer_cfg
        self.preapre_gds()

    def preapre_gds(self) -> None:
        """
        Prepare the GDSII file by creating a library and adding cells with polygons.

        This method initializes a GDSII library and creates a top-level cell. It then iterates
        over the layer configuration and adds polygons to the cell based on the provided layer
        properties. The polygons are created using the points extracted from the geometries.

        Note:
        ----
        The `exclude_from_current` parameter has been deprecated in gdspy and will be removed
        in a future version.
        """
        # The GDSII file is called a library, which contains multiple cells.
        self.gdsii = gdspy.GdsLibrary()
        # Geometry must be placed in cells.
        cell = gdspy.Cell("toplevel", exclude_from_current=True)
        self.gdsii.add(cell)

        for lname, l_property in self.layer_cfg.items():
            polygons = self.zhk_layers[lname]
            for poly in polygons.geoms:
                points = list(poly.exterior.coords)
                gds_poly = gdspy.Polygon(points, **l_property)
                cell.add(gds_poly)

    def save(self):
        """ Save the GDSII file"""
        self.gdsii.write_gds(self.name + '.gds')
        print("Geometries saved successfully.")

    def preview_gds(self):
        """ Preview the GDSII file"""
        warnings.warn("IMPORTANT: preview feature is unstable, kernel might crash in jupyter notebook, use with caution")
        # good only for quick looks
        gdspy.LayoutViewer(self.gdsii)


class Reader_GDS():
    """ helper class to import .gds file into zhk dictionary """

    __slots__ = "filename", "geometries", "gdsii", "cells"

    def __init__(self, filename: str):
        self.filename = filename
        self.geometries = {}
        self.cells = {}
        self.gdsii = gdspy.GdsLibrary(infile=filename)
        self.extract_geometries()

    def extract_geometries(self) -> None:

        cells = {}

        for name, cell in self.gdsii.cells.items():
            layer_names = cell.get_layers()
            print(f"{self.filename} // Layers in cell '{name}': {layer_names}")

            cells[name] = dict.fromkeys(layer_names, MultiPolygon())

            for poly in cell.polygons:
                lname = poly.layers[0]
                cells[name][lname] = union(cells[name][lname], Polygon(poly.polygons[0]))

        self.cells = cells

    def import2zhk(self, cellname: str="toplevel"):
        geoms = self.cells[cellname]
        self.geometries = {"L"+str(k) if isinstance(k, numbers.Number) else k: v for k, v in geoms.items()}

    def plot(self):
        # IMPORTANT
        # unstable, kernel might crash in jupyter notebook, use with caution
        # good only for quick looks
        gdspy.LayoutViewer(self.gdsii)


class Exporter_DXF():
    """ helper class to export zhk dictionary with geometries into .dxf file """

    __slots__ = "name", "zhk_layers", "dxf", "layer_cfg"

    def __init__(self, name: str, zhk_layers: dict, layer_cfg: list) -> None:
        self.name = name
        self.zhk_layers = zhk_layers
        self.layer_cfg = layer_cfg
        self.preapre_dxf()

    def preapre_dxf(self) -> None:
        
        self.dxf = ezdxf.new("R2000")
        msp = self.dxf.modelspace()

        for i, lname in enumerate(self.layer_cfg):
            self.dxf.layers.add(lname, color = i + 1)
            polygons = self.zhk_layers[lname]
            for poly in polygons.geoms:
                points = list(poly.exterior.coords)
                msp.add_lwpolyline(points, dxfattribs={"layer": lname,
                                                       "color": BYLAYER})
        
    def save(self):
        self.dxf.saveas(self.name + ".dxf")
        print("Geometries saved successfully.")


class Reader_DXF():
    """ helper class to import .dxf file into zhk dictionary
    NOTE: currently Arcs are not supported, and points will be ignored
    """

    __slots__ = "filename", "geometries", "dxf"

    def __init__(self, filename: str):
        self.filename = filename
        self.geometries = {}
        self.dxf = ezdxf.readfile(filename)
        self.extract_geometries()

    def extract_geometries(self):
        msp = self.dxf.modelspace()
        self.geometries = msp.groupby(dxfattrib="layer")

        layer_names = self.geometries.keys()
        print(f"{self.filename} // Layers : {layer_names}")

        for k in layer_names:
            self.geometries[k] = self.convert_dxf2shapely(self.geometries[k])
    
    def convert_dxf2shapely(self, dxfentity_list) -> MultiPolygon:
        """ converts dxf entity into shapely MultiPolygon

        Args:
            dxfentity_list (_type_): dxf entity list

        Returns:
            MultiPolygon: converted geometries
        """

        polys = []
        for dxfentity in dxfentity_list:
            coords = []
            for point in dxfentity:
                x, y, _, _, _ = point
                coords.append((x, y))
            if len(coords) > 1:
                # this ignores points
                polys.append(Polygon(coords))

        return MultiPolygon(polys)

    def import2zhk(self):
        return self.geometries


class Reader_Pickle():

    __slots__ = "filename", "geometries"

    def __init__(self, filename: str):
        self.filename = filename
        self.geometries = {}
        self.extract_geometries()

    def extract_geometries(self):
        try:
            with open(self.filename, 'rb') as file:
                self.geometries = pickle.load(file)
        except Exception as e:
            print(f"Error occurred while reading geometries: {e}")


class Exporter_Pickle():

    """ helper class to export zhk dictionary with geometries into .pickle file """

    __slots__ = "name", "zhk_layers"

    def __init__(self, name: str, zhk_layers: dict) -> None:
        self.name = name + '.pickle'
        self.zhk_layers = zhk_layers

    def save(self):
        """ Saves geometry layout of the Entity/Structure in .pickle format

        Args:
            geometries_dict (Entity/Structure): geometry layout
            file_path: name and location of the file
        """

        try:
            with open(self.name, 'wb') as file:
                pickle.dump(self.zhk_layers, file)
            print("Geometries saved successfully.")
        except Exception as e:
            print(f"Error occurred while saving geometries: {e}")
