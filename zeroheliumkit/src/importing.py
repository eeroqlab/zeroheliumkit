import warnings
import numbers
import gdstk
import ezdxf
import pickle
import xml.etree.ElementTree as ET

from ezdxf.colors import BYLAYER
from shapely import Polygon, MultiPolygon, unary_union
from svgpathtools import parse_path, Line, CubicBezier, QuadraticBezier

from .errors import *


def sample_bezier(bezier, num_points=20):
    """
    Sample points along a Bezier curve.
    
    Returns:
        list: A list of (x, y) tuples representing points along the Bezier curve.
    """
    return [(bezier.point(t).real, bezier.point(t).imag) for t in [i / num_points for i in range(num_points + 1)]]


class Exporter_GDS():
    """
    A class for exporting geometries to GDSII format using gdstk.

    Args:
        name (str): The base name of the GDSII file (without extension).
        zhk_layers (dict): A dictionary containing the geometries for each layer.
                           Expected: zhk_layers[lname].geoms yields shapely polygons.
        layer_cfg (dict): A dictionary containing the layer configuration.
                          Expected keys like {"layer": int, "datatype": int}.
    """

    __slots__ = "name", "zhk_layers", "gdsii", "layer_cfg"

    def __init__(self, name: str, zhk_layers: dict, layer_cfg: dict) -> None:
        self.name = name
        self.zhk_layers = zhk_layers
        self.layer_cfg = layer_cfg
        self.preapre_gds()

    def preapre_gds(self) -> None:
        """
        Prepare the GDSII library by creating a top-level cell and adding polygons.

        Notes vs gdspy:
        - gdstk does not have `exclude_from_current`; cells are not automatically "current".
        - gdstk polygons use `layer` and `datatype` (same concepts).
        """
        self.gdsii = gdstk.Library()
        cell = gdstk.Cell("toplevel")
        self.gdsii.add(cell)

        for lname, l_property in self.layer_cfg.items():
            polygons = self.zhk_layers[lname]
            for poly in polygons.geoms:
                points = list(poly.exterior.coords)

                # Optional: shapely exterior repeats the first point at the end.
                # gdstk is fine either way, but removing the duplicate keeps things tidy.
                if len(points) > 1 and points[0] == points[-1]:
                    points = points[:-1]

                gds_poly = gdstk.Polygon(points, **l_property)
                cell.add(gds_poly)

    def save(self):
        """
        Saves the GDSII file.
        """
        self.gdsii.write_gds(self.name + '.gds')
        print("Geometries saved successfully.")


class Reader_GDS():
    """
    Helper class to import .gds file into zhk dictionary using gdstk.

    Args:
        filename (str): Path to the GDSII file.
        cellname (str): Name of the cell to import (default: "toplevel").

    Attributes:
        geometries (dict): Output dict after import2zhk(), mapping "L<layer>" -> MultiPolygon.
        gdsii (gdstk.Library): Loaded gdstk library.
        cells (dict): Dict mapping cellname -> {layer_number -> MultiPolygon}.
    """

    __slots__ = "filename", "geometries", "gdsii", "cells"

    def __init__(self, filename: str, cellname: str = "toplevel"):
        self.filename = filename
        self.geometries = {}
        self.cells = {}
        self.gdsii = gdstk.read_gds(filename)
        self.extract_geometries()
        self.prepare_dict(cellname)

    def extract_geometries(self) -> None:
        cells_out = {}

        # gdstk: library.cells is a list of Cell objects
        for cell in self.gdsii.cells:
            name = cell.name

            # Collect polygons per layer (we'll union at the end per layer)
            by_layer = {}

            # gdstk polygons live in cell.polygons (list of gdstk.Polygon)
            for p in cell.polygons:
                layer = p.layer

                # gdstk.Polygon.points is an Nx2 numpy array-like
                pts = p.points
                # Ensure plain Python list of (x, y)
                points = [(float(x), float(y)) for x, y in pts]

                # Shapely polygon; if degenerate, Polygon(...) may be invalid/empty
                shp = Polygon(points)
                if shp.is_empty:
                    continue

                by_layer.setdefault(layer, []).append(shp)

            layer_numbers = sorted(by_layer.keys())
            print(f"{self.filename} // Layers in cell '{name}': {layer_numbers}")

            # Build MultiPolygon per layer via unary_union
            layer_map = {}
            for layer in layer_numbers:
                merged = unary_union(by_layer[layer]) if by_layer[layer] else MultiPolygon()
                # Ensure MultiPolygon type for consistency
                if merged.geom_type == "Polygon":
                    merged = MultiPolygon([merged])
                layer_map[layer] = merged

            cells_out[name] = layer_map

        self.cells = cells_out

    def prepare_dict(self, cellname: str = "toplevel") -> None:
        geoms = self.cells[cellname]
        self.geometries = {
            ("L" + str(k) if isinstance(k, numbers.Number) else k): v
            for k, v in geoms.items()
        }


class Exporter_DXF():
    """
    Helper class to export zhk dictionary with geometries into .dxf file.

    Args:
        name (str): The name of the DXF file.
        zhk_layers (dict): A dictionary containing the geometries for each layer.
        dxf (ezdxf.DXFDocument): The DXF document object.
        layer_cfg (list): A list containing the layer configuration.
    """

    __slots__ = "name", "zhk_layers", "dxf", "layer_cfg"

    def __init__(self, name: str, zhk_layers: dict, layer_cfg: list) -> None:
        self.name = name
        """The name of the DXF file."""
        self.zhk_layers = zhk_layers
        """A dictionary containing the geometries for each layer."""
        self.layer_cfg = layer_cfg
        """A list containing the layer configuration."""
        self.preapre_dxf()

    def preapre_dxf(self) -> None:
        """
        Prepares the DXF file by creating a new DXF document and adding layers with polygons.
        """
        
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
        """
        Saves the DXF file.
        """
        self.dxf.saveas(self.name + ".dxf")
        print("Geometries saved successfully.")


class Reader_DXF():
    """
    Helper class to import .dxf file into zhk dictionary.

    Args:
        filename (str): The name of the DXF file.
        geometries (dict): A dictionary containing the extracted geometries from the DXF file.
        dxf (ezdxf.DXFDocument): The DXF document object.     

    NOTE: currently Arcs are not supported, and points will be ignored
    """

    __slots__ = "filename", "geometries", "dxf"

    def __init__(self, filename: str):
        self.filename = filename
        self.geometries = {}
        self.dxf = ezdxf.readfile(filename)
        self.extract_geometries()

    def extract_geometries(self):
        """"
        Extract geometries from the DXF file and group them by layer.
        Converts the DXF entities into Shapely MultiPolygon objects.
        """
        msp = self.dxf.modelspace()
        self.geometries = msp.groupby(dxfattrib="layer")

        layer_names = self.geometries.keys()
        print(f"{self.filename} // Layers : {layer_names}")

        for k in layer_names:
            self.geometries[k] = self.convert_dxf2shapely(self.geometries[k])
    
    def convert_dxf2shapely(self, dxfentity_list) -> MultiPolygon:
        """
        Converts dxf entity into a Shapely MultiPolygon.

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
    """
    Helper class to import .pickle file into zhk dictionary.
    
    Args:
        filename (str): The name of the pickle file.
        geometries (dict): A dictionary containing the extracted geometries from the pickle file.
    """

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
    """
    Helper class to export zhk dictionary with geometries into .pickle file.
    
    Args:
        name (str): The name of the pickle file.
        zhk_layers (dict): A dictionary containing the geometries for each layer.
    """

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


class Reader_SVG():
    """
    A class to read and convert SVG files into Shapely polygons.
    Contributor: https://github.com/yneter

    Args:
        svg_file (str): The path to the SVG file to be read.
        geometries (dict): A dictionary containing the extracted geometries from the SVG file.
    """

    __slots__ = "svg_file", "geometries"

    def __init__(self, svg_file: str, bezier_samples: int=20):
        self.svg_file = svg_file
        self.geometries = {"L1": self.svg_to_shapely_polygons(bezier_samples)}

    def svg_to_shapely_polygons(self, bezier_samples=20):
        """
        Extract multiple polygons from an SVG file.
        
        Args:
            bezier_samples (int): Number of points to sample along Bezier curves. Default is 20.
        
        Returns:
            MultiPolygon: A Shapely MultiPolygon object containing all extracted polygons.
        """
        tree = ET.parse(self.svg_file)
        root = tree.getroot()

        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        polygons = []

        # Find all <path> elements and extract points from them
        for path_element in root.findall('.//svg:path', namespace):
            path_data = path_element.attrib['d']
            points = self.extract_points_from_path(path_data, bezier_samples)
            polygons.append(Polygon(points))

        # Find all <polygon> elements
        for polygon_element in root.findall('.//svg:polygon', namespace):
            points_data = polygon_element.attrib['points']
            points = [tuple(map(float, point.split(','))) for point in points_data.split()]
            polygons.append(Polygon(points))

        # Find all <polyline> elements (similar to <polygon>, but not closed)
        for polyline_element in root.findall('.//svg:polyline', namespace):
            points_data = polyline_element.attrib['points']
            points = [tuple(map(float, point.split(','))) for point in points_data.split()]
            if points[0] != points[-1]:
                points.append(points[0])  # Close the polyline to make a polygon
            polygons.append(Polygon(points))

        return MultiPolygon(polygons)


    def extract_points_from_path(self, path_data, bezier_samples=20):
        """
        Extracts points from a single path (handling lines and Bezier curves).
        
        Args:
            path_data (str): The SVG path data string.
            bezier_samples (int): Number of points to sample along Bezier curves. Default is 20.
            
        Returns:
            list: A list of (x, y) tuples representing points along the path.
        """
        svg_path = parse_path(path_data)
        points = []

        for segment in svg_path:
            if isinstance(segment, Line):
                # For lines, just add the start point
                points.append((segment.start.real, segment.start.imag))
            elif isinstance(segment, (CubicBezier, QuadraticBezier)):
                # For Bezier curves, sample points along the curve
                bezier_points = sample_bezier(segment, num_points=bezier_samples)
                points.extend(bezier_points)
            else:
                raise NotImplementedError(f"Segment type {type(segment)} is not handled")

        # Ensure the path is closed by adding the first point to the end if necessary
        if points[0] != points[-1]:
            points.append(points[0])

        return points
