import geopandas as gpd
import matplotlib.pyplot as plt

from tabulate import tabulate
from shapely import LineString, Polygon, MultiPolygon
from .errors import *

class reader_dxf():
    def __init__(self, filename: str, ignore_nonclosed: bool=False):
        self.raw_geometries = gpd.read_file(filename)
        self.extract_dxf_layers()
        self.geometries = self.get_multipolygon_dict(ignore_nonclosed)

    def extract_dxf_layers(self) -> None:
        layer_info = self.raw_geometries[['Layer']]
        layers = list(layer_info.iloc[:,0])
        self.unique_layers = list(set(layers))

        num_polygon_in_layer = []
        for name in self.unique_layers:
            num_polygon_in_layer.append(layers.count(name))

        print(tabulate(zip(self.unique_layers, num_polygon_in_layer), headers=['Layer', 'Num Obj']))

    def get_multipolygon_dict(self, ignore_nonclosed: bool=False):

        # create a dictionary with layers as KEY and list of polygons as VALUE

        LayersGeoms = self.raw_geometries[['Layer', 'geometry']]
        value = lambda layer_name: list(LayersGeoms[LayersGeoms['Layer']==layer_name].iloc[:,1])
        return {key: self.create_multipolygon(value(key), ignore_nonclosed) for key in self.unique_layers}
    
    def create_multipolygon(self, list_of_lines: list, ignore_nonclosed: bool=False) -> list:
        all_polygons =[]
        for item in list_of_lines:
            p = self.convert_to_polygon(item, ignore_nonclosed)
            if p!=None:
                all_polygons.append(p)
        return MultiPolygon(all_polygons)
    
    def convert_to_polygon(self, line: LineString, ignore_nonclosed: bool=False) -> Polygon:
        coords = list(line.coords)
        if coords[0] == coords[-1]:
            return Polygon(coords)
        elif not ignore_nonclosed:
            errorMessage = 'LineString is not closed! check your dxf file and close objects'
            plt.plot(*line.xy)
            plt.title(errorMessage)
            plt.show()
            raise PolygonConverterError(errorMessage)
        else:
            return None
    
    def plot_dxf(self, **kwargs):
        self.raw_geometries.plot(**kwargs)
        plt.show()


if __name__=="__main__":
    chip = reader_dxf('test.dxf')
    chip.plot_dxf()