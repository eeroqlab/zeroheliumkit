import numpy as np

import copy

import shapely
from shapely import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely import (affinity, unary_union, ops, 
                     difference, set_precision, 
                     is_empty, crosses, 
                     intersection, set_coordinates)

from ..helpers.plotting import *
from ..importing import reader_dxf
from ..errors import *
from ..settings import *

from ..dev.geometries import StraightLine, Route, RouteTwoElbows
from ..dev.functions import *
from ..dev.core import *


route_types = {"line": lambda **kwargs: StraightLine(kwargs),
               "elbow": lambda **kwargs: Route(kwargs),
               "zigzag": lambda **kwargs: RouteTwoElbows(kwargs)}


class Line(StraightLine):
    def __init__(self,
                 anchor1: Anchor,
                 anchor2: Anchor,
                 anchor_mid: Anchor=Anchor(),
                 radius: float=10,
                 num_segments: int=10,
                 layers: dict=None):
        length = length_between_points(anchor1.point, anchor2.point)
        super().__init__(length = length,
                         layers = layers,
                         alabel = ("xyz1","xyz2")
                         )
        self.rotate(anchor1.direction)
        self.moveby(xy=anchor1.coords)

class Elbow(Route):
    def __init__(self,
                 anchor1: Anchor,
                 anchor2: Anchor,
                 anchor_mid: Anchor=Anchor(),
                 radius: float=10,
                 num_segments: int=10,
                 layers: dict=None):
        super().__init__(point1 = anchor1.point, 
                         direction1 = anchor1.direction, 
                         point2 = anchor2.point,
                         direction2 = anchor2.direction, 
                         radius = radius, 
                         num_segments = num_segments, 
                         layers = layers,
                         alabel = ("xyz1","xyz2")
                         )

def construct_kwargs(type):
    pass


class SuperStructure(Structure):
    def __init__(self):
        super().__init__()
    
    def route(self, anchors: tuple, type: str, layers: dict):
        
        if type not in route_types.keys():
            raise TypeError(f"'{type}' is not supported. choose from {route_types.keys()}")
        
        if len(anchors) != 2:
            raise TypeError("connection can be made only between two points. Provide only two point labels in 'anchors'")
        
        point1 = self.get_anchor(anchors[0])
        point2 = self.get_anchor(anchors[1])

        route = route_types.get(type)

        connecting_structure = route