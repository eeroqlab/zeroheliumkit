import numpy as np

from ..helpers.plotting import *
from ..errors import *
from ..settings import *

from ..dev.geometries import StraightLine, ElbowLine, SigmoidLine
from ..dev.functions import *
from ..dev.core import *


route_types = {"line": lambda **kwargs: StraightLine(kwargs),
               "elbow": lambda **kwargs: ElbowLine(kwargs),
               "sigmoid": lambda **kwargs: SigmoidLine(kwargs)}


class SuperStructure(Structure):
    def __init__(self, route_config: dict):
        self._route_config = route_config
        super().__init__()
    
    def route_between_two_pts(self, anchors: tuple, layers: dict):
        
        if len(anchors) != 2:
            raise TypeError("connection can be made only between two points. Provide only two point labels in 'anchors'")
        
        point1 = self.get_anchor(anchors[0])
        point2 = self.get_anchor(anchors[1])
        radius = self._route_config.get("radius")
        num_segments = self._route_config.get("num_segments")

        # calculating check parameters
        a, b, c = get_abc_line(point1.point, point2.point)
        angle = np.arctan(-a/b) * 180/np.pi
        if angle > point1.direction:
            mid_dir = point1.direction + 45
        else:
            mid_dir = point1.direction - 45
        # next

        if (point1.direction == point2.direction) and np.abs(angle - point1.direction) < 1e-4:
            connecting_structure = StraightLine(anchors=(point1,point2),
                                                layers=layers)
        elif point1.direction == point2.direction:
            connecting_structure = SigmoidLine(anchor1=point1, 
                                                anchor2=point2, 
                                                mid_direction=mid_dir, 
                                                radius=radius, 
                                                num_segments=num_segments, 
                                                layers=layers)
        else:   
            try:
                connecting_structure = ElbowLine(anchor1=point1, 
                                                 anchor2=point2, 
                                                 radius=radius, 
                                                 num_segments=num_segments, 
                                                 layers=layers)
            except:
                connecting_structure = SigmoidLine(anchor1=point1, 
                                                   anchor2=point2, 
                                                   mid_direction=mid_dir, 
                                                   radius=radius, 
                                                   num_segments=num_segments, 
                                                   layers=layers)

        self.append(connecting_structure)

    def route(self, anchors: tuple, layers: dict):
        for labels in zip(anchors, anchors[1:]):
            self.route_between_two_pts(anchors=labels, layers=layers)
