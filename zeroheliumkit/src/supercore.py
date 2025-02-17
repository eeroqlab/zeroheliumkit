"""
This file contains the implementation of a SuperStructure class, which is a subclass of the Structure class.
The SuperStructure class provides additional methods for routing and adding structures along a skeleton line.
The ContinuousLineBuilder class is also implemented in this file, which is used to build a structure by defining a continuous line.
"""

import numpy as np

from dataclasses import dataclass
from shapely import (line_locate_point, line_interpolate_point, intersection_all,
                     distance, difference, intersection, unary_union)
from shapely import LineString, Polygon

from .anchors import Anchor, MultiAnchor, Skeletone
from .core import Structure, Entity
from .geometries import ArcLine
from .utils import (fmodnew, flatten_lines, create_list_geoms,
                    round_polygon, buffer_line_with_variable_width)
from .functions import get_normals_along_line
from .routing import create_route
from .errors import WrongSizeError


def get_endpoint_indices(x: tuple=(1,0)) -> tuple:
    if x == (1, 0):
        return (None, -1)
    elif x == (0, 1):
        return (1, None)
    else:
        raise ValueError("Provide a valid tuple of (1, 0) or (0, 1).")


@dataclass(slots=True)
class RoutingConfig:
    """
    Represents the configuration for routing.
    Attr:
    -----
        radius (float, optional): The radius of the routing. Defaults to 50.
        num_segments (int, optional): The number of segments in the routing. Defaults to 13.
    """
    radius: float = 50
    num_segments: int = 13


@dataclass(slots=True)
class ObjsAlongConfig:
    """
    Configuration class for objects evenly placed along a line.
    Attr:
    -----
        structure (object): The structure to be added along the skeleton line.
        spacing (float, optional): The spacing between the added objects. Defaults to 100.
        endpoints (tuple or bool, optional): The endpoints of the skeleton line where the objects should be added. 
            If a tuple is provided, it should contain the indices of the desired endpoints. 
            If True, the objects will be added along the entire skeleton line.
            Defaults to True.
        additional_rotation (float, optional): Additional rotation to be applied to the added objects. Defaults to 0.
    """
    structure: Structure
    spacing: float = 100
    endpoints: tuple | bool = True
    additional_rotation: float = 0


class SuperStructure(Structure):
    """A class representing a superstructure.

    This class inherits from the Structure class and 
    provides additional methods for routing and adding structures along the skeleton line.

    Args:
        route_config (dict): Configuration for routing.

    Attributes:
        _route_config (dict): Configuration for routing.

    """

    def __init__(self, route_config: dict):
        self._route_config = route_config
        super().__init__()

    def route(self,
              anchors: tuple,
              layers: dict=None,
              airbridge: Entity | Structure=None,
              extra_rotation: float=0,
              print_status: bool=False,
              rm_anchor: bool | tuple | str=False,
              rm_route: bool=False,
              cap_style: str="flat",
              **kwargs) -> None:
        """ Routes between anchors.
            This method routes between two anchors based on the provided parameters.

        Args:
        ----
        anchors (tuple): Two anchors between which a route is constructed.
        layers (dict): Layer information.
        airbridge (Entity | Structure): Airbridge structure.
        extra_rotation (float): Extra rotation angle.
        print_status (bool): Flag indicating if the status should be printed.
        rm_anchor (bool or str, optional): If True, removes the anchor points after appending. 
                                           If a string is provided, removes the specified anchor point. 
                                           Defaults to False.
        rm_route (bool, optional): If True, removes the route line after appending route polygons. Defaults to False.
        """

        self.route_with_intersection(anchors, layers, airbridge, extra_rotation, print_status, rm_route, cap_style, **kwargs)

        # remove or not to remove anchor
        if rm_anchor==True:
            self.remove_anchor(anchors)
        elif isinstance(rm_anchor, (str, tuple)):
            self.remove_anchor(rm_anchor)


    def add_along_skeletone(self,
                            bound_anchors: tuple,
                            structure: Structure | Entity,
                            locs: list=None,
                            num: int=1,
                            endpoints: bool|tuple=False,
                            normalized: bool=False,
                            additional_rotation: float | int=0,
                            line_idx: int=None) -> None:
        """ Add structures along the skeleton line.

        Args:
        ----
        bound_anchors (tuple): Skeleton region contained between two anchors.
        structure (Structure | Entity): Structure to be added.
        locs (list, optional): List of locations along the skeleton line where the structures will be added. 
            If not provided, the structures will be evenly distributed between the two anchor points.
        num (int, optional): Number of structures to be added. Ignored if locs is provided.
        endpoints (bool|tuple, optional): Whether to include the endpoints of the skeleton line as locations for adding structures. 
            tuple = (1,0): includes start point and exclides end point. Tuple = (0,1) is vice versa.
            Default is False.
        normalized (bool, optional): Whether the locations are normalized along the skeleton line. Default is False.

        Raises:
        ------
        WrongSizeError: If the number of bound_anchors is not equal to 2.

        Example:
        -------
            >>> # Add 5 structures evenly distributed along the skeleton line between anchor1 and anchor2
            >>> add_along_skeleton((anchor1, anchor2), structure, num=5)
            >>> # Add 3 structures at specific locations along the skeleton line between anchor1 and anchor2
            >>> add_along_skeleton((anchor1, anchor2), structure, locs=[0.2, 0.5, 0.8])
        """

        if len(bound_anchors) != 2:
            raise WrongSizeError(f"Provide 2 anchors! Instead {len(bound_anchors)} is given.")

        p1 = self.get_anchor(bound_anchors[0]).point
        p2 = self.get_anchor(bound_anchors[1]).point

        if line_idx:
            line = self.skeletone.lines.geoms[line_idx]
        else:
            line = self.skeletone.lines

        start_point = line_locate_point(line, p1, normalized=True)
        end_point = line_locate_point(line, p2, normalized=True)
        
        extra_rotation = 0
        if locs is None:
            if isinstance(endpoints, tuple):
                i1, i2 = get_endpoint_indices(endpoints)
                locs = np.linspace(start_point, end_point, num=num+1, endpoint=True)[i1:i2]
            elif endpoints == True:
                locs = np.linspace(start_point, end_point, num=num, endpoint=True)
            else:
                locs = np.linspace(start_point, end_point, num=num+2, endpoint=True)[1:-1]
            normalized = True
            extra_rotation = 90

        pts = line_interpolate_point(line, locs, normalized=normalized).tolist()
        normal_angles = get_normals_along_line(line, locs) + extra_rotation   # figure out why extra_rotation is added

        for point, angle in zip(pts, normal_angles):
            s = structure.copy()
            s.rotate(angle + additional_rotation)
            s.moveby(xy=(point.x, point.y))
            self.append(s)

    
    def route_with_intersection(self,
                                anchors: tuple,
                                layers: dict,
                                airbridge: Entity | Structure=None,
                                extra_rotation: float=0,
                                print_status: bool=False,
                                remove_line: bool=False,
                                cap_style: str="flat",
                                **kwargs) -> None:
        """ Creates a route connecting specified anchors and
            adding airbridge when a crossing with the skeleton is expected.

        Args:
        ----
        anchors (tuple): The anchors to route between. Provide labels.
        layers (dict): The layer width information.
        airbridge (Entity | Structure): The airbridge structure.
            Should contain 'in' and 'out' anchors. Defaults to None
        extra_rotation (float, optional): Additional rotation angle for the airbridge. Defaults to 0.
        print_status (bool, optional): Whether to print the status of the route creation. Defaults to False.
        remove_line (bool, optional): Whether to remove created route line. Defaults to False.

        Examples:
        --------
            >>> ss = SuperStructure(route_config={...})
            >>> ss.route_with_intersection(anchors=('A', 'B'),
            >>>                                layers={'layer1': 0.1}, airbridge=airbridge_structure)
        """
        #start and end points
        p_start = self.get_anchor(anchors[0]).point
        p_end = self.get_anchor(anchors[-1]).point

        if airbridge:
            ab_labels = ['in', 'out']
            if ((ab_labels[0] not in airbridge.anchors.labels) or
                (ab_labels[1] not in airbridge.anchors.labels)):
                raise TypeError("airbridge anchors could be only 'in' and 'out'")
        
        # getting route line along the anchor points
        route_line = LineString()
        for labels in zip(anchors, anchors[1:]):
            line = create_route(a1=self.get_anchor(labels[0]),
                                a2=self.get_anchor(labels[1]),
                                radius=self._route_config.get("radius"),
                                num_segments=self._route_config.get("num_segments"),
                                print_status=print_status,
                                **kwargs)
            route_line = flatten_lines(route_line, line)

        # get all intersection points with route line and skeletone
        intersections = intersection_all([route_line, self.skeletone.lines])

        # get valid intesections
        if not intersections.is_empty:
            # create a list of points
            list_of_intersection_points = create_list_geoms(intersections)
            # getting airbridge locations (i.e. removing start and end points)
            ab_locs = [p for p in list_of_intersection_points if p not in [p_start, p_end]]

        ###################################
        #### buffering along the route ####
        ###################################

        if intersections.is_empty or ab_locs==[]:
            # in case of no intersections or no ab_locs make a simple route structure
            self.bufferize_routing_line(route_line, layers, keep_line=False, cap_style=cap_style)
            # remove or not to remove route line
            if not remove_line:
                self.add_line(route_line, chaining=False)

        else:
            # getting list of distances of airbridge locations from starting point
            list_distances = np.asarray(list(map(distance, ab_locs, [p_start]*len(ab_locs))))
            sorted_distance_indicies = np.argsort(list_distances)

            ab_locs_on_skeletone = line_locate_point(self.skeletone.lines, ab_locs, normalized=True)
            ab_angles = get_normals_along_line(self.skeletone.lines, ab_locs_on_skeletone)

            # create route anchor list with temporary anchor list, which will be deleted in the end
            route_anchors = [anchors[0]]
            temporary_anchors = []

            # adding airbridges to superstructure
            for i, idx in enumerate(sorted_distance_indicies):
                ab_coords = (ab_locs[idx].x, ab_locs[idx].y)
                ab_angle = fmodnew(ab_angles[idx] + 90 + extra_rotation)

                ab = airbridge.copy()
                ab.rotate(angle=ab_angle)
                ab.moveby(xy=ab_coords)

                # correcting the orientation of the airbridge if 'in' and 'out' are swapped
                distance2in  = distance(ab.get_anchor("in").point,  self.get_anchor(route_anchors[-1]).point)
                distance2out = distance(ab.get_anchor("out").point, self.get_anchor(route_anchors[-1]).point)
                if distance2out < distance2in:
                    ab.rotate(180, origin=ab_coords)

                for label in ab_labels:
                    temporary_name = str(i) + label
                    ab.modify_anchor(label=label,
                                     new_name=temporary_name)
                    route_anchors.append(temporary_name)
                    temporary_anchors.append(temporary_name)
                self.append(ab)
            route_anchors.append(anchors[1])

            # adding all routes between airbridge anchors
            for labels in zip(route_anchors[::2], route_anchors[1::2]):
                route_line = create_route(a1=self.get_anchor(labels[0]),
                                          a2=self.get_anchor(labels[1]),
                                          radius=self._route_config.get("radius"),
                                          num_segments=self._route_config.get("num_segments"),
                                          print_status=print_status,
                                          **kwargs)
                self.bufferize_routing_line(route_line, layers, keep_line=False)
                # remove or not to remove route line
                if not remove_line:
                    self.add_line(route_line, chaining=False)

            # remove anchors
            self.remove_anchor(temporary_anchors)


    def bufferize_routing_line(self,
                               line: LineString,
                               layers: float | int | list | dict,
                               keep_line: bool=True,
                               cap_style: str="flat") -> None:
        """ Append route to skeleton and create polygons by buffering.

        Args:
        ----
        line (LineString): The route line.
        layers (Union[float, int, list, dict]): The layer information.
            It can be a single value, a list of values, or a dictionary with distances and widths.

        Examples:
        --------
            >>> line = LineString([(0, 0), (1, 1), (2, 2)])
            >>> layers = {'layer1': 0.1, 'layer2': [0.2, 0.3, 0.4]}
            >>> bufferize_routing_line(line, layers)

            >>> line = LineString([(0, 0), (1, 1), (2, 2)])
            >>> layers = {'layer1': {'d': [0, 0.5, 1], 'w': [0.1, 0.2, 0.1], 'normalized': True}}
            >>> bufferize_routing_line(line, layers)

            >>> line = LineString([(0, 0), (1, 1), (2, 2)])
            >>> layers = {'layer1': [0.1, 0.2, 0.3], 'layer2': {'d': [0, 0.5, 1], 'w': [0.2, 0.3, 0.2], 'normalized': False}}
            >>> bufferize_routing_line(line, layers)
        """
        s = Structure()
        if keep_line:
            s.add_line(line)

        if layers:
            for k, width in layers.items():
                if isinstance(width, (int, float)):
                    poly = line.buffer(distance=width/2, cap_style=cap_style)
                elif isinstance(width, (list, np.ndarray)):
                    distances = np.linspace(0, 1, len(width), endpoint=True)
                    poly = buffer_line_with_variable_width(line, distances, width, normalized=True, join_style='flat')
                elif isinstance(width, dict):
                    distances = np.asarray(width.get("d"))
                    widths = np.asarray(width.get("w"))
                    norm = width.get("normalized")
                    poly = buffer_line_with_variable_width(line, distances, widths, normalized=norm, join_style='flat')
                else:
                    raise TypeError("Provide a valid 'layers' dictionary.")
                s.add_layer(k, poly)
            self.append(s)


    def round_sharp_corners(self, area: Polygon, layer: str | list[str], radius: float | int, **kwargs) -> None:
        """ Rounds the sharp corners within the specified area for the given layer(s) by applying a radius.

        Args:
        ----
        area (Polygon): The area within which the corners should be rounded.
        layer (str | list[str]): The layer(s) on which the operation should be performed.
            If a single layer is provided as a string, the operation will be applied to that layer only.
            If multiple layers are provided as a list of strings, the operation will be applied to each layer individually.
        radius (float | int): The radius to be applied for rounding the corners.

        Example:
        -------
            >>> s = ...  # your SuperStructure(route_config={...})
            >>> area = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])
            >>> s.round_sharp_corners(area, "layer1", 2.5)
            >>> # Round the sharp corners for multiple layers
            >>> s.round_sharp_corners(area, ["layer2", "layer3"], 3)
        """
        if isinstance(layer, str):
            layer = [layer]
        for l in layer:
            original = getattr(self, l)
            base = difference(original, area)
            rounded = intersection(original, area.buffer(2*radius, join_style="mitre"))
            rounded = round_polygon(rounded, radius, **kwargs)
            rounded = intersection(rounded, area)
            setattr(self, l, unary_union([base, rounded]))



class ContinuousLineBuilder():
    """ Class for building continuous lines and structures.

    Attr:
    -----
        routing (RoutingConfig): The routing configuration.
        layers (dict): A dictionary containing layer names as keys and buffer widths as values.
        objs_along (ObjsAlongConfig): The configuration for adding objects along the skeleton line.
    
    Example:
    -------
        >>> s = Structure()
        >>> s.add_layer("l1", Square(10))
        >>> clb = ContinuousLineBuilder(routing=RoutingConfig(radius=10, num_segments=10),
        >>>                             layers={"l1": 2, "l2": 4},
        >>>                             objs_along=ObjsAlongConfig(structure=s, spacing=40))
        >>> a = Anchor((100,0), 0, "b")
        >>> clb.start(Anchor((1,2), 30, "a")).turn(-60,14).go(23,65).routeto(a).go(30,110).forward(36).build_all()
    """

    __slots__ = ("routing", "layers", "objs_along",
                 "absolute_angle", "skeletone", "anchors",
                 "starting_coords", "structure")

    def __init__(self,
                 routing: RoutingConfig=None,
                 layers: dict=None,
                 objs_along: ObjsAlongConfig=None):
        self.routing = routing
        self.layers = layers
        self.objs_along = objs_along
        self.absolute_angle = 0


    def __repr__(self):
        name = f"<CONTINUOUSLINEBUILDER {self.layers}>"
        max_length = 75
        if len(name) > max_length:
            return f"{name[: max_length - 3]}...>"

        return name


    def start(self, anchor: Anchor):
        """ Adds the given anchor to the set of anchors and initializes the starting coordinates and absolute angle.

        Args:
        -----
            anchor (Anchor): The anchor object to be added. It should have 'direction' and 'coords' attributes.
            
        Example:
        -----
        >>> anchor = Anchor(coords=(10,20), direction=90, "a")
        >>> instance = ContinuousLineBuilder()
        >>> instance.start(anchor)
        <MyClass object at 0x...>
        """
        self.skeletone = Skeletone()
        self.anchors = MultiAnchor()
        self.structure = Structure()

        self.anchors.add(anchor)
        self.absolute_angle = anchor.direction
        self.starting_coords = anchor.coords
        return self


    def forward(self, length: float=1):
        """ Creates a line and adds to a skeletone.
        
        Args:
        -----
            length (float): The length of the line. Default is 1.
        """

        new_line = LineString([(0,0), (length,0)])
        self.skeletone.add_line(line = new_line,
                                direction = self.absolute_angle,
                                ignore_crossing = True,
                                chaining = True)
        if self.starting_coords:
            self.skeletone.move(*self.starting_coords)
            self.starting_coords = None
        return self


    def turn(self, angle: float=90, radius: float=1, num_segments: int=13):
        """ Creates an arcline and appends to a skeletone.
        
        Args:
        -----
            angle (float): turn angle of the arcline. Default is 90.
            radius (float): radius of the arcline. Default is 1.
            num_segments (int): number of segments of the arcline. Default is 13.
        """
        new_line = ArcLine(centerx = 0,
                           centery = np.sign(angle) * radius,
                           radius = radius,
                           start_angle = -np.sign(angle) * 90,
                           end_angle = -np.sign(angle) * 90 + angle,
                           numsegments = num_segments)
        self.skeletone.add_line(line = new_line,
                                direction = self.absolute_angle,
                                ignore_crossing = True,
                                chaining = True)
        self.absolute_angle = fmodnew(self.absolute_angle + angle)
        return self


    def add_anchor(self, name: str, angle: float=None, type: str=None):
        """ Adds an Anchor at the location of the skeletone end point with given angle and type.
        
        Args:
        -----
            name (str): The name of the anchor.
            angle (float, optional): The angle of the anchor. Defaults to None.
            type (str, optional): The type of angle calculation. Can be "absolute", "relative", or None. Defaults to None.
        """
        _, end_p = self.skeletone.boundary
        if type == "absolute":
            angle = angle
        elif type == "relative":
            angle = fmodnew(self.absolute_angle + angle)
        else:
            angle = self.absolute_angle 
        self.anchors.add(Anchor(end_p, angle, name))
        return self


    def end(self, name: str):
        """ Adds an anchor at the end of the skeletone"""
        self.add_anchor(name)


    def go(self, length: float=1, angle: float=90):
        """ Combines the forward and turn methods in one action.
            routing should be set before using this method.
        
        Args:
        -----
            length (float): The length of the straight section. Default is 1.
            angle (float): The angle of the turn. Default is 90.

        Error:
        -----
            Raises ValueError if routing config is not set.
        """
        if not self.routing:
            raise ValueError("Routing config is not set. Provide dict with keys 'radius' and 'num_segments'")
        self.forward(length)
        self.turn(angle, self.routing.radius, self.routing.num_segments)
        return self


    def routeto(self, anchor: Anchor, **kwargs):
        """ Routes the last point in the skeletone with given anchor.
            routing should be set before using this method.
        
        Args:
        -----
            anchor (Anchor): the anchor point to which the route will be created.
            **kwargs: additional parameters to be passed to create_route method.

        Error:
        -----
            Raises ValueError if routing config is not set.
        """
        if not self.routing:
            raise ValueError("Routing config is not set. Provide dict with keys 'radius' and 'num_segments'")
        self.add_anchor("temp")
        new_line = create_route(self.anchors.point("temp"),
                                anchor,
                                self.routing.radius,
                                self.routing.num_segments,
                                **kwargs)
        self.skeletone.add_line(line = new_line,
                                direction = None,
                                ignore_crossing = True,
                                chaining = False)
        self.skeletone.fix()
        self.absolute_angle = anchor.direction
        return self


    def build_layers(self, layers: dict=None, **kwargs):
        """ Build layers for the structure.

        Args:
        -----
            layers (dict, optional): A dictionary containing layer names as keys and buffer widths as values.
                If not provided, the layers from the object's attribute `layers` will be used.
            **kwargs: Additional keyword arguments to be passed to the `buffer` method.
        """

        if not layers:
            layers = self.layers

        for lname, buffer_width in layers.items():
            poly = self.skeletone.buffer(buffer_width/2, join_style="mitre", **kwargs)
            if self.structure.has_layer(lname):
                self.structure.add_polygon(lname, poly)
            else:
                self.structure.add_layer(lname, poly)

        return self


    def add_along_skeletone(self, **kwargs):
        """ Add objects along the skeleton line.

        Args:
        -----
            structure (object): The structure to be added along the skeleton line.
            spacing (float): The spacing between the added objects.
            endpoints (tuple or bool): The endpoints of the skeleton line where the objects should be added. 
                If a tuple is provided, it should contain the indices of the desired endpoints. 
                If True, the objects will be added along the entire skeleton line.
            additional_rotation (float): Additional rotation to be applied to the added objects.
        """

        for k,v in kwargs.items():
            setattr(self.objs_along, k, v)

        num = int(self.skeletone.length / self.objs_along.spacing)

        p1, p2 = self.skeletone.boundary
        start_point = line_locate_point(self.skeletone.lines, p1, normalized=True)
        end_point = line_locate_point(self.skeletone.lines, p2, normalized=True)

        if isinstance(self.objs_along.endpoints, tuple):
            i1, i2 = get_endpoint_indices(self.objs_along.endpoints)
            locs = np.linspace(start_point, end_point, num=num, endpoint=True)[i1:i2]
        elif self.objs_along.endpoints == True:
            locs = np.linspace(start_point, end_point, num=num, endpoint=True)
        else:
            locs = np.linspace(start_point, end_point, num=num, endpoint=True)[1:-1]


        pts = line_interpolate_point(self.skeletone.lines, locs, normalized=True).tolist()
        normal_angles = get_normals_along_line(self.skeletone.lines, locs)   # figure out why extra_rotation is added

        for point, angle in zip(pts, normal_angles):
            s = self.objs_along.structure.copy()
            s.rotate(angle + self.objs_along.additional_rotation + 90)
            s.moveby(xy=(point.x, point.y))
            self.structure.append(s)

        return self


    def build_all(self):
        """ Build all the layers and add objects along the skeleton line.
        """
        self.build_layers()
        if self.objs_along:
            self.add_along_skeletone()
        return self
