import numpy as np

from shapely import line_locate_point, line_interpolate_point, intersection_all, distance
from shapely import LineString, MultiLineString, Point

from ..dev.geometries import StraightLine, ElbowLine, SigmoidLine
from ..dev.functions import get_abc_line, create_list_geoms, get_normals_along_line
from ..dev.core import Structure, Entity
from ..errors import RouteError, GeometryError


class SuperStructure(Structure):
    """ this class provides more advanced routing options.
        Based on Structure class.
    """
    def __init__(self, route_config: dict):
        self._route_config = route_config
        super().__init__()

    def route(self, anchors: tuple, layers: dict):
        for labels in zip(anchors, anchors[1:]):
            self.route_between_two_pts(anchors=labels, layers=layers)

    def route_between_two_pts(self, anchors: tuple, layers: dict):
        """ makes a route between two anchors.
            specify route config in SuperStructure init stage.

        Args:
            anchors (tuple): two anchors between which a route is constructed
            layers (dict): layer info

        Raises:
            TypeError: if more that two anchors are provided.
        """

        if len(anchors) != 2:
            raise TypeError("Provide only two point labels in 'anchors'")

        point1 = self.get_anchor(anchors[0])
        point2 = self.get_anchor(anchors[1])
        radius = self._route_config.get("radius")
        num_segments = self._route_config.get("num_segments")

        # calculating check parameters
        a, b, _ = get_abc_line(point1.point, point2.point)
        angle = np.arctan(-a/b) * 180/np.pi
        if angle > point1.direction:
            mid_dir = point1.direction + 45
        else:
            mid_dir = point1.direction - 45

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
            except RouteError:
                connecting_structure = SigmoidLine(anchor1=point1,
                                                   anchor2=point2,
                                                   mid_direction=mid_dir,
                                                   radius=radius,
                                                   num_segments=num_segments,
                                                   layers=layers)

        self.append(connecting_structure)


    def add_along_skeletone(self,
                            bound_anchors: tuple,
                            num: int,
                            structure: Structure | Entity):

        if len(bound_anchors) != 2:
            raise ValueError(f"Provide 2 anchors! Instead {len(bound_anchors)} is given.")
        p1 = self.get_anchor(bound_anchors[0]).point
        p2 = self.get_anchor(bound_anchors[1]).point

        #self.fix_line()
        start_point = line_locate_point(self.skeletone, p1, normalized=True)
        end_point = line_locate_point(self.skeletone, p2, normalized=True)

        locs = np.linspace(start_point, end_point, num=num+2, endpoint=True)
        pts = line_interpolate_point(self.skeletone, locs[1:-1], normalized=True).tolist()
        normal_angles = get_normals_along_line(self.skeletone, locs[1:-1]) - 90

        for point, angle in zip(pts, normal_angles):
            s = structure.copy()
            s.rotate(angle)
            s.moveby(xy=(point.x, point.y))
            self.append(s)


    def route_with_intersection(self, anchors: tuple, layers: dict, airbridge: Entity | Structure) -> None:

        if len(anchors) != 2:
            raise TypeError("Provide only two point labels in 'anchors'.")
        
        anchor_labels_airbridge = airbridge.anchorsmod.labels
        if len(anchor_labels_airbridge) != 2:
            raise GeometryError("Airbridge can contain only two in/out anchors.")

        point_start = self.get_anchor(anchors[0]).point
        point_end = self.get_anchor(anchors[1]).point

        route_line = LineString([point_start, point_end])

        # get all intersection points with route line and skeletone     
        intersections = intersection_all([route_line, self.skeletone])
        
        if not intersections.is_empty:
            # create a list of points
            list_intersection_points = create_list_geoms(intersections)
            
            # removing start and end points
            valid_intersections = [p for p in list_intersection_points if p not in [point_start, point_end]]
        
        #######################
        ### creating route ####
        #######################

        if intersections.is_empty or valid_intersections==[]:
            # in case of no intersections make a simple route or no valid_intersections 
            self.route_between_two_pts(anchors, layers)

        else:
            list_distances = np.asarray(list(map(distance, 
                                                 valid_intersections, 
                                                 [point_start]*len(valid_intersections)
                                                 )
                                            )
                                        )
            sorted_distances_indicies = np.argsort(list_distances)

            intersect_locs_on_skeletone = line_locate_point(self.skeletone, 
                                                            valid_intersections,
                                                            normalized=True)
            intersect_normals = get_normals_along_line(self.skeletone, intersect_locs_on_skeletone)
            print(intersect_normals)
            
            route_anchors = [anchors[0]]
            temporary_anchors = []

            for i, idx in enumerate(sorted_distances_indicies):
                airBRDG = airbridge.copy()
                airBRDG.rotate(angle=intersect_normals[idx] + 90)
                airBRDG.moveby(xy=(valid_intersections[idx].x, 
                                     valid_intersections[idx].y))
                for ab_anchor in anchor_labels_airbridge:
                    anchor_temp_name = str(i) + ab_anchor
                    airBRDG.modify_anchor(label=ab_anchor,
                                          new_name=anchor_temp_name)
                    route_anchors.append(anchor_temp_name)
                    temporary_anchors.append(anchor_temp_name)
                
                self.append(airBRDG)

            route_anchors.append(anchors[1])

            for labels in zip(route_anchors[::2], route_anchors[1::2]):
                self.route_between_two_pts(anchors=labels, layers=layers)

            print(valid_intersections)
            print(sorted_distances_indicies)
            print(route_anchors)
            print(temporary_anchors)
            #for point in list_intersection_points:
            #return route_anchors
