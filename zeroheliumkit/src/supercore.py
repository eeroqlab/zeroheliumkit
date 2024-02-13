import numpy as np

from shapely import line_locate_point, line_interpolate_point, intersection_all, distance
from shapely import LineString

from .core import Structure, Entity
from .geometries import StraightLine, ElbowLine, SigmoidLine
from .functions import get_abc_line, create_list_geoms, get_normals_along_line, modFMOD
from .errors import WrongSizeError
from .routing import route_fillet


class SuperStructure(Structure):
    """ this class provides more advanced routing options.
        Based on Structure class.
    """
    def __init__(self, route_config: dict):
        self._route_config = route_config
        super().__init__()

    def route(self,
              anchors: tuple,
              layers: dict=None,
              airbridge: Entity | Structure=None,
              extra_rotation: float=0,
              new_feature: bool=False):
        for labels in zip(anchors, anchors[1:]):
            if airbridge:
                self.route_with_intersection(labels, layers, airbridge, extra_rotation, new_feature)
            else:
                self.route_between_two_pts(anchors, layers)

    def route_between_two_pts(self,
                              anchors: tuple,
                              layers: dict=None) -> None:
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

        # calculating angle of the line between anchors
        a, b, _ = get_abc_line(point1.point, point2.point)
        direction_sign_y = np.sign(point2.point.y - point1.point.y)
        direction_sign_x = -np.sign(point2.point.x - point1.point.x)/2 + 0.5
        if b != 0:
            if a == 0:
                angle = 180 * direction_sign_x
            else:
                angle = np.arctan(-a/b) * 180/np.pi
                if direction_sign_y * np.sign(angle) < 0:
                    angle = -180 * np.sign(angle) + angle
        else:
            angle = 90 * direction_sign_y


        ##############################
        #### MAIN routing choices ####
        ##############################

        if (np.abs(point1.direction - point2.direction) < 1e-4 and
            np.abs(angle - point1.direction) < 1e-4):
            # stright line construction
            # - anchor directions are the same
            # - direction is the same with anchor-anchor angle

            connecting_structure = StraightLine(anchors=(point1,point2),
                                                layers=layers)

        elif np.abs(point1.direction - point2.direction) < 1e-4:
            # sigmoid line construction
            # - anchor directions are the same
            
            # calculating intermediate point direction
            angle_point1reference = modFMOD(angle - point1.direction)
            if (0 < angle_point1reference <= 90) or (-180 < angle_point1reference <= -90):
                #mid_dir = modFMOD(point1.direction + np.abs(angle))
                mid_dir = modFMOD(angle + 20)
            else:
                #mid_dir = modFMOD(point1.direction - np.abs(angle))
                mid_dir = modFMOD(angle - 20)

            connecting_structure = SigmoidLine(anchor1=point1,
                                               anchor2=point2,
                                               mid_direction=mid_dir,
                                               radius=radius,
                                               num_segments=num_segments,
                                               layers=layers)
        else:
            # elbow line construction
            # takes care of all othe possibilities

            connecting_structure = ElbowLine(anchor1=point1,
                                             anchor2=point2,
                                             radius=radius,
                                             num_segments=num_segments,
                                             layers=layers)

        ###################################
        #### END of ROUTE construction ####
        ###################################

        self.append(connecting_structure)


    def add_along_skeletone(self,
                            bound_anchors: tuple,
                            num: int,
                            structure: Structure | Entity) -> None:
        """ add Structures alongth the skeletone line

        Args:
            bound_anchors (tuple): skeletone regoin contained between two anchors
            num (int): number of structures
            structure (Structure | Entity): structure to be added

        Raises:
            ValueError: _description_
        """

        if len(bound_anchors) != 2:
            raise WrongSizeError(f"Provide 2 anchors! Instead {len(bound_anchors)} is given.")

        p1 = self.get_anchor(bound_anchors[0]).point
        p2 = self.get_anchor(bound_anchors[1]).point

        start_point = line_locate_point(self.skeletone, p1, normalized=True)
        end_point = line_locate_point(self.skeletone, p2, normalized=True)

        locs = np.linspace(start_point, end_point, num=num+2, endpoint=True)
        pts = line_interpolate_point(self.skeletone, locs[1:-1], normalized=True).tolist()
        normal_angles = get_normals_along_line(self.skeletone, locs[1:-1]) - 90
        normal_angles = np.asarray([modFMOD(alpha) for alpha in normal_angles])

        for point, angle in zip(pts, normal_angles):
            s = structure.copy()
            s.rotate(angle)
            s.moveby(xy=(point.x, point.y))
            self.append(s)


    def route_with_intersection(self,
                                anchors: tuple,
                                layers: dict,
                                airbridge: Entity | Structure,
                                extra_rotation: float=0,
                                new_feature: bool=False) -> None:
        """ creates route between two anchors when a crossing with skeletone is expected

        Args:
            anchors (tuple): routing between two anchors. provide labels
            layers (dict): layer width information
            airbridge (Entity | Structure): airbridge structure
                                            should contain 'in' and 'out' anchors
        """
        if len(anchors) != 2:
            raise WrongSizeError("Provide only two point labels in 'anchors'.")

        anchor_labels_airbridge = ['in', 'out']

        if ((anchor_labels_airbridge[0] not in airbridge.anchors.labels) or
            (anchor_labels_airbridge[1] not in airbridge.anchors.labels)):
            raise TypeError("airbridge anchors could be only 'in' and 'out'")

        p_start = self.get_anchor(anchors[0]).point
        p_end = self.get_anchor(anchors[1]).point

        # testing a new feature here
        if new_feature:
            route_line, _ = route_fillet(anchor1=self.get_anchor(anchors[0]),
                                      anchor2=self.get_anchor(anchors[1]),
                                      radius=self._route_config.get("radius"),
                                      num_segments=self._route_config.get("num_segments"))
        else:
            route_line = LineString([p_start, p_end])

        # get all intersection points with route line and skeletone
        intersections = intersection_all([route_line, self.skeletone])

        # get valid intesections
        if not intersections.is_empty:
            # create a list of points
            list_intersection_points = create_list_geoms(intersections)

            # removing start and end points
            valid_intersections = [p for p in list_intersection_points if p not in [p_start, p_end]]

        #######################
        ### creating route ####
        #######################

        if intersections.is_empty or valid_intersections==[]:
            # in case of no intersections make a simple route or no valid_intersections
            self.route_between_two_pts(anchors, layers)

        else:
            list_distances = np.asarray(list(map(distance,
                                                 valid_intersections,
                                                 [p_start]*len(valid_intersections)
                                                 )
                                            )
                                        )
            sorted_distance_indicies = np.argsort(list_distances)

            intersect_locs_on_skeletone = line_locate_point(self.skeletone,
                                                            valid_intersections,
                                                            normalized=True)
            intersect_normals = get_normals_along_line(self.skeletone, intersect_locs_on_skeletone)

            route_anchors = [anchors[0]]
            # create temporary anchor list, which will be deleted in the end
            temporary_anchors = []

            # adding airbridges to superstructure
            for i, idx in enumerate(sorted_distance_indicies):
                airBRDG = airbridge.copy()

                airBRDG.rotate(angle=modFMOD(intersect_normals[idx] + 90 + extra_rotation))
                air_loc = (valid_intersections[idx].x, valid_intersections[idx].y)
                airBRDG.moveby(xy=air_loc)

                # correcting the orientation of the airbridge if 'in' and 'out' are swapped
                distance2in  = distance(airBRDG.get_anchor("in").point,  self.get_anchor(route_anchors[-1]).point)
                distance2out = distance(airBRDG.get_anchor("out").point, self.get_anchor(route_anchors[-1]).point)

                if distance2out < distance2in:
                    airBRDG.rotate(180, origin=air_loc)

                for ab_anchor in anchor_labels_airbridge:
                    anchor_temp_name = str(i) + ab_anchor
                    airBRDG.modify_anchor(label=ab_anchor,
                                          new_name=anchor_temp_name)
                    route_anchors.append(anchor_temp_name)
                    temporary_anchors.append(anchor_temp_name)

                self.append(airBRDG)

            route_anchors.append(anchors[1])

            # adding all routes between anchors
            for labels in zip(route_anchors[::2], route_anchors[1::2]):
                self.route_between_two_pts(anchors=labels, layers=layers)

            self.remove_anchor(temporary_anchors)

    ###################################################
    ########### testing new routing ###################
    ###################################################
    
    def route_bezier(self,
              anchors: tuple,
              layers: dict=None,
              airbridge: Entity | Structure=Entity(),
              extra_rotation: float=0,
              starting_length_scale: float=10):
        for labels in zip(anchors, anchors[1:]):
            self.route_with_intersection_bezier(labels, layers, airbridge, extra_rotation, starting_length_scale)
    
    def _add_buffered_route(self,
                            route_line: LineString,
                            layers: dict=None) -> None:
        """ append route to skeletone and buffers to create polygons.

        Args:
            route_line (LineString): route line
            layers (dict): layer info
        """
        routed_structure = Structure()
        routed_structure.add_line(route_line)
        
        # create polygons
        if layers:
            for k, width in layers.items():
                routed_structure.buffer_line(name=k, offset=width/2, cap_style='square')
        
        self.append(routed_structure)


    def route_with_intersection_bezier(self,
                                anchors: tuple,
                                layers: dict,
                                airbridge: Entity | Structure,
                                extra_rotation: float=0,
                                starting_length_scale: float=10) -> None:
        """ creates route between two anchors when a crossing with skeletone is expected

        Args:
            anchors (tuple): routing between two anchors. provide labels
            layers (dict): layer width information
            airbridge (Entity | Structure): airbridge structure
                                            should contain 'in' and 'out' anchors
        """
        if len(anchors) != 2:
            raise WrongSizeError("Provide only two point labels in 'anchors'.")

        anchor_labels_airbridge = ['in', 'out']

        if ((anchor_labels_airbridge[0] not in airbridge.anchors.labels) or
            (anchor_labels_airbridge[1] not in airbridge.anchors.labels)):
            raise TypeError("airbridge anchors could be only 'in' and 'out'")

        p_start = self.get_anchor(anchors[0]).point
        p_end = self.get_anchor(anchors[1]).point

        # testing a new feature here
        route_line, _ = route_fillet(anchor1=self.get_anchor(anchors[0]),
                                     anchor2=self.get_anchor(anchors[1]),
                                     radius=self._route_config.get("radius"),
                                     num_segments=self._route_config.get("num_segments"),
                                     starting_length_scale=starting_length_scale)

        # get all intersection points with route line and skeletone
        intersections = intersection_all([route_line, self.skeletone])

        # get valid intesections
        if not intersections.is_empty:
            # create a list of points
            list_intersection_points = create_list_geoms(intersections)

            # removing start and end points
            valid_intersections = [p for p in list_intersection_points if p not in [p_start, p_end]]

        #######################
        ### creating route ####
        #######################

        if intersections.is_empty or valid_intersections==[]:
            # in case of no intersections make a simple route or no valid_intersections
            self._add_buffered_route(route_line, layers)

        else:
            list_distances = np.asarray(list(map(distance,
                                                 valid_intersections,
                                                 [p_start]*len(valid_intersections)
                                                 )
                                            )
                                        )
            sorted_distance_indicies = np.argsort(list_distances)

            intersect_locs_on_skeletone = line_locate_point(self.skeletone,
                                                            valid_intersections,
                                                            normalized=True)
            intersect_normals = get_normals_along_line(self.skeletone, intersect_locs_on_skeletone)

            route_anchors = [anchors[0]]
            # create temporary anchor list, which will be deleted in the end
            temporary_anchors = []

            # adding airbridges to superstructure
            for i, idx in enumerate(sorted_distance_indicies):
                airBRDG = airbridge.copy()

                airBRDG.rotate(angle=modFMOD(intersect_normals[idx] + 90 + extra_rotation))
                air_loc = (valid_intersections[idx].x, valid_intersections[idx].y)
                airBRDG.moveby(xy=air_loc)

                # correcting the orientation of the airbridge if 'in' and 'out' are swapped
                distance2in  = distance(airBRDG.get_anchor("in").point,  self.get_anchor(route_anchors[-1]).point)
                distance2out = distance(airBRDG.get_anchor("out").point, self.get_anchor(route_anchors[-1]).point)

                if distance2out < distance2in:
                    airBRDG.rotate(180, origin=air_loc)

                for ab_anchor in anchor_labels_airbridge:
                    anchor_temp_name = str(i) + ab_anchor
                    airBRDG.modify_anchor(label=ab_anchor,
                                          new_name=anchor_temp_name)
                    route_anchors.append(anchor_temp_name)
                    temporary_anchors.append(anchor_temp_name)

                self.append(airBRDG)

            route_anchors.append(anchors[1])

            # adding all routes between anchors
            for labels in zip(route_anchors[::2], route_anchors[1::2]):
                route_line, _ = route_fillet(anchor1=self.get_anchor(labels[0]),
                                             anchor2=self.get_anchor(labels[1]),
                                             radius=self._route_config.get("radius"),
                                             num_segments=self._route_config.get("num_segments"),
                                             starting_length_scale=starting_length_scale)
                self._add_buffered_route(route_line, layers)

            self.remove_anchor(temporary_anchors)
