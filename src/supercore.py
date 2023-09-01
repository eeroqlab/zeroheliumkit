import numpy as np

from shapely import line_locate_point, line_interpolate_point, intersection_all, distance
from shapely import LineString

from .core import Structure, Entity
from .geometries import StraightLine, ElbowLine, SigmoidLine
from .functions import get_abc_line, create_list_geoms, get_normals_along_line, modFMOD
from .errors import WrongSizeError


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

    def route_between_two_pts(self,
                              anchors: tuple,
                              layers: dict) -> None:
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
        if b != 0:
            angle = np.arctan(-a/b) * 180/np.pi
        else:
            angle = 90 * np.sign(point2.point.y - point1.point.y)

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
            if (0 < modFMOD(angle - point1.direction) <= 90) or (-180 < modFMOD(angle - point1.direction) <= -90):
                mid_dir = modFMOD(point1.direction + 45)
            else:
                mid_dir = modFMOD(point1.direction - 45)

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

        for point, angle in zip(pts, normal_angles):
            s = structure.copy()
            s.rotate(angle)
            s.moveby(xy=(point.x, point.y))
            self.append(s)


    def route_with_intersection(self,
                                anchors: tuple,
                                layers: dict,
                                airbridge: Entity | Structure,
                                extra_rotation: float=0) -> None:
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
                airBRDG.rotate(angle=intersect_normals[idx] + 90 + extra_rotation)
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

            # adding all routes between anchors
            for labels in zip(route_anchors[::2], route_anchors[1::2]):
                self.route_between_two_pts(anchors=labels, layers=layers)

            self.remove_anchor(temporary_anchors)
