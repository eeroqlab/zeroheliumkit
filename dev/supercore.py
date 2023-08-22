import numpy as np

from shapely import line_locate_point, line_interpolate_point

from ..dev.geometries import StraightLine, ElbowLine, SigmoidLine
from ..dev.functions import get_abc_line, get_angle_between_points
from ..dev.core import Structure, Entity
from ..errors import RouteError


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
        normal_angles = self._get_normals_along_line(locs[1:-1])

        for point, angle in zip(pts, normal_angles):
            s = structure.copy()
            s.rotate(angle)
            s.moveby(xy=(point.x, point.y))
            self.append(s)


    def _get_normals_along_line(self, locs: list) -> list:
        eps = np.abs(locs[1] - locs[0])/10
        pts_up = line_interpolate_point(self.skeletone, locs + eps, normalized=True).tolist()
        pts_down = line_interpolate_point(self.skeletone, locs - eps, normalized=True).tolist()
        tangent_angles = list(map(get_angle_between_points, pts_down, pts_up))

        return np.asarray(tangent_angles)

    def route_with_intersection(self, anchors: tuple, layers: dict) -> None:
        pass