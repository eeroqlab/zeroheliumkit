from math import pi, tanh, cos, tan
import numpy as np

from shapely import Point, LineString, MultiLineString, Polygon
from shapely import affinity, unary_union, box

from .core import Entity, Structure
from .anchors import Anchor, MultiAnchor
from .basics import ArcLine
from .functions import (get_angle_between_points, offset_point, get_intersection_point_bruteforce,
                        modFMOD, midpoint, extract_coords_from_point)
from .settings import GRID_SIZE


# Multi-layer Geometry Classes
# _________________________________________________________________________________

class StraightLine(Entity):
    """ creates a straight line polygon structure

    Args:
        anchors (tuple, optional): two anchors/points/coords. Defaults to None.
        lendir (tuple, optional): length and orientation tuple. Defaults to None.
        layers (dict, optional): layers info. Defaults to {}.
        alabel (tuple, optional): labels of the start and end-points. Defaults to None.
    """
    def __init__(self,
                 anchors: tuple=None,
                 lendir: tuple=None,
                 layers: dict={"one": 1},
                 alabel: tuple=None):

        super().__init__()

        if anchors:
            p1 = extract_coords_from_point(anchors[0])
            p2 = extract_coords_from_point(anchors[1])

        elif lendir:
            length, direction = lendir
            p1 = (0, 0)
            p2 = (length * np.cos(direction * np.pi/180),
                  length * np.sin(direction * np.pi/180))
        else:
            raise AttributeError("provide only one argument: 'anchors' or 'lendir'")

        # create skeletone
        self.skeletone = LineString([p1, p2])

        # create polygons
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square')

        # create anchors
        if alabel:
            angle = get_angle_between_points(p1, p2)
            self.add_anchor([Anchor(point=p1, direction=angle, label=alabel[0]),
                            Anchor(point=p2, direction=angle, label=alabel[1])])


class ArbitraryLine(Structure):
    """ create an arbitrary line geometry based on list of point
        and list of widths which defines the polygon

     Args:
        points (list): list of points along which a polygon will be constructed
        layers (dict): layers info
        alabel (tuple): labels of the start and end-points.
    """
    def __init__(self, points: list, layers: dict, alabel: tuple):

        super().__init__()

        # create skeletone
        self.skeletone = LineString(points)

        # create polygons
        for k, width in layers.items():
            polygon = self.make_base_polygon(points, width)
            setattr(self, k, polygon)

        # create anchors
        if alabel:
            input_angle = get_angle_between_points(points[0], points[1])
            output_angle = get_angle_between_points(points[-2], points[-1])
            self.add_anchor([Anchor(point=points[0], direction=input_angle, label=alabel[0]),
                            Anchor(point=points[-1], direction=output_angle, label=alabel[1])])

    def make_base_polygon(self, points: list, width: list | float) -> Polygon:
        """ creates arbitraryline polygons

        Args:
            points (list): alongth these points polygon structure is constructed
            width (list | float): the width of the polygon defined by list or
                                single value (uniform width)

        Returns:
            polygon (Polygon): polygon structure
        """
        num = len(points)
        if isinstance(width, (int, float)):
            width = np.full(shape=num, fill_value=width, dtype=np.float32)

        p_start_up = offset_point(points[0],
                                  width[0]/2,
                                  get_angle_between_points(points[0], points[1]))
        p_start_down = offset_point(points[0],
                                    -width[0]/2,
                                    get_angle_between_points(points[0], points[1]))
        points_up = [p_start_up]
        points_down = [p_start_down]
        for i in range(1, num - 1):
            points_up.append(self.get_boundary_intersection_point(points[i-1],
                                                                  points[i],
                                                                  points[i+1],
                                                                  width[i-1]/2,
                                                                  width[i]/2,
                                                                  width[i+1]/2))
            points_down.append(self.get_boundary_intersection_point(points[i-1],
                                                                    points[i],
                                                                    points[i+1],
                                                                    -width[i-1]/2,
                                                                    -width[i]/2,
                                                                    -width[i+1]/2))
        points_up.append(offset_point(points[-1],
                                      width[-1]/2,
                                      get_angle_between_points(points[-2], points[-1])))
        points_down.append(offset_point(points[-1],
                                        -width[-1]/2,
                                        get_angle_between_points(points[-2], points[-1])))
        pts = points_up + points_down[::-1]

        return Polygon(pts)

    def get_boundary_intersection_point(self,
                                        point1: Point,
                                        point2: Point,
                                        point3: Point,
                                        distance1: float,
                                        distance2: float,
                                        distance3: float):
        """ given 3 points and offset distances calculates the coordinate of the offset point
        from the middle point
        """
        angle1 = get_angle_between_points(point1, point2)
        angle2 = get_angle_between_points(point2, point3)
        offset_p1 = offset_point(point1, distance1, angle1)
        offset_p2 = offset_point(point2, distance2, angle1)
        offset_p3 = offset_point(point2, distance2, angle2)
        offset_p4 = offset_point(point3, distance3, angle2)

        return get_intersection_point_bruteforce(offset_p1, offset_p2, offset_p3, offset_p4)


class Taper(ArbitraryLine):
    """ creates a taper structure

    Args:
        length (float): length of the tapered section
        layers (dict, optional): layer info. Defaults to None.
        alabel (tuple, optional): labels of the start and end-points. Defaults to None.
    """
    def __init__(self,
                 length: float,
                 layers: dict=None,
                 alabel: tuple=None):

        # preparing dictionary for supplying it into ArbitraryLine class
        for k, v in layers.items():
            w1 = v[0]   # input side width
            w2 = v[1]   # output side width
            if w1==w2:
                # 1. if width are the same -> adding small difference
                # in order to avoid 'division by zero' error
                # 2. later during "setattr" operation this difference will be eliminated
                # by "set_precision" (by default)
                w2 = w2 + GRID_SIZE/10
            layers[k] = np.asarray([w1, w1, w2, w2])

        pts = [(-length/2 - w1, 0),
               (-length/2, 0),
               (length/2, 0),
               (length/2 + w2, 0)
               ]
        super().__init__(pts, layers, alabel)


class Fillet(Structure):
    """ Fillet structure
        creates a rounded route structure from origin (0,0) point towards anchor

    Args:
        anchor (Anchor): endpoint
        radius (float): radius of the rounded section
        num_segments (int): number of segments in arcline
        layers (dict): layers info
        alabel (tuple): labels of the start and end-points
    """
    def __init__(self,
                 anchor: Anchor,
                 radius: float,
                 num_segments: int,
                 layers: dict,
                 alabel: tuple):

        super().__init__()

        # create skeletone
        self.__create_skeletone(anchor, radius, num_segments, layers)

        # create polygons
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2,
                             cap_style='square', join_style=self.__joinstyle)

        # create anchors
        if alabel:
            first, last = self.get_skeletone_boundary()

            self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]),
                            Anchor(point=last, direction=anchor.direction, label=alabel[1])])

    def __get_fillet_params(self, anchor: Anchor, radius: float) -> tuple:
        """ calculates fillet parameters

        Args:
            anchor (Anchor): endpoint of the fillet
            radius (float): radius of the curved section

        Returns:
            tuple: _description_
        """

        ang = anchor.direction * np.pi/180

        # calculating lengths
        if cos(ang)==1:
            length2 = anchor.y - radius
        else:
            length2 = (np.sign(ang)
                       * (anchor.y - np.sign(ang) * radius * (1 - np.cos(ang)))
                       / np.sqrt(1 - np.cos(ang)**2))

        length1 = anchor.x - 1 * length2 * np.cos(ang) - np.sign(ang) * radius * np.sin(ang)

        return length1, length2, anchor.direction

    def __create_fillet_skeletone(self,
                                  length1: float,
                                  length2: float,
                                  direction: float,
                                  radius: float,
                                  num_segments: int) -> None:
        """ create a fillet linestring

        Args:
            length1 (float): length of the first section
            length2 (float): length of the second section placed after arcline
            direction (float): orientation of the endsegment
            radius (float): radius if the arcline
            num_segments (int): num of segments in the arcline
        """

        direction = modFMOD(direction)
        dir_rad = direction * np.pi/180

        # creating first section
        pts = [(0, 0), (length1, 0)]
        self.skeletone = LineString(pts)

        # creating arcline section
        if direction > 0:
            x0, y0, phi0 = (0, radius, 270)
        else:
            x0, y0, phi0 = (0, -radius, 90)
        self.add_line(ArcLine(x0, y0, radius, phi0, phi0 + direction, num_segments))

        # creating second straight line section
        self.add_line(LineString([(0, 0), (length2 * np.cos(dir_rad), length2 * np.sin(dir_rad))]))


    def __create_skeletone(self,
                           anchor: Anchor,
                           radius: float,
                           num_segments: int,
                           layers: dict) -> None:
        """ creating skeletone of the structure

        Args:
            anchor (Anchor): endpoint
            radius (float): radius of the arcline
            num_segments (int): number of segments in the arcline
            layers (dict): layers info
        """

        # calculate fillet params
        extension_length = max(layers.values())     # determines min length of section 2
        length1, length2, direction = self.__get_fillet_params(anchor, radius)
        dir_rad = direction * np.pi/180

        if (length1 > 0 and length2 > 0) and np.sign(anchor.y)==np.sign(np.sin(dir_rad)):
            # valid fillet
            self.__create_fillet_skeletone(length1, length2, direction, radius, num_segments)

            # correction of the last point -> adjusting to anchor point
            self.skeletone = LineString(list(self.skeletone.coords[:-1]) + [anchor.coords])
            self.__joinstyle = 'mitre'

        elif np.sign(anchor.y)==np.sign(np.sin(dir_rad)) and length1 > 0:
            # correcting the radius of the round section to make valid fillet

            print("using smaller radius for routing")
            while length2 < extension_length:
                radius = radius * 0.6
                length1, length2, direction = self.__get_fillet_params(anchor, radius)
            self.__create_fillet_skeletone(length1, length2, direction, radius, num_segments)

            self.skeletone = LineString(list(self.skeletone.coords[:-1]) + [anchor.coords])
            self.__joinstyle = 'mitre'

        else:
            # using linestring for invalid fillet params

            dx = np.abs(anchor.x)/5
            after_startPoint = Point(dx, 0)
            before_endPoint = Point(anchor.x - dx * np.cos(dir_rad),
                                    anchor.y - dx * np.sin(dir_rad))
            self.skeletone = LineString([Point(0,0),
                                         after_startPoint,
                                         before_endPoint,
                                         anchor.point])
            self.__joinstyle = 'round'


class ElbowLine(Fillet):
    def __init__(self,
                 anchor1: Anchor,
                 anchor2: Anchor,
                 radius: float=10,
                 num_segments: int=10,
                 layers: dict={},
                 alabel: tuple=None):

        direction = anchor2.direction - anchor1.direction
        p = affinity.rotate(Point((anchor2.x - anchor1.x, anchor2.y - anchor1.y)),
                            -anchor1.direction,
                            origin=(0,0))

        super().__init__(Anchor(p, direction), radius, num_segments, layers, alabel)

        self.rotate(anchor1.direction)
        self.moveby((anchor1.x, anchor1.y))


class SigmoidLine(Structure):
    def __init__(self,
                 anchor1: Anchor,
                 anchor2: Anchor,
                 mid_direction: float,
                 radius: float=10,
                 num_segments: int=10,
                 layers: dict={},
                 alabel: tuple=None):
        super().__init__()

        anchormid = Anchor(midpoint(anchor1.point, anchor2.point), mid_direction)
        r1 = ElbowLine(anchor1, anchormid, radius, num_segments, layers)
        r2 = r1.copy()
        r2.rotate(angle=180, origin=(anchormid.x, anchormid.y))
        r1.append(r2)
        r1.fix_line()

        if not hasattr(r1.skeletone, "geoms"):
            self.append(r1)
        else:
            dx = np.abs(anchor2.x - anchor1.x)/10
            followup_startPoint = Point(anchor1.x + dx,
                                        anchor1.y + dx * np.tan(anchor1.direction * np.pi/180))
            before_endPoint = Point(anchor2.x - dx,
                                    anchor2.y - dx * np.tan(anchor2.direction * np.pi/180))
            arb_line = ArbitraryLine(points=[anchor1.point,
                                             followup_startPoint,
                                             before_endPoint,
                                             anchor2.point],
                                     layers=layers,
                                     alabel=None)
            self.append(arb_line)

        # create anchors
        if alabel:
            anchor1.label = alabel[0]
            anchor2.label = alabel[1]
            self.add_anchor([anchor1, anchor2])


class claws(Entity):
    def __init__(self,
                 radius: float,
                 offset: float,
                 length: float,
                 layers: dict,
                 alabel: tuple):
        super().__init__()
        r = radius + offset
        self.create_skeletone(offset, radius, length)

        for k, width in layers.items():
            self.add_buffer(name=k, offset=width/2, cap_style='round',
                            join_style='round', quad_segs=20)
        self.anchors = MultiAnchor([Anchor((0,0), 0, alabel[0]),
                                       Anchor((r,0), 0, alabel[1])])

    def create_skeletone(self, offset, radius, length):
        r = radius + offset     # radius of the claw
        angle = 180 * (length/2)/(r * pi)
        if angle > 90:
            d = (length - pi * r)/2
            self.add_line(LineString([(-d, -r), (0, -r)]))
            self.add_line(ArcLine(0, r, r, -90, 90, 50))
            self.add_line(LineString([(0, 0), (-d, 0)]))
        else:
            self.add_line(ArcLine(0, 0, r, -angle, angle, 50))

class uChannelsAngle(Entity):
    def __init__(self,
                 length: float,
                 spacing: float,
                 num: int,
                 angle: float,
                 layers: dict,
                 alabel: tuple):
        super().__init__()

        # create skeletone
        slope = tan(angle * pi/180)
        l = length - spacing * slope * (num - 1)
        pts = lambda i: [(0, 0),
                         (0, l + slope * spacing * i),
                         (spacing, l + slope * spacing * (i + 1)),
                         (spacing, 0)]
        for i in range(num - 1):
            self.add_line(LineString(pts(i)))
        self.add_line(LineString([(0, 0), (0, length), (spacing/2, length)]))

        # create polygon
        for k, width in layers.items():
            self.buffer_line(name=k,
                             offset=width/2,
                             cap_style='flat',
                             join_style='round',
                             quad_segs=3)

        # create anchors
        if alabel:
            first, last = self.get_skeletone_boundary()
            self.add_anchor([Anchor(point=first, direction=90, label=alabel[0]),
                            Anchor(point=last, direction=0, label=alabel[1])])

        # delete skeletone
        self.skeletone = MultiLineString()


class SpiralInductor(Entity):
    def __init__(self,
                 size: float,
                 width: float,
                 gap: float,
                 num_turns: int,
                 smallest_section_length: float,
                 layers: dict,
                 alabel: dict):
        super().__init__()

        # create skeletone
        eps = 0.1 # this removes some artifacts at the corner of the central_pad unary_union process
        radius = width/2
        self._gap = gap
        self._width = width
        coord_init = [(0, 0), (0, -size/2 - width/2),
                      (size/2 + width + gap, -size/2 - width/2)]
        self.skeletone = LineString(coord_init)

        # create polygons
        self.construct_spiral(size, radius, num_turns, smallest_section_length)
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='round',
                             join_style='mitre', quad_segs=2)

        central_pad = box(-size/2, -size/2, size/2, size/2).buffer(self._width+eps, join_style=1)
        central_pad.simplify(0.2, preserve_topology=True)
        for k in layers.keys():
            united = unary_union([getattr(self, k), central_pad])
            setattr(self, k, united)

        # create anchors
        if alabel:
            first, last = self.get_skeletone_boundary()
            self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]),
                            Anchor(point=last, direction=0, label=alabel[1])])

    def num_segments(self, R: float, smallest_segment: float):
        ''' limits the maximum number of segments in arc '''
        return int(10*tanh(pi * R/2/smallest_segment/20))

    def construct_spiral(self, size: float, radius: float, num_turns:int, ls: float):

        self.add_line(ArcLine(0, radius, radius, 270, 360, self.num_segments(radius, ls)))
        self.add_line(LineString([(0, 0), (0, size/2)]))

        for _ in range(num_turns):
            radius = radius + self._gap + self._width
            self.add_line(LineString([(0, 0), (0, size/2)]))
            self.add_line(ArcLine(-radius, 0, radius, 0, 90, self.num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (-size, 0)]))
            self.add_line(ArcLine(0, -radius, radius, 90, 180, self.num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (0, -size)]))
            self.add_line(ArcLine(radius, 0, radius, 180, 270, self.num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (size + self._width + self._gap, 0)]))
            self.add_line(ArcLine(0, radius, radius, 270, 360, self.num_segments(radius, ls)))
            self.add_line(LineString([(0, 0), (0, size/2)]))
        self.add_line(LineString([(0, 0), (2*self._gap + 2*self._width, 0)]))


class IDC(Entity):
    def __init__(self,
                 length: float,
                 spacing: float,
                 num: int,
                 layers: dict,
                 alabel: tuple):
        super().__init__()

        pts = [(0, 0), (spacing/2, 0), (spacing/2,length),
               (spacing/2, -length), (spacing/2, 0), (spacing, 0)]
        self.skeletone = LineString(pts)
        for _ in range(num):
            self.add_line(LineString(pts))

        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square',
                             join_style='round', quad_segs=2)

        # create anchors
        if alabel:
            first, last = self.get_skeletone_boundary()
            self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]),
                            Anchor(point=last, direction=0, label=alabel[1])])
