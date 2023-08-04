import numpy as np

from math import sqrt, pi, tanh, cos, sin, tan, atan
from shapely import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
from shapely import affinity, unary_union, box

from .core import *
from .functions import *


# Collection of different Geometries
# _________________________________________________________________________________

def ArcLine(centerx, centery, radius, start_angle, end_angle, numsegments=10):
    theta = np.radians(np.linspace(start_angle, end_angle, num=numsegments, endpoint=True))
    x = centerx + radius * np.cos(theta)
    y = centery + radius * np.sin(theta)
    return LineString(np.column_stack([x, y]))

def Meander(length: float, radius: float, direction: float, num_segments: int=100) -> LineString:
    """ Angles in degrees """

    coord_init = [(0, 0), (0, length/2)]
    s = Entity()
    s.add_line(LineString(coord_init))
    s.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))
    s.add_line(LineString([(0, 0), (0, -length)]))
    s.add_line(ArcLine(radius, 0, radius, 180, 360, num_segments))
    s.add_line(LineString([(0, 0), (0, length/2)]))
    s.rotate(direction, origin= (0, 0))

    return s.skeletone

def MeanderHalf(length: float, radius: float, direction: float, num_segments: int=100) -> LineString:
    """ Angles in degrees """

    coord_init = [(0, 0), (0, length/2)]
    s = Entity()
    s.add_line(LineString(coord_init))
    s.add_line(ArcLine(radius, 0, radius, 180, 0, num_segments))
    s.add_line(LineString([(0, 0), (0, -length/2)]))
    s.rotate(direction, origin= (0, 0))

    return s.skeletone

def PinchGate(arm_w: float, arm_l: float, length: float, width: float) -> Polygon:
    pts = [(-arm_w/2, arm_w/2), 
            (arm_l - length/32, arm_w/2), 
            (arm_l, length/8), 
            (arm_l, 7*length/16), 
            (arm_l + width/3, length/2),
            (arm_l + width, length/2),
            (arm_l + width, -length/2),
            (arm_l + width/3, -length/2),
            (arm_l, -7*length/16),
            (arm_l, -length/8),
            (arm_l - length/32, -arm_w/2),
            (-arm_w/2, -arm_w/2)
            ]
    return Polygon(pts)

def Rectangle(a: float, b: float) -> Polygon:
    return Polygon([(-a/2, -b/2), (-a/2, b/2), (a/2, b/2), (a/2, -b/2)])

def Square(s: float) -> Polygon:
    return Rectangle(s, s)

def RegularPolygon(edge: float, n: int) -> Polygon:
    xy = []
    angle = 2 * np.pi / n
    radius = edge/np.sin(angle/2) * 0.5
    for i in range(n):
        x = radius * np.cos(i * angle)
        y = radius * np.sin(i * angle)
        xy.append((x, y))

    return Polygon(xy)


# Multi-layer Geometry Classes
# _________________________________________________________________________________

class StraightLine(Entity):
    def __init__(self, 
                 length: float, 
                 layers: dict,
                 alabel: tuple):
        super().__init__()

        pts = [(0, 0), (length, 0)]
        self.skeletone = LineString(pts)

        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square')

        first, last = self.get_skeletone_boundary()
        self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]), 
                         Anchor(point=last, direction=0, label=alabel[1])])
        

class ArbitraryLine(Entity):
    def __init__(self, points: list, layers: dict, alabel: tuple):
        super().__init__()
        
        self.skeletone = LineString(points)

        for k, width in layers.items():
            polygon = self.make_base_polygon(points, width)
            setattr(self, k, polygon)
        
        input_angle = angle_between_points(points[0], points[1])
        output_angle = angle_between_points(points[-2], points[-1])
        self.add_anchor([Anchor(point=points[0], direction=input_angle, label=alabel[0]), 
                         Anchor(point=points[-1], direction=output_angle, label=alabel[1])])

    def make_base_polygon(self, points: list, width: list | float):
        num = len(points)
        if isinstance(width, (int, float)):
            width = np.full(shape=num, fill_value=width, dtype=np.float32)
        p_start_up = offset_point(points[0], width[0]/2, angle_between_points(points[0], points[1]))
        p_start_down = offset_point(points[0], -width[0]/2, angle_between_points(points[0], points[1]))
        points_up = [p_start_up]
        points_down = [p_start_down]
        for i in range(1, num - 1):
            points_up.append(self.get_boundary_intersection_point(points[i-1], points[i], points[i+1],
                                                            width[i-1]/2, width[i]/2, width[i+1]/2
                                                            ))
            points_down.append(self.get_boundary_intersection_point(points[i-1], points[i], points[i+1],
                                                            -width[i-1]/2, -width[i]/2, -width[i+1]/2
                                                            ))
        points_up.append(offset_point(points[-1], width[-1]/2, angle_between_points(points[-2], points[-1])))
        points_down.append(offset_point(points[-1], -width[-1]/2, angle_between_points(points[-2], points[-1])))
        pts = points_up + points_down[::-1]
        
        return Polygon(pts)

    def get_boundary_intersection_point(self, p1 :Point, p2: Point, p3: Point, w1: float, w2: float, w3: float):
        a1 = angle_between_points(p1, p2)
        a2 = angle_between_points(p2, p3)
        o1 = offset_point(p1, w1, a1)
        o2 = offset_point(p2, w2, a1)
        o3 = offset_point(p2, w2, a2)
        o4 = offset_point(p3, w3, a2)
        return get_intersection_point_bruteforce(o1, o2, o3, o4)
    

class Taper(ArbitraryLine):
    def __init__(self, 
                 length: float, 
                 layers: dict=None,
                 alabel: tuple=None):

        # preparing dictionary for ArbitraryLine class
        for k, v in layers.items():
            w1 = v[0]   # input side width
            w2 = v[1]   # output side width
            if w1==w2:
                # if width are the same -> adding small difference in order to avoid division by zero error
                # later during "setattr" operation this difference will be eliminated by "set_precision" (default)
                w2 = w2 + GRID_SIZE/10
            layers[k] = np.asarray([w1, w1, w2, w2])

        pts = [(-length/2 - w1, 0),
               (-length/2, 0),
               (length/2, 0),
               (length/2 + w2, 0)
               ]
        super().__init__(pts, layers, alabel)



class uChannelsAngle(Entity):
    def __init__(self, 
                 length: float, 
                 spacing: float, 
                 num: int, 
                 angle: float, 
                 layers: dict,
                 alabel: tuple):
        super().__init__()
        
        slope = tan(angle * pi/180)
        l = length - spacing * slope * (num - 1)
        pts = lambda i: [(0, 0), (0, l + slope * spacing * i), (spacing, l + slope * spacing * (i + 1)), (spacing, 0)]
        for i in range(num - 1):
            self.add_line(LineString(pts(i)))
        self.add_line(LineString([(0, 0), (0, length), (spacing/2, length)]))
        
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='flat', join_style='round', quad_segs=3)

        first, last = self.get_skeletone_boundary()
        self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]), 
                         Anchor(point=last, direction=0, label=alabel[1])])


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

        eps = 0.1 # this removes some artifacts at the corner of the central_pad unary_union process
        radius = width/2
        self._gap = gap
        self._width = width
        coord_init = [(0, 0), (0, -size/2 - width/2), (size/2 + width + gap, -size/2 - width/2)]
        self.skeletone = LineString(coord_init)
        
        self.construct_spiral(size, radius, num_turns, smallest_section_length)
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='round', join_style='mitre', quad_segs=2)
        
        central_pad = box(-size/2, -size/2, size/2, size/2).buffer(self._width+eps, join_style=1).simplify(0.2, preserve_topology=True)
        for k in layers.keys():
            united = unary_union([getattr(self, k), central_pad])
            setattr(self, k, united)
        
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

        pts = [(0, 0), (spacing/2, 0), (spacing/2,length), (spacing/2, -length), (spacing/2, 0), (spacing, 0)]
        self.skeletone = LineString(pts)
        for _ in range(num):
            self.add_line(LineString(pts))

        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square', join_style='round', quad_segs=2)
        
        first, last = self.get_skeletone_boundary()
        self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]), 
                         Anchor(point=last, direction=0, label=alabel[1])])


class Fillet(Entity):
    def __init__(self, 
                 length1: float, 
                 length2: float, 
                 radius: float, 
                 direction: float, 
                 num_segments: int, 
                 layers: dict,
                 alabel: tuple):
        super().__init__()
        
        pts = [(0, 0), (length1, 0)]
        self.skeletone = LineString(pts)
        if direction>0:
            x0, y0, phi0 = (0, radius, 270)
        else:
            x0, y0, phi0 = (0, -radius, 90)
        self.add_line(ArcLine(x0, y0, radius, phi0, phi0 + direction, num_segments))
        self.add_line(LineString([(0, 0), (length2 * np.cos(direction*np.pi/180), length2 * np.sin(direction*np.pi/180))]))
        for k, width in layers.items():
            self.buffer_line(name=k, offset=width/2, cap_style='square', join_style='mitre')
        
        first, last = self.get_skeletone_boundary()
        self.add_anchor([Anchor(point=first, direction=0, label=alabel[0]), 
                         Anchor(point=last, direction=direction, label=alabel[1])])


class Route(Fillet):
    def __init__(self, 
                 point1: Point, 
                 direction1: float, 
                 point2: Point,
                 direction2: float, 
                 radius: float=10, 
                 num_segments: int=10, 
                 layers: dict=None,
                 alabel: tuple=None):
        
        direction = direction2 - direction1
        angle_rad = direction * pi/180
        p = affinity.rotate(Point((point2.x - point1.x, point2.y - point1.y)), -direction1, origin=(0,0))

        if np.abs(angle_rad) < np.abs(atan(p.y/p.x)) or np.sign(p.y)!=np.sign(direction):
            raise ValueError("cannot make a route, choose a different type of routing")
        
        if cos(angle_rad)==1:
            length2 = p.y - radius
        else:    
            length2 = np.abs(p.y - np.sign(direction) * radius * (1 - cos(angle_rad)))/sqrt(1 - cos(angle_rad)**2)
            
        length1 = p.x - 1 * length2 * cos(angle_rad) - np.sign(direction) * radius * sin(angle_rad)
        
        if length1 < 0 or length2 < 0:
            raise ValueError(f"cannot make route, make radius={radius} smaller")
        
        super().__init__(length1, length2, radius, direction, num_segments, layers, alabel)

        self.rotate(direction1)
        self.moveby((point1.x, point1.y))
        #lnames = self.layer_names()
        #for a in lnames:
        #    setattr(self, a, affinity.rotate(getattr(self, a), direction1, origin=(0,0)))
        #    setattr(self, a, affinity.translate(getattr(self, a), xoff=point1.x, yoff=point1.y))

        #self.anchorsmod.rotate(direction1)
        #self.anchorsmod.move

class RouteTwoElbows(Entity):
    def __init__(self, 
                 point1: Point, 
                 direction1: float, 
                 point2: Point,
                 direction2: float, 
                 mid_direction: float,
                 radius: float=10, 
                 num_segments: int=10, 
                 layers: dict=None,
                 alabel: tuple=None):
        
        mid_p = midpoint(point1, point2)
        mid_dir = mid_direction
        r1 = Route(point1, direction1, mid_p, mid_dir, radius, num_segments, layers, (alabel[0], "mid"))
        r2 = Route(mid_p, mid_dir, point2, direction2, radius, num_segments, layers, ("mid", alabel[1]))
        for a in r1.layer_names():
            geom_1 = getattr(r1, a)
            geom_2 = getattr(r2, a)
            setattr(self, a, unary_union([geom_1, geom_2]))
        self.anchorsmod = MultiAnchor([Anchor(point1, direction1, alabel[0]), 
                                       Anchor(point2, direction2, alabel[1])])


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
        #self.skeletone = MultiLineString([self.skeletone, LineString([(r, 0), (2*r, 0)])])
        for k, width in layers.items():
            self.add_buffer(name=k, offset=width/2, cap_style='round', join_style='round', quad_segs=20)
        self.anchorsmod = MultiAnchor([Anchor((0,0), 0, alabel[0]), 
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