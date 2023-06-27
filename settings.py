from math import sqrt

from shapely import (Point, MultiPoint, 
                     LineString, MultiLineString, 
                     Polygon, MultiPolygon)

# plot size
GM      = (sqrt(5) - 1.0)/2.0
W       = 8.0
H       = W*GM
SIZE    = (W, H)
SIZE_L  = (1.5*W, 1.5*H)

# plot colors
BLUE        = '#6699cc'
GRAY        = '#cccccc'
DARKGRAY    = '#333333'
YELLOW      = '#ffcc33'
YELLOW2     = '#fee08d'
GREEN       = '#339933'
RED         = '#ff3333'
BLACK       = '#000000'
WHITE       = '#ffffff'

COLORS = [BLUE, YELLOW2, GRAY, GREEN, DARKGRAY]

# grouped geometry types
PTS_CLASSES   = [Point, MultiPoint]
LINE_CLASSES  = [LineString, MultiLineString]
PLG_CLASSES   = [Polygon, MultiPolygon]

# constants
hbar = 1.0546 * 1e-34
qe = 1.602 * 1e-19
me = 9.109 * 1e-31
l0 = 1e-6
Escale = hbar**2/2/me/l0**2