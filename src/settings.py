from math import sqrt

from shapely import (LineString, MultiLineString,
                     Polygon, MultiPolygon)
from .anchors import Anchor, MultiAnchor

GRID_SIZE = 1e-4
EPS = 1e-6

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
VIOLET      = '#7671D3'
CYAN        = '#5FCFA7'
ORANGE      = '#FFA275'
LIGHTGRAY   = '#F1F1F1'

COLORS = [BLUE, YELLOW2, GRAY, GREEN, DARKGRAY, RED, BLACK]

# grouped geometry types
PTS_CLASSES   = [Anchor, MultiAnchor]
LINE_CLASSES  = [LineString, MultiLineString]
PLG_CLASSES   = [Polygon, MultiPolygon]
