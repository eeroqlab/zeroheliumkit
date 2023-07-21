import numpy as np

from shapely import Polygon, MultiPolygon
from shapely import unary_union

from .fonts import _glyph, _indentX, _indentY


def polygonize_text(text: str="abcdef", size: float=1000) -> MultiPolygon:
    """ Converts given text to a MultiPolygon object

    Args:
        text (str, optional): text in str format. Defaults to "abcdef".
        size (float, optional): defines the size of the text. Defaults to 1000.

    Returns:
        MultiPolygon: converted text into MultiPolygon object
    """

    scaling = size/1000
    xoffset = 0
    yoffset = 0
    MULTIPOLY = []

    for line in text.split("\n"):
        
        for c in line:
            ascii_val = ord(c)
            
            if c==" ":
                xoffset += 500 * scaling

            elif (33 <= ascii_val <= 126) or (ascii_val == 181):
                multipolygon = []
                for poly in _glyph.get(ascii_val):
                    coords = np.array(poly) * scaling
                    coords[:, 0] += xoffset
                    coords[:, 1] += yoffset
                    multipolygon.append(Polygon(coords))

                mpolygon = unary_union(MultiPolygon(multipolygon))
                _, _, xmax, _ = mpolygon.bounds
                xoffset = xmax + _indentX * scaling
                MULTIPOLY.append(mpolygon)
            else:
                valid_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"

                raise ValueError(
                        'Warning, no geometry for character "%s" with ascii value %s. '
                        "Valid characters: %s"
                        % (chr(ascii_val), ascii_val, valid_chars)
                    )
            
        yoffset -= _indentY * scaling
        xoffset = 0
    
    return unary_union(MULTIPOLY)