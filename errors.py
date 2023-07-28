class PolygonConverterError(Exception):
    """A geometry is invalid or topologically incorrect."""

class TopologyError(Exception):
    """A geometry topologically incorrect."""

class FreefemError(Exception):
    pass