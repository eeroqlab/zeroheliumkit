class PolygonConverterError(Exception):
    """A geometry is invalid or topologically incorrect."""

class TopologyError(Exception):
    """A geometry topologically incorrect."""

class FreefemError(Exception):
    """
    Exception raised for errors related to FreeFEM.

    Attributes:
        message -- explanation of the error
    """
    pass

class RouteError(Exception):
    """
    Exception raised for errors related to routing.

    Attributes:
        message -- explanation of the error
    """
    pass

class GeometryError(Exception):
    """
    Exception raised for errors related to geometry calculations.

    Attributes:
        message -- explanation of the error
    """
    pass

class WrongSizeError(Exception):
    """
    Exception raised when an operation is performed with an incorrect size.
    """
    pass