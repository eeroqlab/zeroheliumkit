import functools

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

class DeprecatedAPIError(RuntimeError):
    """Raised when a hard-deprecated API is called."""

def hard_deprecated(*, alternative: str, since: str):
    """
    Immediately prevents use of the decorated function.
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to include instance identifier if possible
            if hasattr(self, "name"):
                instance_name = self.name
            else:
                instance_name = self.__class__.__name__
            
            alt_msg = f"{instance_name}{alternative}"
            msg = (f"{func.__name__}() was deprecated since {since} and is no longer available.\n"
                   f"Call {alt_msg} instead."
                   )
            raise DeprecatedAPIError(msg)
        return wrapper
    return deco