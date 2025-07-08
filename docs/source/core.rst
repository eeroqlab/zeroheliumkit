Core.py
===========

The following classes exist in the Core module for geometry handling:

.. list-table:: Core.py
   :widths: 25 50
   :header-rows: 0

   * - :doc:`Base`
     - Base class that handles all low-level operations on layers and colors.
   * - :doc:`Entity`
     - Represents a collection of shapely objects linked together and provides methods for geometrical operations.
   * - :doc:`Structure`
     - A subclass of Entity that represents layers with collections of geometries (Points, LineStrings, Polygons, etc.).
   * - :doc:`GeomCollection`
     - A subclass of Structure that represents a collection of geometries.