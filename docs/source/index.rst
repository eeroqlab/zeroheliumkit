.. ZeroHeliumKit documentation master file

ZeroHeliumKit
======================================


.. image:: _static/zhk.png
  :align: center
  :width: 300


ZeroHeliumKit is a Python-based framework tailored for designing planar geometries used 
in microfabricated chip structures and conducting integrated electric field computations. 
It enables you to intuitively assemble complex microdevice layouts from simple elements in 
a “Lego-style” design flow.

Geometry creation and assembly part is based on `Shapely <https://github.com/shapely/shapely>`__ 
python package, which provides fast boolean operations on geometrical objects. ZeroHeliumKit 
enables simple integration with `Gmsh <https://gmsh.info>`__ (mesh generation toolkit) and 
`FreeFEM++ <https://freefem.org>`__ (finite element solver for partial differential equations) 
to perform electric field calculations on designed geometrical structures.


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   user_manual
   examples


.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   anchors
   core
   supercore
   geometries
   utils
   functions
   plotting
   routing
   importing
   fem
