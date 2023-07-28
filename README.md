# ZeroHeliumKit
---
ZeroHeliumKit is a Python module for planar geometry creation and manipulations designed for building microfabricated chips, and integrated electric field calculations. ZeroHeliumKit allows to create complex structure designs from simple geometries and assemble them in an intuitive Lego-style level. Geometry creation and assembly part is based on [Shapely](https://github.com/shapely/shapely) python package, which provides fast boolean operations on geometrical objects. ZeroHeliumKit enables simple integration with [Gmsh](https://gmsh.info) (mesh generation toolkit) and [FreeFEM++](https://freefem.org) (finite element solver for partial differential equations) to perform electric field calculations on designed geometrical structures. 

---

## Installing

Before cloning repository install neseccary packages:
```shell
pip install -r requirements.txt
```
Latest FreeFEM++ software can be found [here](https://github.com/FreeFem/FreeFem-sources/releases).

---

## Troubleshooting
#### Unresolved import warnings in Vscode
If you're getting a warning about an unresolved import, then create a file `.vscode/settings.json` in the workspace with the contents:
```json
{
    "python.analysis.extraPaths": ["./sources"]
}
```
Here a project uses a `sources` directory.