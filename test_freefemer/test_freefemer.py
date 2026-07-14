from zeroheliumkit.fem.freefemer import EDPpreparer
import os

os.makedirs("/tmp/zhktest/geo", exist_ok=True)

yaml_content = """
savedir: /tmp/zhktest
meshfile: /tmp/zhktest/geo/dot.msh
physicalSurfaces:
   mid: 5
   out: 6
   top: 7
physicalVolumes:
   VACUUM: 1
   DIELECTRIC: 2
   METAL: 3
   HELIUM: 4
dielectric_constants:
   VACUUM: 1.0
   DIELECTRIC: 11.0
   METAL: 1.0
   HELIUM: 1.057
ff_polynomial: 2
adaptation_config:
  mesh_adaptation: true
  anisotropic_adaptation: false
  n_adapt: 3
  err_target: 0.01
extract_opt:
  - name: result1
    quantity: phi
    plane: xy
    coordinate1: [-10, 10, 201]
    coordinate2: [-10, 10, 201]
    coordinate3: [1.6]
    curvature_config: null
"""

with open("/tmp/zhktest/dot.yaml", "w") as f:
    f.write(yaml_content)

edps = EDPpreparer("/tmp/zhktest/dot.yaml")

with open("/tmp/zhktest/edp/ff_mid.edp", "r") as f:
    print(f.read())