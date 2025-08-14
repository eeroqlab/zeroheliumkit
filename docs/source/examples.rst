Examples
============

Explore ZeroHeliumKit's functionality by running each cell to test out the code.
Below is an example of how to create geometries, configure meshing, and run FreeFem++ calculations.


Basic Geometry and Layer Definition
------------------------------------

Start by importing the required packages and defining the geometry.

.. code-block:: python

    plot_config = {
        "gnd": BLUE,
        "etch": GRAY,
        "top": ORANGE,
        "bottom": YELLOW2,
        "opening": GREEN,
        "air": GRAY,
        "skeletone": WHITE,
        "anchors": RED
    }

    save_dir = "dump/"
    config_dir = "config/"

    device = GeomCollection(layers=Reader_Pickle(save_dir+"trap_v2_7.pickle").geometries)
    device.add_layer("dielectric", Square(20))
    device.cut_polygon("dielectric", device.etch)
    device.add_polygon("top", device.dielectric)
    device.add_layer("wafer", Square(20))
    device.colors = ColorHandler(plot_config)

    device.quickplot(show_idx=True)

    d_wafer = 12.5
    d_metal1 = 0.113
    d_metal2 = 0.057
    d_diel  = 0.717 - d_metal2 + d_metal1
    d_vac   = 20
    d_He    = d_diel + d_metal2


Geometry Extrusion Configuration
---------------------------------

Define the vertical extrusion layers and associate them with physical volumes for meshing.

.. code-block:: python

    extrude_config = {
        'wafer':        gmshLayer_info('wafer', -d_wafer, d_wafer, 'DIELECTRIC'),
        'trap':         gmshLayer_info('bottom', 0, d_metal1, 'METAL'),
        'dielectric':   gmshLayer_info('dielectric', 0, d_diel, 'DIELECTRIC'),
        'top':          gmshLayer_info('top', d_diel, d_metal2, 'METAL'),
        'helium':       gmshLayer_info('wafer', 0, d_He, 'HELIUM', ('trap', 'dielectric', 'top')),
        'vacuum':       gmshLayer_info('wafer', d_He, d_vac, 'VACUUM', ('dielectric', 'top'))
    }


Electrode Configuration
------------------------

Specify which surfaces act as electrodes and group them accordingly.

.. code-block:: python

    electrodes_config = {
        'trap':         physSurface_info('bottom', [0], 'trap'),
        'resU':         physSurface_info('top', [9], 'top'),
        'resD':         physSurface_info('top', [6], 'top'),
        'splitgateD':   physSurface_info('top', [2], 'top'),
        'splitgateU':   physSurface_info('top', [3], 'top'),
        'unload':       physSurface_info('top', [8], 'top'),
        'gnd':          physSurface_info('top', [0,1,4,5,7,10], 'top'),
    }


Meshing
--------

Define meshing parameters and create the mesh using GMSHmaker.

.. code-block:: python

    scale = 4

    mesh_cfg = [
        {"Thickness": 4, "VIn": scale * 0.6/1, "VOut": 2, "box": [-11, 11, -5, 5, -5, 5]},
        {"Thickness": 2, "VIn": scale * 0.15/1, "VOut": 2, "box": [-4, 4, -3.5, 3.5, -1, 3]}
    ]

    meshMKR = GMSHmaker(
        layout=device,
        extrude_config=extrude_config,
        electrodes_config=electrodes_config,
        mesh_params=mesh_cfg,
        filename="dot"
    )
    meshMKR.disable_consoleOutput()
    meshMKR.create_mesh(dim=3)
    meshMKR.finalize()

    He_level = d_He


FreeFem++
----------------------

Set up FreeFem run and execute simulation

.. code-block:: python
    
   var_eps = {
        'DIELECTRIC': 11.0,
        'METAL': 1.0,
        'HELIUM': 1.057,
        'VACUUM': 1.0
    }

    ffc = FFconfigurator(
        config_file="config/dot.yaml",
        dielectric_constants=var_eps,
        ff_polynomial=2,
        extract_opt=[
            ExtractConfig('ONE', 'phi', 'xy', (-1.5,1.5,5), (-1.5,1.5,5), coordinate3=0.83),
            ExtractConfig('TWO', 'Ey', 'xy', (-1.5,1.5,5), (-1.5,1.5,5), coordinate3=[1, 2, 3]),
            ExtractConfig('THREE', 'phi', 'xy', (-1.5,1.5,5), (-1.5,1.5,5), coordinate3=0.5)
        ]
    )

    pyff = FreeFEM(config_file="config/dot.yaml", run_from_notebook=True)
    await pyff.run(cores=4, print_log=False, single_data_file=True, freefem_path=":/path/to/FreeFem++")