"""
==============================================================================
SAGA GIS algorithm provider
Initialization of SAGA algorithm provider

STRUCTURE:
    data_manager
        grids

    tools
        climate
        grids
        geostatistics
        imagery
        import_export
        projection
        shapes
        tables
        terrain_analysis

    user_tools
        hydrology
        methods

    utilities
        data_validation
        files



How to start?
First define your python environment directory where saga_cmd is stored

ENV = '.../saga-6.4.0_x64/'
from pysaga import environment as env
env.set_env(ENV)

then a message must appear showing the saga version:

SAGA Version: 6.4.0 (64 bit)


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# Import modules
from ._provider import SAGAEnvironment as _SAGAenv
from . import data_manager
from . import tools
from . import user_tools
from . import utilities

# Define environment
# environment is the system manager for SAGA algorithms from console
# environment must be defined when pysaga starts
environment = _SAGAenv()

# Set environment to access SAGA tools libraries
# environment is shared between all libraries
tools.climate._env          = environment
tools.grids._env            = environment
tools.geostatistics._env    = environment
tools.imagery._env          = environment
tools.import_export._env    = environment
tools.projection._env       = environment
tools.shapes._env           = environment
tools.tables._env           = environment
tools.terrain_analysis._env = environment

