"""
SAGA GIS algorithm provider
Initialization of SAGA algorithm provider

STRUCTURE:
    data_manager
        grids

    tools
        climate
        grids
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


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
from _provider import SAGAEnvironment as _SAGAenv
import data_manager
import tools
import user_tools
import utilities

# Define environment
environment = _SAGAenv()

# Set environment
tools.climate._env = environment
tools.grids._env = environment
tools.import_export._env = environment
tools.projection._env = environment
tools.shapes._env = environment
tools.tables._env = environment
tools.terrain_analysis._env = environment

# Set GridObj
tools.grids._io = data_manager.grids
tools.import_export._io = data_manager.grids
tools.shapes._io = data_manager.grids

user_tools.hydrology._io = data_manager.grids
user_tools.methods._io = data_manager.grids

# Set utilities.files as _files
tools.climate._files = utilities.files
tools.grids._files = utilities.files
tools.import_export._files = utilities.files
tools.projection._files = utilities.files
tools.shapes._files = utilities.files
tools.tables._files = utilities.files
tools.terrain_analysis._files = utilities.files

user_tools.methods._files = utilities.files
data_manager.grids._files = utilities.files

# Set utilities.data_validation as _validation
tools.climate._validation = utilities.data_validation
tools.grids._validation = utilities.data_validation
tools.import_export._validation = utilities.data_validation
tools.projection._validation = utilities.data_validation
tools.shapes._validation = utilities.data_validation
tools.tables._validation = utilities.data_validation
tools.terrain_analysis._validation = utilities.data_validation

user_tools.hydrology._validation = utilities.data_validation

# Set projection methods
data_manager.grids._files = utilities.files
data_manager.grids._crs_from_epsg = tools.projection.crs_from_epsg

