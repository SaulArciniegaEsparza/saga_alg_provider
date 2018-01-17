"""
SAGA GIS algorithm provider
Initialization of SAGA algorithm provider

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
tools.terrain_analysis._env = environment
tools.tables._env = environment

# Set GridObj
tools.grids._io = data_manager.grids
tools.import_export._io = data_manager.grids

user_tools.hydrology._io = data_manager.grids
user_tools.methods._io = data_manager.grids

# Set utilities.files as _files
tools.climate._files = utilities.files
tools.grids._files = utilities.files
tools.import_export._files = utilities.files

user_tools.methods._files = utilities.files

# Set utilities.data_validation as _validation
tools.climate._validation = utilities.data_validation
tools.grids._validation = utilities.data_validation
tools.import_export._validation = utilities.data_validation
