"""
==============================================================================
SAGA GIS algorithm provider
Spatial and Geostatistics tools:
    statistics_grid
    statistics_kriging
    statistics_points
    statistics_regression



Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# env is the provider class
# import modules
import sys as _sys
import os as _os
import numpy as _np
import pandas as _pd

from ..utilities import files as _files
from ..utilities import data_validation as _validation

_Frame = type(_pd.DataFrame())  # get pandas DataFrame Type
_Serie = type(_pd.Series())     # get pandas Serie Type

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: statistics_grid
# ==============================================================================

def satatistics_for_grids(outfolder, basename, grids, stats='mean', pctl=50,
                          weights=None, resampling=3):
    """
    Calculates statistical properties (arithmetic mean, minimum, maximum, variance,
    standard deviation) for each cell position for the values of the selected grids.
    Optionally you can supply a list of grids with weights (Only for SAGA > 4.0).
    If you want to use weights, the number of value and weight grids have to be the
    same Value and weight grids are associated by their order in the lists. Weight
    grids have not to share the grid system of the value grids. In case that no weight
    can be obtained from a weight grid for value, that value will be ignored.


    library: statistics_grid  tool: 4


    INPUTS:
        outfolder       [string] output folder to save results
        basename        [string] basename set to all output grids
        grids           [list, tuple] input grids used to compute statistics
        stats           [string, list, tuple] statistic or list of statistics to compute
                          Valid statistics are: 'mean' (default), 'min', 'max', 'range', 'sum',
                          'var', 'std', 'stdl' (mean-std), 'stdu' (mean+std), 'pctl'
        pctl            [int] percentile to compute [0,100], by default 50. Only if pctl is
                          input in stats parameter
        weights         [list, tuple] list of grids to use as weights. Must be equal length than
                          grids. Only available for SAGA versions from 4 to current
        resampling      [int] resampling method for weights
                          [0] Nearest Neighbour
                          [1] Bilinear Interpolation
                          [2] Bicubic Spline Interpolation
                          [3] B-Spline Interpolation
    """
    # Check input files
    if type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('Wrong grids argument type!')

    if type(weights) in (list, tuple):
        if len(weights) != len(grids):
            raise ValueError('grids and weights must have the same length!')
        weights = _validation.input_file(weights, 'grid', False)
        weights_list = ';'.join(weights)

    # Check base name
    basefile = _os.path.join(outfolder, basename)

    # Check statistics
    if type(stats) is str:
        stats = [stats]
    elif type(stats) not in (tuple, list):
        raise TypeError('Input stats parameter must be a list/tuple or a string')
    stats_names = ['MEAN', 'MIN', 'MAX', 'RANGE', 'SUM', 'VAR', 'STD',
                   'STDL', 'STDU', 'PCTL']
    stats_list = ['MEAN', 'MIN', 'MAX', 'RANGE', 'SUM', 'VAR', 'STDDEV',
                  'STDDEVLO', 'STDDEVHI', 'PCTL']
    output_stats = []
    for key in stats:
        if key.upper() in stats_names:
            ids = stats_names.index(key.upper())
            output_stats.extend([
                '-' + stats_list[ids],
                '{}_{}.sgrd'.format(basefile, key.lower())
            ])
        else:
            raise ValueError('Statistic {} is not available!'.format(key))

    # Check inputs
    pctl = _validation.input_parameter(pctl, 50, vrange=[0, 100], dtypes=[int])
    resampling = _validation.input_parameter(resampling, 50, vrange=[0, 100], dtypes=[int])

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'statistics_grid', '4', '-GRIDS', grids_list, '-PCTL', pctl]
    cmd.extend(output_stats)
    if int(_env.saga_version[0]) >= 4 and weights is not None:  # only for SAGA version upper to 4
        cmd.extend(['-WEIGHTS', weights_list, '-RESAMPLING', resampling])

    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))


def zonal_grid_statistics(stats, zones, categories=None, grids=None,
                          aspect=None, shortnames=True):
    """
    The tool calculates zonal statistics and reports these in a table. The tool can
    be used to create a contingency table of unique condition units (UCUs).
    These units are delineated from a zonal grid (e.g. sub catchments) and optional
    categorical grids (e.g. landcover, soil, ...).
    The tool has four different modes of operation:
    (1) only a zonal grid is used as input. This results in a simple contingency table
        with the number of grid cells in each zone.
    (2) a zonal grid and additional categorical grids are used as input. This results
        in a contingency table with the number of cells in each UCU.
    (3) a zonal grid and additional grids with continuous data are used as input. This
        results in a contingency table with the number of cells in each zone and some
        simple statistics for each zone. The statistics are calculated for each continuous grid.
    (4) a zonal grid, additional categorical grids and additional grids with continuous
        data are used as input. This results in a contingency table with the number of
        cells in each UCU and the corresponding statistics for each continuous grid.


    library: statistics_grid  tool: 5


    INPUTS:
        stats           [string] output statistics table
        zones           [string] input zonal grid
        categories      [string, list, tuple] optional categorical grids
        grids           [string, list, tuple] optional grids list with continuous data
        aspect          [string] optional aspect grid
        shortnames      [bool] if True, short names are used, By default False
    """
    # Check inputs
    zones = _validation.input_file(zones, 'grid', False)

    if not (stats.endswith('.txt') or stats.endswith('.csv')):
        stats += '.csv'

    if categories is None:
        category_list = 'NULL'
    elif type(categories) is str:
        categories = [_validation.input_file(categories, 'grid', False)]
        category_list = categories[0]
    elif type(categories) in (list, tuple):
        categories = _validation.input_file(categories, 'grid', False)
        category_list = ';'.join(categories)
    else:
        raise TypeError('Wrong argument type to categories!')

    if grids is None:
        grids_list = 'NULL'
    elif type(grids) is str:
        grids = [_validation.input_file(grids, 'grid', False)]
        grids_list = grids[0]
    elif type(grids) in (list, tuple):
        grids = _validation.input_file(grids, 'grid', False)
        grids_list = ';'.join(grids)
    else:
        raise TypeError('Wrong argument type to grids!')

    if aspect is None:
        aspect = 'NULL'
    elif type(aspect) is str:
        aspect = _validation.input_file(zones, 'grid', False)
    else:
        raise TypeError('Wrong argument type to grids!')

    # Check inputs
    shortnames = str(int(shortnames))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'statistics_grid', '5', '-ZONES', zones,
           '-CATLIST', category_list, '-STATLIST', grids_list, '-ASPECT',
           aspect, '-OUTTAB', stats, '-SHORTNAMES', shortnames]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().f_code.co_name, _env.errlog))

