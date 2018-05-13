"""
SAGA GIS algorithm provider
Terrain Analysis tools
    ta_channels
    ta_hydrology
    ta_lighting
    ta_morphometry
    ta_preprocessor


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# env is the provider class
import sys as _sys
import os as _os
from copy import deepcopy as _deepcopy
import shapefile as _shp
import numpy as _np

from . import tables as _tables
from . import shapes as _shapes
from . import grids as _grids

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: ta_channels
# ==============================================================================


def channel_network(dem, flowdir=None, init_grid=None, channels=None,
                    gridchannels=None, gridchandir=None, init_value=100,
                    init_method=2, min_len=10):
    """
    Derives a channel network based on gridded digital elevation data

    library: ta_channels  tool: 0

    INPUTS
     dem            [string] input dem grid (.sgrd or .tif)
     flowdir        [string] optional input flow direction grid (.sgrd or .tif)
     init_grid      [string] optional input conditional grid for initial flow (.sgrd or .tif)
     channels       [string] optional output shape file of channel network (.sgrd)
     gridchannels   [string] optional output grid of channel network (.sgrd)
     gridchandir    [string] optional output grid of channel network flow direction (.sgrd)
     init_value     [int, float] value for initiate flow from init_grid
     init_method    [int] method for initiate flow using init_grid
                     [0] Less than
                     [1] Equals
                     [2] Greater than (default)
     min_len        [int, float] minimum longitude of channel segments
    """
    # Wrong output parameters
    if channels is None and gridchannels is None and gridchandir is None:
        print('At least an output file must be especified!')
        return (False)
    # Check inputs
    dem = _validation.input_file(dem, 'grid', False)
    if channels is None:  # check channels
        channels = 'NULL'
    else:
        flowdir = _validation.input_file(flowdir, 'grid', False)
    if channels is None:  # check channels
        channels = 'NULL'
    else:
        channels = _validation.output_file(channels, 'vector')
    if gridchannels is None:  # check gridchannels
        gridchannels = 'NULL'
    else:
        gridchannels = _validation.output_file(gridchannels, 'grid')
    if gridchandir is None:  # check gridchandir
        gridchandir = 'NULL'
    else:
        gridchandir = _validation.output_file(gridchandir, 'grid')
    if init_grid is None:  # check init_grid
        init_grid = 'NULL'
    else:
        init_grid = _validation.input_file(init_grid, 'grid', False)
    if init_method < 0 or init_method > 2:  # set default init_method
        init_method = 2
    # convert to strings
    init_value = str(init_value)
    init_method = str(init_method)
    min_len = str(min_len)
    # Create cmd
    cmd = ['saga_cmd', 'ta_channels', '0', '-SHAPES', channels, '-CHNLNTWRK',
           gridchannels, '-CHNLROUTE', gridchandir, '-ELEVATION', dem, '-SINKROUTE',
           flowdir, '-INIT_GRID', init_grid, '-INIT_METHOD', init_method,
           '-INIT_VALUE', init_value, '-MINLEN', min_len]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [channels, gridchannels, gridchandir])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def watershed_basins(outgrid, dem, channels, minsize=0, sinkroute=None):
    """
    Watershed Basins delimitation on a dem grid

    library: ta_channels  tool: 1

    INPUTS
     outgrid     [string] output basins grid file (.sgrd)
     dem         [string] input dem grid (.sgrd or .tif)
     channels    [string] input channel network grid (.sgrd or .tif)
     minsize     [int, float] minimum size of basins
     sinkroute   [string] sink route grid for flat terrain
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    channels = _validation.input_file(channels, 'grid', False)
    if sinkroute is None:
        sinkroute = 'NULL'
    else:
        sinkroute = _validation.input_file(sinkroute, 'grid', False)
    # restringe minsize
    if minsize < 0:
        minsize = 0
    minsize = str(minsize)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_channels', '1', '-ELEVATION', dem,
           '-CHANNELS', channels, '-SINKROUTE', sinkroute, '-BASINS',
           outgrid, '-MINSIZE', minsize]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outgrid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: ta_hydrology
# ==============================================================================


def flow_accumulation(outgrid, dem, method=0, unit=0, sinkroute=None,
                      material=None, weights=None):
    """
    Top-down processing of cells for calculation of flow accumulation

    library: ta_hydrology  tool: 0

    INPUTS
     outgrid     [string] output flow accumulation grid (.sgrd)
     dem         [string] input dem grid (.sgrd or .tif)
     method      [int] method for flow direction estimation
                  [0] Deterministic 8
                  [1] Rho 8
                  [2] Braunschweiger Reliefmodell
                  [3] Deterministic Infinity
                  [4] Multiple Flow Direction
                  [5] Multiple Triangular Flow Directon
                  [6] Multiple Maximum Downslope Gradient Based Flow Directon
     unit        [int] accumulation cell unit
                  [0] number of cells
                  [1] cell area
     sinkroute   [string] optional input grid with sink route, only if dem is
                  not corrected
     material    [srting] optional input grid with accumulation material
     weights     [srting] optional input grid with cells weights
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    if sinkroute is None:  # check sinkroute
        sinkroute = 'NULL'
    else:
        sinkroute = _validation.input_file(sinkroute, 'grid', False)
    if material is None:  # check material
        material = 'NULL'
    else:
        material = _files.default_file_ext(material, 'grid', False)
    if weights is None:  # check weights
        weights = 'NULL'
    else:
        weights = _validation.input_file(weights, 'grid', False)
    # Check input parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 6], dtypes=[int])
    unit = _validation.input_parameter(unit, 0, vrange=[0, 1], dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', 'ta_hydrology', '0', '-FLOW', outgrid, '-ELEVATION', dem,
           '-METHOD', method, '-FLOW_UNIT', unit, '-SINKROUTE', sinkroute,
           '-ACCU_MATERIAL', material, '-WEIGHTS', weights]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outgrid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def upslope_area(output, points, dem, sinkroute=None, field=None,
                 method=0, converge=1.1, single_area=False):
    """
    Delimitation of the upslope contributing area for a point or a cloud of points

    library: ta_hydrology  tool: 4

    INPUTS
     output      [string] output grid of upslope area. When single_area is True, only
                  a single grid is returned. When single_area is False and more than
                  one point is input, multiple grids are returned and output is used
                  as a base name (.sgrd)
     points      [list,array,string] input points. Input points can be a [x,y]
                  list, tuple or array for a single point. For multiple points
                  insert a [[x1,y1], [x2,y2],..,[xn,yn]] list/tuple/array.
                  A points shape file can be used as input
                  file of points
     dem         [string] input dem file (.sgrd or .tif)
     sinkroute   [string] optional input sink routes (.sgrd or .tif)
     field       [string,int] if points is a string, the column values of the
                  attribute tables is used as outgrid names. If field is int
                  then the column id is used
     method      [int] flow direction method
                  [0] Deterministic 8
                  [1] Deterministic Infinity
                  [2] Multiple Flow Direction
     converge    [int,float] threshold value for method=2
     single_area [bool] if True, a single area is returned combining all
                  upslope areas for input points. In other case, upslope
                  area for each grid is returned. If True points must be
                  a shapefile
    """
    # Check inputs
    output = _validation.output_file(output, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    if type(sinkroute) is str:
        sinkroute = _validation.input_file(sinkroute, 'grid', False)
    else:
        sinkroute = 'NULL'
    # Check parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 2], dtypes=[int])
    converge = str(converge)

    if single_area:  # single upslope area
        if type(points) is str:
            # Check points shapefile
            points = _files.default_file_ext(points, 'vector')
            # Temporal grid points
            temporal_grid = _files.create_filename(_env.workdir, 'sgrd', 'points_temporal')
            _grids.shapes_to_grid(temporal_grid, points, grid_extent=dem)
            # Create cmd
            cmd = ['saga_cmd', 'ta_hydrology', '4', '-ELEVATION', dem, '-SINKROUTE',
                   sinkroute, '-METHOD', method, '-CONVERGE', converge, '-AREA',
                   output, '-TARGET', temporal_grid]
            flag = _env.run_command_logged(cmd)
            # Check if output grid has crs file
            _validation.validate_crs(dem, [output])

        else:
            raise TypeError('points must be a shapefile!')

    else:            # multiple upslope areas
        # Create cmd incomplete
        cmd = ['saga_cmd', 'ta_hydrology', '4', '-ELEVATION', dem, '-SINKROUTE',
               sinkroute, '-METHOD', method, '-CONVERGE', converge]
        # Check points and create output names
        path = _os.path.dirname(output)
        base = _os.path.basename(output).split('.')[0]
        if type(points) in [list, tuple, _np.ndarray]:  # points is a list of a point
            points = _np.array(points, dtype=_np.float32)
            if points.ndim == 1 and points.size == 2:
                output = _validation.output_file(output, 'grid')
                x, y = str(points[0]), str(points[1])
                # run command
                cmd.extend(['-AREA', output, '-TARGET_PT_X', x, '-TARGET_PT_Y', y])
                flag = _env.run_command_logged(cmd)
            elif points.ndim == 2 and points.shape[1] == 2:  # points is a list of points
                for i in range(points.shape[0]):
                    output = _os.path.join(path, base + '_' + str(i) + '.sgrd')
                    x, y = str(points[i][0]), str(points[i][1])
                    # run command
                    cmdf = _deepcopy(cmd)
                    cmdf.extend(['-AREA', output, '-TARGET_PT_X', x, '-TARGET_PT_Y', y])
                    flag = _env.run_command_logged(cmdf)
            else:
                raise TypeError('Wrong input points argument!')
        elif type(points) is str:  # points is a shapefile
            points = _files.default_file_ext(points, 'vector')
            layer = _shp.Reader(points, 'r')
            if layer.shapeType == 1:  # verify if shapefiles is a point type goemetry
                shapes = layer.shapes()
                nshapes = len(shapes)  # number of shapes
                # get atribute field name
                table = _tables.get_attribute_table(points)  # get attribute table
                if field is not None:  # look for a field
                    if type(field) is int:
                        columns = table.columns
                        field = columns[field]
                    labels = table[field].values
                    labels = [str(label) for label in labels]

                if field is None:  # field is None, create
                    labels = [str(label) for label in range(nshapes)]

                # iterate over shapes
                for i in range(nshapes):
                    output = _os.path.join(path, base + '_' + labels[i] + '.sgrd')
                    x, y = str(shapes[i].points[0][0]), str(shapes[i].points[0][1])
                    # run command
                    cmdf = _deepcopy(cmd)
                    cmdf.extend(['-AREA', output, '-TARGET_PT_X', x, '-TARGET_PT_Y', y])
                    flag = _env.run_command_logged(cmdf)

            else:
                layer = None  # close shape connection
                raise TypeError('The shape file is not a point type!')

            layer = None  # close shape connection

        else:
            raise TypeError('Wrong input points argument!')

    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: ta_lighting
# ==============================================================================


def analytical_hillshading(outgrid, dem, method=0, azimuth=315, declination=45,
                           exaggeration=4, shadow=1, ndirs=8, radius=10):
    """
    Analytical hillshading calculation

    library: ta_lighting  tool: 1

    INPUTS
     outgrid      [string] output hillshading grid (.sgrd)
     dem          [string] input grid dem (.sgrd o .tif)
     method       [int] shading method
                   [0] Standard (default)
                   [1] Standard (max. 90Degree)
                   [2] Combined Shading
                   [3] Ray Tracing
                   [4] Ambient Occlusion
     azimuth      [float] direction of the light source, measured in degree
                   clockwise from the North direction. Only for method=0,1,2,3
     declination  [float] height of the light source, measured in degree above
                   the horizon. Only for method=0,1,2,3
     exaggeration [float] the terrain exaggeration factor allows one to increase
                   the shading contrasts in flat areas. Only for method=0,1,2,3
     shadow       [int] shadow method (Only for method=3)
                   [0] slim: to trace grid node's shadow
                   [1] fat (default): to trace the whole cell's shadow
     ndirs        [int] number of directions. Only for method=4
     radius       [float] search radius. Only for method=4
    """
    # Check inputs
    # dem = self.default_file_ext(dem, 'grid')
    outgrid = _validation.output_file(outgrid, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    method, shadow, ndirs = int(method), int(shadow), int(ndirs)
    method = _validation.input_parameter(method, 0, vrange=[0, 4], dtypes=[int])
    shadow = _validation.input_parameter(shadow, 1, vrange=[0, 1], dtypes=[int])
    ndirs = _validation.input_parameter(ndirs, 2, gt=2, dtypes=[int])
    # convert to string
    method, shadow = str(method), str(shadow)
    azimuth, declination = str(azimuth), str(declination)
    exaggeration, ndirs, radius = str(exaggeration), str(ndirs), str(radius)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_lighting', '0', '-ELEVATION', dem, '-SHADE',
           outgrid, '-METHOD', method, '-AZIMUTH', azimuth, '-DECLINATION',
           declination, '-EXAGGERATION', exaggeration, '-SHADOW', shadow,
           '-NDIRS', ndirs, '-RADIUS', radius]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outgrid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: ta_morphometry
# ==============================================================================


def dem_slope_aspect(dem, outslope, outaspect, method=6, sunits=0, aunits=0):
    """
    Calculates the local morphometric terrain parameters slope and aspect

    library: ta_morphometry  tool: 0

    INPUTS
     dem          [string] input grid dem (.sgrd o .tif)
     outslope     [string] output slope (.sgrd)
     outaspect    [string] output aspect (.sgrd)
     method       [int] method for slope and aspect calculation. Default 6
                   [0] maximum slope (Travis et al. 1975)
                   [1] maximum triangle slope (Tarboton 1997)
                   [2] least squares fitted plane (Horn 1981, Costa-Cabral & Burgess 1996)
                   [3] 6 parameter 2nd order polynom (Evans 1979)
                   [4] 6 parameter 2nd order polynom (Heerdegen & Beran 1982)
                   [5] 6 parameter 2nd order polynom (Bauer, Rohdenburg, Bork 1985)
                   [6] 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
                   [7] 10 parameter 3rd order polynom (Haralick 1983)
     sunits       [int] slope units
                   [0] radians (default)
                   [1] degree
                   [2] percent
     aunits       [int] aspect units
                   [0] radians
                   [1] degree
    """
    # Check inputs
    outslope = _validation.output_file(outslope, 'grid')
    outaspect = _validation.output_file(outaspect, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    # Check parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 7], dtypes=[int])
    sunits = _validation.input_parameter(sunits, 0, vrange=[0, 2], dtypes=[int])
    aunits = _validation.input_parameter(aunits, 0, vrange=[0, 1], dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_morphometry', '0', '-ELEVATION', dem, '-SLOPE',
           outslope, '-ASPECT', outaspect, '-METHOD', method, '-UNIT_SLOPE',
           sunits, '-UNIT_ASPECT', aunits]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outslope, outaspect])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: ta_preprocessor
# ==============================================================================


def sink_drainage_route_detection(outgrid, dem, threshold=0):
    """
    Detects flow direction in sinks

    library: ta_preprocessor  tool: 1

    INPUTS
     outgrid     [string] output grid sink drainage route (.sgrd)
     dem         [string] input grid dem (.sgrd or .tif)
     threshold   [int, float] if threshold=0 this parameter is ignored. The
                  parameter describes the maximum depth of a sink to be considered
                  for removal [map units]. This makes it possible to exclude
                  deeper sinks from filling.
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    if threshold != 0:
        op = 1
    else:
        op = 0
    op, threshold = str(op), str(threshold)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_preprocessor', '1', '-ELEVATION', dem,
           '-SINKROUTE', outgrid, '-THRESHOLD', op, '-THRSHEIGHT', threshold]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outgrid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def sink_removal(outdem, dem, sinkroute=None, method=0, threshold=0):
    """
    Remove sinks from a digital elevation model

    library: ta_preprocessor  tool: 2

    INPUTS
     outgrid     [string] output dem that has to be processed (.sgrd)
     dem         [string] input grid dem (.sgrd or .tif)
     sinkroute   [string] optional sink routes grid (.sgrd or .tif)
     method      [int] sinks removal method
                  [0] Deepen Drainage Routes (default)
                  [1] Fill Sinks
     threshold   [int, float] if threshold=0 this parameter is ignored. The
                  parameter describes the maximum depth of a sink to be considered
                  for removal [map units]. This makes it possible to exclude
                  deeper sinks from filling.
    """
    # Check inputs
    outdem = _validation.output_file(outdem, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    if sinkroute is not None:
        sinkroute = _validation.output_file(sinkroute, 'grid')
    else:
        sinkroute = 'NULL'
    # Check parameters
    method = _validation.input_parameter(method, 0, vrange=[0, 1], dtypes=[int])
    if threshold != 0:
        op = 1
    else:
        op = 0
    op, threshold, method = str(op), str(threshold), str(int(method))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_preprocessor', '2', '-DEM', dem, '-SINKROUTE',
           sinkroute, '-DEM_PREPROC', outdem, '-METHOD', method, '-THRESHOLD',
           op, '-THRSHEIGHT', threshold]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outdem])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def fill_sinks_wangliu(dem, outdem=None, outflowdir=None, outwshed=None,
                       minslope=0.1):
    """
    Fill sinks and calculate flow direction and watersheds areas with the
    Wang and Liu method
    The method was enhanced to allow the creation of hydrologic sound elevation
    models, i.e. not only to fill the depression(s) but also to preserve a
    downward slope along the flow path.

    library: ta_preprocessor  tool: 4

    INPUTS
     dem          [string] input dem file (.sgrd or .tif)
     outdem       [string] output filled dem (.sgrd)
     outflowdir   [string] output flow direction grid (.sgrd)
     outwshed     [string] output delineated watersheds basins grid (.sgrd)
     minslope     [float] Minimum slope gradient to preserve from cell to cell
                   with a value of zero sinks are filled up to the spill
                   elevation (which results in flat areas). Unit [Degree]
    """
    # Check inputs
    dem = _validation.input_file(dem, 'grid', False)
    if outdem is None:
        outdem = 'NULL'
    else:
        outdem = _validation.output_file(outdem, 'grid')
    if outflowdir is None:
        outflowdir = 'NULL'
    else:
        outflowdir = _validation.output_file(outflowdir, 'grid')
    if outwshed is None:
        outwshed = 'NULL'
    else:
        outwshed = _validation.output_file(outwshed, 'grid')
    minslope = str(minslope)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_preprocessor', '4', '-ELEV', dem,
           '-FILLED', outdem, '-FDIR', outflowdir, '-WSHED', outwshed,
           '-MINSLOPE', minslope]
    # Run
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outdem, outflowdir, outwshed])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def fill_sinks_wangliuXXL(outdem, dem, minslope=0.1):
    """
    Fill sinks  for large data sets using an algorithm proposed by Wang & Liu
    to identify and fill surface depressions in digital elevation models.
    The method was enhanced to allow the creation of hydrologic sound elevation
    models, i.e. not only to fill the depression(s) but also to preserve a
    downward slope along the flow path.

    library: ta_preprocessor  tool: 5

    INPUTS
     outgrid     [string] output dem that has to be processed (.sgrd)
     dem         [string] input grid dem (.sgrd or .tif)
    """
    # Check inputs
    outdem = _validation.output_file(outdem, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    minslope = str(minslope)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_preprocessor', '5' '-ELEV', dem, '-FILLED',
           outdem, '-MINSLOPE', minslope]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outdem])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def burn_stream_network_into_dem(outdem, dem, streams, flowdir=None,
                                 method=0, epsilon=1.0):
    """
    Burns a stream network into a Digital Elevation Model (DEM). Stream cells
    have to be coded with valid data values, all other cells should be set to
    no data value. First two methods decrease. The third method ensures a
    steady downstream gradient. An elevation decrease is only applied, if a
    downstream cell is equally high or higher. You should provide a grid with
    flow directions for determination of downstream cells. The self.sink_dainage_route()
    tool offers such flow directions.

    library: ta_preprocessor  tool: 6

    INPUTS
     outdem      [string] output processed grid dem (.sgrd)
     dem         [string] input grid dem (.sgrd or .tif)
     streams     [string] input grid os hapefile streams network (.sgrd or .tif).
                  If streams is a shapefile it must have .shp extension
     flowdir     [string] optional flow direction. Only applies if method=2
     method      [int] burn stream network method
                  [0] simply decrease cell's value by epsilon (default)
                  [1] lower cell's value to neighbours minimum value minus epsilon
                  [2] trace stream network downstream
     epsilon     [float] epsilon parameter for method=0 and 1
    """
    # Check inputs
    outdem = _validation.output_file(outdem, 'grid')
    dem = _validation.input_file(dem, 'grid', False)
    streams_poly = None
    if streams.endswith('.shp'):
        streams_poly = streams
        streams = 'auxiliar_stream.sgrd'
        if _env.workdir is not None:
            streams = _os.path.join(_env.workdir, streams)
        _grids.shapes_to_grid(streams, streams_poly, grid_extent=dem)
    else:
        streams = _validation.input_file(streams, 'grid', False)
    if flowdir is not None:
        flowdir = _validation.input_file(flowdir, 'grid', False)
    else:
        flowdir = 'NULL'
    if method < 0 or method > 2:
        method = 0
    method = str(int(method))
    epsilon = str(epsilon)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'ta_preprocessor', '6', '-DEM', dem, '-BURN',
           outdem, '-STREAM', streams, '-METHOD', method, '-EPSILON', epsilon,
           '-FLOWDIR', flowdir]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete auxuliar layers
    if streams_poly is not None:
        _files.delete_files(streams)
    # Check if output grid has crs file
    _validation.validate_crs(dem, [outdem])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def dem_carve(outgrid, ingrid, inlines, width=10, depth=1):
    """
    Takes a line shapefile and transforms it to raster and subtracts depth
    from the output grid

    library: user defined

    INPUT
     outgrid     [string] output grid dem with extracted line depth (.sgrd)
     ingrid      [string] input dem (.sgrd or .tif)
     inlines     [string] input lines shapefile
     width       [float] input river width
     depth       [float] inlines depth
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    inlines = _files.default_file_ext(inlines, 'vector', False)

    # convert to strings
    width, depth = str(width), str(depth)
    # Apply grid_buffer to lines
    auxshape = 'auxiliar_shape_lines.shp'
    if _env.workdir is not None:
        auxshape = _os.path.join(_env.workdir, auxshape)
    flag = _shapes.shapes_buffer(auxshape, inlines, dist=width, dissolve=True)
    # Rasterize grid_buffer
    auxgrid = 'auxiliar_grid_river.sgrd'
    if _env.workdir is not None:
        auxgrid = _os.path.join(_env.workdir, auxgrid)
    flag = _grids.shapes_to_grid(auxgrid, auxshape, value_method=0, grid_extent=ingrid)
    # Cumpute dem difference
    flag = _grids.calculator(outgrid, [ingrid, auxgrid], use_nodata=True,
                             formula='g1 - g2 * {}'.format(depth))
    # Delete auxiliar files
    _files.delete_files([auxshape, auxgrid])
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))

