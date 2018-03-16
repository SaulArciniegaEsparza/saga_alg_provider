"""
SAGA GIS algorithm provider
Import and Export tools
    io_gdal
    io_grid


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import sys as _sys
import os as _os

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: io_gdal
# ==============================================================================


def gdal_import_raster(outgrid, inraster, transform=True, method=3):
    """
    Convert a gdal supported raster format to saga grid format

    library: io_gdal  tool: 0

    INPUTS
     outgrid       [string] output grid file name or basename for multiple bands
     inraster      [string] input raster file name
     transform     [bool] activate/desactivate grid resampling
     method        [int] resampling method
                    [0] Nearest Neighbour
                    [1] Bilinear Interpolation
                    [2] Bicubic Spline Interpolation
                    [3] B-Spline Interpolation (default)
    """
    # Check inputs
    outgrid = os.path.splitext(outgrid)[0]
    method = _validation.input_parameter(method, 0, vrange=[0, 3], dtypes=[int])
    transform = str(int(transform))
    method = str(method)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '0', '-FILES', inraster, '-GRIDS', outgrid,
           '-TRANSFORM', transform, '-RESAMPLING', method, '-SELECTION', '0']
    if _env.saga_version[0] not in ['2', '3', '4']:
        cmd.extend(['-MULTIPLE', '0'])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def gdal_export_geotiff(outraster, ingrid, op=None):
    """
    Export multiple grids in a geotiff raster

    library: io_gdal  tool: 2

    INPUTS
     outraster     [string] output geotiff file (.tif)
     ingrid        [string, list] if ingrid is a string, only a grid is exported
                    if ingrid is a list, multiple grids are added to the output
                    raster file
     op            [string] Creation options must contain a space separated list
                    of key-value pairs (K=V)
    """
    # Check inputs
    outraster = _validation.output_file(outraster, 'tif')
    if type(ingrid) is str:
        ingrid = _files.default_file_ext(ingrid, 'grid', False)
    elif type(ingrid) is list:
        ingrid = [_files.default_file_ext(grid, 'grid', False) for grid in ingrid]
        ingrid = ';'.join(ingrid)
    # Create cmd
    cmd = ['saga_cmd', 'io_gdal', '2', '-GRIDS', ingrid, '-FILE', outraster]
    if op is not None:
        cmd.extend(['-OPTIONS', op])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def gdal_export_shapes(outshape, inshape, formatid=0):
    """
    Export vectorial data

    library: io_gdal  tool: 4

    INPUTS
     outshape      [string] output vector layer with file extension
     inshape       [string] input vector layer
     formatid      [int] output format identifier. Since formatid depends of the
                    SAGA version and GDAL version, identifier must be consulted
                    in SAGA GUI
    """
    # Check inputs
    formatid = str(formatid)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '4', '-SHAPES', inshape, '-FILE',
           outshape, '-FORMAT', formatid]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def gdal_export_shapes_to_kml(outfile, inshape):
    """
    Export a shapefile to kml format

    library: io_gdal  tool: 5

    INPUTS
     outshape      [string] output kml vector layer
     inshape       [string] input vector layer
    """
    # Check inputs
    outfile = _validation.output_file(outfile, 'kml')
    inshape = _validation.input_file(inshape, 'vector', True)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '5', '-SHAPES', inshape, '-FILE',
           outfile]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def gdal_import_netcdf(folder, infile, transform=True, resampling=3):
    """
    Import a netcdf file into multiple grids

    library: io_gdal  tool: 6

    INPUTS
     folder        [string] output grids folder
     infile        [string] input netcdf file
     transform     [boolean] apply coordinate transformation
     resampling    [int] interpolation method to use if grid needs to be aligned
                    to coordinate system. Available methods:
                    [0] Nearest Neighbour
                    [1] Bilinear Interpolation
                    [2] Bicubic Spline Interpolation
                    [3] B-Spline Interpolation (default)
    """
    # Check inputs
    transform = str(int(transform))
    resampling = _validation.input_parameter(resampling, 3, vrange=[0, 3], dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '6', '-FILE', infile, '-SAVE_FILE',
           '1', '-SAVE_PATH', folder, '-TRANSFORM', transform, '-RESAMPLING',
           resampling]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: io_grid
# ==============================================================================

def export_surfer_grid(outgrid, ingrid, out_format=0, nodata=False):
    """
    Export grid to Golden Software's Surfer grid format

    INPUTS
     outgrid       [string] output Golden Software grid
     outgrid       [string] input grid or geotif
     format        [int] output format file
                    [0] binary (default)
                    [1] ASCII
     nodata        [bool] if True, use Surfer's No-Data value, in other case
                    use the input grid No-Data value
    """
    outgrid = _validation.output_file(outgrid, 'grd')
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Input parameters
    out_format = _validation.input_parameter(out_format, 0, vrange=[0, 1], dtypes=[int])
    nodata = str(int(nodata))
    # Create cmd
    cmd = ['saga_cmd', 'io_grid', '2', '-FILE', outgrid, '-GRID', ingrid,
           '-NODATA', nodata, '-FORMAT', out_format]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_surfer_grid(outgrid, ingrid, nodata=None):
    """
    Import grid from Golden Software's Surfer grid format

    INPUTS
     outgrid       [string] output saga grid
     ingrid        [string] input Golden Software grid
     nodata        [int, float] if None (default) uses Surfer's No Data Value,
                    in other case, user defined value is used as no data
    """
    outgrid = _validation.output_file(outgrid, 'sgrd')
    ingrid = _validation.input_file(ingrid, 'grd', True)
    if nodata is None:
        nodata = '0'
        nodataval = '-99999.0'
    else:
        nodataval = str(nodata)
        nodata = '1'
    # Create cmd
    cmd = ['saga_cmd', 'io_grid', '3', '-FILE', ingrid, '-GRID', outgrid,
           '-NODATA', nodata, '-NODATA_VAL', nodataval]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_grid_from_xyz(outgrid, infile, skip=0, cellsize=1., delimiter='\t'):
    """

    :return:
    """
    outgrid = _validation.output_file(outgrid, 'sgrd')
    infile = _validation.input_file(infile, 'txt', False)
    # Input parameters
    cellsize = str(cellsize)
    if _env.saga_version[0] in ['2', '3']:
        if skip == 0:
            caption = '0'
        else:
            caption = '1'
        default_delim = {' ': 0, '\t': 1, ',': 2, ';': 3}
        if delimiter in default_delim:
            separator = default_delim[delimiter]
        else:
            raise ValueError('{} is not a valid column delimiter'.format(delimiter))
    else:
        default_delim = {' ': 1, '\t': 4, ',': 2, ';': 3}
        skip = str(int(skip))
        if delimiter in default_delim:
            separator = default_delim[delimiter]
        else:
            separator = '5'
            user_separator = str(delimiter)
    # Create cmd
    cmd = ['saga_cmd', 'io_grid', '6', '-FILE', infile, '-GRID', outgrid,
           '-CELLSIZE', cellsize, '-SEPARATOR', separator]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-CAPTION', caption])
    else:
        cmd.extend(['-SKIP', skip, '-USER', user_separator])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_clip_grid(outgrid, ingrid, polygon=None, cellsize=0, keep_type=False):
    """
    Import, clip and resampling an input raster

    library: io_grid  tool: 16

    INPUTS
     outgrid       [string] output grid file name
     ingrid        [string] input grid file
     polygon       [string] optional polygon shapefile. Shape extent is used to
                    clip the input grid
     cellsize      [int, float] optional cellsize. If cellsize is major to 0
                    then grid is resampled
     keep_type     [boolean] keep data type
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'kml')

    if polygon is None:
        polygon = 'NULL'
    else:
        polygon = _validation.input_file(polygon, 'vector', False)
    if cellsize <= 0:  # restrict no negative cellsize
        cellsize = 0
        resample = 0
    else:  # apply resampling
        resample = 1
    resample, cellsize = str(resample), str(cellsize)
    keep_type = str(int(keep_type))
    # Create cmd
    cmd = ['saga_cmd', 'io_grid', '16', '-FILES', ingrid, '-GRIDS', outgrid,
           '-CLIP', polygon, '-KEEP_TYPE', keep_type, '-RESAMPLE', resample,
           '-CELLSIZE', cellsize]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))

