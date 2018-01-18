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


def gdal_import_raster(outgrid, inraster, band=0, transform=True, method=3):
    """
    Convert a gdal supported raster format to saga grid format

    library: io_gdal  tool: 0

    INPUTS
     outgrid       [string] output grid file name
     inraster      [string] input raster file name
     band          [int, list] if band is an integer, the band number is imported
                    if band is a list of integers, multiple bands are imported
                    if band is None, all bands are imported (need gdal library)
     transform     [boolean] activate/desactivate grid resampling
     method        [int] resampling method
                    [0] Nearest Neighbour
                    [1] Bilinear Interpolation
                    [2] Bicubic Spline Interpolation
                    [3] B-Spline Interpolation (default)
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')

    method = _validation.input_parameter(method, 0, vrange=[0, 3], dtypes=[int])
    if type(band) is int:
        band = str(band)
    elif type(band) is list:
        # get grid base name
        outgrid_base = outgrid.replace('.sgrd', '')
        outgrid = [outgrid_base + str(b) + '.sgrd' for b in band]
        band = [str(b) for b in band]
        band = ';'.join(band)
        outgrid = ';'.join(outgrid)
    else:
        # get raster info
        grid_obj = _io.GridObj(inraster)
        rinfo = grid_obj.get_grid_info()
        grid_obj.close_connection()

        nbands = rinfo['bands']  # number of bands
        outgrid_base = _os.path.splitext(outgrid)[0]
        outgrid = [outgrid_base + str(b) + '.sgrd' for b in range(nbands)]
        band = [str(b) for b in range(nbands)]
        band = ';'.join(band)
        outgrid = ';'.join(outgrid)
    transform = str(int(transform))
    method = str(method)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '0', '-FILES', inraster, '-GRIDS', outgrid,
           '-SELECTION', band, '-TRANSFORM', transform, '-RESAMPLING', method]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def export_geotiff(outraster, ingrid):
    """
    Export multiple grids in a geotiff raster

    library: io_gdal  tool: 2

    INPUTS
     outraster     [string] output geotiff file (.tif)
     ingrid        [string, list] if ingrid is a string, only a grid is exported
                    if ingrid is a list, multiple grids are added to the output
                    raster file
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
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def export_shapes(outshape, inshape, formatid=0):
    """
    Export vectorial data

    library: io_gdal  tool: 4

    INPUTS
     outshape      [string] output vector layer
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


def export_shapes_to_kml(outfile, inshape):
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


def import_netcdf(folder, infile, transform=True, resampling=3):
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


def import_clip_geotiff(outgrid, ingrid, polygon=None, cellsize=0, keep_type=False):
    """
    Import, clip and resampling a geotiff raster

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

