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
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '2', '-GRIDS', ingrid, '-FILE', outraster]
    if op is not None:
        cmd.extend(['-OPTIONS', op])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def gdal_import_shapes(outshape, infile, gtype=0):
    """
    Imports vector data from various file/database formats using GDAL

    library: io_gdal  tool: 3

    INPUTS
     outshape      [string] output shapefile (.shp)
     infile        [string] input vector file
     gtype         [int] geometry type. For SAGA versions 2 and 3 available types:
                    [0] automatic (default)
                    [1] wkbPoint
                    [2] wkbPoint25D
                    [3] wkbMultiPoint
                    [4] wkbMultiPoint25D
                    [5] wkbLineString
                    [6] wkbLineString25D
                    [7] wkbMultiLineString
                    [8] wkbMultiLineString25D
                    [9] wkbPolygon
                    [10] wkbPolygon25D
                    [11] wkbMultiPolygon
                    [12] wkbMultiPolygon25D
                    [13] wkbGeometryCollection
                    [14] wkbGeometryCollection25D
                   For SAGA versions 4 or newer
                    [0] Automatic (default)
                    [1] Point
                    [2] Point (2.5D)
                    [3] Multi-Point
                    [4] Multi-Point (2.5D)
                    [5] Line
                    [6] Line (2.5D)
                    [7] Polyline
                    [8] Polyline (2.5D)
                    [9] Polygon
                    [10] Polygon (2.5D)
                    [11] Multi-Polygon
                    [12] Multi-Polygon (2.5D)
    """
    # Check inputs and outputs
    outshape = _validation.output_file(outshape, 'shp')
    # Input parameters
    if _env.saga_version[0] in ['2', '3']:
        gtype = _validation.input_parameter(gtype, 0, vrange=[0, 14], dtypes=[int])
    else:
        gtype = _validation.input_parameter(gtype, 0, vrange=[0, 12], dtypes=[int])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_gdal', '3', '-SHAPES', outshape, '-FILES', infile,
           '-GEOM_TYPE', gtype]
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
    cmd = ['saga_cmd', '-f=q', '-f=q', 'io_gdal', '4', '-SHAPES', inshape, '-FILE',
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
    cmd = ['saga_cmd', '-f=q', '-f=q', 'io_gdal', '5', '-SHAPES', inshape, '-FILE',
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
    cmd = ['saga_cmd', '-f=q', '-f=q', 'io_gdal', '6', '-FILE', infile, '-SAVE_FILE',
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

def export_esri_grid(outfile, ingrid, out_format=1, georef=0, prec=4, decsep=0):
    """
    Export grid to ESRI's Arc/Info grid format.

    library: io_grid  tool: 0

    INPUTS
     outfile       [string] output ESRI grid
     ingrid        [string] input grid
     out_format    [int] output format file
                    [0] binary
                    [1] ASCII (default)
     georef        [int] Geo-Reference method. The grids geo-reference must
                     be related either to the center or the corner of its lower left grid cell
                     [0] corner (default)
                     [1] center
     prec          [int] number of decimals when writing floating point values in ASCII format.
     decsep        [int] ASCII decimal separator. Applies also to the binary format header file.
    """
    # Input output files
    ingrid = _validation.input_file(ingrid, 'grid', False)
    # Input parameters
    out_format = _validation.input_parameter(out_format, 1, vrange=[0, 1], dtypes=[int])
    georef = _validation.input_parameter(georef, 0, vrange=[0, 1], dtypes=[int])
    decsep = _validation.input_parameter(decsep, 0, vrange=[0, 1], dtypes=[int])
    prec = str(int(prec))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_grid', '0', '-FILE', outfile, '-GRID', ingrid,
           '-FORMAT', out_format, '-GEOREF', georef, '-PREC', prec, '-DECSEP', decsep]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_esri_grid(outgrid, infile, dtype=2, nodata=None):
    """
    Import grid from ESRI's Arc/Info grid format

    library: io_grid  tool: 1

    INPUTS
     outgrid       [string] output saga grid
     infile        [string] input ESRI grid
     dtype         [int] target grid type
                    [0] Integer (2 byte)
                    [1] Integer (4 byte)
                    [2] Floating Point (4 byte) (default)
                    [3] Floating Point (8 byte)
     nodata        [int, float] if None, use input file's nodata value.
                    In other case, user defined nodata value is used
    """
    outgrid = _validation.output_file(outgrid, 'sgrd')
    infile = _validation.input_file(infile, 'asc', True)
    # Input parameters
    dtype = _validation.input_parameter(dtype, 2, vrange=[0, 3], dtypes=[int])
    if nodata is None:
        nodata = '0'
        nodataval = '-99999.0'
    else:
        nodataval = str(nodata)
        nodata = '1'
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_grid', '1', '-FILE', infile, '-GRID', outgrid, '-GRID_TYPE',
           dtype, '-NODATA', nodata, '-NODATA_VAL', nodataval]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def export_surfer_grid(outgrid, ingrid, out_format=0, nodata=False):
    """
    Export grid to Golden Software's Surfer grid format

    library: io_grid  tool: 2

    INPUTS
     outgrid       [string] output Golden Software grid
     outgrid       [string] input grid or geotif
     out_format    [int] output format file
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
    cmd = ['saga_cmd', '-f=q', 'io_grid', '2', '-FILE', outgrid, '-GRID', ingrid,
           '-NODATA', nodata, '-FORMAT', out_format]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_surfer_grid(outgrid, ingrid, nodata=None):
    """
    Import grid from Golden Software's Surfer grid format

    library: io_grid  tool: 3

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
    cmd = ['saga_cmd', '-f=q', 'io_grid', '3', '-FILE', ingrid, '-GRID', outgrid,
           '-NODATA', nodata, '-NODATA_VAL', nodataval]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def export_grid_to_xyz(outfile, grids, header=False, nodata=False):
    """
    Export grids to a xyz file

    library: io_grid  tool: 5

    INPUTS
     outfile       [string] output xyz file
     grids         [list] list of grids
     header        [bool] if True write field names
     nodata        [bool] if True, skip NoData cells from the output. In this case,
                    the first input grid will perform like a mask
    """
    # Check input and output files
    outfile = _validation.output_file(outfile, 'xyz')
    grids = _validation.input_file(grids, 'grid', False)
    grid_list = ';'.join(grids)
    # Input parameters
    header, nodata = str(int(header)), str(int(nodata))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_grid', '5', '-FILENAME', outfile, '-GRIDS', grid_list]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-CAPTION', header, '-EX_NODATA', nodata])
    else:
        cmd.extend(['-HEADER', header, '-NODATA', nodata])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def import_grid_from_xyz(outgrid, infile, skip=0, cellsize=1., delimiter='\t'):
    """
    Import grid from a xyz file

    library: io_grid  tool: 6

    INPUTS
     outgrid       [string] output saga grid
     ingrid        [string] input xyz table file
     skip          [int] number of rows to skip at the beginning of the table.
                    skip must be equal or greater than 0. For SAGA versions older than
                    4.0.0 only header is ignored (skip=1)
    cellsize       [int, float] target cellsize
    delimiter      [string] input file column delimiter. For SAGA versions 2 and 3, only
                    ' ', '\t', ',' and ';' delimiters are available
    """
    outgrid = _validation.output_file(outgrid, 'sgrd')
    infile = _validation.input_file(infile, 'xyz', False)
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
    cmd = ['saga_cmd', '-f=q', 'io_grid', '6', '-FILE', infile, '-GRID', outgrid,
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
    outgrid = _validation.output_file(outgrid, 'grid')

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
    cmd = ['saga_cmd', '-f=q', 'io_grid', '16', '-FILES', ingrid, '-GRIDS', outgrid,
           '-CLIP', polygon, '-KEEP_TYPE', keep_type, '-RESAMPLE', resample,
           '-CELLSIZE', cellsize]
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


# ==============================================================================
# Library: io_grid_image
# ==============================================================================

def export_grid_to_image(outfile, ingrid, shade=None, file_kml=False, color_method=0,
                         palette=0, colors=100, inv_color=False, std=2, srange=None,
                         table=None, shade_trans=40, sbrange=None):
    """
    Save a grid as image.
    Optionally, a shade grid can be overlayed and it's transparency and brightness can be adjuste

    library: io_grid_image  tool: 2

    INPUTS
     outfile       [string] output image file ('.png', '.jpg', '.tif', '.bpm')
     ingrid        [string] input grid file or geotif
     shade         [string] optional grid to use as shadow basemap
     file_kml      [bool] if True create kml file
     color_method  [int] colouring method
                    [0] stretch to grid's standard deviation (default)
                    [1] stretch to grid's value range
                    [2] stretch to specified value range
                    [3] lookup table
                    [4] rgb coded values
     palette       [int] color palette
                    [0] DEFAULT (default)
                    [1] DEFAULT_BRIGHT
                    [2] BLACK_WHITE
                    [3] BLACK_RED
                    [4] BLACK_GREEN
                    [5] BLACK_BLUE
                    [6] WHITE_RED
                    [7] WHITE_GREEN
                    [8] WHITE_BLUE
                    [9] YELLOW_RED
                    [10] YELLOW_GREEN
                    [11] YELLOW_BLUE
                    [12] RED_GREEN
                    [13] RED_BLUE
                    [14] GREEN_BLUE
                    [15] RED_GREY_BLUE
                    [16] RED_GREY_GREEN
                    [17] GREEN_GREY_BLUE
                    [18] RED_GREEN_BLUE
                    [19] RED_BLUE_GREEN
                    [20] GREEN_RED_BLUE
                    [21] RAINBOW
                    [22] NEON
                    [23] TOPOGRAPHY
                    [24] ASPECT_1
                    [25] ASPECT_2
                    [26] ASPECT_3
     colors        [int] number of colors in color palette
     inv_color     [bool] if True, revert color palette
     std           [int, float] standard deviation for color_method == 0
     srange        [tuple, list] stretch value range [min, max] for color_method == 2
     table         [string] input lookup table for color_method == 3
     shade_trans   [int, float] shade grid transparency (0<=transparency<=100). 40 as default
     sbrange       [tuple, list] shade brightness [min, max]
    """
    # Check input and output files
    outfile = _files.default_file_ext(outfile, ftype='png', force=False)
    ingrid = _validation.input_file(ingrid, 'grid', False)
    if shade is None:
        shade = 'NULL'
    else:
        shade = _validation.input_file(shade, 'grid', False)
    if table is None:
        table = 'NULL'
    else:
        table = _validation.input_file(table, 'txt', False)
    # Input parameters
    file_kml, inv_color = str(int(file_kml)), str(int(inv_color))
    color_method = _validation.input_parameter(color_method, 0, vrange=[0, 4], dtypes=[int])
    palette = _validation.input_parameter(palette, 0, vrange=[0, 26], dtypes=[int])
    colors = str(int(colors))
    std = str(std)
    shade_trans = _validation.input_parameter(shade_trans, 40, vrange=[0, 100], dtypes=[int, float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_grid_image', '0', '-GRID', ingrid, '-SHADE', shade, '-FILE', outfile,
           '-FILE_KML', file_kml, '-COLOURING', color_method, '-COL_PALETTE', palette, '-COL_COUNT',
           colors, '-COL_REVERT', inv_color, '-STDDEV', std, '-LUT', table]
    if color_method == 2 and type(srange) in (list, tuple):
        if len(srange) == 2:
            cmd.extend(['-STRETCH_MIN', str(min(srange)), '-STRETCH_MAX', str(max(srange))])
    if shade != 'NULL' and type(sbrange) in (list, tuple):
        cmd.extend(['-SHADE_TRANS', shade_trans])
        if len(sbrange) == 2:
            cmd.extend(['-SHADE_BRIGHT_MIN', str(min(sbrange)), '-SHADE_BRIGHT_MAX', str(max(sbrange))])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def export_grid_to_kml(outfile, ingrid, shade=None, kml_option=2, color_method=0,
                       palette=0, colors=100, inv_color=False, std=2, srange=None,
                       table=None, shade_trans=40, sbrange=None):
    """
    Export a grid to kml
    Uses 'Export Image' tool to create the image file.
    Automatically projects raster to geographic coordinate system, if its projection is known and not geographic.

    library: io_grid_image  tool: 2

    INPUTS
     outfile       [string] output image file ('.png', '.jpg', '.tif', '.bpm')
     ingrid        [string] input grid file or geotif
     shade         [string] optional grid to use as shadow basemap
     kml_option    [int] output file option
                    [0] kml and image files
                    [1] kmz, kml and image files
                    [2] kmz file
     color_method  [int] colouring method
                    [0] stretch to grid's standard deviation (default)
                    [1] stretch to grid's value range
                    [2] stretch to specified value range
                    [3] lookup table
                    [4] rgb coded values
     palette       [int] color palette
                    [0] DEFAULT (default)
                    [1] DEFAULT_BRIGHT
                    [2] BLACK_WHITE
                    [3] BLACK_RED
                    [4] BLACK_GREEN
                    [5] BLACK_BLUE
                    [6] WHITE_RED
                    [7] WHITE_GREEN
                    [8] WHITE_BLUE
                    [9] YELLOW_RED
                    [10] YELLOW_GREEN
                    [11] YELLOW_BLUE
                    [12] RED_GREEN
                    [13] RED_BLUE
                    [14] GREEN_BLUE
                    [15] RED_GREY_BLUE
                    [16] RED_GREY_GREEN
                    [17] GREEN_GREY_BLUE
                    [18] RED_GREEN_BLUE
                    [19] RED_BLUE_GREEN
                    [20] GREEN_RED_BLUE
                    [21] RAINBOW
                    [22] NEON
                    [23] TOPOGRAPHY
                    [24] ASPECT_1
                    [25] ASPECT_2
                    [26] ASPECT_3
     colors        [int] number of colors in color palette
     inv_color     [bool] if True, revert color palette
     std           [int, float] standard deviation for color_method == 0
     srange        [tuple, list] stretch value range [min, max] for color_method == 2
     table         [string] input lookup table for color_method == 3
     shade_trans   [int, float] shade grid transparency (0<=transparency<=100). 40 as default
     sbrange       [tuple, list] shade brightness [min, max]
    """
    # Check input and output files
    outfile = _files.default_file_ext(outfile, ftype='png', force=False)
    ingrid = _validation.input_file(ingrid, 'grid', False)
    if shade is None:
        shade = 'NULL'
    else:
        shade = _validation.input_file(shade, 'grid', False)
    if table is None:
        table = 'NULL'
    else:
        table = _validation.input_file(table, 'txt', False)
    # Input parameters
    kml_option = _validation.input_parameter(kml_option, 2, vrange=[0, 2], dtypes=[int])
    inv_color = str(int(inv_color))
    color_method = _validation.input_parameter(color_method, 0, vrange=[0, 4], dtypes=[int])
    palette = _validation.input_parameter(palette, 0, vrange=[0, 26], dtypes=[int])
    colors = str(int(colors))
    std = str(std)
    shade_trans = _validation.input_parameter(shade_trans, 40, vrange=[0, 100], dtypes=[int, float])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'io_grid_image', '2', '-GRID', ingrid, '-SHADE', shade, '-FILE', outfile,
           '-OUTPUT', kml_option, '-COLOURING', color_method, '-COL_PALETTE', palette, '-COL_COUNT',
           colors, '-COL_REVERT', inv_color, '-STDDEV', std, '-LUT', table]
    if color_method == 2 and type(srange) in (list, tuple):
        if len(srange) == 2:
            cmd.extend(['-STRETCH_MIN', str(min(srange)), '-STRETCH_MAX', str(max(srange))])
    if shade != 'NULL' and type(sbrange) in (list, tuple):
        cmd.extend(['-SHADE_TRANS', shade_trans])
        if len(sbrange) == 2:
            cmd.extend(['-SHADE_BRIGHT_MIN', str(min(sbrange)), '-SHADE_BRIGHT_MAX', str(max(sbrange))])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))
