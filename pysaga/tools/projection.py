"""
==============================================================================
SAGA GIS algorithm provider
Projection tools
    User osr tools
    pj_proj4


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# Import modules
import sys as _sys
import os as _os
import shutil as _shutil
import numpy as _np

from ..utilities import files as _files
from ..utilities import data_validation as _validation

try:
    import osr as _osr
    import ogr as _ogr
except:
    print('gdal library could not be imported, some tools in PySaga would be disabled!')

_ERROR_TEXT = ('Error running "{}()", please check the error file: {}')


# ==============================================================================
# Library: User osr tools
# ==============================================================================


def crs_from_epsg(code, asproj4=False):
    """
    Create a coordinate reference system text from a EPSG code
    Need gdal and osr

    INPUTS
     code      [int] EPSG code
     asproj4   [bool] if asproj4 is True, output crs is a proj4 text.
                In other case, output crs is a wkt text
    OUTPUTS
     crs       [string] crs text
    """
    # Getcoordinate reference system
    crs = _osr.SpatialReference()
    crs.ImportFromEPSG(code)
    if asproj4:
        crs = crs.ExportToProj4()
    else:
        crs = crs.ExportToWkt()
    return(crs)


def reproject_points(points, in_crs, out_crs):
    """
    Reprojection of points using _osr

    INPUTS
     points       [list, tuple, np.ndarray] [x, y] point array
                   For multiple points use [[x1, y1], [x2, y2], ...]
     in_crs       [int, str] input coordinate reference system
                   If in_crs is an integer it is used as EPSG code
                   If in_crs can be a wkt string
     out_crs      [int, str] output coordinate reference system
                   If in_crs is an integer it is used as EPSG code
                   If in_crs can be a wkt string
    OUTPUTS
     newpoints     output numpy array with transformed points
    """
    # Check inputs
    if type(points) in [list, tuple, _np.ndarray]:
        points = _np.array(points)
        if points.ndim == 1:
            points = _np.array([points])
        if points.ndim > 2:
            raise TypeError('points must be a bidimensional array!')
        if points.ndim == 2 and points.shape[1] != 2:
            raise TypeError('points must be a bidimensional array with two columns!')
    else:
        raise TypeError('Bad points parameter type {}'.format(str(type(points))))

    if type(in_crs) is int:
        in_crs = crs_from_epsg(in_crs)
    elif type(in_crs) != str:
        raise TypeError('Bad in_crs parameter type {}'.format(str(type(in_crs))))

    if type(out_crs) is int:
        out_crs = crs_from_epsg(out_crs)
    elif type(out_crs) != str:
        raise TypeError('Bad out_crs parameter type {}'.format(str(type(out_crs))))

    # Spatial reference
    crsin = _osr.SpatialReference()
    crsin.ImportFromWkt(in_crs)
    crsout = _osr.SpatialReference()
    crsout.ImportFromWkt(out_crs)

    # Create geometry and transform
    newpoints = []  # output points
    coordtrans = _osr.CoordinateTransformation(crsin, crsout)
    for x, y in points:
        geometry = _ogr.Geometry(_ogr.wkbPoint)
        geometry.AddPoint(x, y)
        geometry.Transform(coordtrans)
        newpoints.append([geometry.GetX(), geometry.GetY()])

    # Output array
    newpoints = _np.array(newpoints)
    return(newpoints)


# ==============================================================================
# Library: pj_proj4
# ==============================================================================


def set_crs(grids=None, shapes=None, precise=False, crs_method=0,
            proj="+proj=longlat +datum=WGS84 +no_defs"):
    """
    The module allows define the Coordinate Reference System (CRS) of
    the supplied data sets. The module applies no transformation to the data
    sets, it just updates their CRS metadata

    library: pj_proj4  tool: 0

    INPUTS
     grids         [string, list, tuple] grid or list of grids file names which
                    crs will be defined
     shapes        [string, list, tuple] shape or list of shape file names which
                    crs will be defined
     precise       [boolean] precise datum conversion. False as default
     crs_method    [int] Get CRS Definition from
                    [0] Proj4 Parameters (default)
                    [1] Well Known Text File (.prj)
     proj          [string] crs definition information that depends of crs_method
                    crs_method=0 [string] proj is a proj4 parameters string.
                                 By default proj="+proj=longlat +datum=WGS84 +no_defs"
                    crs_method=1 [string] proj is a well known text (.prj extension)
                                 In this case, a .shp or .sgrd file can be used and
                                 its associated .prj file is taken
    """
    # Check inputs
    if grids is None and shapes is None:
        return TypeError('Input grids and shapes can\'t be None!')

    # check grids
    if grids is None:
        grids = 'NULL'
    if type(grids) is str:
        grids = [_validation.input_file(grids, 'grid', False)]
    elif type(grids) in [list, tuple]:
        grids = _validation.input_file(grids, 'grid', False)
    else:
        return TypeError('Bad grids type: {}'.format(str(type(grids))))

    # check shapes
    if shapes is None:
        shapes = 'NULL'
    elif type(shapes) is str:
        shapes = [_validation.input_file(shapes, 'vector', False)]
    elif type(shapes) in [list, tuple]:
        shapes = _validation.input_file(shapes, 'vector', False)
    else:
        return TypeError('Bad shapes type: {}'.format(str(type(shapes))))

    # default methods
    crs_method = _validation.input_parameter(crs_method, 0, vrange=[0, 1], dtypes=[int])

    # Set CRS
    if crs_method == 1:
        # Get proj file
        proj = _validation.input_file(proj, 'prj', True)

        # check if file exist
        if not _os.path.exists(proj):
            return(False)

        # associate grids proj file
        if type(grids) is list:
            for grid in grids:
                new_proj = _files.default_file_ext(grid, 'prj')
                try:  # copy prj file
                    _shutil.copy(proj, new_proj)
                except:  # maybe the input is the same that output
                    pass

        # associate shapes proj file
        if type(shapes) is list:
            for shape in shapes:
                new_proj = _files.default_file_ext(shape, 'prj')
                try:  # copy prj file
                    _shutil.copy(proj, new_proj)
                except:  # maybe the input is the same that output
                    pass
        flag = True
    else:
        # Convert to strings
        shapes = ';'.join(shapes)
        grids = ';'.join(grids)
        crs_method = str(crs_method)
        precise = str(int(precise))
        proj = str(proj)

        # Create cmd
        cmd = ['saga_cmd', '-f=q', 'pj_proj4', '0', '-CRS_METHOD', crs_method,
               '-CRS_PROJ4', proj, '-PRECISE', precise, '-GRIDS', grids,
               '-SHAPES', shapes]

        # Run command
        flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def shape_transformation(outshape, inshape, precise=False, crs_method=0,
                         proj="+proj=longlat +datum=WGS84 +no_defs"):
    """
    Coordinate transformation for shapes using a proj4 text
    Projection routines make use of the Proj.4 Cartographic Projections library

    library: pj_proj4  tool: 2

    INPUTS
     outshape      [string] output shape file name
     outshape      [string] input shape file name
     prcise        [boolean] precise datum conversion. False as default
     crs_method    [int] Get CRS Definition from
                    [0] Proj4 Parameters (default)
                    [1] Well Known Text File (.prj)
     proj          [string] crs definition information that depends of crs_method
                    crs_method=0 [string] proj is a proj4 parameters string.
                                 By default proj="+proj=longlat +datum=WGS84 +no_defs"
                    crs_method=1 [string] proj is a well known text (.prj extension)
                                  In this case, a .shp or .sgrd file can be used and
                                  its associated .prj file is taken
    """
    # Check inputs
    inshape = _validation.input_file(inshape, 'vector', False)
    outshape = _validation.output_file(outshape, 'vector')
    precise = str(precise)

    # create prj file
    if crs_method == 1:
        proj = _validation.input_file(proj, 'vector', False)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'pj_proj4', '2', '-PRECISE', precise, '-SOURCE',
           inshape, '-TARGET', outshape]
    if crs_method == 0:  # use a Proj4 parameters
        cmd.extend(['-CRS_METHOD', '0', '-CRS_PROJ4', proj])
    else:  # use a well known text file
        proj = _files.default_file_ext(proj, 'prj')
        cmd.extend(['-CRS_METHOD', '0', '-CRS_FILE', proj])

    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))


def grid_transformation(outgrid, ingrid, crs_method=0, resampling=4, keep_type=True,
                        precise=False, proj="+proj=longlat +datum=WGS84 +no_defs",
                        cellsize=0, grid_extent=None):
    """
    Coordinate transformation for grids usign a proj4 text
    Projection routines make use of the Proj.4 Cartographic Projections library

    library: pj_proj4  tool: 4

    INPUTS
     outgrid       [string] output grid file name
     ingrid        [string] input grid file name
     crs_method    [int] Get CRS Definition from
                    [0] Proj4 Parameters (default)
                    [1] Well Known Text File (.prj)
     resampling    [int] interpolation method for grid resampling
                    [0] Nearest Neigbhor
                    [1] Bilinear Interpolation
                    [2] Inverse Distance Interpolation
                    [3] Bicubic Spline Interpolation
                    [4] B-Spline Interpolation (default)
     keep_type     [boolean] preserve original data type. True as default
     prcise        [boolean] precise datum conversion. False as default
     proj          [string] crs definition information that depends of crs_method
                    crs_method=0 [string] proj is a proj4 parameters string.
                                 By default proj="+proj=longlat +datum=WGS84 +no_defs"
                    crs_method=1 [string] proj is a well known text (.prj extension)
                                  In this case, a .shp or .sgrd file can be used and
                                  its associated .prj file is used
     cellsize      [int] output grid resolution in output projected coordinate system.
                     If cellsize is zero, cellsize is automatic calculated using the
                     output projection
     grid_extent   [string] output grid extent in output projected coordinate system
    """

    # Check inputs
    ingrid = _validation.input_file(ingrid, 'grid', False)
    outgrid = _validation.output_file(outgrid, 'grid')

    # check interpolation method
    resampling = _validation.input_parameter(resampling, 4, vrange=[0, 4], dtypes=[int])

    # convert to strings
    if crs_method == 1:  # well known text
        proj = _validation.input_file(proj, 'prj', True)

    keep_type = str(int(keep_type))
    precise = str(int(precise))
    proj = str(proj)

    # check for grid system as grid extent
    grid_extent = _validation.validate_gridsystem(grid_extent)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'pj_proj4', '4', '-PRECISE', precise, '-KEEP_TYPE',
           keep_type, '-RESAMPLING', resampling, '-SOURCE', ingrid]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-TARGET_GRID', outgrid])
    else:
        cmd.extend(['-GRID', outgrid])
    if crs_method == 0:  # use a Proj4 parameters
        cmd.extend(['-CRS_METHOD', '0', '-CRS_PROJ4', proj])
    elif crs_method == 1:  # use a well known text file
        cmd.extend(['-CRS_METHOD', '0', '-CRS_FILE', proj])

    if grid_extent is None:
        if cellsize > 0:
            cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', str(cellsize)])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])
    # Run command
    flag = _env.run_command_logged(cmd)
    if not flag:
        raise EnvironmentError(_ERROR_TEXT.format(_sys._getframe().
                                                  f_code.co_name, _env.errlog))
