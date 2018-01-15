"""
SAGA GIS algorithm provider
Shapes tools

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import os as _os
import pandas.core.series.Series as _Series
from collections import OrderedDict as _OrderedDict

import files as _files
import projection as _projection
import tables as _tables


#==============================================================================
# Library: shapes_grid
#==============================================================================

# Add grid values to points
# library: shapes_grid  tool: 0
# INPUTS:
#  outpoints     [string] output point shape with grid values
#  points        [string] input point shape
#  grid          [string, list] grid of list of grids for extract values
#  method        [int] method for resampling grid
#                 [0] Nearest Neighbour
#                 [1] Bilinear Interpolation
#                 [2] Bicubic Spline Interpolation
#                 [3] B-Spline Interpolation
#  delete_old    [boolean] if is True, fields of original shape are deleted
#  field_id      [int, string] field of the points index
def grid_values_to_points(outpoints, points, grid, method=0,
                          delete_old=False, field_id=None):
    # Check inputs
    outpoints = _files.default_file_ext(outpoints, 'vector')
    points = _files.default_file_ext(points, 'vector')
    if type(grid) is str:
        grid = _files.default_file_ext(grid, 'grid', False)
    elif type(grid) in [list, tuple]:
        # join a list of grids
        grid = list(grid)
        grid = [_files.default_file_ext(name, 'grid', False) for name in grid]
        grid = ';'.join(grid)
    # default method
    if method < 0 or method > 3:
        method = 0
    method = str(method)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '0', '-SHAPES', points, '-GRIDS',
           grid, '-RESULT', outpoints, '-RESAMPLING', method]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete fields?
    if delete_old:
        table = _tables.get_attribute_table(points)
        fields = table.columns.tolist()
        if field_id is not None:
            if type(field_id) is str:
                fields.delete(field_id)
            elif type(field_id) is int:
                del(fields[field_id])
        flag = _tables.delete_fields(outpoints, outpoints, fields)

    # Check if output grid has crs file
    if not _files.has_crs_file(outpoints):  # set first input layer crs
        _projection.set_crs(shapes=outpoints, crs_method=1, proj=points);
    return(flag)  # grid_values_to_points()


# Zonal grid statistics. For each polygon statistics based on all covered grid cells will be calculated
# library: shapes_grid  tool: 2
# INPUTS:
#  outshape      [string] output point shape with grid values
#  polygons      [string] input point shape
#  grid          [string, list] grid of list of grids for extract values
#  method        [int] method for compute the polygon's statistics
#                 [0] simple and fast (default)
#                 [1] polygon wise (cell centers)
#                 [2] polygon wise (cell area)
#                 [3] polygon wise (cell area weighted)
#  delete_old    [boolean] if is True, fields of original shape are deleted
#  field_id      [int, string] field index of the original shape. By default
#                 field_id is no considered
#  paralel       [boolean] use multiple cores. By default False
#  Available statistics and its default value: Mean (True), Min (True), Max (True)
#                 Sum (False), Count (False), Range (False), Var (False),
#                 Std (False), Quantile (by default 0 <not calculated>,
#                 minimum:0, maximum: 50)
def grid_statistics_for_polygons(outshape, polygons, grid, method=0,
                                 naming=1, delete_old=False, field_id=None,
                                 paralel=False, Mean=False, Min=True, Max=True,
                                 Sum=False, Count=False, Range=False, Var=False,
                                 Std=False, Quantile=0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'grid')
    polygons = _files.default_file_ext(polygons, 'vector')
    if type(grid) is str:
        grid = _files.default_file_ext(grid, 'grid', False)
    elif type(grid) in [list, tuple]:
        # join a list of grids
        grid = list(grid)
        grid = [_files.default_file_ext(name, 'grid', False) for name in grid]
        grid = ';'.join(grid)
    # methods
    if naming < 0 or naming > 1:
        naming = 1
    if method < 0 or method > 3:
        method = 0
    # convert to strings
    naming = str(naming)
    method = str(method)
    Mean, Min, Max = str(int(Mean)), str(int(Min)), str(int(Max))
    Sum, Count, Range = str(int(Sum)), str(int(Count)), str(int(Range))
    Var, Std = str(int(Var)), str(int(Std))
    Quantile = str(Quantile)
    paralel = str(int(paralel))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '2', '-GRIDS', grid, '-POLYGONS',
           polygons, '-RESULT', outshape, '-NAMING', naming, '-METHOD', method,
           '-COUNT', Count, '-MIN', Min, '-MAX', Max, '-RANGE', Range, '-SUM',
           Sum, '-MEAN', Mean, '-VAR', Var, '-STDDEV', Std, '-QUANTILE',
           Quantile, '-PARALLELIZED', paralel]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete fields?
    if delete_old:
        table = _tables.get_attribute_table(polygons)
        fields = table.columns.tolist()
        if field_id is not None:
            if type(field_id) is str:
                fields.delete(field_id)
            elif type(field_id) is int:
                del (fields[field_id])
        flag = _tables.delete_fields(outshape, outshape, fields)

        # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=polygons);
    return(flag)  # grid_statistics_for_polygons()


# Derive contour lines (isolines) from grid
# library: shapes_grid  tool: 5
# INPUTS:
#  ingrid     [string] inputd grid file
#  lines      [string] optional contour lines shape file
#  polygons   [string] optional contour polygons shape file
#  ztep       [float] equidistance for countours
#  zmin       [float] minimum contour value
#  zmax       [float] maximum contour value
#  scale      [float] interpolation scale. Set greater one for line smoothing
#  add_z      [bool] if add_z is False, vertex with x,y coordinates are used, if add_z
#              is True, x,y,z is used
#  split      [bool] split lines and polygons parts
def contour_lines_from_grid(ingrid, lines=None, polygons=None, zstep=10, zmin=0,
                            zmax=100, scale=1, add_z=False, split=True):
    # Check inputs
    if lines is None and polygons is None:
        raise ValueError('At least, lines or polygons must be created!')
    ingrid = _files.default_file_ext(ingrid, 'grid')
    if type(lines) is str:
        lines = _files.default_file_ext(lines, 'vector')
    if type(polygons) is str:
        polygons = _files.default_file_ext(polygons, 'vector')
    # Convert to strings
    zstep, zmin, zmax = str(zstep), str(zmin), str(zmax)
    scale, add_z, split = str(scale), str(int(add_z)), str(int(split))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '5', '-GRID', ingrid, '-VERTEX', add_z,
           '-SCALE', scale, '-POLY_PARTS', split, '-LINE_PARTS', split]
    if lines is not None:
        cmd.extend(['-CONTOUR', lines])
    if polygons is not None:
        cmd.extend(['-POLYGONS', polygons])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if lines is not None:
        if not _files.has_crs_file(lines):  # set first input layer crs
            _projection.set_crs(shapes=lines, crs_method=1, proj=ingrid);
    if polygons is not None:
        if not _files.has_crs_file(polygons):  # set first input layer crs
            _projection.set_crs(shapes=polygons, crs_method=1, proj=ingrid);
    return(flag)  # contour_lines_from_grid()


# Vectorising grid classes
# library: shapes_grid  tool: 6
# INPUTS
#  outshape     [string] output polygons shape file
#  ingrid       [string] input grid
#  method       [int] select how the grid will vectorizing
#                [0] one single class specified by class identifier
#                [1] all classes (by default)
#  classid      [int, float] it applies if method=0. classid takes only the
#                cells that contains this value
#  vertices     [boolean] keep vertices on straight lines
def vectorising_grid_classes(outshape, ingrid, method=1, classid=0,
                             vertices=False):
    # Check input
    outshape = _files.default_file_ext(outshape, 'vector')
    ingrid = _files.default_file_ext(ingrid, 'grid')
    # check methods
    if method < 0 or method > 1:
        method = 1
    method = str(method)
    classid = str(classid)
    vertices = str(int(vertices))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '6', '-GRID', ingrid, '-POLYGONS',
           outshape, '-CLASS_ALL', method, '-CLASS_ID', classid, '-ALLVERTICES',
           vertices]
    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=ingrid);
    return(flag)  # vectorising_grid_classes()


# Clip grid with polygons
# library: shapes_grid  tool: 7
# INPUT
#  outgrid    [string] output clipped grid
#  ingrid     [string] input grid
#  polygon    [string] input polygon shape file
#  extent     [int] method for extension adjustment
#              [None] default option for older versions of SAGA GIS
#              [0] original extent
#              [1] polygons extent (version 4.0 or newer)
#              [2] crop to data (default)
def clip_grid_with_polygons(outgrid, ingrid, polygon, extent=2):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    polygon = _files.default_file_ext(polygon, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '7', '-INPUT', ingrid, '-OUTPUT',
           outgrid, '-POLYGONS', polygon]

    if _env.saga_version[0] in ['2', '3']:  # extension method for older versions
        if extent < 0 or extent > 1:
            extent = 1  # default method
        extent = str(extent)

        cmd.extend(['-NODATA', extent])

    else:  # extension method for newer versions
        if extent < 0 or extent > 2:
            extent = 2  # default method
        extent = str(extent)

        cmd.extend(['-EXTENT', extent])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # clip_grid_with_polygons()


# Creates a polygon (rectangle) from a grid system's extent
# library: shapes_grid  tool: 10
# INPUT
#  outshape      [string] output shape extent
#  grid_system   [dict, OrderDict, string] dictionary of grid system or .sgrd file
#  method        [int] border method
#                 [0] grid cells (default)
#                 [1] grid nodes
#  proj          [string] proj4 parameters to define outshape projection
#  proj_file     [string] well known file (.prj) to denfine outshape projection
# NOTE: if grid_system is a .sgrd file, projection could be extracted from its associate
#       .prj file
def grid_system_extent(outshape, grid_system, method=0, proj=None, proj_file=None):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    grid = None
    if type(grid_system) in [dict, _OrderedDict]:
        gs = grid_system.copy()
    elif type(grid_system) is str:
        grid = _files.default_file_ext(grid_system, 'grid')
        gs = _files.grid_system(grid)
    else:
        raise TypeError('Bad gris_system data type <{}>'.format(str(type(grid_system))))
    if method < 0 or method > 1:
        method = 0
    method = str(method)

    # Get grid system parameters
    nx, ny = str(int(gs['CELLCOUNT_X'])), str(int(gs['CELLCOUNT_Y']))
    x, y = str(gs['POSITION_XMIN']), str(gs['POSITION_YMIN'])
    dxy = str(gs['CELLSIZE'])
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '10', '-PARAMETERS_GRID_SYSTEM_NX', nx,
           '-PARAMETERS_GRID_SYSTEM_NY', ny, '-PARAMETERS_GRID_SYSTEM_X', x,
           '-PARAMETERS_GRID_SYSTEM_Y', y, '-PARAMETERS_GRID_SYSTEM_D', dxy,
           '-SHAPES', outshape, '-CELLS', method]

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if proj is not None:
        _projection.set_crs(shapes=outshape, crs_method=0, proj=proj);
    elif proj_file is not None:
        _projection.set_crs(shapes=outshape, crs_method=1, proj=proj_file);
    elif grid is not None:
        _projection.set_crs(shapes=outshape, crs_method=1, proj=grid);
    return(flag)  # grid_system_extent()


# Clips the input grid with the (rectangular) extent of a shapefile. The clipped grid
# will have the extent of the shapefile.
# library: shapes_grid  tool: 11
# INPUT
#  outgrid    [string] output clipped grid
#  ingrid     [string] input grid
#  inshapes   [string] input shape file
#  method     [int] border method
#              [0] grid cells (default)
#              [1] grid nodes
#              [2] align to grid system
def clip_grid_with_rectangle(outgrid, ingrid, inshape, method=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    inshape = _files.default_file_ext(inshape, 'vector')
    if method < 0 or method > 2:
        method = 0
    method = str(method)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_grid', '11', '-INPUT', ingrid,
           '-SHAPES', inshape, '-OUTPUT', outgrid, '-BORDER', method]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # clip_grid_with_rectangle()


#==============================================================================
# Library: shapes_lines
#==============================================================================

# Line-Polygons intersection. Tool for clip lines with polygons
# library: shapes_lines  tool: 3
# INPUTS
#  lines           [string] input lines shape file
#  polygons        [string] input polygons shape file
#  outshape        [string] output intersection lines
#  difference      [string] optional output difference lines with polygons.
#                   Only works with new SAGA versions. By default difference
#                   layer is not generated
#  attributes      [int] attributes inherid to intersection result
#                   For old SAGA versions
#                    [0] one multi-line per polygon
#                    [1] keep original line attributes
#                   For new SAGA versions
#                    [0] attributes from polygon
#                    [1] line attributes
#                    [2] line and polygon attributes
#                   By default attributes=1
def line_polygon_intersection(lines, polygons, outshape, difference=False,
                              attributes=0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    lines = _files.default_file_ext(lines, 'vector')
    polygons = _files.default_file_ext(polygons, 'vector')
    if type(difference) is str:
        difference = _files.default_file_ext(difference, 'vector')
    else:
        difference = _os.path.splitext(outshape)[0] + '_diff'
        difference = _files.default_file_ext(difference, 'vector')
    if _env.saga_version[0] == '2':  # old saga version
        if attributes < 0 or attributes > 1:
            attributes = 1  # default method for old SAGA versions
    else:                            # new saga version
        if attributes < 0 or attributes > 2:
            attributes = 1  # default method for new SAGA versions
    # convert to strings
    attributes = str(attributes)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_lines', '3', '-LINES', lines, '-POLYGONS',
           polygons]
    if _env.saga_version[0] == '2':  # old saga version
        cmd.extend(['-INTERSECT', outshape, '-METHOD', attributes])
    else:
        cmd.extend(['-ATTRIBUTES', attributes, '-INTERSECT', outshape,
                    '-DIFFERENCE', difference])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=lines);
    if difference is not None and int(_env.saga_version[0]) > 2:
        if not _files.has_crs_file(difference):  # set first input layer crs
            _projection.set_crs(shapes=difference, crs_method=1, proj=lines);
    return(flag)  # line_polygon_intersection()


# Lines and polygons simplification implementing
# library: shapes_lines  tool: 4
# INPUTS
#  outshape    [string] output lines or polygons simpliefied
#  inshape     [string] input lines or polygons shapefile
#  tol         [float] Maximum deviation allowed between original and
#               simplified curve [map units]. By default tol=1.0
def line_simplification(outshape, inshape, tol=1.0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    # convert to string
    tol = str(tol)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_lines', '4', '-LINES', inshape, '-OUTPUT',
           outshape, '-TOLERANCE', tol]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # line_simplification()


# Dissolve lines shapes. Shapes with the same attribute can be dissolved
# INPUTS
# library: shapes_lines  tool: 5
#  outshape    [string] output dissolved lines
#  inshape     [string] input lines
#  fields      [int, str, list] field index or name of the attribute field
#               for dissolve lines with the same value. fields can be a list
#               of fields indexe of name (maximum 3 fields can be used).
#               If fields is None (default value), all lines are merged
def line_dissolve(outshape, inshape, fields=None):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    # fields
    method = '0'  # dissolve lines with same attribute
    if type(fields) in [int, str]:
        fields = [str(fields)]
    elif type(fields) in [list, tuple]:
        fields = [str(field) for field in fields]
    elif fields is None:
        fields = ['NULL']
        method = '1'  # dissolve all lines
    else:
        raise TypeError('Wrong fields parameter <{}>'.format(str(type(fields))))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_lines', '5', '-LINES', inshape, '-DISSOLVED',
           outshape, '-ALL', method]
    # add fields
    for i in range(min([3, len(fields)])):
        cmd.extend(['-FIELD_%d' % (i + 1), str(fields[i])])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # line_dissolve()


# Line smoothing using different method
# library: shapes_lines  tool: 7
# INPUTS
#  outshape        [string] output smoothed lines
#  inshape         [string] input lines
#  method          [int] smoothing method
#                   [0] basic SIA model
#                   [1] improved SIA model
#                   [2] Gaussian Filtering (default)
#  sigma           [float] standard deviation os the Gaussian Filter (method [2])
#                   Parameter sigma must be higher than 0.5
#  sensitivity     [int] half the size of the moving window [vertex count],
#                   controls smoothing sensitivity. sensitivity must be higher
#                   than 1 (3 as default) and applies for methods [0] and [1]
#  iterations      [int] number of smoothing iterations. Minimum 1, 10 as default
#                   and applies for methods [0] and [1]
#  preservation    [float] number of smoothing iterations. Minimum 1, 10 as default
#                   and works with method [1]
def line_smoothing(outshape, inshape, method=2, sigma=2, sensitivity=3,
                   iterations=10, preservation=10):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    # set default values
    if method < 0 or method > 2:
        method = 2
    if sigma < 0.5:
        sigma = 0.5  # minimum value
    if sensitivity < 1:
        sensitivity = 1  # minimum value
    if iterations < 1:
        iterations = 1  # minimum
    if preservation < 1:
        preservation = 1  # minimum value
    # conter to strings
    method, sigma, preservation = str(method), str(int(sigma)), str(preservation)
    sensitivity, iterations = str(int(sensitivity)), str(int(iterations))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_lines', '7', '-LINES_IN', inshape, '-LINES_OUT',
           outshape, '-METHOD', method, '-SENSITIVITY', sensitivity, '-ITERATIONS',
           iterations, '-PRESERVATION', preservation]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # line_smoothing()


#==============================================================================
# Library: shapes_points
#==============================================================================

# Create points shape from table using X, Y, Z fields
# library: shapes_points  tool: 0
# INPUTS:
#  output       [string] output points shape file
#  table        [string] input table file name
#  X            [int, str] index or name of field with X coordinate
#  Y            [int, str] index or name of field with Y coordinate
#  Z            [int, str] optional index or name of field with Z value. As
#                default Z is not consider
#  proj         [string] optional proj4 parameters
#  proj_file    [string] copy the .proj file of a layer
def convert_table_to_points(output, table, X, Y, Z=None, proj=None,
                            proj_file=None):
    # Check inputs
    output = _files.default_file_ext(output, 'vector')
    if Z is None:
        Z = '-1'
    else:
        Z = str(Z)
    X, Y = str(X), str(Y)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '0', '-POINTS', output,
           '-TABLE', table, '-X', X, '-Y', Y, '-Z', Z]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if proj is not None:
        _projection.set_crs(shapes=output, crs_method=0, proj=proj);
    elif proj_file is not None:
        _projection.set_crs(shapes=output, crs_method=1, proj=proj_file);
    return(flag)  # convert_table_to_points()


# Computes distances between pairs of points.
# library: shapes_points  tool: 3
# INPUTS:
#  out_table    [string] output distance table (.txt)
#  points       [string] points layer
#  id_points    [id, string] attribute id or name as point id
#  max_dist     [int, float] maximum distance for search points.
#                If max_dist is zero, then complete input points matrix is computed
#  near         [string] additional points layer to compute distance
#  id_near      [id, string] attribute id or name as near point id
def point_distances(out_table, points, id_points=0, max_dist=0,
                    near=None, id_near=0):
    # Check inputs
    out_table = _files.default_file_ext(out_table, 'txt')
    points = _files.default_file_ext(points, 'vector')
    if type(near) is str:
        near = _files.default_file_ext(near, 'vector')
    else:
        near = "NULL"
    # Methods
    id_points, id_near = str(id_points), str(id_near)
    op = True  # compute distance matrix
    if max_dist <= 0:
        max_dist = 0
        op = False
    max_dist = str(max_dist)

    # Create cmd and run
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '3', '-POINTS', points,
           '-ID_POINTS', id_points, '-NEAR', near, '-ID_NEAR', id_near,
           '-DISTANCES', out_table, '-MAX_DIST', max_dist]
    if op:
        cmd.extend(['-FORMAT', '1'])
    else:
        cmd.extend(['-FORMAT', '0'])

    # Run algorithm
    flag = _env.run_command_logged(cmd)
    return(flag)  # point_distances()


# Converts lines to points. Optionally inserts additional
# points in user defined distances
# library: shapes_points  tool: 5
# INPUTS:
#  output    [string] output shape file
#  lines     [string] shape file of points
#  insert    [int] additional insert additional points method
#             [0] per line segment (default)
#             [1] per line
#             [2] from line center
#  dist      [int, float] point insertion distance [map units]
#             If dist is zero, then additional points are not added
#  order     [boolean] if order is True, add point order
def convert_lines_to_points(output, lines, insert=0, dist=0, order=False):
    # Check inputs
    output = _files.default_file_ext(output, 'vector')
    lines = _files.default_file_ext(lines, 'vector')
    # Aditional inputs
    if insert < 0 or insert > 2:
        insert = 0  # default method
    insert = str(int(insert))

    op = False  # option for additinal points
    if dist < 0:
        dist = 0
    if dist == 0:
        op = True
    dist = str(dist)
    order = str(int(order))

    # Create cmd and run
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '5', '-LINES', lines, '-POINTS', output,
           '-ADD_POINT_ORDER', order]
    if op:
        cmd.extend(['-ADD', '1', '-METHOD_INSERT', insert, '-DIST', dist])
    else:
        cmd.extend(['-ADD', '0'])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=lines);
    return(flag)  # convert_lines_to_points()


# Add coordinates to a shape file of points
# library: shapes_points  tool: 6
# INPUTS:
#  output    [string] output shape file
#  points    [string] shape file of points
def add_coordinates_to_points(output, points):
    # Check inputs
    output = _files.default_file_ext(output, 'vector')
    points = _files.default_file_ext(points, 'vector')
    # Create cmd and run
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '6', '-INPUT', points, '-OUTPUT', output]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=points);
    return(flag)  # add_coordinates_to_points()


# Remove duplicate points
# library: shapes_points  tool: 7
# INPUTS:
#  output    [string] output shape file with clipped points
#  points    [string] shape file of points
#  keep      [int] point to keep
#             [0] first point (default)
#             [1] last point
#             [2] point with minimum attribute value
#             [3] point with maximum attribute value
#  field     [int, string] attribute id or name for keep method [2] and [3]
#  values    [int] numeric attribute values
#             [0] take value from the point to be kept (default)
#             [1] minimum value of all duplicates
#             [2] maximum value of all duplicates
#             [3] mean value of all duplicates
def remove_duplicate_points(output, points, keep=0, field=0, values=0):
    # Check inputs
    output = _files.default_file_ext(output, 'vector')
    points = _files.default_file_ext(points, 'vector')
    # Check methods
    if keep < 0 or keep > 3:
        keep = 0  # default method
    if values < 0 or values > 3:
        values = 0  # default method
    keep, field = str(keep), str(field)
    values = str(values)

    # Create cmd and run
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '7', '-POINTS', points, '-RESULT', output,
           '-FIELD', field, '-METHOD', keep, '-NUMERIC', values]
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=points);
    return(flag)  # add_coordinates_to_points()


# Clip a point cloud using polygons
# library: shapes_points  tool: 8
# INPUTS:
#  output    [string] output shape file with clipped points
#  points    [string] shape file of points
#  polygons  [string] shape file of polygons
#  field     [int,string] attribute id number or name of the
#             input polygon. By default 0
def clip_points_with_polygons(output, points, polygons, field=0):
    # Check inputs
    output = _files.default_file_ext(output, 'vector')
    points = _files.default_file_ext(points, 'vector')
    polygons = _files.default_file_ext(polygons, 'vector')
    # Run command
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '8', '-CLIPS', output, '-POINTS',
           points, '-POLYGONS', polygons, '-FIELD', str(field), '-METHOD', '0']
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=points);
    return(flag)  # clip_points_with_polygons()


# Retrieves for each point the selected attributes from those polygon, which contain the point.
# library: shapes_points  tool: 10
# INPUTS
#  output     [string] output shape file with clipped points
#  points     [string] shape file of points
#  polygons   [string] shapefile of polygons
#  fields     [int,string, list, tuple] attribute id number or name of the input polygon.
#              fields can be a list or tuple of integers or strings
def add_polygon_attributes_to_points(outshape, points, polygons, fields=0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    points = _files.default_file_ext(points, 'vector')
    polygons = _files.default_file_ext(polygons, 'vector')
    # Check fields
    if type(fields) in [int, str]:
        fields = str(fields)
    elif type(fields) in [list, tuple]:
        fields = [str(field) for field in fields]
        fields = ",".join(fields)
    else:
        raise TypeError("Parameter fields must be a string, intenger or list/tuple")
    # Run command
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '10', '-OUTPUT', outshape, '-POINTS',
           points, '-POLYGONS', polygons, '-FIELDS', fields]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=points);
    return(flag)  # add_polygon_attributes_to_points()


# Convert Multipoints to Points
# library: shapes_points  tool: 15
# INPUTS:
#  output    [string] output points shape file
#  points    [string] input multipoints shape file
def convert_multipoints_to_points(output, multipoints):
    # Check inputs
    multipoints = _files.default_file_ext(multipoints, 'vector')
    output = _files.default_file_ext(output, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '15', '-POINTS', output,
           '-MULTIPOINTS', multipoints]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=multipoints);
    return(flag)  # convert_multipoints_to_points()


# Creates Thiessen or Voronoi polygons for given point data set
# library: shapes_points  tool: 16
# INPUTS:
#  output    [string] output shape file with thiessen polygons
#  points    [string] shape file of points
#  frame     [float] frame size in projection units
def thiessen_polygons(output, points, frame=10.0):
    # Check inputs
    points = _files.default_file_ext(points, 'vector')
    output = _files.default_file_ext(output, 'vector')
    frame = str(frame)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '16', '-POINTS', points,
           '-POLYGONS', output, '-FRAME', frame]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(output):  # set first input layer crs
        _projection.set_crs(shapes=output, crs_method=1, proj=points);
    return(flag)  # thiessen_polygons()


# Snap points to points
# library: shapes_points  tool: 19
# INPUTS:
#  out_points   [string] output snapped points
#  out_moves    [string] output distance shape file
#  points       [string] input points shape file
#  snap         [string] input snap points shape file
#  dist         [float] search distance [map units]
def snap_points_to_points(out_points, out_moves, points, snap, dist=0):
    # Check inputs
    out_points = _files.default_file_ext(out_points, 'vector')
    out_moves = _files.default_file_ext(out_moves, 'vector')
    points = _files.default_file_ext(points, 'vector')
    snap = _files.default_file_ext(snap, 'vector')
    # Convert to strings
    dist = str(dist)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '18', '-INPUT', points,
           '-SNAP', snap, '-OUTPUT', out_points, '-MOVES', out_moves,
           '-DISTANCE', dist]
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(out_points):  # set first input layer crs
        _projection.set_crs(shapes=out_points, crs_method=1, proj=points);
    if not _files.has_crs_file(out_moves):  # set first input layer crs
        _projection.set_crs(shapes=out_moves, crs_method=1, proj=points);
    return(flag)  # snap_points_to_lines()


# Snap points to lines
# library: shapes_points  tool: 19
# INPUTS:
#  out_points   [string] output snapped points
#  out_moves    [string] output distance shape file
#  points       [string] input points shape file
#  lines        [string] input snap lines shape file
#  dist         [float] search distance [map units]
def snap_points_to_lines(out_points, out_moves, points, lines, dist=0):
    # Check inputs
    out_points = _files.default_file_ext(out_points, 'vector')
    out_moves = _files.default_file_ext(out_moves, 'vector')
    points = _files.default_file_ext(points, 'vector')
    lines = _files.default_file_ext(lines, 'vector')
    # Convert to strings
    dist = str(dist)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '19', '-INPUT', points,
           '-SNAP', lines, '-OUTPUT', out_points, '-MOVES', out_moves,
           '-DISTANCE', dist]
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(out_points):  # set first input layer crs
        _projection.set_crs(shapes=out_points, crs_method=1, proj=points);
    if not _files.has_crs_file(out_moves):  # set first input layer crs
        _projection.set_crs(shapes=out_moves, crs_method=1, proj=points);
    return(flag)  # snap_points_to_lines()


# library: shapes_points  tool: 20
# INPUTS:
#  out_points   [string] output snapped points
#  out_moves    [string] output distance shape file
#  points       [string] input points shape file
#  grids        [string] input grid to span points
#  dist         [float] search distance [map units]
#  method       [int] method for search distance shape
#                [0] circle (default)
#                [1] square
#  extreme      [int] method for search extreme value
#                [0] minimum
#                [1] maximum (default)
def snap_points_to_grid(out_points, out_moves, points, grid, dist=0,
                        method=0, extreme=1):
    # Check inputs
    out_points = _files.default_file_ext(out_points, 'vector')
    out_moves = _files.default_file_ext(out_moves, 'vector')
    points = _files.default_file_ext(points, 'vector')
    grid = _files.default_file_ext(grid, 'grid')

    # Convert to strings
    dist = str(dist)
    if method < 0 or method > 1:
        method = 0  # default method
    if extreme < 0 or extreme > 1:
        extreme = 0  # default method
    method, extreme = str(method), str(extreme)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_points', '20', '-INPUT', points,
           '-GRID', grid, '-OUTPUT', out_points, '-MOVES', out_moves,
           '-DISTANCE', dist, '-SHAPE', method, '-EXTREME', extreme]
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(out_points):  # set first input layer crs
        _projection.set_crs(shapes=out_points, crs_method=1, proj=points);
    if not _files.has_crs_file(out_moves):  # set first input layer crs
        _projection.set_crs(shapes=out_moves, crs_method=1, proj=points);
    return(flag)  # snap_points_to_grid()


#==============================================================================
# Library: shapes_polygons
#==============================================================================

# Creates a points layer containing the centroids of the input polygon layer
# library: shapes_polygons  tool: 1
# INPUTS
#  outshape           [string] output centroids of polygons
#  inshape            [string] input polygons layer
#  parts              [boolean] set True if you want centroids for each parts
def polygon_centroids(outshape, inshape, parts=False):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    parts = str(int(parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '1', '-POLYGONS', inshape,
           '-CENTROIDS', outshape, '-METHOD', parts]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # polygon_centroids()


# Calculates statistics over all points falling in a polygon
# library: shapes_polygons  tool: 2
# INPUTS
#  outpolygons      [string] output shape file of polygons with point statistics
#  points           [string] input points shape
#  polygons         [string] input polygons
#  fields           [int, str, list] index or name of the field for the statistic
#                    computing. fields can be a list with the name or index of
#                    multiple fields
#  field_names      [int] field naming method for statistics (for example sum of X)
#                    [0] variable type + original name (SUM_X)
#                    [1] original name + variable type (X_SUM)
#                    [2] original name (X)
#                    [3] variable type (SUM)
# available statistics (and their default values): mean (Mean=True), sum (Sum=False),
#  variance (Var=False), standard deviation (Std=False), minimum (Min=False),
#  maximum (Max=False), count points (Count=False)
def point_statistics_for_polygons(outpolygons, points, polygons, fields=0,
                                  field_names=1, Mean=True, Sum=False, Var=False,
                                  Std=False, Min=False, Max=False, Count=False):
    # Check inputs
    outpolygons = _files.default_file_ext(outpolygons, 'vector')
    points = _files.default_file_ext(points, 'vector')
    polygons = _files.default_file_ext(polygons, 'vector')
    # fields
    if type(fields) in [int, str]:
        fields = str(fields)
    elif type(fields) in [list, tuple]:
        fields = [str(field) for field in fields]
        fields = ','.join(fields)
    else:
        raise TypeError('Bad fields paramter type <{}>'.format(str(type(fields))))
    # convert to strings
    Mean, Sum, Var, Std = str(int(Mean)), str(int(Sum)), str(int(Var)), str(int(Std))
    Min, Max, Count = str(int(Min)), str(int(Max)), str(int(Count))
    if field_names < 0 or field_names > 3:
        field_names = 1  # default value
    names = str(field_names)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '4', '-POINTS', points, '-POLYGONS', polygons,
           '-STATISTICS', outpolygons, '-FIELDS', fields, '-SUM', Sum, '-AVG', Mean,
           '-VAR', Var, '-DEV', Std, '-MIN', Min, '-MAX', Max, '-NUM', Count,
           '-FIELD_NAME', names]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outpolygons):  # set first input layer crs
        _projection.set_crs(shapes=outpolygons, crs_method=1, proj=points);
    return(flag)  # point_statistics_for_polygons()


# Converts lines to polygons. Line arcs are closed to polygons simply by
# connecting the last point with the first.
# library: shapes_polygons  tool: 3
# INPUTS
#  outshape       [string] output line layers
#  inshape        [string] input polygons
#  single         [boolean] set True to create single multipart polygon
#  merge          [boolean] set True to merge line parts to one polygon
def convert_lines_to_polygons(outshape, inshape, single=True, merge=True):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    single = str(int(single))
    merge = str(int(merge))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '3', '-LINES', inshape,
           '-POLYGONS', outshape, '-SINGLE', single, '-MERGE', merge]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # convert_lines_to_polygons()


# Merges polygons, which share the same attribute value, and (optionally)
# dissolves borders between adjacent polygon parts. If no attribute or
# combination of attributes is chosen, all polygons will be merged.
# library: shapes_polygons  tool: 5
# INPUTS
#  outshape       [string] output merged polygons
#  polygons       [string] input polygons
#  fields         [int, str, list] field name or field index for merge polygons
#                  with the same value. Parameter fields can be a list of field
#                  names or indexes. If fields is None (default) all polygons are
#                  merged. For old_versions of SAGA, only 3 fields can be used
#  keep_bounds   [boolean] keep polygons boundaries
#  min_area      [float] minimum of merged polygons (only for new versions of SAGA)
#  stat_fields   [int, str, list] index, name or list of fields for statistics
#                 estimation for merged polygons. If stat_fields is None, fields
#                 statistics are not computing
#  field_names   [int] statistics fields names
#                 [0] variable type + original name (default)
#                 [1] original name + variable type
#                 [2] original name
#                 [3] variable type
# Available fields and their default values: sum (Sum=False), mean (Mean=False),
# minimum (Min=False), maximum (Max=False), range (Range=False), variance (Var=False)
# standard deviation (Std=False), values list of merged polygons (Listing=False) (list is
# separated with a0|a1|...|an), number of values of merged polygons (Count=False)
def polygon_dissolve(outshape, polygons, fields=None, keep_bounds=False, min_area=0,
                     stat_fields=None, Sum=False, Mean=False, Min=False, Max=False, Range=False,
                     Var=False, Std=False, Listing=False, Count=False, field_names=0):
    # Check inputs
    polygons = _files.default_file_ext(polygons, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # fields
    if type(fields) in [int, str]:
        fields = [str(fields)]  # create fields list
    elif type(fields) in [list, tuple]:
        fields = [str(field) for field in fields]  # convert arguments to strings
    elif fields is None:
        fields = ['NULL']
    else:
        raise TypeError('Bad fields type <{}>'.format(str(type(fields))))
    # statistics fields
    if type(stat_fields) in [str, int]:
        stat_fields = str(stat_fields)
    elif type(stat_fields) in [list, tuple]:
        stat_fields = [str(field) for field in stat_fields]
        stat_fields = ','.join(stat_fields)  # convert to text
    elif stat_fields is None:
        stat_fields = 'NULL'
    else:
        raise TypeError('Bad stat_fields type <{}>'.format(str(type(stat_fields))))
    # convert inputs to string
    keep_bounds = str(int(keep_bounds))
    min_area = str(min_area)
    Sum, Mean, Min, Max = str(int(Sum)), str(int(Mean)), str(int(Min)), str(int(Max))
    Range, Var, Std, Listing = str(int(Range)), str(int(Var)), str(int(Std)), str(int(Listing))
    Count = str(int(Count))
    if field_names < 0 or field_names > 3:
        field_names = 0  # default value
    field_names = str(field_names)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '5', '-POLYGONS', polygons,
           '-DISSOLVED', outshape, '-BND_KEEP', keep_bounds, '-STAT_FIELDS', stat_fields,
           '-STAT_SUM', Sum, '-STAT_AVG', Mean, '-STAT_MIN', Min, '-STAT_MAX', Max,
           '-STAT_RNG', Range, '-STAT_DEV', Std, '-STAT_VAR', Var, '-STAT_LST', Listing,
           '-STAT_NUM', Count, '-STAT_NAMING', field_names]
    if _env.saga_version[0] in ['2', '3']:  # old version of SAGA GIS
        for i in range(min([3, len(fields)])):
            cmd.extend(['-FIELD_%d' % (i + 1), str(fields[i])])
    elif _env.saga_version[0] in ['4', '5']:  # new version of SAGA GIS
        fields = ','.join(fields)
        cmd.extend(['-FIELDS', fields, '-MIN_AREA', min_area])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=polygons);
    return(flag)  # polygon_dissolve()


# Convert Polygon/Line vertices to points
# library: shapes_polygons  tool: 6
# INPUTS
#  outshape           [string] output points layer
#  inshape            [string] input lines or polygons layer
def convert_vertices_to_points(outshape, inshape):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '6', '-SHAPES', inshape,
           '-POINTS', outshape]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # convert_vertices_to_points()


# Compute various indices describing the shape of polygons.
# Indices are based on area, perimeter, maximum distance between the vertices of a polygon.
# library: shapes_polygons  tool: 7
# INPUTS
#  outshape           [string] output polygons with indices
#  inshape            [string] input polygons layer
def polygon_shape_indices(outshape, inshape):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '7', '-SHAPES', inshape,
           '-INDEX', outshape]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # polygon_shape_indices()


# library: shapes_polygons  tool: 8
# INPUTS
#  outshape           [string] splited polygons
#  polygons           [string] input polygons layer
#  lines              [string] input lines layer
def polygon_line_intersection(outshape, polygons, lines):
    # Check inputs
    polygons = _files.default_file_ext(polygons, 'vector')
    lines = _files.default_file_ext(lines, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '8', '-POLYGONS', polygons,
           '-LINES', lines, '-INTERSECT', outshape]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=polygons);
    return(flag)  # polygon_line_intersection()


# Splits parts of multipart polygons into separate polygons.
# This can be done only for islands (outer rings) or for all parts
# (inner and outer rings) by checking the 'lakes' option.
# library: shapes_polygons  tool: 10
# INPUTS
#  outshape           [string] output clipped layer
#  inshape            [string] clipper polygon
#  layes              [boolean] set True if you want separate all parts (inner
#                      and outer rings), if False only islands are separated.
def polygon_parts(outshape, inshape, lakes=True):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    lakes = str(int(lakes))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '10', '-POLYGONS', inshape,
           '-PARTS', outshape, '-LAKES', lakes]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # polygon_parts()


# Clipping of vector layers with a polygon layer
# library: shapes_polygons  tool: 11
# INPUTS
#  outshape           [string] output clipped layers
#  clip_shape         [string] clipper polygon
#  input_shape        [string] layer to be clipped
#  dissolve           [boolean] set True if you want dissolve clipped shapes
def polygon_clipping(outshape, clip_shape, input_shape, dissolve=True):
    # Check inputs
    clip_shape = _files.default_file_ext(clip_shape, 'vector')
    input_shape = _files.default_file_ext(input_shape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    dissolve = str(int(dissolve))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '11', '-CLIP', clip_shape, '-S_INPUT', input_shape,
           '-S_OUTPUT', outshape, '-DISSOLVE', dissolve, '-MULTIPLE', '0']
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=clip_shape);
    return(flag)  # polygon_clipping()


# Self-intersection of one layer's polygons
# library: shapes_polygons  tool: 12
# INPUTS
#  outshape       [string] output intersected polygons
#  inshape        [string] polygons to intersect
#  field          [string, int] field ID or field name of attribute table to
#                  name new polygons intersection.
def polygon_selfintersection(outshape, inshape, field=None):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    if field is None:
        field = -1
    field = str(field)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '12', '-POLYGONS', inshape,
           '-INTERSECT', outshape, '-ID', field]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # polygon_clipping()


# Calculates the geometric intersection of the overlayed polygon layers,
# i.e. layer A and layer B.
# library: shapes_polygons  tool: 14
# INPUTS
#  outshape       [string] output intersected polygons
#  layerA         [string] input shape
#  layerB         [string] input shape
#  split_parts    [boolean] set True if you want multipart polygons to become separate polygons.
def intersect(outshape, layerA, layerB, split_parts=True):
    # Check inputs
    layerA = _files.default_file_ext(layerA, 'vector')
    layerB = _files.default_file_ext(layerB, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    split = str(int(split_parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '14', '-A', layerA, '-B', layerB,
           '-RESULT', outshape, '-SPLIT', split]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layerA);
    return(flag)  # intersect()


# Calculates the geometric difference of the overlayed polygon layers,
# i.e. layer A less layer B.
# library: shapes_polygons  tool: 15
# INPUTS
#  outshape       [string] output difference polygons
#  layerA         [string] input shape
#  layerB         [string] input shape
#  split_parts    [boolean] set True if you want multipart polygons to become separate polygons.
def difference(outshape, layerA, layerB, split_parts=True):
    # Check inputs
    layerA = _files.default_file_ext(layerA, 'vector')
    layerB = _files.default_file_ext(layerB, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    split = str(int(split_parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '15', '-A', layerA, '-B', layerB,
           '-RESULT', outshape, '-SPLIT', split]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layerA);
    return(flag)  # difference()


# Calculates the symmetrical geometric difference of the overlayed polygon layers,
# i.e. layer A less layer B plus layer B less layer A.
# library: shapes_polygons  tool: 16
# INPUTS
#  outshape       [string] output difference polygons
#  layerA         [string] input shape
#  layerB         [string] input shape
#  split_parts    [boolean] set True if you want multipart polygons to become separate polygons.
def symetrical_difference(outshape, layerA, layerB, split_parts=True):
    # Check inputs
    layerA = _files.default_file_ext(layerA, 'vector')
    layerB = _files.default_file_ext(layerB, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    split = str(int(split_parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '16', '-A', layerA, '-B', layerB,
           '-RESULT', outshape, '-SPLIT', split]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layerA);
    return(flag)  # symetrical_difference()


# Calculates the geometric union of the overlayed polygon layers,
# i.e. the intersection plus the symmetrical difference of layers A and B.
# library: shapes_polygons  tool: 17
# INPUTS
#  outshape       [string] output union polygons
#  layerA         [string] input shape
#  layerB         [string] input shape
#  split_parts    [boolean] set True if you want multipart polygons to become separate polygons.
def polygon_union(outshape, layerA, layerB, split_parts=True):
    # Check inputs
    layerA = _files.default_file_ext(layerA, 'vector')
    layerB = _files.default_file_ext(layerB, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    split = str(int(split_parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '17', '-A', layerA, '-B', layerB,
           '-RESULT', outshape, '-SPLIT', split]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layerA);
    return(flag)  # polygon_union()


# Updates features of layer A with the features of layer B, i.e. all features
# of layer B will be supplemented with the difference of layer A less layer B plus.
# It is assumed, that both input layers share the same attribute structure.
# library: shapes_polygons  tool: 18
# INPUTS
#  outshape       [string] output union polygons
#  layerA         [string] input shape
#  layerB         [string] input shape
#  split_parts    [boolean] set True if you want multipart polygons to become separate polygons.
def polygon_update(outshape, layerA, layerB, split_parts=True):
    # Check inputs
    layerA = _files.default_file_ext(layerA, 'vector')
    layerB = _files.default_file_ext(layerB, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Convert to strings
    split = str(int(split_parts))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '18', '-A', layerA, '-B', layerB,
           '-RESULT', outshape, '-SPLIT', split]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layerA);
    return(flag)  # polygon_union()


# Removes invalid polygons, i.e. polygons with less than three vertices,
# and merges polygons belonging spatially together, i.e. forming outer and
# inner rings. Inner rings are not preserved as separate polygon, but become
# new part of the polygon forming the outer ring.
# library: shapes_polygons  tool: 21
# INPUTS
#  outshape       [string] output polygons
#  inshape        [string] input polygons
def flatten_polygon_layer(outshape, inshape):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    outshape = _files.default_file_ext(outshape, 'vector')
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_polygons', '21', '-INPUT', inshape,
           '-OUTPUT', outshape]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output shape has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # difference()


#==============================================================================
# Library: shapes_tools
#==============================================================================

# Merge vector layers in a shapefile
# library: shapes_tools  tool: 2
# INPUTS
#  outshape       [string] output shapefile
#  shape_list     [list] list of shapefiles that will be merge
#  addsource      [boolean] if addsource is True, a field with the
#                  name of the merged layers is added
#  match          [boolean] match fields by name
def merge_layers(outshape, shape_list, addsource=False, match=True):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    shape_list = [_files.default_file_ext(name, 'vector') for name in shape_list]
    # convert to text
    slist = ';'.join(shape_list)
    addsource = str(int(addsource))
    match = str(int(match))
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_tools', '2', '-INPUT', slist, '-MERGED',
           outshape, '-SRCINFO', addsource, '-MATCH', match]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=shape_list[0]);
    return(flag)  # merge_layers()


# Select shapes by attributes and save them in a new shape file
# library: shapes_tools  tool: 3, 6
# INPUTS
#  saveas        [string] output selected features shapefile
#  inshape       [string] input shape file
#  expression    [string, pandas.core.serie] expression parameter can be a
#                 conditional string compatible with pandas, as example
#                 expression='(field_X > 100) and (field_Y <=200) '
#                 expression also can be a pandas serie with bool data type
def select_by_attributes(saveas, inshape, expression):
    # Check inputs
    saveas = _files.default_file_ext(saveas, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    # Get expression
    table = _tables.get_attribute_table(inshape)  # get attribute table
    if type(expression) is _Series:
        if expression.dtype == 'bool':
            table['COND'] = expression.astype('int')
        else:
            raise TypeError('Pandas serie must have data type bool')
    elif type(expression) is str:
        table['COND'] = 0  # create new field
        cond = table.query(expression)
        table.loc[cond.index, 'COND'] = 1  # set rows with expression True
    else:
        raise TypeError('Wrong expression data type <{}>'.format(type(expression)))
    _tables.create_attribute_table(_files.default_file_ext(inshape, 'dbf'), table)
    # Create batch file
    t = 'shapes_tools 3 -SHAPES="{}" -FIELD="COND" -EXPRESSION="a = 1" -METHOD=0'
    t += '\nshapes_tools 6 -INPUT="{}" -OUTPUT="{}"'
    t = t.format(inshape, inshape, saveas)
    filename = _os.path.join('batch.txt')
    if _env.workdir is not None:
        filename = _os.path.join(_env.workdir, filename)
    with open(filename, 'w') as fid:
        fid.write(t)
    # Create cmd
    cmd = ['saga_cmd', '-f=s', filename]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete auxiliar data
    _os.remove(filename)
    _tables.delete_fields(inshape, inshape, fields='COND')
    # Check if output grid has crs file
    if not _files.has_crs_file(saveas):  # set first input layer crs
        _projection.set_crs(shapes=saveas, crs_method=1, proj=inshape);
    return(flag)  # select_by_attributes()


# Select shapes by location and save them in a new shape file
# library: shapes_tools  tool: 5, 6
# INPUTS
#  saveas        [string] output selected features shapefile
#  inshape       [string] input shape file with shapes to select
#  locations     [string] input shape file with areas to select features from
#                 inshape parameter
#  intersect     [bool] select from intersect
#  within        [bool] are completely within
#  contain       [bool] completely contain
#  centroid_in   [bool] have their centroid in locations shapes
#  centroid_of   [bool] contain the centeroid of locations shapes
def select_by_location(saveas, inshape, locations, intersect=False,
                       within=False, contain=False, centroid_in=False,
                       centroid_of=False):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    saveas = _files.default_file_ext(saveas, 'vector')
    # Create batch file
    t = ''
    if intersect:  # intersect
        c = '\nshapes_tools 5 -SHAPES="{}" -LOCATIONS="{}" -CONDITION=0 -METHOD=1'
        c = c.format(inshape, locations)
        t += c
    if within:  # are completely within
        c = '\nshapes_tools 5 -SHAPES="{}" -LOCATIONS="{}" -CONDITION=1 -METHOD=1'
        c = c.format(inshape, locations)
        t += c
    if contain:  # completely contain
        c = '\nshapes_tools 5 -SHAPES="{}" -LOCATIONS="{}" -CONDITION=2 -METHOD=1'
        c = c.format(inshape, locations)
        t += c
    if centroid_in:  # have their centroid in
        c = '\nshapes_tools 5 -SHAPES="{}" -LOCATIONS="{}" -CONDITION=3 -METHOD=1'
        c = c.format(inshape, locations)
        t += c
    if centroid_of:  # contain the centeroid of
        c = '\nshapes_tools 5 -SHAPES="{}" -LOCATIONS="{}" -CONDITION=4 -METHOD=1'
        c = c.format(inshape, locations)
        t += c
    c = '\nshapes_tools 6 -INPUT="{}" -OUTPUT="{}"'.format(inshape, saveas)
    t += c
    filename = _os.path.join('batch.txt')
    if _env.workdir is not None:
        filename = _os.path.join(_env.workdir, filename)
    with open(filename, 'w') as fid:
        fid.write(t)
    # Create cmd
    cmd = ['saga_cmd', '-f=s', filename]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete auxiliar data
    _os.remove(filename)
    # Check if output grid has crs file
    if not _files.has_crs_file(saveas):  # set first input layer crs
        _projection.set_crs(shapes=saveas, crs_method=1, proj=inshape);
    return(flag)  # copy_shapes_by_location()


# Create a shape graticule
# library: shapes_tools  tool: 13
# INPUTS
#  outshape       [string] output graticule shapefile
#  layer          [string] optional shapefile. If layer is not None,
#                  graticule is created using the layer extent. Output
#                  graticule takes the layer's crs
#  gtype          [int] type of graticule shapes
#                  [0] Lines
#                  [1] Rectangles
#  dx, dy         [float] graticule division width
#  extent         [list, tuple, array] graticule extent in the case that
#                  parameter layer is None. extent must be a 4 element object
#                  [xmin, xmax, ymin, ymax]
#  align          [int] cells alignment
#                  [0] bottom-left
#                  [1] top-left
#                  [2] bottom-right
#                  [3] top-right
#                  [4] centered
#  proj           [string] optional proj4 parameters. Only if layer is None
#  proj_file      [string] copy the .proj file of a layer. Only if layer is None
def create_graticule(outshape, layer=None, gtype=1, dx=10, dy=10,
                     extent=[-180, 180, -90, 90], align=0, proj=None, proj_file=None):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    if layer is None:
        layer = 'NULL'
    else:
        layer = _files.default_file_ext(layer, 'vector')
    if gtype < 0 or gtype > 1:
        gtype = 1
    if len(extent) != 4:
        raise ValueError('Parameter extent must have 4 elements [xmin, xmax, ymin, ymax]!')
    # convert to text
    gtype = str(gtype)
    dx, dy = str(dx), str(dy)
    extent = [str(value) for value in extent]
    align = str(align)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_tools', '12', '-GRATICULE_LINE', outshape, '-GRATICULE_RECT', outshape,
           '-TYPE', gtype, '-EXTENT', layer, '-DIVISION_X', dx, '-DIVISION_Y', dy, '-ALIGNMENT', align,
           '-EXTENT_X_MIN', extent[0], '-EXTENT_X_MAX', extent[1], '-EXTENT_Y_MIN', extent[2], '-EXTENT_Y_MAX',
           extent[3]]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if layer != 'NULL':  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=layer);
    elif proj is not None:
        _projection.set_crs(shapes=outshape, crs_method=0, proj=proj);
    elif proj_file is not None:
        _projection.set_crs(shapes=outshape, crs_method=1, proj=proj_file);
    return(flag)  # create_graticule()


# Creates a new shapefile using the location of the shapes
# library: shapes_tools  tool: 13
# INPUTS
#  outshape       [string] output shapefile
#  inshape        [string] input shape to be selected
#  extent_shape   [string] shapefile for the selection of shapes in inshapes parameter
#  condition      [int] conditional used by shapes selection
#                  [0] completely contained
#                  [1] intersects
#                  [2] center
#  extent         [int] use extent_shape parameter as:
#                  [0] shapes layer extent (default)
#                  [1] polygons
#  overlap        [float] minimum overlapping area as percentage of the total
#                  size of the input shape. Applies only if extent=1.
#                  For old versions of SAGA this parameter is ignored.
def copy_shapes_by_location(outshape, inshape, extent_shape, condition=0,
                            extent=0, overlap=50):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    extent_shape = _files.default_file_ext(extent_shape, 'vector')
    if condition < 0 or condition > 2:
        condition = 0  # default method
    extent += 2
    if extent < 2 or extent > 3:
        extent = 2  # default method
    # convert to string
    condition = str(condition)
    extent = str(extent)
    overlap = str(overlap)
    # Create cmd
    if _env.saga_version[0] in ['2']:  # method for old versions of SAGA
        cmd = ['saga_cmd', '-f=q', 'shapes_tools', '13', '-SHAPES', inshape,
               '-METHOD', condition, '-TARGET', extent, '-CUT', outshape,
               '-SHAPES_SHAPES', extent_shape, '-POLYGONS_POLYGONS', extent_shape]
    elif _env.saga_version[0] in ['3', '4', '5']:  # new versions of SAGA
        cmd = ['saga_cmd', '-f=q', 'shapes_tools', '13', '-SHAPES', inshape,
               '-METHOD', condition, '-EXTENT', extent, '-CUT', outshape,
               '-OVERLAP', overlap, '-SHAPES_EXT', extent_shape,
               '-POLYGONS', extent_shape]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # copy_shapes_by_location()


# Split shapes using a field in attribute table
# library: shapes_tools  tool: 17
# INPUTS:
#  outshape  [string] output shape file. Automatic name extension is generated
#             for each unique value in field attribute
#  inshape   [string] input shape file name
#  field     [int,string] attribute id number or name of inshape parameter
def split_shapes_by_attribute(outshape, inshape, field=0):
    # Check inputs
    inshape = _files.default_file_ext(inshape, 'vector')
    # Get atribute table
    table = _tables.get_attribute_table(inshape)
    if type(field) is int:
        label = table.columns
        field = label[0]
    table.sort_values(by=field, inplace=True)  # sort table
    labels = table[field].unique()  # get unique values
    # Create output names
    path = _os.path.dirname(outshape)
    base = _os.path.basename(outshape).split('.')[0]
    outshape = [_os.path.join(path, base + '_' + str(name) + '.shp') for name in labels]
    outshape = ';'.join(outshape)
    # Create and run cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_tools', '17', '-CUTS', outshape,
           '-TABLE', inshape, '-FIELD', str(field)]
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # split_shapes_by_attribute()


# Vector based buffer construction
# library: shapes_tools  tool: 18
# INPUTS
#  outshape    [string] output buffer shape
#  inshape     [string] output shapefile
#  dist        [int, float] define a single buffer distance for all features
#  field       [int, string] if field is None, dist parameter is used
#               If field is the index or name of the column in the attribute
#               table to use as a dist parameter for each feature
#  scale       [float] if field parameter is not None, scale is a factor
#               that multiplies the value of the attribute table
#  dissolve    [boolean] if dissolve is False (default) and nzones=1, then
#               buffer layer is storage for each feature. If dissolve
#               is True, a single buffer shape is created
#  nzones      [int] number of buffer zones
#  inner       [boolean] applies a inner buffer (only for polygons)
#  darc        [float] arc vertex distance [Degree]
def shapes_buffer(outshape, inshape, dist=100, field=None,
                  scale=1.0, dissolve=False, nzones=1, inner=False, darc=5.0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    inshape = _files.default_file_ext(inshape, 'vector')
    if nzones < 1:
        nzones = 1  # minimum value
    # convert to strings
    dist = str(dist)
    scale = str(scale)
    nzones = str(nzones)
    inner = str(int(inner))
    darc = str(darc)
    dissolve = str(int(dissolve))
    if field is None:
        field = '-1'
    else:
        field = str(field)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_tools', '18', '-SHAPES', inshape,
           '-BUFFER', outshape, '-DIST_FIELD_DEFAULT', dist, '-DIST_FIELD',
           field, '-DIST_SCALE', scale, '-NZONES', nzones, '-DARC', darc,
           '-DISSOLVE', dissolve, '-POLY_INNER', inner]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(shapes=outshape, crs_method=1, proj=inshape);
    return(flag)  # shapes_buffer()


# Convert a delimited text file (.txt or .csv) to a point shape file
# library: shapes_tools  tool: 22
# INPUTS
#  outshape    [string] output pints shape file
#  intable     [string] input delimited table .txt (tabulated) or .csv
#  field_id    [int, string] column number or name of feature ID
#  field_x     [int, string] column number or name of X coordinate
#  field_y     [int, string] column number or name of Y coordinate
#  proj        [string] proj parameter text. See set_crs() method
#  proj_file   [string] .prj file with projection information
def generate_shapes_from_table(outshape, intable, field_id=0, field_x=1,
                               field_y=2, proj=None, proj_file=None):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector')
    # convert to string
    field_id = str(field_id)
    field_x = str(field_x)
    field_y = str(field_y)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'shapes_tools', '22', '-INPUT', intable,
           '-OUTPUT', outshape, '-FIELD_ID', field_id, '-FIELD_X', field_x,
           '-FIELD_Y', field_y, '-SHAPE_TYPE', '0']
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if proj is not None:  # set first input layer crs
        _projection.set_crs(grids=outshape, crs_method=0, proj=proj);
    elif proj_file is not None:
        _projection.set_crs(shapes=outshape, crs_method=1, proj=proj_file);
    return(flag)  # convert_table_to_points()