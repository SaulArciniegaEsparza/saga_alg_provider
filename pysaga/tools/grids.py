"""
SAGA GIS algorithm provider
Grids tools

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# env is the provider class
import os as _os
import numpy as _np
import pandas.core.series.Series as _Series
import itertools as _itertools

import files as _files
import projection as _projection
from tables import get_attribute_table as _get_attribute_table
import _gridsio as _io


#==============================================================================
# Library: grid_analysis
#==============================================================================

# Calculation of accumulated cost, either isotropic or anisotropic, if direction
# of maximum cost is specified.
# library: grid_analysis  tool: 0
# INPUTS
#  accumulated        [string] output grid of accumulated cost
#  allocation         [string] output allocation grid for destinations
#  cost               [string] input local cost grid
#  destinations       [string] must be shape file or a grid file with the location
#                      of destinations
#  direction          [string] optional input grid with the direction of maximum cost
#  dir_unit           [int] units of direction grid
#                      [0] radians (default)
#                      [1] degree
#  dir_k              [int, float] k factor for effective friction
#                      stated friction ^ {cos(DifAngle)^k}
#  threshold          [int, float] Threshold for different route By default 0
def accumulated_cost(accumulated, allocation, cost, destinations, direction=None,
                     dir_unit=0, dir_k=2, threshold=0):
    # Check inputs
    accumulated = _files.default_file_ext(accumulated, 'grid')
    allocation = _files.default_file_ext(allocation, 'grid')
    cost = _files.default_file_ext(cost, 'grid', False)

    if direction is None:
        direction = 'NULL'
    else:
        direction = _files.default_file_ext(direction, 'grid', False)

    if destinations.endswith('.shp'):
        dest_points = True
    else:
        destinations = _files.default_file_ext(destinations, 'grid', False)
        dest_points = False

    # Convert to strings
    if dir_unit < 0 or dir_unit > 1:
        dir_unit = 0

    dir_unit, dir_k = str(dir_unit), str(dir_k)
    threshold = str(threshold)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '0', '-COST', cost, '-ACCUMULATED',
           accumulated, '-ALLOCATION', allocation, '-DIR_MAXCOST', direction,
           '-DIR_UNIT', dir_unit, '-DIR_K', dir_k, '-THRESHOLD', threshold]
    if dest_points:
        cmd.extend(['-DEST_TYPE', '0', '-DEST_POINTS', destinations])
    else:
        cmd.extend(['-DEST_TYPE', '1', '-DEST_GRID', destinations])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(accumulated):
        _projection.set_crs(grids=accumulated, crs_method=1, proj=cost);
    if not _files.has_crs_file(allocation):
        _projection.set_crs(grids=allocation, crs_method=1, proj=cost);
    return(flag)  # accumulated_cost()


# Compute least cost path profile(s). It takes an accumulated cost surface grid and
# a point shapefile as input. Each point in the shapefile represents a source for
# which the least cost path is calculated.
# library: grid_analysis  tool: 5
# INPUTS
#  grid              [string] accumulated cost grid
#  points            [string] source points shape file
#  field_id          [int, string] attribute id or name to be used as name of multiple
#                     source points
#  profile_lines     [string] output lines profiles shape file. For multiple source points
#                     profile_lines is used a basename plus basename
#  profile_points    [string] output points profiles shape file. For multiple source points
#                     points_lines is used a basename plus basename
#  values            [string, list, tuple] single or multiple grid files which values are
#                      to the outputs as additional table fields
def least_cost_paths(grid, points, profile_lines=None, profile_points=None, field_id=0,
                     values=None):
    # Check inputs
    grid = _files.default_file_ext(grid, 'grid', False)
    points = _files.default_file_ext(points, 'vector')
    if type(values) is str:
        values = [values]
    elif type(values) in [list, tuple]:
        values = [_files.default_file_ext(value, 'grid', False) for value in values]
        values = ';'.join(values)
    else:
        values = 'NULL'

    # Get attribute table
    table = _get_attribute_table(points)  # get attribute table
    field_values = table.ix[:, field_id]  # get field values

    # Output file names
    if profile_lines is not None:
        base_line = _os.path.splitext(profile_lines)[0]
    if profile_points is not None:
        base_points = _os.path.splitext(profile_points)[0]

    out_line, out_points = [], []
    if len(field_values) == 1:
        if profile_lines is not None:
            out_line.append(base_line + '.shp')
        if profile_points is not None:
            out_points.append(base_points + '.shp')

    else:
        for field_value in field_values:
            if profile_lines is not None:
                out_line.append(base_line + '_' + str(field_value) + '.shp')
            if profile_points is not None:
                out_points.append(base_points + '_' + str(field_value) + '.shp')

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '5', '-DEM', grid, '-SOURCE',
           points, '-VALUES', values]
    if out_line:
        line_list = ';'.join(out_line)
        cmd.extend(['-LINE', line_list])
    if out_points:
        points_list = ';'.join(out_points)
        cmd.extend(['-POINTS', points_list])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    for filename in out_line:
        if not _files.has_crs_file(filename):
            _projection.set_crs(grids=filename, crs_method=1, proj=points);
    for filename in out_points:
        if not _files.has_crs_file(filename):
            _projection.set_crs(grids=filename, crs_method=1, proj=points);

    return(flag)  # least_cost_paths()



# library: grid_analysis  tool: 6
# INPUTS
#  out_dist      [string] output distance grid
#  out_dir       [string] output direction grid
#  initial       [string, list, tuple] initial state grid or list of grids
#  final         [string, list, tuple] final state grid or list of grids
#  angle         [boolean] if True angle is calculated. Only available when
#                 exact two initial grids are compared
#  change        [boolean] if True output of change vector is compared
#  color         [boolean] If True a classification color table for out_grid
#                 is created in the same path that out_dir grid
def change_vector_analysis(out_dist, out_dir, initial, final, angle=True,
                           change=False, color=False):
    # Check inputs
    out_dist = _files.default_file_ext(out_dist, 'grid')
    out_dir = _files.default_file_ext(out_dir, 'grid')

    if type(initial) is str:
        initial = [_files.default_file_ext(initial, 'grid', False)]
    elif type(initial) in [list, tuple]:
        initial = [_files.default_file_ext(grid, 'grid', False) for grid in initial]
    else:
        raise TypeError('initial must be a grid or a list of grids!')

    if type(final) is str:
        final = [_files.default_file_ext(final, 'grid', False)]
    elif type(final) in [list, tuple]:
        final = [_files.default_file_ext(grid, 'grid', False) for grid in final]
    else:
        raise TypeError('final must be a grid or a list of grids!')

    if len(initial) != len(final):
        raise TypeError('initial and final must have the same number of grids!')

    initial_list = ';'.join(initial)
    final_list = ';'.join(final)

    # Additional inputs
    angle, change = str(int(angle)), str(int(change))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '6', '-A', initial_list,
           '-B', final_list, '-DIST', out_dist, '-DIR', out_dir,
           '-ANGLE', angle, '-C_OUT', change]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(out_dist):
        _projection.set_crs(grids=out_dist, crs_method=1, proj=initial[0]);
    if not _files.has_crs_file(out_dir):
        _projection.set_crs(grids=out_dir, crs_method=1, proj=initial[0]);

    # Output color classification for out_dir
    if color:
        combination = list(_itertools.product(['-', '+'], repeat=len(initial)))
        colors = _np.random.randint(3165924, 12656600, (len(combination)))
        color_file = _os.path.splitext(out_dir)[0] + 'Direction_classification.txt'
        with open(color_file, 'w') as fout:
            fout.write('COLOR\tNAME\tDESCRIPTION\tMINIMUM\tMAXIMUM\n')
            for i in range(len(combination)):
                row = list(combination[i])
                row.reverse()
                value_id = ''.join(row)
                fout.write('%d\t"%s"\t""\t%.4f\t%.4f\n' % (colors[i], value_id, i, i))

    return(flag)  # change_vector_analysis()


# Covered Distance between grids
# library: grid_analysis  tool: 7
# INPUTS
#  outgrid       [string] output covered distance grid
#  grids         [list, tuple] input grid list
def covered_distance(outgrid, grids):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grids = [_files.default_file_ext(grid, 'grid', False) for grid in grids]
    grid_list = ';'.join(grids)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '7', '-INPUT', grid_list,
           '-RESULT', outgrid]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # covered_distance()


# Creates a new grid containing the ID of the grid with the extreme value
# library: grid_analysis  tool: 9
# INPUTS
#  outgrid       [string] output covered distance grid
#  grids         [list, tuple] input grid list
#  method        [int] criteria for extreme value
#                 [0] Maximum (default)
#                 [1] Minimum
def layer_of_extreme_value(outgrid, grids, method=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grids = [_files.default_file_ext(grid, 'grid', False) for grid in grids]
    grid_list = ';'.join(grids)

    # Convert methods to strings
    if method < 0 or method > 1:
        method = 0
    method = str(method)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '9', '-GRIDS', grid_list,
           '-RESULT', outgrid, '-CRITERIA', method]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # layer_of_extreme_value()


# Analytical Hierarchy Process for multiple criteria decision making
# library: grid_analysis  tool: 10
# INPUTS
#  outgrid           [string] output grid
#  grids             [list, tuple] list of input grids
#  pairwise          [list, tuple, array, string] .txt table delimited by tabs and headers. Also can
#                     be a list, tuple or array with the factor weights. pairwise must content a
#                     squared array with the same number of rows and cols as input grids.
#                     Each value in the pairwise array can be expressed as aij, where aij is higher
#                     for more important properties.
#                     When i=j aij is equal to 1. For the upper triangle values are aij and the
#                     lower triangle matrix values are aji = 1/aij
#                     Preferences in numerical values for comparisons are expressed as:
#                      1/9    Extremely less important
#                      1/7    Very strongly less important
#                      1/5    Strongly less important
#                      1/3    Moderately less important
#                      1      Equal importance
#                      3      Moderately more important
#                      5      Strongly more important
#                      7      Very strongly more important
#                      9      Extremely more important
def analytical_hierarchy_process(outgrid, grids, pairwise):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')

    if type(grids) not in [list, tuple]:
        raise TypeError('grids must be a list or tuple with at least 2 elements!')
    else:
        if len(grids) < 2:
            raise TypeError('grids must be a list or tuple with at least 2 elements!')

    grids = [_files.default_file_ext(grid, 'grid', False) for grid in grids]
    ng = len(grids)
    grids_list = ';'.join(grids)

    # Pairwise Comparisons
    if type(pairwise) in [list, tuple]:
        pairwise = _np.array(pairwise, dtype=_np.float32)

    if type(pairwise) is _np.ndarray:
        r, c = pairwise.shape
        if (r != ng) or (c != ng):
            raise TypeError('matrix must be a squared array considering the number of input grids!')

        # Create temporary file
        data = pairwise.copy()
        pairwise = _files.create_filename(_env.workdir, 'txt', 'auxiliar_AHP')

        # write file
        header = '\t'.join([str(j) for j in range(ng)])
        _np.savetxt(pairwise, data, fmt='%.4f', header=header, delimiter='\t', comments='')

    # check if pairwise is a file
    if type(pairwise) is str:
        if not _os.path.exists(pairwise):
            raise IOError('pairwise file does not exist!')
    else:
        raise TypeError('pairwise must be a .txt file delimited by tabs or a squared array!')

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '10', '-GRIDS', grids_list,
           '-OUTPUT', outgrid, '-TABLE', pairwise]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # analytical_hierarchy_process()


# Ordered Weighted Averaging (OWA)
# library: grid_analysis  tool: 11
# INPUTS
#  outgrid       [string] output grid
#  grids         [list, tuple] input grid list
#  weights       [list, tuple] input weights for each grid
def ordered_weighted_averaging(outgrid, grids, weights):
    # Check inputs
    if len(grids) != len(weights):
        raise IOError('Number of grids must be equal to number of weights!')

    # Create equation
    for i in range(len(weights)):
        if i == 0:
            eq = '%g * g%d' % (weights[i], i + 1)
        else:
            eq += '+ %g * g%d' % (weights[i], i + 1)

    # Calculate averange
    flag = calculator(outgrid, grids, formula=eq, data_type=7)
    return(flag)  # ordered_weighted_averaging()


# Derive soil texture classes from sand, silt and clay contents
# library: grid_analysis  tool: 14
# INPUTS
#  outgrid       [string] output texture grid
#  sand          [string] input grid of sand content given as percentage
#  silt          [string] input grid of silt content given as percentage
#  clay          [string] input grid of clay content given as percentage
#  method        [int] texture classification scheme
#                 [0] USDA (dafult) For SAGA 2. and 3. this is the unique method
#                 [1] Germany KA5
#                 [2] Belgium/France
#  color         [boolean] If True a classification color table is created in
#                 the same path that outgrid
def soil_texture_classification(outgrid, sand, silt, clay, method=0, color=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    sand = _files.default_file_ext(sand, 'grid', False)
    silt = _files.default_file_ext(silt, 'grid', False)
    clay = _files.default_file_ext(clay, 'grid', False)

    # Check method
    if method < 0 or method > 2:
        method = 0
    if _env.saga_version[0] in ['2', '3']:
        method = 0

    scheme = str(method)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_analysis', '14', '-SAND', sand,
           '-SILT', silt, '-CLAY', clay, '-TEXTURE', outgrid]

    if _env.saga_version[0] not in ['2', '3']:
        cmd.extend(['-SCHEME', scheme])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=sand);

    # Create classification file
    if color:  # write classification table
        color_file = _os.path.splitext(outgrid)[0]
        if method == 0:  # SCHEME USDA
            color_file += 'USDA_color_table.txt'
            with open(color_file, 'w') as fid:
                fid.write(
"""COLOR	NAME	DESCRIPTION	MINIMUM	MAXIMUM
16711680	"C"	"Clay"	1.000000	1.000000
16762880	"SiC"	"Silty Clay"	2.000000	2.000000
13158400	"SiCL"	"Silty Clay Loam"	3.000000	3.000000
16711880	"SC"	"Sandy Clay"	4.000000	4.000000
13158600	"SCL"	"Sandy Clay Loam"	5.000000	5.000000
13139839	"CL"	"Clay Loam"	6.000000	6.000000
65280	"Si"	"Silt"	7.000000	7.000000
8388479	"SiL"	"Silt Loam"	8.000000	8.000000
8355711	"L"	"Loam"	9.000000	9.000000
255	"S"	"Sand"	10.000000	10.000000
8323327	"LS"	"Loamy Sand"	11.000000	11.000000
8355784	"SL"	"Sandy Loam"	12.000000	12.000000
"""
                )
        elif method == 1:  # SCHEME GERMANY KA5
            color_file += 'Germany_KA5_color_table.txt'
            with open(color_file, 'w') as fid:
                fid.write(
"""COLOR	NAME	DESCRIPTION	MINIMUM	MAXIMUM
1251832	"Ss"	"Reiner Sand"	1.000000	1.000000
3365368	"Su2"	"Schwach schluffiger Sand"	2.000000	2.000000
4020718	"Sl2"	"Schwach lehmiger Sand"	3.000000	3.000000
5865445	"Sl3"	"Mittel lehmiger Sand"	4.000000	4.000000
2628066	"St2"	"Schwach toniger Sand"	5.000000	5.000000
6137332	"Su3"	"Mittel schluffiger Sand"	6.000000	6.000000
8185332	"Su4"	"Stark schluffiger Sand"	7.000000	7.000000
9627103	"Slu"	"Schluffig-lemiger Sand"	8.000000	8.000000
6586330	"Sl4"	"Stark lehmiger Sand"	9.000000	9.000000
4728521	"St3"	"Mittel toniger Sand"	10.000000	10.000000
11068873	"Ls2"	"Schwach sandiger Lehm"	11.000000	11.000000
9351881	"Ls3"	"Mittel sandiger Lehm"	12.000000	12.000000
7238345	"Ls4"	"Stark sandiger Lehm"	13.000000	13.000000
11717810	"Lt2"	"Schwach toniger Lehm"	14.000000	14.000000
9597605	"Lts"	"Sandig-toniger Lehm"	15.000000	15.000000
6235826	"Ts4"	"Stark sandiger Ton"	16.000000	16.000000
7939737	"Ts3"	"Mittel sandiger Ton"	17.000000	17.000000
15400949	"Uu"	"Reiner Schluff"	18.000000	18.000000
11534324	"Us"	"Sandiger Schluff"	19.000000	19.000000
14680037	"Ut2"	"Schwach toniger Schluff"	20.000000	20.000000
15007706	"Ut3"	"Mittel toniger Schluff"	21.000000	21.000000
11730911	"Uls"	"Sandig-lehmiger Schluff"	22.000000	22.000000
15531978	"Ut4"	"Stark toniger Schluff"	23.000000	23.000000
13565891	"Lu"	"Schluffiger Lehm"	24.000000	24.000000
13421721	"Lt3"	"Mittel toniger Lehm"	25.000000	25.000000
15531938	"Tu3"	"Mittel schluffiger Ton"	26.000000	26.000000
16187318	"Tu4"	"Stark schluffiger Ton"	27.000000	27.000000
10430066	"Ts2"	"Schwach sandiger Ton"	28.000000	28.000000
12939890	"Tl"	"Lehmiger Ton"	29.000000	29.000000
15320184	"Tu2"	"Schwach schluffiger Ton"	30.000000	30.000000
14760763	"Tt"	"Reiner Ton"	31.000000	31.000000
"""
                )
        else:  # SCHEME BELGIUM-FRANCE
            color_file += 'Belgium-France_color_table.txt'
            with open(color_file, 'w') as fid:
                fid.write(
"""COLOR	NAME	DESCRIPTION	MINIMUM	MAXIMUM
13003117	"U"	"Heavy Clay"	1.000000	1.000000
10136506	"E"	"Clay"	2.000000	2.000000
15531997	"A"	"Silt Loam"	3.000000	3.000000
10747877	"L"	"Sandy Loam"	4.000000	4.000000
6927856	"P"	"Light Sandy Loam"	5.000000	5.000000
3951335	"S"	"Loamy Sand"	6.000000	6.000000
1778421	"Z"	"Sand"	7.000000	7.000000
"""
                )

    return(flag)  # soil_texture_classification()


#==============================================================================
# Library: grid_calculus
#==============================================================================

# Normalize the values of a grid. Rescales all grid values to fall in the range
# 'Minimum' to 'Maximum' defined by user
# library: grid_calculus  tool: 0
# INPUTS
#  outgrid      [string] output grid calculus
#  ingrid       [string] input grid file
#  drange       [list] two elements list [minimum, maximum]. If None drange
#                is set as [0, 1]
def normalization(outgrid, ingrid, drange=None):
    # Check inputs
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    outgrid = _files.default_file_ext(outgrid, 'grid')

    if drange is None:
        drange = [0, 1]
    if type(drange) in [list, tuple, _np.ndarray]:
        if len(drange) != 2:
            raise TypeError('Bad drange number of elements!')
    else:
        raise TypeError('Bad drange type <{}>'.format(str(type(drange))))
    vmin, vmax = str(min(drange)), str(max(drange))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '0', '-INPUT', ingrid, '-OUTPUT',
           outgrid, '-RANGE_MIN', vmin, '-RANGE_MAX', vmax]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)   # normalization()


# Calculates a new grid based on existing grids and a mathematical formula
# library: grid_calculus  tool: 1
# INPUTS
#  outgrid      [string] output grid calculus
#  grids        [string, list] grid name or list of grids with the same
#                grid system. In formula, each grid must be named with 'g'
#                follow with the grid number starting with 1. For example
#                if grids=['grid.sgrd', 'other_grid.sgrd'], in formula
#                must be g1 and g2, respectively
#  formula      [string] grids formula. By default formula='g1 + 10'. When
#                grids and hgrids are input, formula must have g1,g2,...gn
#                and h1,h2,...,hn, depending of the number of grids
#  hgrids       [string, list] additional grid name or list of grids with a
#                different grid system that grids. In formula, each grid
#                must be named with 'h' follow of the grid number starting with 1
#  data_type    [int] output grid data type
#                [0] bit
#                [1] unsigned 1 byte integer
#                [2] signed 1 byte integer
#                [3] unsigned 2 byte integer
#                [4] signed 2 byte integer
#                [5] unsigned 4 byte integer
#                [6] signed 4 byte integer
#                [7] 4 byte floating point number (default)
#                [8] 8 byte floating point number
#  resampling   [int] resampling method for hgrids
#                [0] Nearest Neighbour
#                [1] Bilinear Interpolation
#                [2] Bicubic Spline Interpolation
#                [3] B-Spline Interpolation
#  use_nodata   [boolean] if is True include NoData cells in the calculation
# The following operators are available for the formula definition:
#  +             Addition
#  -             Subtraction
#  *             Multiplication
#  /             Division
#  abs(x)        Absolute Value
#  mod(x, y)     Returns the floating point remainder of x/y
#  int(x)        Returns the integer part of floating point value x
#  sqr(x)        Square
#  sqrt(x)       Square Root
#  exp(x)        Exponential
#  pow(x, y)     Returns x raised to the power of y
#  x ^ y         Returns x raised to the power of y
#  ln(x)         Natural Logarithm
#  log(x)        Base 10 Logarithm
#  pi()          Returns the value of Pi
#  sin(x)        Sine
#  cos(x)        Cosine
#  tan(x)        Tangent
#  asin(x)       Arcsine
#  acos(x)       Arccosine
#  atan(x)       Arctangent
#  atan2(x, y)   Arctangent of x/y
#  gt(x, y)      Returns true (1), if x is greater than y, else false (0)
#  x > y         Returns true (1), if x is greater than y, else false (0)
#  lt(x, y)      Returns true (1), if x is less than y, else false (0)
#  x < y         Returns true (1), if x is less than y, else false (0)
#  eq(x, y)      Returns true (1), if x equals y, else false (0)
#  x = y         Returns true (1), if x equals y, else false (0)
#  and(x, y)     Returns true (1), if both x and y are true (i.e. not 0)
#  or(x, y)      Returns true (1), if at least one of both x and y is true (i.e. not 0)
#  ifelse(c, x, y)  Returns x, if condition c is true (i.e. not 0), else y
#  rand_u(x, y)  Random number, uniform distribution with minimum x and maximum y
#  rand_g(x, y)  Random number, Gaussian distribution with mean x and standard deviation y
#  xpos(), ypos()   Get the x/y coordinates for the current cell
#  row(), col()  Get the current cell's column/row index
#  nodata()      Returns resulting grid's no-data value
def calculator(outgrid, grids, formula='g1 + 1', hgrids=None,
               data_type=7, resampling=3, use_nodata=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    # check input grids with the same grid system
    if type(grids) is str:
        grids = _files.default_file_ext(grids, 'grid', False)
        gridsys = grids
    elif type(grids) in [list, tuple]:
        gridsys = _files.default_file_ext(grids[0], 'grid')
        grids = ';'.join([_files.default_file_ext(grid, 'grid') for grid in grids])
    else:
        raise TypeError('Wrong grids type <{}>'.format(type(grids)))
    # check for additional grids
    if type(hgrids) is str:
        hgrids = _files.default_file_ext(hgrids, 'grid')
    elif type(hgrids) in [list, tuple]:
        hgrids = ';'.join([_files.default_file_ext(grid, 'grid') for grid in hgrids])
    else:
        hgrids = 'NULL'
    if type(formula) is not str:
        raise TypeError('Wrong formula type <{}>'.format(type(formula)))
    if data_type < 0 or data_type > 8:
        data_type = 7  # default data type
    if resampling < 0 or resampling > 3:
        resampling = 3
    # convert to strings
    data_type = str(data_type)
    resampling = str(resampling)
    nodata = str(int(use_nodata))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '1', '-GRIDS', grids, '-RESULT',
           outgrid, '-XGRIDS', hgrids, '-FORMULA', formula, '-TYPE', data_type,
           '-RESAMPLING', resampling, '-USE_NODATA', nodata]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=gridsys);
    return(flag)  # calculator()


# Cellwise addition of grid values
# library: grid_calculus  tool: 8
# INPUTS
#  outgrid        [string] output sum grid
#  grids          [string, list, tuple] input grids list
#  nodata         [boolean] if True counts no data as zero
def grids_sum(outgrid, grids, nodata=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    if type(grids) in [list, tuple]:
        grids = [_files.default_file_ext(grid, 'grid') for grid in grids]
    else:
        raise TypeError('Wrong grids type <{}>'.format(type(grids)))
    grid_list = ';'.join(grids)

    # Check additional inputs
    nodata = str(int(nodata))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '8', '-GRIDS', grid_list,
           '-RESULT', outgrid, '-NODATA', nodata]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # grids_sum()


# Cellwise multiplication of grid values
# library: grid_calculus  tool: 9
# INPUTS
#  outgrid        [string] output product grid
#  grids          [list, tuple] input grids list
#  nodata         [boolean] if True counts no data as zero
def grids_product(outgrid, grids, nodata=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    if type(grids) in [list, tuple]:
        grids = [_files.default_file_ext(grid, 'grid') for grid in grids]
    else:
        raise TypeError('Wrong grids type <{}>'.format(type(grids)))
    grid_list = ';'.join(grids)

    # Check additional inputs
    nodata = str(int(nodata))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '9', '-GRIDS', grid_list,
           '-RESULT', outgrid, '-NODATA', nodata]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # grids_product()


# Standardise the values of a grid. The standard score (z) is calculated
# as raw score (x) less arithmetic mean (m) divided by standard deviation (s).
# z = (x - m) * s
# library: grid_calculus  tool: 10
# INPUTS
#  outgrid      [string] output grid file
#  grid         [string] input grid file
#  stretch      [float] Stretch factor
def standardisation(outgrid, grid, stretch=1.0):
    # Check inputs
    ingrid = _files.default_file_ext(grid, 'grid', False)
    outgrid = _files.default_file_ext(outgrid, 'grid')

    # Check additional inputs
    stretch = str(stretch)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '10', '-INPUT', grid,
           '-OUTPUT', outgrid, '-STRETCH', stretch]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # standardisation()


# Translates grid values into fuzzy set membership as preparation for fuzzy set analysis
# library: grid_calculus  tool: 11
# INPUTS
#  outgrid      [string] output fuzzify grid
#  ingrid       [string] input grid file
#  coefs        [dict] values of the boundary of a membership function
#                coefs must contain keys 'A', 'B', 'C' and 'D' where 0 <= A <= B <= C <= D
#                Values lower than A will be set to 0.
#                Values between A and B increase from 0 to 1.
#                Values between B and C will be set to 1.
#                Values between C and D decrease from 1 to 0.
#                Values greater than D will be set to 0.
#  method       [int] Membership transition function Type
#                [0] linear (default)
#                [1] sigmoidal
#                [2] j-shaped
#  invert       [boolean] if True fuzzy classification is inverted
#  adjust       [boolean] automatically adjust control points to grid's data range
def fuzzify(outgrid, grid, coefs, method=0, invert=False, adjust=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)

    # Additional inputs
    if method < 0 or method > 2:
        method = 0

    method = str(method)
    invert, adjust = str(int(invert)), str(int(adjust))

    # Check parameters
    keys = ['A', 'B', 'C', 'D']
    dcoefs = dict.fromkeys(keys, 0)
    dcoefs.update(coefs)

    for i in range(4):
        if dcoefs[keys[i]] < 0:
            dcoefs[keys[i]] = 0
        if i > 0: # Check if coefs[i] is grater than coefs[i-1]
            if dcoefs[keys[i]] < dcoefs[keys[i - 1]]:
                dcoefs[keys[i]] = dcoefs[keys[i - 1]]
        dcoefs[keys[i]] = str(dcoefs[keys[i - 1]])

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '11', '-INPUT', grid, '-OUTPUT',
           outgrid, '-AUTOFIT', adjust]

    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-TYPE', method, '-A', dcoefs['A'], '-B', dcoefs['B'],
                    '-C', dcoefs['C'], '-D', dcoefs['D']])
    else:
        cmd.extend(['-TRANSITION', method, '-INVERT', invert, '-METHOD', '2',
                    '-INC_MIN', dcoefs['A'], '-INC_MAX', dcoefs['B'],
                    '-DEC_MIN', dcoefs['C'], '-DEC_MAX', dcoefs['D']])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # fuzzify()


# Calculates the intersection (min operator) for each grid cell
# of the selected grids
# library: grid_calculus  tool: 12
# INPUTS
#  outgrid        [string] output fuzzy grid
#  grids          [string, list, tuple] input grid file or grid list
#  operator       [int] operator type
#                  [0] min(a, b) (non-interactive) (default)
#                  [1] a * b
#                  [2] max(0, a + b - 1)
def fuzzy_intersection(outgrid, grids, operator=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    if type(grids) is str:
        grids = [_files.default_file_ext(grids, 'grid', False)]
    elif type(grids) in [list, tuple]:
        grids = [_files.default_file_ext(grid, 'grid') for grid in grids]
    else:
        raise TypeError('Wrong grids type <{}>'.format(type(grids)))
    grid_list = ';'.join(grids)

    # Convert to string
    if operator < 0 or operator > 2:
        operator = 0
    operator = str(operator)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '12', '-GRIDS', grid_list,
           '-AND', outgrid, '-TYPE', operator]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # fuzzy_intersection()


# Calculates the union (max operator) for each grid cell
# of the selected grids.
# library: grid_calculus  tool: 13
# INPUTS
#  outgrid        [string] output fuzzy grid
#  grids          [string, list, tuple] input grid file or grid list
#  operator       [int] operator type
#                  [0] max(a, b) (non-interactive) (default)
#                  [1] a + b - a * b
#                  [2] min(1, a + b)
def fuzzy_union(outgrid, grids, operator=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    if type(grids) is str:
        grids = [_files.default_file_ext(grids, 'grid', False)]
    elif type(grids) in [list, tuple]:
        grids = [_files.default_file_ext(grid, 'grid') for grid in grids]
    else:
        raise TypeError('Wrong grids type <{}>'.format(type(grids)))
    grid_list = ';'.join(grids)

    # Convert to string
    if operator < 0 or operator > 2:
        operator = 0
    operator = str(operator)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_calculus', '13', '-GRIDS', grid_list,
           '-OR', outgrid, '-TYPE', operator]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
    return(flag)  # fuzzy_union()


#==============================================================================
# Library: grid_filter
#==============================================================================

# Simple standard filters for grids
# library: grid_filter  tool: 0
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  method           [int] filter method
#                    [0] Smooth (default)
#                    [1] Sharpen
#                    [2] Edge
#  kernel           [int] shape of the filter kernel
#                    [0] Square
#                    [1] Circle (default)
#  radius           [int] kernel radius in cells
def simple_filter(outgrid, grid, method=0, kernel=1, radius=2):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    if method < 0 or method > 2:
        method = 0
    if kernel < 0 or kernel > 1:
        kernel = 1
    method, kernel, radius = str(method), str(kernel), str(int(radius))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '0', '-INPUT', grid,
           '-RESULT', outgrid, '-METHOD', method]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-MODE', kernel, '-RADIUS', radius])
    else:
        cmd.extend(['-KERNEL_TYPE', kernel, '-KERNEL_RADIUS', radius])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # simple_filter()


# Smoothing grids using a Gaussian filter for remove detail and noise.
# library: grid_filter  tool: 1
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  sigma            [int, float] degree of smoothing is determined given
#                    by the standard deviation. By default sigma is 200
#  kernel           [int] kernel type
#                    [0] Square
#                    [1] Circle (default)
#  radius           [int, float] kernel radius in cells. For higher standard
#                    deviations you need a greater Radius
def gaussian_filter(outgrid, grid, sigma=200, kernel=1, radius=5):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    if kernel < 0 or kernel > 1:
        kernel = 1  # default method
    sigma, kernel, radius = str(sigma), str(kernel), str(radius)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '1', '-INPUT', grid,
           '-RESULT', outgrid, '-SIGMA', sigma]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-MODE', kernel, '-RADIUS', radius])
    else:
        cmd.extend(['-KERNEL_TYPE', kernel, '-KERNEL_RADIUS', radius])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # gaussian_filter()


# User defined filter matrix
# library: grid_filter  tool: 4
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  matrix           [string, list, tuple, array] matrix must be a .txt file with
#                    headers delimited with tabs, containing a 3x3 matrix.
#                    matrix can be a 3x3 list, tuple or numpy array
#  absolute         [boolean] If True absolute weighting is used
def user_defined_filter(outgrid, grid, matrix=None, absolute=True):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to strings
    absolute = str(int(absolute))

    # Filter matrix
    if matrix is None:  # default value
        matrix = [[0.25, 0.5, 0.25],
                  [0.5, -1.0, 0.50],
                  [0.25, 0.5, 0.25]]

    if type(matrix) in [list, tuple]:
        matrix = _np.array(matrix, dtype=_np.float32)

    if type(matrix) is _np.ndarray:
        r, c = matrix.shape
        if (r != 3) or (c != 3):
            raise TypeError('matrix must be a 3x3 array, list or tuple!')

        # Create temporary file
        data = matrix.copy()
        matrix = _files.create_filename(_env.workdir, 'txt', 'auxiliar_filter')

        # write file
        _np.savetxt(matrix, data, fmt='%.4f', header='1\t2\t3', delimiter='\t', comments='')

    # check if matrix is a file
    if type(matrix) is str:
        if not _os.path.exists(matrix):
            raise IOError('matrix file does not exist')
    else:
        raise TypeError('matrix must be a .txt file delimited by tabs or a 3x3 array')

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '4', '-INPUT', grid,
           '-RESULT', outgrid, '-FILTER', matrix, '-ABSOLUTE', absolute]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # user_defined_filter()


# Filter for identify groups of grid cells having the same data value
# library: grid_filter  tool: 5
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  min_size         [int] threshold of minimum grouped cells
def filter_clumps(outgrid, grid, min_size=10):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    min_size = str(min_size)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '5', '-GRID', grid,
           '-OUTPUT', outgrid, '-THRESHOLD', min_size]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # filter_clumps()


# Morphological filter for grids
# library: grid_filter  tool: 8
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  method           [int] filter method
#                    [0] Dilation: returns the maximum
#                    [1] Erosion: returns the minimum
#                    [2] Opening: applies first an erosion followed by a dilation
#                    [3] Closing: applies first an dilation followed by a erosion
#  kernel           [int] shape of the filter kernel
#                    [0] Square
#                    [1] Circle (default)
#  radius           [int] kernel radius in cells
def morphological_filter(outgrid, grid, method=0, kernel=1, radius=2):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    if method < 0 or method > 3:
        method = 0
    if kernel < 0 or kernel > 1:
        kernel = 1
    method, kernel, radius = str(method), str(kernel), str(int(radius))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '8', '-INPUT', grid,
           '-RESULT', outgrid, '-METHOD', method]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-MODE', kernel, '-RADIUS', radius])
    else:
        cmd.extend(['-KERNEL_TYPE', kernel, '-KERNEL_RADIUS', radius])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # morphological_filter()


# Rank filter for grids
# library: grid_filter  tool: 9
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  rank             [int, float] percent of the rank. Set 50 percent
#                    to apply a median filter
#  kernel           [int] shape of the filter kernel
#                    [0] Square
#                    [1] Circle (default)
#  radius           [int] kernel radius in cells
def rank_filter(outgrid, grid, rank=50, kernel=1, radius=2):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    if kernel < 0 or kernel > 1:
        kernel = 1
    rank, kernel, radius = str(rank), str(kernel), str(int(radius))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '9', '-INPUT', grid,
           '-RESULT', outgrid, '-RANK', rank]
    if _env.saga_version[0] in ['2', '3']:
        cmd.extend(['-MODE', kernel, '-RADIUS', radius])
    else:
        cmd.extend(['-KERNEL_TYPE', kernel, '-KERNEL_RADIUS', radius])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # rank_filter()


# Grid denoising using the algorithm of Sun et al. (2007)
# library: grid_filter  tool: 10
# INPUTS
#  outgrid          [string] output grid
#  grid             [string] input grid
#  sigma            [float] threshold value (0<=sigma<=1)
#  iter             [int] number of iterations for normal updating
#  viter            [int] number of iterations for vertex updating
#  vertex           [boolean] if True, vertex is used as face neighbourhood.
#                    In other case, edge is used as face neighbourhood.
#  zonly            [boolean] if True only Z-Direction position is updated
def mesh_denoise(outgrid, grid, sigma=0.9, iter=5, viter=50, vertex=True, zonly=False):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    # Convert to string
    if sigma < 0 or sigma > 1.0:
        sigma = 0.9
    if vertex:
        nb_cv = '0'
    else:
        nb_cv = '1'
    sigma, iter = str(sigma), str(int(iter))
    viter, zonly = str(int(viter)), str(int(zonly))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_filter', '10', '-INPUT', grid,
           '-OUTPUT', outgrid, '-SIGMA', sigma, '-ITER', iter,
           '-VITER', viter, '-NB_CV', nb_cv, '-ZONLY', zonly]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # mesh_denoise()


#==============================================================================
# Library: grid_gridding
#==============================================================================

# Converts shape files to grids
# library: grid_gridding  tool: 0
# INPUTS
#  outgrid          [string] output grid
#  inshape          [string] input shape file
#  value_method     [int] output grid values
#                    [0] data / no-data (default)
#                    [1] index number
#                    [2] attribute
#  field            [int, str] attribute index or name
#  multiple_values  [int] method for multiple values
#                    [0] first
#                    [1] last (default)
#                    [2] minimum
#                    [3] maximum
#                    [4] mean
#  line_type        [int] type of lines
#                    [0] thin
#                    [1] thick (default)
#  poly_type        [int] type of polygons
#                    [0] node
#                    [1] cell (default)
#  data_type        [int] preferred data type of output grid
#                    [0] Integer (1 byte)
#                    [1] Integer (2 byte)
#                    [2] Integer (4 byte)
#                    [3] Floating Point (4 byte) (default)
#                    [4] Floating Point (8 byte)
#  cellsize         [float] output grid cellsize. Only if grid_extent is None
#  grid_extent      [string] input grid to take its grid system. In this case
#                    cellsize is not considered
def shapes_to_grid(outgrid, inshape, value_method=0, field=0, multiple_values=1,
                   line_type=1, poly_type=1, data_type=3, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    inshape = _files.default_file_ext(inshape, 'vector')
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
    else:
        grid_extent = 'NULL'
    # default methods
    if value_method < 0 or value_method > 2:
        value_method = 2  # set default method
    if type(field) in [int, str]:
        field = str(field)
    else:
        field = '0'  # set default field
    if multiple_values < 0 or multiple_values > 4:
        multiple_values = 1
    if line_type < 0 or line_type > 1:
        line_type = 1
    if poly_type < 0 or poly_type > 1:
        poly_type = 1
    if data_type < 0 or data_type > 4:
        data_type = 3
    # convert to strings
    value_method, multiple_values = str(value_method), str(multiple_values)
    line_type, poly_type = str(line_type), str(poly_type)
    data_type, cellsize = str(data_type), str(cellsize)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '0', '-INPUT', inshape, '-GRID',
           outgrid, '-OUTPUT', value_method, '-FIELD', field, '-MULTIPLE',
           multiple_values, '-LINE_TYPE', line_type, '-POLY_TYPE', poly_type,
           '-GRID_TYPE', data_type]
    if grid_extent == 'NULL':  # use a cellsize
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:  # use a grid extent
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        if grid_extent == 'NULL':
            _projection.set_crs(grids=outgrid, crs_method=1, proj=grid_extent);
        else:
            _projection.set_crs(grids=outgrid, crs_method=1, proj=inshape);
    return(flag)


# Interpolation of points using inverse distance weighted (IDW)
# library: grid_gridding  tool: 1
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  method           [int] interpolation method
#                    [0] no distance weighting
#                    [1] inverse distance to a power (default)
#                    [2] exponential
#                    [3] gaussian weighting
#  weight           [float] if method=1 weight is the idw power, if method=2
#                    or 3, weight is the gaussian and exponential weighting
#                    bandwidth
#  offset           [boolean] if offset is True and method=1, calculates
#                    weights for distance plus one, avoiding division by zero
#                    for zero distances
#  smethod          [int] search range method
#                    [0] local (default)
#                    [1] global
#  sradius          [float] if smethod=0, sradius is the local maximum search
#                    distance given in map units
#  spoints          [int] search points method
#                    [0] maximum number of nearest points (default)
#                    [1] all points within search distance
#  spointsmin       [int] minimum number of points to use
#  spointsmax       [int] maximum number of nearest points
#  sdirection       [int] search direction method
#                    [0] all directions (default)
#                    [1] quadrants
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_system is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def inverse_distance_weighted(outgrid, points, field=0, method=1,
                              weight=2, offset=False, smethod=0, sradius=1000,
                              spoints=0, spointsmin=1, spointsmax=20,
                              sdirection=0, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')
    # get field name or index
    if type(field) is not str:
        field = str(field)
    # interpolation method
    if method < 0 or method > 3:
        method = 1  # default method
        weight = 2  # default weight parameter
    # search range method
    if smethod < 0 or smethod > 1:
        smethod = 0  # default method
    # search points method
    if spoints < 0 or spoints > 1:
        spoints = 0  # default method
    # search direction method
    if sdirection < 0 or sdirection > 1:
        sdirection = 0  # default method

    # convert parameters to string
    method, weight, offset = str(method), str(weight), str(int(offset))
    smethod, sradius = str(smethod), str(sradius)
    spoints, spointsmin, spointsmax = str(spoints), str(spointsmin), str(spointsmax)
    sdirection, cellsize = str(sdirection), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '1', '-POINTS', points,
           '-TARGET_OUT_GRID', outgrid, '-FIELD', field, '-SEARCH_RANGE', smethod,
           '-SEARCH_RADIUS', sradius, '-SEARCH_POINTS_ALL', spoints, '-SEARCH_POINTS_MIN',
           spointsmin, '-SEARCH_POINTS_MAX', spointsmax, '-SEARCH_DIRECTION',
           sdirection, '-DW_WEIGHTING', method, '-DW_IDW_POWER', weight,
           '-DW_IDW_OFFSET', offset, '-DW_BANDWIDTH', weight]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # inverse_distance_weighted()


# Interpolation of points using nearest neighbour
# library: grid_gridding  tool: 2
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def nearest_neighbour(outgrid, points, field=0, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')

    # Convert to strings
    field, cellsize = str(field), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '2', '-POINTS', points,
           '-FIELD', field, '-TARGET_OUT_GRID', outgrid]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # nearest_neighbour()


# Interpolation of points using natural neighbour
# library: grid_gridding  tool: 3
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  method           [int] interpolation method
#                    [0] Linear
#                    [1] Sibson (default)
#                    [2] Non-Sibsonian
#  weight           [int] minimum weight. Restricts extrapolation by assigning
#                    minimal allowed weight for a vertex (normally "-1";
#                    lower values correspond to lower reliability;
#                    "0" means no extrapolation)
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def natural_neighbour(outgrid, points, field=0, method=1, weight=0,
                      cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')

    # Convert to strings
    if method < 0 or method > 2:
        method = 1
    if weight > 0:
        weight = 0

    method, weight = str(int(method)), str(weight)
    field, cellsize = str(field), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '3', '-POINTS', points,
           '-FIELD', field, '-TARGET_OUT_GRID', outgrid, '-METHOD',
           method, '-WEIGHT', weight]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # natural_neighbour()


# Interpolation of points using modified quadratic shepard
# library: grid_gridding  tool: 4
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  neighbors        [int] quadratic neighbors (>5)
#  weight           [int] weighting neighbors (>3)
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def modified_quadratic_shepard(outgrid, points, field=0, neighbors=13, weight=19,
                               cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')

    # Convert to strings
    if neighbors < 5:
        neighbors = 5
    if weight < 3:
        weight = 3

    neighbors, weight = str(int(neighbors)), str(int(weight))
    field, cellsize = str(field), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '4', '-POINTS', points,
           '-FIELD', field, '-TARGET_OUT_GRID', outgrid, '-QUADRATIC_NEIGHBORS',
           neighbors, '-WEIGHTING_NEIGHBORS', weight]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # modified_quadratic_shepard()


# Interpolation of points using Delaunay Triangulation.
# library: grid_gridding  tool: 5
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def triangulation(outgrid, points, field=0, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')

    # Convert to strings
    field, cellsize = str(field), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '5', '-POINTS', points,
           '-FIELD', field, '-TARGET_OUT_GRID', outgrid]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # triangulation()


# Kernel density estimation
# library: grid_gridding  tool: 6
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  kernel           [int] interpolation method
#                    [0] quartic kernel (default)
#                    [1] gaussian kernel
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def kernel_density_estimation(outgrid, points, field=0, radius=1.0, kernel=0,
                              cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')
    # get field name or index
    if type(field) is not str:
        field = str(field)
    # interpolation method
    if kernel < 0 or kernel > 3:
        kernel = 0  # default method

    radius, kernel, cellsize = str(radius), str(kernel), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '6', '-POINTS', points,
           '-POPULATION', field, '-RADIUS', radius, '-KERNEL', kernel,
           '-TARGET_OUT_GRID', outgrid]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # kernel_density_estimation()


# Interpolation of points using angular distance weighted (ADW)
# library: grid_gridding  tool: 7
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  method           [int] interpolation method
#                    [0] no distance weighting
#                    [1] inverse distance to a power (default)
#                    [2] exponential
#                    [3] gaussian weighting
#  weight           [float] if method=1 weight is the idw power, if method=2
#                    or 3, weight is the gaussian and exponential weighting
#                    bandwidth
#  offset           [boolean] if offset is True and method=1, calculates
#                    weights for distance plus one, avoiding division by zero
#                    for zero distances
#  smethod          [int] search range method
#                    [0] local (default)
#                    [1] global
#  sradius          [float] if smethod=0, sradius is the local maximum search
#                    distance given in map units
#  spoints          [int] search points method
#                    [0] maximum number of nearest points (default)
#                    [1] all points within search distance
#  spointsmin       [int] minimum number of points to use
#  spointsmax       [int] maximum number of nearest points
#  sdirection       [int] search direction method
#                    [0] all directions (default)
#                    [1] quadrants
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def angular_distance_weighted(outgrid, points, field=0, method=1,
                              weight=2, offset=False, smethod=0, sradius=1000,
                              spoints=0, spointsmin=1, spointsmax=20,
                              sdirection=0, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')
    # get field name or index
    if type(field) is not str:
        field = str(field)
    # interpolation method
    if method < 0 or method > 3:
        method = 1  # default method
        weight = 2  # default weight parameter
    # search range method
    if smethod < 0 or smethod > 1:
        smethod = 0  # default method
    # search points method
    if spoints < 0 or spoints > 1:
        spoints = 0  # default method
    # search direction method
    if sdirection < 0 or sdirection > 1:
        sdirection = 0  # default method

    # convert parameters to string
    method, weight, offset = str(method), str(weight), str(int(offset))
    smethod, sradius = str(smethod), str(sradius)
    spoints, spointsmin, spointsmax = str(spoints), str(spointsmin), str(spointsmax)
    sdirection, cellsize = str(sdirection), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_gridding', '7', '-POINTS', points,
           '-TARGET_OUT_GRID', outgrid, '-FIELD', field, '-SEARCH_RANGE', smethod,
           '-SEARCH_RADIUS', sradius, '-SEARCH_POINTS_ALL', spoints, '-SEARCH_POINTS_MIN',
           spointsmin, '-SEARCH_POINTS_MAX', spointsmax, '-SEARCH_DIRECTION',
           sdirection, '-DW_WEIGHTING', method, '-DW_IDW_POWER', weight,
           '-DW_IDW_OFFSET', offset, '-DW_BANDWIDTH', weight]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # angular_distance_weighted()


#==============================================================================
# Library: grid_spline
#==============================================================================

# Interpolation of points using Thin Plate Spline
# Creates a 'Thin Plate Spline' function for each grid point based on
# all of the scattered data points that are within a given distance
# library: grid_spline  tool: 1
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  smethod          [int] search range method
#                    [0] local (default)
#                    [1] global
#  sradius          [float] if smethod=0, sradius is the local maximum search
#                    distance given in map units
#  spoints          [int] search points method
#                    [0] maximum number of nearest points (default)
#                    [1] all points within search distance
#  spointsmin       [int] minimum number of points to use
#  spointsmax       [int] maximum number of nearest points
#  sdirection       [int] search direction method
#                    [0] all directions (default)
#                    [1] quadrants
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_system is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def thin_plate_spline(outgrid, points, field=0, smethod=0, sradius=1000,
                      spoints=0, spointsmin=1, spointsmax=20,
                      sdirection=0, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')
    # get field name or index
    if type(field) is not str:
        field = str(field)
    # search range method
    if smethod < 0 or smethod > 1:
        smethod = 0  # default method
    # search points method
    if spoints < 0 or spoints > 1:
        spoints = 0  # default method
    # search direction method
    if sdirection < 0 or sdirection > 1:
        sdirection = 0  # default method

    # convert parameters to string
    smethod, sradius = str(smethod), str(sradius)
    spoints, spointsmin, spointsmax = str(spoints), str(spointsmin), str(spointsmax)
    sdirection, cellsize = str(sdirection), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_spline', '1', '-SHAPES', points,
           '-TARGET_OUT_GRID', outgrid, '-FIELD', field, '-SEARCH_RANGE', smethod,
           '-SEARCH_RADIUS', sradius, '-SEARCH_POINTS_ALL', spoints, '-SEARCH_POINTS_MIN',
           spointsmin, '-SEARCH_POINTS_MAX', spointsmax, '-SEARCH_DIRECTION',
           sdirection]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # thin_plate_spline()


# Interpolation of points using multilevel B-Spline
# The algorithm makes use of a coarse-to-fine hierarchy of control
# lattices to generate a sequence of bicubic B-spline functions,
# whose sum approaches the desired interpolation function.
# library: grid_spline  tool: 4
# INPUTS
#  outgrid          [string] output interpolated grid
#  points           [string] input points shapefile
#  field            [int, str] attribute index or name to interpolate
#  method           [int] refinement method
#                    [0] without B-spline refinement
#                    [1] with B-spline refinement (default)
#  error            [float] threshold error
#  level            [int] maximum level (1 <= level <= 14)
#  cellsize         [int, float] output cell size for interpolated grid.
#                    If grid_extent is not None, cellsize is ignored
#  grid_extent      [string] input grid file to take its grid system as
#                    the outgrid extent.
def multilevel_BSpline(outgrid, points, field=0, method=1, error=0.0001,
                       level=11, cellsize=100, grid_extent=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    points = _files.default_file_ext(points, 'vector')

    # Convert to strings
    if method < 0 or method > 1:
        method = 1
    if level < 1 or level > 14:
        level = 11

    method, error, level = str(method), str(error), str(level)
    field, cellsize = str(field), str(cellsize)

    # check for grid system as grid extent
    if type(grid_extent) is str:
        grid_extent = _files.default_file_ext(grid_extent, 'grid')
        if not _os.path.exists(grid_extent):
            grid_extent = None  # file extent does not exist

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_spline', '4', '-SHAPES', points,
           '-FIELD', field, '-TARGET_OUT_GRID', outgrid, '-METHOD',
           method, '-EPSILON', error, '-LEVEL_MAX', level]
    if grid_extent is None:
        cmd.extend(['-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize])
    else:
        cmd.extend(['-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid_extent])

    # Run command
    flag = _env.run_command_logged(cmd)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=points);
    return(flag)  # multilevel_BSpline()


#==============================================================================
# Library: grid_tools
#==============================================================================

# Grid resampling using a grid system or a cellsize
# library: grids_tools  tool: 0
# INPUTS
#  outgrid       [string] output grid file name
#  ingrid        [string] input grid file name
#  scale_up      [int] upscaling grid method
#                 [0] Nearest Neighbour
#                 [1] Bilinear Interpolation
#                 [2] Bicubic Spline Interpolation
#                 [3] B-Spline Interpolation
#                 [4] Mean Value
#                 [5] Mean Value (cell area weighted)
#                 [6] Minimum Value
#                 [7] Maximum Value
#                 [8] Majority
#  scale_down    [int] downscaling grid method
#                 [0] Nearest Neighbour
#                 [1] Bilinear Interpolation
#                 [2] Bicubic Spline Interpolation
#                 [3] B-Spline Interpolation
#  grid          [string] optional grid file name. If grid is not None,
#                 output grid is resampling with the grid system
#  cellsize      [int] if grid is None, output grid is resampling using
#                 the input cellsize value
#  fit           [integer] method for resampling
#                 [0] fix the node position
#                 [1] fix the cell center position
#  keep          [boolean] preserve input data type
def resampling(outgrid, ingrid, scale_up=5, scale_down=3,
               grid=None, cellsize=100, fit=0, keep=True):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    if grid is not None:
        grid = _files.default_file_ext(grid, 'grid', False)
    if scale_up > 8 or scale_up < 0:  # set default scale_up method
        scale_up = 5
    if scale_down > 3 or scale_down < 0:  # set default scale_down method
        scale_down = 3
    # convert to string
    scale_up = str(scale_up)
    scale_down = str(scale_down)
    cellsize = str(cellsize)
    fit = str(fit)
    keep = str(int(keep))
    # Create cmd
    if grid is not None:  # grid system from file
        cmd = ['saga_cmd', '-f=q', 'grid_tools', '0', '-INPUT', ingrid,
               '-SCALE_UP', scale_up, '-SCALE_DOWN', scale_down, '-KEEP_TYPE', keep,
               '-TARGET_DEFINITION', '1', '-TARGET_TEMPLATE', grid, '-OUTPUT', outgrid]
    else:  # cellsize input
        cmd = ['saga_cmd', '-f=q', 'grid_tools', '0', '-INPUT', ingrid,
               '-SCALE_UP', scale_up, '-SCALE_DOWN', scale_down, '-KEEP_TYPE', keep,
               '-TARGET_DEFINITION', '0', '-TARGET_USER_SIZE', cellsize,
               '-TARGET_USER_FITS', keep, '-OUTPUT', outgrid]
    # Run cmd
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # resampling()


# Merges multiple grids into a grid
# library: grids_tools  tool: 3
# INPUTS
#  outgrid       [string] output merged grid
#  gridlist      [list] list of grid files to merge (.sgrd or .tif)
#  resolution    [float] if resolution is different to 0, resolution parameter
#                 is used for resampling
#  dtype         [int] data storage type
#                 [0] 1 bit
#                 [1] 1 byte unsigned integer
#                 [2] 1 byte signed integer
#                 [3] 2 byte unsigned integer
#                 [4] 2 byte signed integer
#                 [5] 4 byte unsigned integer
#                 [6] 4 byte signed integer
#                 [7] 4 byte floating point (default)
#                 [8] 8 byte floating point
#  resampling    [int] resampling method
#                 [0] Nearest Neighbour
#                 [1] Bilinear Interpolation
#                 [2] Bicubic Spline Interpolation
#                 [3] B-Spline Interpolation (default)
#  overlap       [int] overlaping grids treatment
#                 [0] first grid
#                 [1] last grid (default)
#                 [2] minimum value
#                 [3] maximum value
#                 [4] mean value
#                 [5] blend boundary
#                 [6] feathering
def mosaicking(outgrid, gridlist, resolution=0, dtype=7,
               resampling=3, overlap=1):
    # Check inputs
    basename = _os.path.basename(outgrid).split('.')[0]
    outgrid = _files.default_file_ext(outgrid, 'grid')
    gridlist = [_files.default_file_ext(filename, 'grid', False) for filename in gridlist]
    gridl = ';'.join(gridlist)  # list of grids
    if dtype < 0 or dtype > 7:
        dtype = 7
    if resampling < 0 or resampling > 3:
        resampling = 3
    if overlap < 0 or overlap > 6:
        overlap = 1
    dtype = str(7)
    resampling = str(resampling)
    overlap = str(overlap)
    # Grid Resolution
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '3', '-GRIDS', gridl, '-NAME',
           basename, '-TYPE', dtype, '-RESAMPLING', resampling, '-OVERLAP',
           overlap, '-TARGET_DEFINITION', '0', '-TARGET_OUT_GRID', outgrid]
    if resolution > 0:
        resolution = str(resolution)
        cmd.extend(['-TARGET_USER_SIZE', resolution])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=gridlist[0]);
    return(flag)  # mosaicking()


# Create a grid of constant values
# library: grids_tools  tool: 4
# INPUTS
#  outgrid          [string] output grid file of constant value (.sgrd)
#  grid_extent      [string] input grid to use as grid extent (.sgrd or .tif)
#  value            [int, float] value of the constant grid
#  dtype            [int] data type of the constant grid
#                    [0] bit
#                    [1] unsigned 1 byte integer
#                    [2] signed 1 byte integer
#                    [3] unsigned 2 byte integer
#                    [4] signed 2 byte integer
#                    [5] unsigned 8 byte integer
#                    [6] signed 8 byte integer
#                    [7] 4 byte floating point number (default)
#                    [8] 8 byte floating point number
def constant_grid(outgrid, grid_extent, value, dtype=7):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid_extent = _files.default_file_ext(grid_extent, 'grid', False)
    value = str(value)
    if dtype < 0 or dtype > 8:
        dtype = 7
    dtype = str(dtype)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '4', '-CONST', value, '-TYPE',
           dtype, '-DEFINITION', '1', '-TEMPLATE', grid_extent, '-OUT_GRID',
           outgrid]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid_extent);
    return(flag)  # constant_grid()


# Fill gaps of a grid with data from another grid
# library: grids_tools  tool: 5
# INPUTS
#  outgrid      [string] output grid file name (.sgrd)
#  ingrid       [string] input grid file name (.sgrd or .tif)
#  patch        [string] patch grid to fill ingrid (.sgrd or .tif)
#  resampling   [int] resampling method
#                [0] Nearest Neighbour
#                [1] Bilinear Interpolation
#                [2] Bicubic Spline Interpolation
#                [3] B-Spline Interpolation (default)
def patching(outgrid, ingrid, patch, resampling=3):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    patch = _files.default_file_ext(patch, 'grid', False)
    if resampling < 0 or resampling > 3:
        resampling = 3
    resampling = str(resampling)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '5', '-ORIGINAL', ingrid,
           '-ADDITIONAL', patch, '-COMPLETED', outgrid, '-RESAMPLING',
           resampling]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # patching()


# Close gaps of a grid data set (i.e. eliminate no data values)
# library: grids_tools  tool: 7
# INPUTS
#  outgrid      [string] output grid file name
#  ingrid       [string] input grid file name
#  mask         [string] optional grid mask file
#  threshold    [float] tension threshold
def close_gaps(outgrid, ingrid, mask=None, threshold=0.1):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    if type(mask) is str:
        mask = _files.default_file_ext(mask, 'grid', False)
    else:
        mask = 'NULL'
    threshold = str(threshold)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '7' '-INPUT', ingrid, '-MASK',
           mask, '-RESULT', outgrid, '-THRESHOLD', threshold]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # close_gaps()


# Creates buffers around features in a grid. The output buffer grid cell
# values refer to 1 inside the buffer, 2 for feature location.
# library: grids_tools  tool: 8
# INPUTS
#  outgrid      [string] output grid file name
#  ingrid       [string] input grid file name
#  dist         [int, float] buffer distance in map units. If dist=0 (default)
#                grid values are used as buffer distance.
def buffer(outgrid, ingrid, dist=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    if dist == 0:  # distance from grid
        method = '1'
    else:  # fixed distance
        method = '0'
    dist = str(dist)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '8', '-FEATURES', ingrid, '-BUFFER',
           outgrid, '-TYPE', method, '-DISTANCE', dist]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # buffer()


# Change grid values using equalities or ranges
# library: grids_tools  tool: 12
# INPUTS
#  outgrid       [string] output grid file name
#  ingrid        [string] input grid file name
#  identity      [list, tuple, array] identity can be a [new_value, old_value]
#                 array/list/tuple for change a single value. For multiple
#                 values you can use an array like [[new1, old1], [new2, ol2], ...]
#  vrange        [list, tuple, array] vrange can be a [new_value, old_low, old_high]
#                 array/list/tuple for change a single range. For multiple
#                 ranges you can use an array like [[new1, low1, high1],
#                 [new2, low2, high2], ...]
# NOTE: identity and vrange parameters can be used at the same time
def change_values(outgrid, ingrid, identity=None, vrange=None):
    # Check inputs
    if identity is None and vrange is None:
        raise TypeError('Parameters identity or vrange must be input!')
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    # Work depending of Saga Versions
    # tested for saga versions 2.3.1 and 3.0.0
    if _env.saga_version[0] in ['2', '3']:
        if type(identity) in [list, tuple, _np.ndarray]:  # identity
            identity = _np.array(identity, dtype=_np.float32)
            if identity.ndim == 1:
                identity = identity.reshape((1, 2))
            # create matrix
            matrix = _np.zeros((identity.shape[0], 3), dtype=_np.float32)
            matrix[:] = identity[:, [1, 1, 0]]  # [low, high, new]
            # write reclass file
            if _env.workdir is None:
                aux_file = 'reclass_aux_file.txt'
            else:
                aux_file = _os.path.join(_env.workdir, 'reclass_aux_file.txt')
            header = 'Low\tHigh\tNew'
            _np.savetxt(aux_file, matrix, fmt='%g', delimiter='\t', comments='',
                        header=header)
            # create cmd
            cmd = ['saga_cmd', '-f=q', 'grid_tools', '12', '-GRID_IN', ingrid,
                   '-GRID_OUT', outgrid, '-METHOD', '0', '-LOOKUP', aux_file]
            # Run command
            flag = _env.run_command_logged(cmd)
            _os.remove(aux_file)

        if type(vrange) in [list, tuple, _np.ndarray]:  # range of values
            vrange = _np.array(vrange, dtype=_np.float32)
            if vrange.ndim == 1:
                vrange = vrange.reshape((1, 3))
            # create matrix
            matrix = _np.zeros((vrange.shape[0], 3), dtype=_np.float32)
            matrix[:] = vrange[:, [1, 2, 0]]  # [low, high, new]
            # write reclass file
            if _env.workdir is None:
                aux_file = 'reclass_aux_file.txt'
            else:
                aux_file = _os.path.join(_env.workdir, 'reclass_aux_file.txt')
            header = 'Low\tHigh\tNew'
            _np.savetxt(aux_file, matrix, fmt='%g', delimiter='\t', comments='',
                        header=header)
            # create cmd
            cmd = ['saga_cmd', '-f=q', 'grid_tools', '12', '-GRID_IN', ingrid,
                   '-GRID_OUT', outgrid, '-METHOD', '2', '-LOOKUP', aux_file]
            # Run command
            flag = _env.run_command_logged(cmd)
            # remove reclass table
            _os.remove(aux_file)

    # tested for saga versions 4.0.1 and 5.0.0
    if _env.saga_version[0] in ['4', '5', '6']:
        if type(identity) in [list, tuple, _np.ndarray]:  # identity
            identity = _np.array(identity, dtype=_np.float32)
            if identity.ndim == 1:
                identity = identity.reshape((1, 2))
            # create matrix
            matrix = identity[:]  # [new, old]
            # write reclass file
            if _env.workdir is None:
                aux_file = 'reclass_aux_file.txt'
            else:
                aux_file = _os.path.join(_env.workdir, 'reclass_aux_file.txt')
            header = 'New\tOld'
            _np.savetxt(aux_file, matrix, fmt='%g', delimiter='\t', comments='',
                        header=header)
            # create cmd
            cmd = ['saga_cmd', '-f=q', 'grid_tools', '12', '-INPUT', ingrid,
                   '-OUTPUT', outgrid, '-METHOD', '0', '-IDENTITY', aux_file]
            # Run command
            flag = _env.run_command_logged(cmd)
            _os.remove(aux_file)

        if type(vrange) in [list, tuple, _np.ndarray]:  # range of values
            vrange = _np.array(vrange, dtype=_np.float32)
            if vrange.ndim == 1:
                vrange = vrange.reshape((1, 3))
            # create matrix
            matrix = vrange[:]  # [new, low, high]
            # write reclass file
            if _env.workdir is None:
                aux_file = 'reclass_aux_file.txt'
            else:
                aux_file = _os.path.join(_env.workdir, 'reclass_aux_file.txt')
            header = 'Low\tHigh\tNew'
            _np.savetxt(aux_file, matrix, fmt='%g', delimiter='\t', comments='',
                        header=header)
            # create cmd
            cmd = ['saga_cmd', '-f=q', 'grid_tools', '12', '-INPUT', ingrid,
                   '-OUTPUT', outgrid, '-METHOD', '1', '-RANGE', aux_file]
            # Run command
            flag = _env.run_command_logged(cmd)
            # remove reclass table
            _os.remove(aux_file)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # change_values()


# Change grid nodata values using multiple values o ranges. Nodata values
# are defined as -99999
# library: grids_tools  tool: derived from 12
# Depends of: change_grid_values()
# INPUTS
#  outgrid  [string] output grid
#  ingrid   [string] input grid file
#  value    [int, float, list, array] if value is int or float, a
#            single value is changed. If value is a list, tuple or
#            a numpy 1d array, all values are set as novalues in
#            the grid. If vrange is not None, value is ignored
#  vrange   [list, array] if vrange is a 1d array or a list with 2
#            elements, a single range is used. For multiple ranges
#            you can use [[min1, max1], [min2, max2],...]
def change_nodata_values(outgrid, ingrid, value=None, vrange=None):
    # Check inputs
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    outgrid = _files.default_file_ext(outgrid, 'grid')

    # Equality conditions
    if value is not None:
        if type(value) in [int, float]:
            value = _np.array([-99999, value])
        elif type(value) in [list, tuple, _np.ndarray]:
            value = _np.array(value)
            aux = _np.full(len(value), -99999, dtype=_np.float32)
            value = _np.array(zip(aux, value))
        # Change nodata values
        flag = change_values(outgrid, ingrid, identity=value)

    # Range conditions
    if vrange is not None:
        vrange = _np.array(vrange)
        if vrange.ndim == 1:
            value = _np.array([-99999, vrange.min(), vrange.max()])
        elif vrange.ndim > 1:
            value = _np.array([[-99999, x.min(), x.max()] for x in vrange])
        # Change nodata values
        flag = change_values(outgrid, ingrid, vrange=value)

    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # change_nodata_values()


# Reclassification of grid values using a single value o ranges. All values
# that are not considered in the reclassification are set as NaNs
# library: grids_tools  tool: derived from 15
# INPUTS
#  outgrid     [string] output grid file
#  ingrid      [string] input grid file
#  single      [list, tuple, array] two elements list/array with [new_value,old_value]
#               single is None as default, that means that single value is not.
#  vrange      [list, tuple, array] two elements list/array with [new_value,
#               old_min, old_max]. For multiple ranges you can use [[new1, min1, max1],
#               [new2, min2, max2], ...,[new_n, minx_n, max_n]]. If vrange is
#               None it is not used for reclassify
#  smethod     [int] operator for single value
#               [0] =  (default)
#               [1] <
#               [2] <=
#               [3] >=
#               [4] >
#  rmethod     [int] operator for range values
#               [0] min <= value < max
#               [1] min <= value <= max
#               [2] min < value <= max
#               [3] min < value < max
#  other       [float] value for all these grid values that are not considered
#               in the reclassification. If other is None (default),
#               that values are set at the same value that nodata values
#  nodata      [float] value for no data values. If nodata is None, original
#               nodata value is used
# NOTE: single and vrange parameters can't be used at the same time
def reclassify_values(outgrid, ingrid, single=None, vrange=None,
                      smethod=0, rmethod=0, other=None, nodata=None):
    # Check inputs
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    outgrid = _files.default_file_ext(outgrid, 'grid')
    # check values
    if single is None and vrange is None:
        raise TypeError('Parameters single and vrange cant be None!')
    # check single parameter
    if single is not None:
        if type(single) in [list, tuple, _np.ndarray]:
            if len(single) != 2:
                raise TypeError('single parameter must have 2 elements!')
                # check vrange parameter
    if vrange is not None:
        if type(vrange) in [list, tuple, _np.ndarray]:
            vrange = _np.array(vrange, dtype=_np.float32)
            if vrange.ndim == 1 and len(vrange) == 3:
                vrange = _np.array([vrange], dtype=_np.float32)
            elif vrange.ndim == 2 and vrange.shape[1] != 3:
                raise TypeError('vrange parameter must have rows with 3 elements!')
        else:
            raise TypeError('Bad vrange parameter type <{}>'.format(str(type(vrange))))
    if smethod < 0 or smethod > 4:
        smethod = 0
    if rmethod < 0 or rmethod > 4:
        rmethod = 0
    smethod = str(smethod)
    rmethod = str(rmethod)
    # Get grid system info
    if nodata is None:
        gs = _io.grid_system(ingrid)
        nodata = gs['NODATA_VALUE']
    if other is None:
        other = nodata
    nodata, other = str(nodata), str(other)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '15', '-INPUT', ingrid, '-RESULT',
           outgrid, '-NODATAOPT', '1', '-NODATA', nodata, '-RESULT_NODATA_CHOICE',
           '1', '-RESULT_NODATA_VALUE', nodata, '-OTHEROPT', '1', '-OTHERS', other]

    table_file = None
    if single is not None:
        new, old = str(single[0]), str(single[1])
        cmd.extend(['-METHOD', '0', '-OLD', old, '-NEW', new, '-SOPERATOR',
                    smethod])

    elif vrange is not None:
        if vrange.shape[0] == 1:
            new, vmin, vmax = str(vrange[0, 0]), str(vrange[0, 1]), str(vrange[0, 2])

            cmd.extend(['-METHOD', '1', '-MIN', vmin, '-MAX', vmax, '-RNEW',
                        new, '-ROPERATOR', rmethod])

        else:
            # create table name
            table_file = 'aux_table_reclass.txt'
            if _env.workdir is not None:
                table_file = _os.path.join(_env.workdir, table_file)
            # save
            _np.savetxt(table_file, vrange, delimiter='\t', fmt='%.6f',
                        header='new\tmin\tmax', comments='')
            # create cmd
            cmd.extend(['-METHOD', '3', '-RETAB_2', table_file, '-TOPERATOR',
                        rmethod, '-F_MIN', '1', '-F_MAX', '2', '-F_CODE', '0'])
    # Run command
    flag = _env.run_command_logged(cmd)
    # delete auxiliar table
    if table_file is not None:
        _os.remove(table_file)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # reclassify_values()


# Crop grids to valid data cells
# library: grids_tools  tool: 17
# INPUTS
#  outgrid       [string] output grif
#  ingrid        [string] input grid
def crop_to_data(outgrid, ingrid):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '17', '-INPUT', ingrid, '-OUTPUT',
           outgrid]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # crop_to_data()


# Invert data/no data of a grid
# library: grids_tools  tool: 18
# INPUTS
#  outgrid       [string] output grid
#  ingrid        [string] input grid
def invert_grid_nodata(outgrid, ingrid):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '18', '-INPUT', ingrid,
           '-OUTPUT', outgrid]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # invert_grid_nodata()


# Creates an index grid according to the cell values either in ascending
# or descending order.
# library: grids_tools  tool: 21
# INPUTS
#  outgrid        [string] output index grid
#  grid           [string] input grid file
#  order          [int] sorting order method
#                   [0] ascending (default)
#                   [1] descending
def grid_cell_index(outgrid, grid, order=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)

    # Check additional inputs
    if order < 0 or order > 1:
        order = 0
    order = str(order)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '21', '-GRID', grid, '-INDEX',
           outgrid, '-ORDER', order]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # grid_cell_index()


# Creates several grids using a classified grid and a table with data values
# One grid is created for each column using the input table (except to the ID column)
# and takes for each class the value specified in the column
# library: grids_tools  tool: 22
# INPUTS
#  outgrid        [string] basename for the output grids. Number of output grids
#                  depends of the number of columns in table
#  grid           [string] input classified grid
#  table          [string, list, tuple, array, DataFrame] input .txt table file delimited by
#                  tabs and with 1 line of headers. table can be a nxm numpy array, list or tuple
#                  in which case, field_id must be an integer. table can also be a pandas DataFrame
#  field_id       [int, string] field id or name of the input table. When table is an array,
#                  list or tuple, field_id must be an integer.
def grids_from_classified_grid(outgrid, grid, table, field_id=0):
    # Check inputs
    grid = _files.default_file_ext(grid, 'grid', False)

    # Check table
    if type(table) in [list, tuple, _np.ndarray]:  # list or array as table
        table = _np.array(table)
        # get names for output grids
        names = ['values_' + str(j) for j in range(table.shape[1] - 1)]

        # header for output file
        header = names[:]
        header.insert(field_id, 'ID')
        header = '\t'.join(header)

        # temporary table file
        filename = _files.create_filename(_env.workdir, 'txt', 'Auxiliar_grid_reclass')
        _np.savetxt(filename, table, fmt='%.6f', delimiter='\t', header=header, comments='')

    elif type(table) is _Series:  # Pandas Data Frame
        # Get names for output grids
        header = list(table.columns)

        if type(field_id) is str:
            field_name = field_id
        else:
            field_idx = field_id
            field_name = header[field_idx]

        names = header[:]
        names.remove(field_name)

        # temporary table file
        filename = _files.create_filename(_env.workdir, 'txt', 'Auxiliar_grid_reclass')
        table.to_csv(filename, '\t', index=False)

    elif type(table) is str:  # Input table file
        filename = table
        # Get header
        with open(filename, 'r') as fid:
            line = fid.readline()
            header = line.replace('\n', '').split('\t')

        # Get names for output grids
        if type(field_id) is str:
            field_name = field_id
        else:
            field_idx = field_id
            field_name = header[field_idx]

        names = header[:]
        names.remove(field_name)

    else:  # Wrong input argument
        raise TypeError('Wrong table parameter. table must be an array, a Pandas'\
                        'DataFrame or a .txt file delimited by tabs')

    # Create output grids file names
    basename = _os.path.splitext(outgrid)[0]
    out_grids = [basename + name + '.sgrd' for name in names]
    grid_list = ';'.join(out_grids)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '22', '-TABLE', filename,
           '-ID_FIELD', field_id, '-CLASSES', grid, '-GRIDS', grid_list]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    for outgrid in out_grids:
        if not _files.has_crs_file(outgrid):  # set first input layer crs
            _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # grids_from_classified_grid()


# Creates a new user specified Grid System for use with other tools
# library: grids_tools  tool: 23
# INPUTS
#  outgrid       [string] output grid
#  value         [int, float] initialization value to create a constant grid
#  cellsize      [int, float] cellsize of the grid system
#  adjust        [int] method to adjust the extent to match the cellsize
#                 [0] extent to cell size (default)
#                 [1] cell size to left-right extent
#                 [2] cell size to bottom-top extent
#  xoffset       [int, float] lower left corner in W-E. Positive values result
#                 in a shift in E direction and negative in W direction
#  yoffset       [int, float] lower left corner in S-N. Positive values result
#                 in a shift in N direction and negative in S direction
#  corner        [list, tuple, array] 4 element object tah defines the lower-left
#                 corner of the grid system. corner must contain [xmin, cols, ymin, rows]
#  extent        [list, tuple, array] 4 element object tah defines the lower-left
#                 corner and the upper-right corner of the grid system.
#                 extent must contain [xmin, xmax, ymin, ymax]
#  grids         [string, list, tuple] grid file or list of grid files to define the
#                 grid system extension
#  shapes        [string, list, tuple] shape file or list of shape files to define the
#                 grid system extension
# NOTE: first input parameter (corner, extent, grids, shapes) is used to define the
# grid system extension
def create_grid_system(outgrid, value=0, cellsize=10, adjust=0, xoffset=0, yoffset=0,
                       corner=None, extent=None, grids=None, shapes=None):
    # Check inputs
    outgrid = outgrid = _files.default_file_ext(outgrid, 'grid')

    # Convert to string
    if xoffset == 0 and yoffset == 0:
        offset = False
    else:
        offset = True
    if adjust < 0 or adjust > 2:
        adjust = 0

    xoffset, yoffset = str(xoffset), str(yoffset)
    value, cellsize, adjust = str(value), str(cellsize), str(adjust)

    # Initialize cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '23', '-GRID', outgrid, '-INIT',
           value, '-CELLSIZE', cellsize, '-ADJUST']
    if offset:
        cmd.extend(['-USEOFF', '1', '-XOFFSET', xoffset, '-YOFFSET', yoffset])

    # Check extension method
    if type(corner) in [list, tuple, _np.ndarray]:
        if len(corner) != 4:
            raise TypeError('corner must be a 4 element list, tuple or array')

        xmin, cols, ymin, rows = [str(x) for x in extent]

        cmd.extend(['-M_EXTENT', '0', '-XMIN', xmin, '-NX', cols,
                    '-YMIN', ymin, '-NY', rows])

    elif type(extent) in [list, tuple, _np.ndarray]:
        if len(extent) != 4:
            raise TypeError('extent must be a 4 element list, tuple or array')

        xmin, xmax, ymin, ymax = [str(x) for x in extent]

        cmd.extend(['-M_EXTENT', '1', '-XMIN', xmin, '-XMAX', xmax,
                    '-YMIN', ymin, '-YMAX', ymax])

    elif type(grids) is not None:
        if type(grids) is str:
            grids = [_files.default_file_ext(grids, 'grid', False)]
        elif type(grids) in [list, tuple]:
            grids = [_files.default_file_ext(grid, 'grid', False) for grid in grids]
        else:
            raise TypeError('grids must be a grid file or a list of grids')

        grid_list = ';'.join(grids)

        cmd.extend(['-M_EXTENT', '3', '-GRIDLIST', grid_list])

    elif type(shapes) is not None:
        if type(shapes) is str:
            shapes = [_files.default_file_ext(shapes, 'vector')]
        elif type(shapes) in [list, tuple]:
            shapes = [_files.default_file_ext(shape, 'grid', False) for shape in shapes]
        else:
            raise TypeError('shapes must be a shape file or a list of shape')

        shape_list = ';'.join(shapes)

        cmd.extend(['-M_EXTENT', '2', '-SHAPESLIST', shape_list])

    else:
        raise TypeError('One way to generate the grid system must be input')

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        if type(grids) in [list, tuple]:
            _projection.set_crs(grids=outgrid, crs_method=1, proj=grids[0]);
        elif type(shapes) in [list, tuple]:
            _projection.set_crs(grids=outgrid, crs_method=1, proj=shapes[0]);
    return (flag)  # masking(()



# Mask a grid with other grid
# library: grids_tools  tool: 24
# INPUTS
#  outgrid       [string] output masked grid
#  ingrid        [string] input grid
#  mask          [string] mask grid
def masking(outgrid, ingrid, mask):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    ingrid = _files.default_file_ext(ingrid, 'grid', False)
    mask = _files.default_file_ext(mask, 'grid', False)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '24', '-GRID', ingrid, '-MASK',
           mask, '-MASKED', outgrid]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=ingrid);
    return(flag)  # masking()


# Close gaps of a grid data set (i.e. eliminate no data values)
# using Spline interpolation
# library: grids_tools  tool: 25
# INPUTS
#  outgrid        [string] output grid file name
#  grid           [string] input grid file name
#  mask           [string] optional grid mask file
#  cells          [int] maximum number of cells to process. Is ignored if set to zero
#  points         [int] maximum number of points
#  local_points   [int] number of points for the local interpolation
#  extended       [boolean] if True, extended neighbourhood  is used
#  neighbours     [int] neighbours method
#                  [0] Neumann (default)
#                  [1] Moore
#  radius         [int] number of cells to use as radius
#  relaxation     [int, float] relaxation value
def close_gaps_with_spline(outgrid, grid, mask=None, cells=0, points=100,
                           local_points=20, extended=False, neighbours=0,
                           radius=0, relaxation=0):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)
    if type(mask) is str:
        mask = _files.default_file_ext(mask, 'grid', False)
    else:
        mask = 'NULL'

    # Check additional inputs
    if neighbours < 0 or neighbours > 1:
        neighbours = 0
    cells, points, local_points = str(int(cells)), str(int(points)), str(int(local_points))
    extended, neighbours = str(int(extended)), str(int(neighbours))
    radius, relaxation = str(int(radius)), str(int(relaxation))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '25', '-GRID', grid, '-MASK',
           mask, '-CLOSED', outgrid, '-MAXGAPCELLS', cells, '-MAXPOINTS', points,
           '-LOCALPOINTS', local_points, '-EXTENDED', extended, '-NEIGHBOURS',
           neighbours, '-RADIUS', radius, '-RELAXATION', relaxation]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # close_gaps_with_spline()


#
# library: grids_tools  tool: 28
# INPUTS
#  outgrid        [string] output grid file
#  grid           [string] input grid file
#  method         [int] operation to do
#                  [0] shrink (default)
#                  [1] expand
#                  [2] shrink and expand
#                  [3] expand and shrink
#  mode           [int] search mode
#                  [0] Square
#                  [1] Circle (default)
#  radius         [int] number of cells to use as radius distance
#  expand         [int] expand method
#                  [0] minimum
#                  [1] maximum
#                  [2] mean
#                  [3] majority (default)
def shrink_and_expand(outgrid, grid, method=0, mode=1, radius=10, expand=3):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)

    # Check additional inputs
    if method < 0 or method > 3:
        method = 0
    if mode < 0 or mode > 1:
        mode = 1
    if expand < 0 or expand > 3:
        expand = 3

    method, mode = str(int(method)), str(int(mode))
    radius, expand = str(int(radius)), str(int(expand))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '28', '-INPUT', grid, '-RESULT',
           outgrid, '-OPERATION', method, '-CIRCLE', mode, '-RADIUS', radius,
           '-EXPAND', expand]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return (flag)  # copy_grid()



# Copy a grid file
# library: grids_tools  tool: 33
# INPUTS
#  outgrid        [string] output grid file
#  grid           [string] input grid file
def copy_grid(outgrid, grid):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '33', '-GRID', grid, '-COPY', outgrid]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # copy_grid()


# Invert a grid data values, highest value becomes the lowest and vice versa
# library: grids_tools  tool: 34
# INPUTS
#  outgrid        [string] output grid file
#  grid           [string] input grid file
def invert_grid(outgrid, grid):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    grid = _files.default_file_ext(grid, 'grid', False)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'grid_tools', '34', '-GRID', grid, '-INVERSE', outgrid]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=grid);
    return(flag)  # invert_grid()

