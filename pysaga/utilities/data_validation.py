"""
SAGA GIS algorithm provider
Validation of input and output parameters
"""


import os as _os
import shutil as _shutil

import files as _files


# ==============================================================================
# Validate input files
# ==============================================================================


def input_file(filename, ftype='grid', force=False):
    """
    Input files validation. Verifies if input files' name is a string, set a default
    file extension and verifies if input files exist

    INPUTS:
      filename   [string, list, tuple] input file name or list of files
      ftype      [string] file extension
      force      [bool] if True, file extension is forced. By default False
    OUTPUTS:
      in_file    [as filename] fixed input filename or list of files
    """
    if type(filename) is str:
        files_list = [filename]
    elif type(filename) in [list, tuple]:
        files_list = list(filename)
    else:
        raise TypeError('Filename must be a string or list!')

    in_file = []  # input files fixed
    for filename in files_list:
        if type(filename) is not str:
            raise TypeError('Wrong input file < {} >'.format(filename))
        file_fixed = _files.default_file_ext(filename, ftype, force)
        if not _os.path.exists(file_fixed):
            raise IOError('File < {} > does not exist'.format(file_fixed))
        in_file.append(file_fixed)
    if len(in_file) == 1:  # only one input
        in_file = in_file[0]
    return(in_file)


def output_file(filename, ftype='grid'):
    """
    Output file validation. Verifies if output path exist, if not is created.
    Forces to the output file to avoid conflicts with other modules

    INPUTS:
      filename     [string] full file name
      ftype        [string] file extension to be forced
    OUTPUTS:
      out_file     [string] output file name.
    """
    if type(filename) is str:
        files_list = [filename]
    elif type(filename) in [list, tuple]:
        files_list = list(filename)
    else:
        raise TypeError('Filename must be a string or list!')

    out_file = []  # output files fixed
    for filename in files_list:
        if type(filename) is not str:
            raise TypeError('Wrong input file < {} >'.format(filename))
        file_fixed = _files.default_file_ext(filename, ftype, True)
        path = _os.path.dirname(file_fixed)
        if not _os.path.exists(path):
            _os.makedirs(path)
        out_file.append(file_fixed)
    if len(out_file) == 1:  # only one file
        out_file = out_file[0]
    return(out_file)


def input_parameter(value, default, vrange=None, value_list=None, lt=None, gt=None,
                    length=None, asstr=True, dtypes=[int, float]):
    """
    Parameters validation for models computation. When an input value is out of range
    a default value is used

    INPUTS:
      value        [int, float, str] input value
      default      [as value] default value in case that an erroneous value is input
      vrange       [list, tuple] 2 elements object that contains the minimum and maximum for value
                    If range is a tuple, extreme values are not reached   min < value < max
                    If range is a list, extreme values not reached   min <= value <= max
      value_list   [tuple, list] possible values list
      lt           [same as value] maximum value (value <= lt)
      gt           [same as value] minimum value (value >= gt)
      length       [int] only if value is a tuple or list, len is the number of elements
                    expected in value
      asstr        [bool] if True, output value is converted to string. By default True
      dtypes       [list] type of data. By default (int, float)
    OUTPUT:
      value        [as value] validated output value
    """
    # Check limits and available values
    flag = False  # Flag for default value
    if type(value) not in dtypes:
        value = default
        return(value)
    if vrange is not None:
        if type(vrange) is tuple:
            if value <= min(vrange) or value >= max(vrange):
                flag = True
        if type(vrange) is list:
            if value < min(vrange) or value > max(vrange):
                flag = True
    if value_list is not None:
        if type(value_list) in (list, tuple):
            if value not in value_list:
                flag = True
    if lt is not None:
        if value > lt:
            flag = True
    if gt is not None:
        if value < gt:
            flag = True
    if type(value) in (tuple, list):
        if len(value) != length:
            value = default
    # Set default value
    if flag:
        value = default
    # Convert to string?
    if asstr:
        value = str(value)
    return(value)


def validate_crs(source, files):
    """
    Copies the .prj file associated to the source and copy it to a set
    of grids or shapes with missing projection.
    Useful for some methods that don't set any crs to output files

    INPUTS:
      source      [string] input projection source. The .prj extension is forced
      files       [list, tuple] set of files names to associate the source .prj file
    """
    source = _files.default_file_ext(source, 'prj')
    if _os.path.exists(source):
        if type(files) in (list, tuple):
            for filename in files:
                if not _files.has_crs_file(filename):
                    new_proj = _files.default_file_ext(filename, 'prj')
                    try:  # copy prj file
                        _shutil.copy(source, new_proj)
                    except:  # maybe the input is the same that output
                        pass
    # End Function


def validate_gridsystem(grid_system):
    """
    Validate a grid_system file with .sgrd extension

    INPUTS:
      grid_system     [string] input .sgrd file
    OUTPUTS:
      grid_system     [string] fixed grid_system file
                       If grid_system is invalid None is returned
    """
    if type(grid_system) is str:
        grid_system = _files.default_file_ext(grid_system, 'grid')
        if not _os.path.exists(grid_system):
            grid_system = None  # file extent does not exist
    else:
        grid_system = None
    return(grid_system)

