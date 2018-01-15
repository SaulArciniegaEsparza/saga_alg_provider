"""
SAGA GIS algorithm provider
Validation of input and output parameters
"""


import os as _os
import numpy as _np

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
        if not _os.path.exists(filename):
            raise IOError('File < {} > does not exist'.format(file_fixed))
        in_file.append(file_fixed)
    if len(in_file):  # only one input
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
    if len(out_file):  # only one file
        out_file = out_file[0]
    return(out_file)

