"""
SAGA GIS algorithm provider
Attribute tables tools

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import os as _os
import pandas as _pd
import numpy as _np
import shapefile as _shp

import files as _files
import projection as _projection


#==============================================================================
# Library: user tools
#==============================================================================

# Return a shape file attribute table as a pandas DataFrame
# INPUTS
#  shapefile  [string] vector file name
# OUTPUTS
#  table      [DataFrame] pandas DataFrame object
def get_attribute_table(shapefile):
    # Check if shapefile exist
    shapefile = _files.default_file_ext(shapefile, 'vector')
    if not _os.path.exists(shapefile):
        raise TypeError('Shape file does not exist!')
    # Define default data types
    deftypes = {'F': float, 'N': int, 'C': str, 'L': bool}
    # Get field names
    s = _shp.Reader(shapefile, 'r')
    fields = [s.fields[i][0] for i in range(1, len(s.fields))]
    dtypes = [s.fields[i][1] for i in range(1, len(s.fields))]
    table = _pd.DataFrame(data=s.records(), columns=fields)
    # Convert Series data types
    for i in range(len(fields)):
        field = fields[i]
        try:
            table[field] = table[field].astype(deftypes[dtypes[i]])
        except:
            pass
    s = None  # close shapefile connection
    return(table)


# Get the fieldnames of the attribute table of a shapefile
# INPUTS
#  shapefile   [string] input shape file
#  onlynames   [boolean] get only field names ignoring type data
# OUTPUTS
#  fields      [list] list with field names and data type
def get_attribute_table_fields(shapefile, onlynames=True):
    # Check inputs
    shapefile = _files.default_file_ext(shapefile, 'vector')
    if _os.path.exists(shapefile):
        s = _shp.Reader(shapefile, 'r')  # read shapefile
        if onlynames:
            fields = [s.fields[i][0] for i in range(1, len(s.fields))]
        else:
            fields = [s.fields[i] for i in range(1, len(s.fields))]
        s = None
    return(fields)


#==============================================================================
# Library: table_tools
#==============================================================================

# Create an attribute table (.dbf) for a shape file using
# of from a pandas DataFrame
# library: user defined, uses table_tools  tool: 2 and 11
# INPUTS
#  outable     [string] output attribute table (.dbf)
#  intable     [string, DataFrame] input table (.csv, .txt) or pandas DataFrame
def create_attribute_table(outable, intable):
    # Check inputs
    auxfile = None
    if type(intable) is _pd.DataFrame:
        auxfile = 'auxiliar_table_file.csv'
        if (_env.workdir is not None):
            auxfile = _os.path.join(_env.workdir, auxfile)
        intable.to_csv(auxfile)
        intable = auxfile
    elif _os.path.splitext(intable)[1] not in ['.csv', '.txt', '.dbf']:
        raise IOError('Wrong file extension <{}>'.format(_os.path.splitext(intable)[1]))
    outable = _files.default_file_ext(outable, 'dbf')
    # Convert table
    cmd = ['saga_cmd', '-f=q', 'table_tools', '2', '-INPUT', intable, '-OUTPUT', outable,
           '-FIELD', '0']
    flag = _env.run_command_logged(cmd)
    # delete aditional field
    cmd = ['saga_cmd', '-f=q', 'table_tools', '11', '-TABLE', outable,
           '-FIELDS', 'ENUM_ID', '-OUT_TABLE', outable]
    _env.run_command_logged(cmd);
    # Delete auxiliar file
    if auxfile is not None:
        _os.remove(auxfile)
    return(flag)


# Join attributes from two tables and save it in a new table or shapefile
# library: table_tools  tool: 3 and 4
# INPUTS
#  table1         [string] table (.dbf, .csv, .txt) or shape file which table 2 will
#                  be joined
#  table2         [string, DataFrame] table (.dbf, .csv, .txt) or shapefile,
#                  or pandas DataFrame that will be joined to table1
#  output         [string] optional output file with the same file extension that
#                  table1. If output is None, table1 is overwrited
#  id1            [int, string] table1 field identifier
#  id2            [int, string] table2 field identifier. id1 and id2 must have the same
#                  values for the tables joining
#  fields         [list] list of fields (can be integers or field names) of the
#                  fields that will be joined from table2 to table1. If fields
#                  is None, all fields from table 2 are jpined to table1
#  keep_all       [boolean] keep all values
#  sensitive      [boolean] case for sensitive string comparison. Only for new
#                  SAGA versions
def join_attributes_from_tables(table1, table2, output=None, id1=0, id2=0,
                                fields=None, keep_all=True, sensitive=True):
    # Check inputs
    if fields is None:
        fields_all = '1'
        fields = 'NULL'
    else:
        fields_all = '0'
        fields = ','.join([str(field) for field in fields])
    # convert to text
    keep_all = str(int(keep_all))
    sensitive = str(int(sensitive))
    id1, id2 = str(id1), str(id2)
    # Get input types
    if type(table1) is str:  # table1 is a file
        ext1 = _os.path.splitext(table1)[1]  # get file1 extension
    else:
        raise TypeError('Bad type <{}> for table1 parameter'.format(str(type(table1))))

    aux2 = None  # in case of pandas DataFrame create a temporal file
    if type(table2) is str:  # table2 is a file
        ext2 = _os.path.splitext(table1)[1]  # get file1 extension
    elif type(table2) is _pd.DataFrame:  # table 2 is a pandas DataFrame
        # create temporal file
        aux2 = 'auxiliar_dataframe_table.csv'
        ext2 = '.csv'
        if _env.workdir is not None:
            aux2 = _os.path.join(_env.workdir, aux2)
        # write file
        table2.to_csv(aux2)
        table2 = aux2
    else:
        raise TypeError('Bad type <{}> for table2 parameter'.format(str(type(table2))))

    # Overwrite table1?
    if output is None:
        output = table1
        ext3 = ext1
    elif type(output) is str:
        ext3 = _os.path.splitext(output)[1]  # get file1 extension
        if not bool(ext2):
            output = output + '.shp'
    else:
        raise TypeError('Bad type <{}> for output parameter'.format(str(type(output))))
    # Comparison of file extensions
    if ext1 != ext3:
        raise NameError('table1 and output file must have the same extensions!')
    # Create cmd for each case
    if ext3 == '.shp':  # create a shapefile output
        cmd = ['saga_cmd', '-f=q', 'table_tools', '4']
    else:  # create a table output
        cmd = ['saga_cmd', '-f=q', 'table_tools', '3']
        # add more parameters
    cmd.extend(['-TABLE_A', table1, '-TABLE_B', table2, '-RESULT', output,
                '-ID_A', id1, '-ID_B', id2, '-FIELDS_ALL', fields_all,
                '-FIELDS', fields, '-KEEP_ALL', keep_all])
    if int(_env.saga_version[0]) > 2:
        cmd.extend(['-CMP_CASE', sensitive])
    # Run command
    flag = _env.run_command_logged(cmd)
    # Delete auxiliar file
    if aux2 is not None:
        _os.remove(aux2)
    # End function
    return(flag)  # join_attributes_from_tables()


# Change field data types in an attribute table
# library: table_tools  tool: 7
# INPUTS
#  table      [string] input table (.dbf, .csv, .txt)
#  fields     [int, str, list] field index or name from the attribute table
#              fields can be a list with multiple fields index or names
#  types     [int, list] types is the new data type. If types is an integer
#             all input fields are set with the same data type. If types
#             is a list, it must have the same number of element of fields
#             Available choises of data type:
#                [0] string
#                [1] date
#                [2] color
#                [3] unsigned 1 byte integer
#                [4] signed 1 byte integer
#                [5] unsigned 2 byte integer
#                [6] signed 2 byte integer
#                [7] unsigned 4 byte integer
#                [8] signed 4 byte integer
#                [9] unsigned 8 byte integer
#                [10] signed 8 byte integer
#                [11] 4 byte floating point number
#                [12] 8 byte floating point number
#                [13] binary
#  outshapes  [string] output shape file with deleted fields
def change_field_type(table, fields, types, output=None):
    # Check inputs
    if type(fields) in [int, str]:
        fields = [str(fields)]
    elif type(fields) in [list, tuple, _np.ndarray]:
        fields = [str(field) for field in fields]
    else:
        raise TypeError('Bad fields parameter type <{}>'.format(str(type(fields))))
    if type(types) is int:
        types = str(types)
    elif type(types) in [list, tuple, _np.ndarray]:
        types = [str(dtype) for dtype in types]
    else:
        raise TypeError('Bad types parameter type <{}>'.format(str(type(types))))
    # check dimensions
    if type(fields) is list and type(types) is list:
        if len(fields) != len(types):
            raise ValueError('Wrong inputs, fields and types must have the same number of elements!')
    if output is None:
        output = table
    # Create cmd and run for each field
    for i in range(len(fields)):
        field = fields[i]
        if type(types) is list:
            dtype = types[i]
        else:
            dtype = types
        # cmd
        cmd = ['saga_cmd', '-f=q', 'table_tools', '7', '-TABLE', table, '-FIELD',
               field, '-OUTPUT', output, '-TYPE', dtype]
        # run
        flag = _env.run_command_logged(cmd)
    return(flag)  # change_field_type()


# Deletes selected fields from a table or shapefile
# library: table_tools  tool: 11
# INPUTS
#  outshapes  [string] output shape file with deleted fields
#  inshape    [string] input shape file
#  fields     [int,string,list] if fields is a int/string, a single field is
#              deleted, if fields is a list, multiple fields are deleted
def delete_fields(outshape, inshape, fields=0):
    # Check inputs
    outshape = _files.default_file_ext(outshape, 'vector', False)
    inshape = _files.default_file_ext(inshape, 'vector', False)
    # check fields
    if type(fields) in [int, str]:
        fields = [fields]
    fields = [str(field) for field in fields]  # convert to strings
    fields = ','.join(fields)
    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'table_tools', '11', '-TABLE', inshape,
           '-FIELDS', fields, '-OUT_SHAPES', outshape]
    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if _files.has_crs_file(outshape):  # set first input layer crs
        _projection.set_crs(grids=outshape, crs_method=1, proj=inshape);
    return(flag)  # delete_fields()