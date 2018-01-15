"""
SAGA GIS algorithm provider
Files management

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import os as _os
import shutil as _shutil


#==============================================================================
# Define files and path functions
#==============================================================================

# Delete all files in a folder associates with an input file or list of files
# INPUTS
#  files   string or list of files
def delete_files(files):
    # Check inputs
    if type(files) is str:
        files = [files]
    if type(files) is not list:
        raise TypeError('Wrong input parameter!')
    # Delete files
    for f in files:
        path = _os.path.dirname(f)
        name = _os.path.basename(_os.path.splitext(f)[0])
        # get list of files in directory
        if path:
            filedir = _os.listdir(path)
        else:
            filedir = _os.listdir(_os.curdir)
        for f1 in filedir:
            name1 = _os.path.splitext(f1)[0]
            if name == name1:
                # delete files
                _os.remove(_os.path.join(path, f1))


# Copy a layer to other path. All associated files (.prj, .dbf., etc) are copied
# INPUTS
#  outlayer   [string] full name of the output layer. If path is not in the filename,
#              copies are made in the same path of the original files
#  inlayer    [string] input layer to be copied. No extension is needed
#  delete     [boolean] if delete is True original files are deleted after copy
def copy_layer(outlayer, inlayer, delete=False):
    # Get file parts
    path1 = _os.path.dirname(inlayer)
    path2 = _os.path.dirname(outlayer)
    fullname1 = _os.path.splitext(inlayer)[0]  # full name without extension
    name1 = _os.path.basename(fullname1)  # get basename
    fullname2 = _os.path.splitext(outlayer)[0]  # full name without extension
    if not path2:  # copy to the same path
        path2 = path1
        fullname2 = _os.path.join(path2, fullname2)
    # Get associated files
    filedir = _os.listdir(path1)  # get list of files
    for f1 in filedir:
        filename, ext = _os.path.splitext(f1)  # get filename without extension
        name = _os.path.basename(filename)
        if name == name1:
            # copy file to another path
            _shutil.copy(_os.path.join(path1, f1), fullname2 + ext)
            if delete:  # delete original files
                _os.remove(_os.path.join(path1, f1))


# Associate a file with its default extension
# INPUTS
#  filename    [string] original file name
#  ftype       [string] type of file: 'grid', 'vector', 'saga' or other file extension
#  force       [boolean] if True, file extent is overwritten
# OUTPUTS
#  output      [string] new file name
def default_file_ext(filename, ftype='grid', force=True):
    output = filename
    # Define default extensions
    defext = {'grid': '.sgrd', 'vector': '.shp', 'saga': '.sdat'}
    # Create new filename
    ftype = ftype.lower()
    file_ext = _os.path.splitext(filename)[1]  # get file extension
    if not file_ext:
        force = True
    if force:
        if defext.has_key(ftype):
            output = _os.path.splitext(filename)[0] + defext[ftype]
        else:
            output = _os.path.splitext(filename)[0] + '.' + ftype
    return(output)


# Checks if a grid or shape file has an associate .prj file
# INPUTS
#  filename    [string] input grid or shape file
# OUTPUTS
#  flag        [boolean] flag is True if filename has an associate .prj file
def has_crs_file(filename):
    basename = _os.path.splitext(filename)[0]
    basename += '.prj'  # create prj filename
    flag = _os.path.exists(basename)
    return(flag)


# Create a no repeated file name in a folder
# INPUTS
#  path       [string] file path. If None current dir is used
#  ext        [string] file extend
#  basename   [string] base file name. As default basename is aux
# OUTPUTS
#  filename   [string] output non repeated file name
def create_filename(path=None, ext='txt', basename='aux'):
    # Check inputs
    if path is None:  # get current dir
        path = _os.getcwd()

    if not _os.path.exists(path):
        raise IOError('Input path does not exist!')
    basename, ext = str(basename), str(ext)
    if ext.count('.') == 0:
        ext = '.' + ext  # add dot

    # Create file
    filename = _os.path.join(path, basename + ext)
    cnt = -1
    while _os.path.exists(filename):
        cnt += 1
        filename = _os.path.join(path, basename + '_' + str(cnt) + ext)
    return(filename)


# Get a list of files contained in a folder using constraints
# INPUTS
#  folder      [str] folder path with files
#  ext         [str] file extension
#  contains    [str] check if file base name contains an specific text
#  start       [str] check if file base name starts with
#  ends        [str] check if file base name ends with
# OUTPUTS
#  filter      [list] filtered list of files
def file_list(folder, ext='sgrd', contains='', start='', ends=''):
    if not _os.path.exists(folder):
        raise IOError('Folder path "{}" does not exist.'.format(folder))

    # Check inputs
    if not ext.startswith('.'):
        ext = '.' + ext

    # Get files list
    files = _os.listdir(folder)
    if not files:
        print('Folder "{}" is empty'.format(folder))
    # Filter files
    filter = []
    for filename in files:
        basename, fileext = _os.path.splitext(filename)
        if ((fileext == ext) and (basename.find(contains) != -1) and
                (basename.startswith(start)) and (basename.endswith(ends))):
            filter.append(filename)
    return(filter)
