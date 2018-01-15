"""
SAGA GIS algorithm provider
Time Grids provider

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import os as _os
import json as _json
import pandas as _pd
import numpy as _np
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta

import grids as _io


### Time series grid class
class TimeGridObj(object):

    def __init__(self, filename=None):
        # set file attributes
        self._reset_attributes()

        # file connection
        self.connect(filename)

    # Iterate method
    def __iter__(self):
        yield 'title', self.title
        yield 'description', self.description
        yield 'units', self.units
        yield 'folder', self.folder
        yield 'fill_method', self.fill_method
        yield 'fill_value', self.fill_value
        yield 'extremes', self.extremes
        yield 'extreme_value', self.extreme_value

    # Reset attributes
    def _reset_attributes(self):
        # set file attributes
        self.title = ''
        self.description = ''
        self.units = ''
        self.folder = ''
        # set default dataset
        self.dataset = None
        # set methods
        self.fill_method = 0  # set filling method
        self.fill_value = 0  # set filling value
        self.extremes = 0  # set extremes method
        self.extreme_value = 0  # set extreme value

    # Read a temporal grid file .tgrid
    def connect(self, filename):
        if filename and type(filename) is str:
            # Check file
            if not _os.path.exists(filename):
                raise IOError('')

            # Get file parts
            folder = _os.path.dirname(filename)
            basename, ext = _os.path.basename(filename).split('.')

            # Check file extent
            if ext != 'tgrid':
                raise IOError('Bad file extension {}. File must have a .tgrid extension.'.format(ext))

            # Load file
            try:
                with open(filename, 'r') as fid:
                    data = _json.load(fid)  # read file
                # extract data frame
                gridlist = data.pop('dataset')
                df1 = _pd.read_json(gridlist)
                dataset = _pd.DataFrame(data=df1.values, columns=['grids'])
                dataset.index = _pd.to_datetime(df1.index)
                dataset.index.name = 'date'
            except:
                print('Error with file {}'.format(filename))
                return

            # Change attributes
            self.title = basename
            self.folder = folder
            self.__dict__.update(data)
            self.dataset = dataset
            print('Successful Connection')

    # Save Time Grid object to a .tgrid file using folder and title
    # to create file name (folder/title.tgrid)
    # INPUTS
    #  title     [string] optional dataset title. title is used as file name
    #  folder    [string] optional dataset folder. folder must be the folder
    #             the grids are contained
    def register(self, title=None, folder=None):
        # Check values
        if type(self.dataset) is not _pd.DataFrame:
            raise TypeError('Time Grids Object does not contain rasters!')

        if title is None and self.title == '':
            raise ValueError('Paratemer title must be input.')
        elif title is not None:
            if type(title) is str:
                self.title = title
            else:
                raise TypeError('Parameter folder must be a string')

        if folder is None and self.folder == '':
            raise ValueError('Paratemer folder must be input.')
        elif folder is not None:
            if type(folder) is str:
                if _os.path.exists(folder):
                    self.folder = folder
                else:
                    raise IOError('folder "{}" does not exist'.format(folder))
            else:
                raise TypeError('Parameter folder must be a string')

        # Register raster data time
        output = {
            'description' :  self.description,
            'units' : self.units,
            'fill_method' : self.fill_method,
            'fill_value' : self.fill_value,
            'extremes' : self.extremes,
            'extreme_values' : self.extreme_value,
            'dataset' : self.dataset.to_json()
                  }
        # output file name
        filename = _os.path.join(self.folder, self.title + '.tgrid')
        with open(filename, 'w') as fid:
            _json.dump(output, fid)
        print('Successful Register')


### Define functions

def register_grids(title, folder, gridlist, time=None, initial=None,
                   dtime=None, namefmt=None, fill_method=0, fill_value=0,
                   extremes=0, extreme_value=0, description='', units=''):
    # Check inputs and create time vector
    op = -1  # type of input date

    if type(time) in [list, tuple]:
        time = list(time)
        if type(time[0]) is not _datetime:
            raise TypeError('Parameter time must have datetime elements')
        op = 0  # input time

    elif initial is not None and dtime is not None:
        if type(initial) is not _datetime:
            raise TypeError('Parameter initial must be a datetime')
        if type(dtime) not in [_timedelta, int]:
            raise TypeError('Parameter initial must be a timedelta')
        elif type(dtime) is int:
            dtime = _timedelta(dtime)
        op = 1

    elif namefmt is not None:
        if type(namefmt) is str:
            raise TypeError('Parameter namefmt must be a string')
        op = 2

    if op == -1:
        raise IOError('Time values missing (time, initial, dtime, namefmt)')

    # Get time list
    if op == 1:  # use initial and dtime
        nl = len(gridlist)
        time = [initial + dtime * cnt for cnt in range(nl)]
    elif op == 2:
        time = []
        for filename in gridlist:
            basename = _os.path.basename(_os.path.splitext(filename)[0])
            time.append(_datetime.strptime(basename, namefmt))

    # Check time and grid list
    if len(time) != len(gridlist):
        raise IOError('time and gridlist must have the same number of elements')

    # Register time grids
    df = _pd.DataFrame(data=gridlist, columns=['grids'])
    df.index = _pd.to_datetime(time)  # register time index
    df.index.name = 'date'
    # sort by date
    df.sort_index(ascending=True, inplace=True)

    # Create object
    title = _os.path.basename(_os.path.splitext(title)[0])
    tgo = TimeGridObj()                # create temporal object
    tgo.dataset = df                   # save pandas frame
    tgo.folder = folder                # set folder data base
    tgo.title = title                  # set title
    tgo.fill_method = fill_method      # set filling method
    tgo.fill_value = fill_value        # set filling value
    tgo.extremes = extremes            # set extremes method
    tgo.extreme_value = extreme_value  # set extreme value
    tgo.description = description      # set description
    tgo.units = units                  # set units

    # Register file
    tgo.register()
    # register_grids()



