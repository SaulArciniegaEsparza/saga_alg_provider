"""
SAGA GIS algorithm provider
Climate tools

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# env is the provider class
import os as _os
import numpy as _np

import files as _files
import projection as _projection


#==============================================================================
# Library: climate_tools
#==============================================================================

# Estimation of daily potential evapotranspiration from daily average,
# minimum and maximum temperatures using Hargreave's empirical equation
# library: climate_tools  tool: 8
# INPUTS
#  outgrid         [string] output PET grid file
#  tmean           [string] input mean temperature grid
#  tmin            [string] input minimum temperature grid
#  tmax            [string] input maximum temperature grid
#  month           [int] input month from 1 (January) to 12 (December)
#  day             [int] input day of month. If day is None, potential
#                   evapotranspiration es estimated at monthly scale
def grid_ETpot(outgrid, tmean, tmin, tmax, month=1, day=None):
    # Check inputs
    outgrid = _files.default_file_ext(outgrid, 'grid')
    tmean = _files.default_file_ext(tmean, 'grid', False)
    tmin = _files.default_file_ext(tmin, 'grid', False)
    tmax = _files.default_file_ext(tmax, 'grid', False)

    # Check additional inputs
    if day is None:  # only month
        option = '1'
        day = 1
    else:  # day
        option = '0'

    month -= month  # fix month
    month, day = str(int(month)), str(int(day))

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'climate_tools', '8', '-T', tmean, '-T_MIN', tmin,
           '-T_MAX', tmax, '-PET', outgrid, '-MONTH', month, '-DAY', day,
           '-TIME', option]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(outgrid):  # set first input layer crs
        _projection.set_crs(grids=outgrid, crs_method=1, proj=tmean);
    return(flag)  # grid_ETpot()


# Computes sunrise, sunset and day length
# Sunrise is the instant at which the upper edge of the Sun appears
# over the horizon as a result of Earth's rotation.
# Sunset is the instant at which disappearance of the Sun below the
# horizon as a result of Earth's rotation.
# library: climate_tools  tool: 9
# INPUTS
#  sunrise         [string] output sunrise grid file [hours]
#  sunset          [string] output sunset grid file [hours]
#  day_len         [string] output day length grid file [hours]
#  grid            [string] input grid file
#  year            [int] year
#  month           [int] month
#  day             [int] day
#  time            [int] time method
#                   [0] local (default)
#                   [1] world
def sunrise_and_sunset(sunrise, sunset, day_len, grid, year=2017,
                       month=12, day=21, time=0):
    # Check inputs
    grid = _files.default_file_ext(grid, 'grid', False)
    sunrise = _files.default_file_ext(sunrise, 'grid')
    sunset = _files.default_file_ext(sunset, 'grid')
    day_len = _files.default_file_ext(day_len, 'grid')

    # Check date and time
    if _env.saga_version[0] in ['2', '3', '4', '5']:
        date = '%s/%s/%d' % (str(day).zfill(2), str(month).zfill(2), year)
    else:
        date = '%d-%s-%s' % (year, str(month).zfill(2), str(day).zfill(2))

    if time < 0 or time > 1:
        time = 0
    time = str(time)

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'climate_tools', '9', '-TARGET', grid, '-SUNRISE',
           sunrise, '-SUNSET', sunset, '-LENGTH', day_len, '-DAY', date, '-TIME', time]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    if not _files.has_crs_file(sunrise):
        _projection.set_crs(grids=sunrise, crs_method=1, proj=grid);
    if not _files.has_crs_file(sunset):
        _projection.set_crs(grids=sunset, crs_method=1, proj=grid);
    if not _files.has_crs_file(day_len):
        _projection.set_crs(grids=day_len, crs_method=1, proj=grid);
    return(flag)  # sunrise_and_sunset()

