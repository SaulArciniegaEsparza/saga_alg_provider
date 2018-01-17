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


# ==============================================================================
# Library: climate_tools
# ==============================================================================


def grid_ETpot(outgrid, tmean, tmin, tmax, month=1, day=None):
    """
    Estimation of daily potential evapotranspiration from daily average,
    minimum and maximum temperatures using Hargreave's empirical equation

    library: climate_tools  tool: 8

    INPUTS
     outgrid         [string] output PET grid file
     tmean           [string] input mean temperature grid
     tmin            [string] input minimum temperature grid
     tmax            [string] input maximum temperature grid
     month           [int] input month from 1 (January) to 12 (December)
     day             [int] input day of month. If day is None, potential
                      evapotranspiration es estimated at monthly scale
    """
    # Check inputs
    outgrid = _validation.output_file(outgrid, 'grid')
    tmean = _validation.input_file(tmean, 'grid', False)
    tmin = _validation.input_file(tmin, 'grid', False)
    tmax = _validation.input_file(tmax, 'grid', False)

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
    _validation.validate_crs(tmean, [outgrid])
    return(flag)  # grid_ETpot()


def sunrise_and_sunset(sunrise, sunset, day_len, grid, year=2017,
                       month=12, day=21, time=0):
    """
    Computes sunrise, sunset and day length
    Sunrise is the instant at which the upper edge of the Sun appears
    over the horizon as a result of Earth's rotation.
    Sunset is the instant at which disappearance of the Sun below the
    horizon as a result of Earth's rotation.

    library: climate_tools  tool: 9

    INPUTS
     sunrise         [string] output sunrise grid file [hours]
     sunset          [string] output sunset grid file [hours]
     day_len         [string] output day length grid file [hours]
     grid            [string] input grid file
     year            [int] year
     month           [int] month
     day             [int] day
     time            [int] time method
                      [0] local (default)
                      [1] world
    """
    # Check inputs
    sunrise = _validation.output_file(sunrise, 'grid')
    sunset = _validation.output_file(sunset, 'grid')
    day_len = _validation.output_file(day_len, 'grid')
    grid = _validation.input_file(grid, 'grid', False)

    # Check date and time
    if _env.saga_version[0] in ['2', '3', '4', '5']:
        date = '%s/%s/%d' % (str(day).zfill(2), str(month).zfill(2), year)
    else:
        date = '%d-%s-%s' % (year, str(month).zfill(2), str(day).zfill(2))

    time = _validation.input_parameter(int(time), 0, vrange=[0, 1], dtypes=[int])

    # Create cmd
    cmd = ['saga_cmd', '-f=q', 'climate_tools', '9', '-TARGET', grid, '-SUNRISE',
           sunrise, '-SUNSET', sunset, '-LENGTH', day_len, '-DAY', date, '-TIME', time]

    # Run command
    flag = _env.run_command_logged(cmd)
    # Check if output grid has crs file
    _validation.validate_crs(grid, [sunrise, sunset, day_len])
    return(flag)  # sunrise_and_sunset()

