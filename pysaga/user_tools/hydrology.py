"""
SAGA GIS algorithm provider
User defined methods used in

User tools needs gdal and osr
User tools can work with memory layers

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# Import modules
import os as _os
import numpy as _np

from scipy.sparse.linalg import spsolve as _spsolve
#from scipy.sparse.linalg import lsqr as _lsqr

import _gridsio as _io
import files as _files


# ==============================================================================
# Hydrology analysis methods
# ==============================================================================


def flow_path_distance(flowdir, dem=None, saveas=None,):
    """
    Compute distance from a cell to the output cell in a flow direction grid
    INPUTS
     flowdir    [str, GridObj] input flow direction grid. flowdir can be a
                 grid file name or a GridObj
     dem        [str, GridObj] optional elevation input. dem can be a
                 grid file name or a GridObj. If dem is not None, a X, Y, Z
                 distance is computed, in other way, only a X, Y distance is
                 computed
     saveas     [str] optional output grid file name. If saveas is None, then
                 a GridObj is returned
    OUTPUTS
     outgrid    [GridObj] output distance GridObj (only if saveas is not None)
    """

    # Check inputs
    method = 0  # use only flow direction
    # check flowdir
    if type(flowdir) is str:
        if not _os.path.exists(flowdir):
            raise IOError('flowdir file "{}" does not exists'.format(flowdir))
        else:
            flowdir = _io.GridObj(flowdir)  # read raster file
    elif type(flowdir) is not _io.GridObj:
        raise TypeError('Bad flowdir parameter type {}'.format(str(type(flowdir))))

    # Output file
    if type(saveas) is str:
        saveas = _files.default_file_ext(saveas, 'sdat', True)
    else:
        saveas = None

    # check dem
    if dem is not None:
        if type(dem) is str:
            if _os.path.exists(dem):
                dem = _io.GridObj(dem)  # read raster file
                method = 1

                # Check grids compatibility
                if not dem.is_compatible(flowdir):
                    raise IOError("Input grids don't have the same extent or projection")

            else:
                raise IOError('DEM file "{}" does not exists'.format(dem))
        elif type(dem) is not _io.GridObj:
            raise TypeError('Bad dem parameter type {}'.format(str(type(dem))))

    # Get flow direction grid
    Z1 = flowdir.get_data()
    r1, c1 = _np.shape(Z1)
    Z1[Z1 == -1] = _np.nan  # change output cells
    # check flow direction data
    if _np.sum(Z1 > 7) > 0:
        raise IOError('Bad type of flow direction grid')
    # Cell with values
    pixels = _io.cell_with_value(Z1)  # row and cols with data

    # Get elevation from grid
    if dem is not None:
        Z2 = dem.get_data()
        r2, c2 = _np.shape(Z2)
        Z1[_np.isnan(Z1)] = 0  # change wells without data
        elevation = Z1[pixels[:, 0],
                       pixels[:, 1]]  # cells with elevation
        del(Z2)  # delete dem

    # get coordinates
    coors = flowdir.pixel2coor(pixels)
    X, Y = coors[:, 0], coors[:, 1]

    # Cells connectivity
    Ind, Indc = cell_direction(Z1, pixels)

    if method == 1:  # 3D distance
        dist = ((X[Ind] - X[Indc]) ** 2.0 + (Y[Ind] - Y[Indc]) ** 2.0 +
                (elevation[Ind] - elevation[Indc]) ** 2.0) ** 0.5

    elif method == 0:  # 2D distance
        dist = ((X[Ind] - X[Indc]) ** 2.0 + (Y[Ind] - Y[Indc]) ** 2.0) ** 0.5

    del(X, Y)  # delete vars

    # Compute cell distances
    # Solve system   A * dist = distance
    A = matrix_coefficients(Ind, Indc)
    distance = _spsolve(A, dist)  # problems with memory
    # distance = _lsqr(A, dist, iter_lim=100, atol=1e-10, btol=1e-10)[0]
    longitude = _np.full((r1, c1), -99999.0)
    longitude[pixels[:, 0], pixels[:, 1]] = distance
    # delete vars
    del(Z1, distance, dist, pixels, A, Ind, Indc)

    # Output data
    outgrid = _io.create_virtualraster(longitude, geotransform=flowdir.geotransform,
                                       proj=flowdir.projectionref, noval=-99999.0,
                                       dtype=float)
    if saveas is not None:
        outgrid.to_file(filename=saveas, driver='SAGA')
        outgrid.close()
    else:
        return(outgrid)
    # flow_path_distance()


def concentration_time(flowdir, dem, distance, method=0, cell_acc=0,
                       threshold=500, saveas=None):
    """
    Compute concentration time from each cell until output cells
    Concentration time is calculated in hours

    INPUTS
     flowdir    [str, GridObj] input flow direction grid. flowdir can be a
                 grid file name or a GridObj
     dem        [str, GridObj] elevation grid. dem can be a
                 grid file name or a GridObj
     distance   [str, GridObj] 2D flow path distance. distance can be a
                 grid file name or a GridObj. distance units must be in meters
     method     [int] concentration time method
                 [0] Kirpich concentration time (default)
                 [1] Chow concentration time
     cell_acc   [int] accumulative time method
                 [0] Ponderation using maximum distance (default)
                 [1] Compute concentration time from each cell
                     to output cells (unstable for flat areas)
                 [2] Accumulative cells time until output cell
                     (unstable for flat areas)
     threshold [int, float] maximum concentration time allowed (hours)
     saveas    [str] optional output grid file name. If saveas is None, then
                a GridObj is returned
    OUTPUTS
     outgrid   [GridObj] output concentration time (in hours) GridObj
                (only if saveas is not None)
    """
    # Check inputs
    # check flowdir grid
    if type(flowdir) is str:
        if not _os.path.exists(flowdir):
            raise IOError('flowdir file "{}" does not exists'.format(flowdir))
        else:
            flowdir = _io.GridObj(flowdir)  # read raster file
    elif type(flowdir) is not _io.GridObj:
        raise TypeError('Bad flowdir parameter type {}'.format(str(type(flowdir))))

    # check dem grid
    if type(dem) is str:
        if not _os.path.exists(dem):
            raise IOError('dem file "{}" does not exists'.format(dem))
        else:
            dem = _io.GridObj(dem)  # read raster file
    elif type(flowdir) is not _io.GridObj:
        raise TypeError('Bad dem parameter type {}'.format(str(type(dem))))

    # check distance grid
    if type(distance) is str:
        if not _os.path.exists(distance):
            raise IOError('distance file "{}" does not exists'.format(dem))
        else:
            distance = _io.GridObj(distance)  # read raster file
    elif type(distance) is not _io.GridObj:
        raise TypeError('Bad distance parameter type {}'.format(str(type(distance))))

    # Check methods
    assert 0 <= method <= 1
    assert 0 <= cell_acc <= 2

    # Output file
    if type(saveas) is str:
        saveas = _files.default_file_ext(saveas, 'sdat', True)
    else:
        saveas = None

    # Check compatibility
    if not (dem.is_compatible(flowdir) and dem.is_compatible(distance)):
        raise IOError("Input grids don't have the same extent or projection")

    # Get grids data
    Z1 = dem.get_data()               # elevation
    Z2 = flowdir.get_data()           # flow direction
    Z3 = distance.get_data()          # distance
    r, c = Z2.shape

    # Get cells of analysis
    # get pixels with values
    pixels = _io.cell_with_value(Z2)
    Elev = Z1[pixels[:, 0], pixels[:, 1]]   # elevation
    Dist2 = Z3[pixels[:, 0], pixels[:, 1]]  # distance
    del(Z1, Z3)  # delete data

    # Remove nans
    Elev[_np.isnan(Elev)] = 0
    Dist2[_np.isnan(Dist2)] = 0

    # Cells conectivity
    Ind, Indc = cell_direction(flowdir, pixels)
    A = matrix_coefficients(Ind, Indc)

    # Get coordinates and distance between cells
    coors = flowdir.pixel2coor(pixels)
    X, Y = coors[:, 0], coors[:, 1]
    # compute from cell to cell distance
    dist = ((X[Ind] - X[Indc]) ** 2.0 + (Y[Ind] - Y[Indc]) ** 2.0) ** 0.5
    dist[dist == 0] = 1e-2  # avoid 0 distance
    del (X, Y, Z2)

    # Compute equivalent slope and concentration time
    if cell_acc == 0 or cell_acc == 1:
        # compute equivalent trapezoidal area
        atrap = (Elev[Ind] + Elev[Indc]) * dist / 2  # trap area
        D1 = _spsolve(A, atrap)         # cumulative trapz area
        # D1 = _lsqr(A, atrap, iter_lim=100, atol=1e-10, btol=1e-10)[0]
        H1 = _np.min(Elev)              # min elevation
        H2 = 2 * D1 / Dist2 - H1        # equivalent elevation
        MSM = _np.abs(H2 - H1) / Dist2  # slope of equivalent trapz
        # Compute concentration time
        if method == 0:  # Kirpich method
            tc = 0.0003455 * (Dist2 / MSM ** 0.5) ** 0.77
        elif method == 1:  # Chow method
            tc = 0.01 * (Dist2 / (MSM * 100) ** 0.5) ** 0.64

        if cell_acc == 0:  # ponderate concentration time
            max_index = _np.argmax(Dist2)
            tc = tc[max_index] * Dist2 / Dist2[max_index]

    elif cell_acc == 2:  # Cell by cell time accumulation
        # slope between cells
        slope = _np.abs((Elev[Ind] - Elev[Indc]) / dist)
        slope[slope <= 1e-4] = 1e-4  # avoid flat terrains
        # compute concentration time between cells
        if method == 0:  # Kirpich method
            tc1 = 0.0003455 * (dist / slope ** 0.5) ** 0.77
        elif method == 1:  # Chow method
            tc1 = 0.01 * (dist / (slope * 100) ** 0.5) ** 0.64
        # compute cummulative time
        tc = _spsolve(A, tc1)
        #tc = _lsqr(A, tc1, iter_lim=100, atol=1e-10, btol=1e-10)[0]

    # Save on matrix
    tc[tc > threshold] = -99999.0  # delete high values
    time_matrix = _np.full((r, c), -99999.0)
    time_matrix[pixels[:, 0], pixels[:, 1]] = tc

    # Create GridObj
    # Output data
    outgrid = _io.create_virtualraster(time_matrix, geotransform=flowdir.geotransform,
                                       proj=flowdir.projectionref, noval=-99999.0,
                                       dtype=float)
    if saveas is not None:
        outgrid.to_file(filename=saveas, driver='SAGA')
        outgrid.close()
    else:
        return(outgrid)
    # concentration_time()
