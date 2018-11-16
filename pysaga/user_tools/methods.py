"""
==============================================================================
SAGA GIS algorithm provider
Auxiliary functions for user tools


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
==============================================================================
"""

# import modules
import numpy as _np
from scipy.sparse import lil_matrix as _lil_matrix
from scipy.sparse import eye as _eye

from ..data_manager import grids as _io
from ..utilities import files as _files


# ==============================================================================
# Flow direction solver methods
# ==============================================================================


def cell_direction(flowdir, pixels=None):
    """
    Get next cell in flow direction matrix

    INPUTS
     flowdir       [np.ndarray, GridObj] flow direction array or GridObj
     pixels        [list, tuple, np.ndarray] pixel row and col array [row, col]
                    for multiple pixels use [[row1, col1], [row2, col2],...]
                   If pixels is None, all pixels in data are used
    OUTPUTS
     Ind           [np.ndarray] index of actual pixel
     Indc          [np.ndarray] index of next pixel considering flow direction
                    If flow direction is not in 0 <= flowdir <= 7 then
                    cell is considered as a output flow cell with Indc == Ind
    """
    # Check inputs
    if type(flowdir) is _io.GridObj:
        flowdir = flowdir.get_data()
    elif type(flowdir) is not _np.ndarray:
        raise TypeError('Wrong pixels parameter type {}'.format(str(type(pixels))))
    data = flowdir.astype(int)
    r, c = data.shape
    del(flowdir)   # delete flow direction

    if pixels is None:
        rows, cols = _np.where(data)
    elif type(pixels) in [list, tuple, _np.ndarray]:
        pixels = _np.array(pixels, dtype=int)
    else:
        raise TypeError('Bad pixels parameter type {}'.format(str(type(pixels))))
    if pixels.ndim == 1:
        pixels = _np.array([pixels], dtype=int)
    elif pixels.ndim == 2:
        if pixels.shape[1] != 2:
            raise TypeError('Bad number of columns in pixels parameter')
    else:
        raise TypeError('Bad shape of pixels parameter')

    # Get rows and cols
    npixels = pixels.shape[0]
    rows, cols = pixels[:, 0], pixels[:, 1]
    Ind = _np.arange(0, npixels, dtype=int)  # actual cell index
    Indmat = _np.full((r, c), -1)            # matrix index
    Indmat[rows, cols] = Ind                 # set index number

    # Get cell connection
    flowd = _np.array(data[rows, cols], dtype=int)  # get flow direction
    nextpixels = pixels.copy()
    # North-East
    mask = flowd == 1
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] - 1
        nextpixels[mask, 1] = cols[mask] + 1
    # East
    mask = flowd == 2
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] + 0
        nextpixels[mask, 1] = cols[mask] + 1
    # South-East
    mask = flowd == 3
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] + 1
        nextpixels[mask, 1] = cols[mask] + 1
    # South
    mask = flowd == 4
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] + 1
        nextpixels[mask, 1] = cols[mask] + 0
    # South-West
    mask = flowd == 5
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] + 1
        nextpixels[mask, 1] = cols[mask] - 1
    # West
    mask = flowd == 6
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] + 0
        nextpixels[mask, 1] = cols[mask] - 1
    # North-West
    mask = flowd == 7
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] - 1
        nextpixels[mask, 1] = cols[mask] - 1
    # North
    mask = (flowd == 0) | (flowd == 8)
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = rows[mask] - 1
        nextpixels[mask, 1] = cols[mask] + 0

    # Correct indexes
    mask = nextpixels[:, 0] > r - 1
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = r - 1
    mask = nextpixels[:, 0] < 0
    if _np.sum(mask) > 0:
        nextpixels[mask, 0] = 0

    mask = nextpixels[:, 1] > c - 1
    if _np.sum(mask) > 0:
        nextpixels[mask, 1] = c - 1
    mask = nextpixels[:, 1] < 0
    if _np.sum(mask) > 0:
        nextpixels[mask, 1] = 0

    # Connection
    Indc = Indmat[nextpixels[:, 0], nextpixels[:, 1]]

    # Output cells. Set same index than Ind
    mask = Indc == -1
    if _np.sum(mask) > 0:
        Indc[mask] = Ind[mask]

    return(Ind, Indc)  # cell_direction()


def matrix_coefficients(Ind, Indc):
    """
    Get coefficient matrix for accumulative models

    INPUTS
      Ind, Indc    indices estimated with cell_direction function
    OUTPUTS
      A            conectitity indices matrix
    """
    n = len(Ind)
    A1 = _lil_matrix((n, n), dtype=int)
    A1[Ind, Indc] = 1
    A1[Ind, Ind] = 0
    A2 = _eye(n)
    A = A2 - A1
    A = A.tocsr()
    return(A)  # matrix_coefficients()

