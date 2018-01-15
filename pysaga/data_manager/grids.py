"""
SAGA GIS algorithm provider
Gdal tools for raster analysis

Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City
"""

# env is the provider class
import os as _os
from copy import deepcopy as _deepcopy
from collections import OrderedDict as _OrderedDict
import numpy as _np
import pandas as _pd

import files as _files
from _projection import crs_from_epsg as _crs_from_epsg
try:
    import gdal as _gdal
    import gdalconst as _gdalconst
except:
    print('gdal library could not be imported, some tools in PySaga would be disabled!')


# ==============================================================================
# Define grid object with gdal module
# ==============================================================================

# Main class
class GridObj(object):
    def __init__(self, filename=None):
        self._reset_attributes()

        if filename is not None:
            self.read_file(filename)

    # Create iterable object
    def __iter__(self):
        yield 'filename', self.filename
        yield 'drivername', self.drivername
        yield 'geotransform', self.geotransform
        yield 'projectionref', self.projectionref
        yield 'xsize', self.xsize
        yield 'ysize', self.ysize
        yield 'bands', self.bands

    # Text special function
    def __repr__(self):
        text = 'Grid Object'
        for key, value in self.__dict__.iteritems():
            text += '\n' + key + ':'
            text += '\n' + str(value)
        return(text)

    # Reset raster connection
    def _reset_attributes(self):
        self.filename = ''
        self.driver = None
        self.drivername = ''
        self.geotransform = ()
        self.projectionref = ''
        self.xsize = 0
        self.ysize = 0
        self.bands = 0

    # Raster connection using gdal library
    def read_file(self, filename):
        # Check grids file
        if filename.endswith('.sgrd'):  # change sgrd to sdat
            filename = _files.default_file_ext(filename, 'sdat', True)

        # Read raster with gdal
        raster = _gdal.Open(filename, _gdal.GA_Update)
        if raster is None:
            raise IOError('{} can not be readed as a raster'.format(filename))

        # save properties
        self.driver = raster  # gdal dataset
        self.filename = filename  # save filename
        rinfo = self.get_grid_info()  # get raster properties
        self.__dict__.update(rinfo)  # update values
        del(raster)
        # read_raster()

    # Raster file disconnection
    def close(self):
        if type(self.driver) is _gdal.Dataset:
            self.driver = None
            self._reset_attributes()

    # Use gdal library for extract raster information
    # OUTPUT
    #  rinfo     [dict] raster information dictionary
    def get_grid_info(self):
        # Check grid connection
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get raster properties
        geotrans = self.driver.GetGeoTransform()  # get transformation
        projref = self.driver.GetProjectionRef()  # get projection
        drivername = self.driver.GetDriver().ShortName  # get file driver
        xsize = self.driver.RasterXSize  # get x size
        ysize = self.driver.RasterYSize  # get y size
        bands = self.driver.RasterCount  # count number of layers
        # Output data
        rinfo = {'geotransform' : geotrans,
                 'projectionref' : projref,
                 'drivername' : drivername,
                 'xsize' : xsize,
                 'ysize' : ysize,
                 'bands' : bands}
        return(rinfo)  # grid_info()

    # Get grid coordinates of whole data or from selected points
    # INPUTS
    #  row, col    [int, list] input row and cols of pixels. If row=col=None, all pixels
    #               coordinates are returned
    #  slice       [boolean] if slice is True, row and col is used as extent
    # OUTPUT
    #  X, Y        [np.array] X and Y coordinate matrix
    def get_coordinates(self, extent=None, index=False):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Get geo transformation
        gt = _deepcopy(self.geotransform)

        # Define get coordinates sub-function
        def get_coors_matrix(rows, cols):
            # Get coordinates matrix
            X = gt[0] + cols * gt[1] + rows * gt[2]
            Y = gt[3] + cols * gt[4] + rows * gt[5]
            return(X, Y)

        # Extract by extent or by pixels
        if extent is None:
            cols, rows = _np.meshgrid(_np.arange(self.xsize),
                                      _np.arange(self.ysize))
            X, Y = get_coors_matrix(rows, cols)  # get coordinates

        elif type(extent) in [list, tuple, _np.ndarray]:  # use extent
            if not index:  # use coordinates
                # convert extent to rows and cols
                points = [[extent[0], extent[2]], [extent[1], extent[3]]]
                pixels = self.coor2pixel(points)
                # convert pixel to extent
                extent = [pixels[0, 1], pixels[1, 1],
                          pixels[1, 0], pixels[0, 0]]

            # Check extent
            assert 0 <= extent[0] < extent[1]
            assert extent[0] < extent[1] <= self.xsize
            assert 0 <= extent[2] < extent[3]
            assert extent[2] < extent[3] <= self.ysize
            # Get pixel indexes
            cols, rows = _np.meshgrid(_np.arange(extent[0], extent[1] + 1),
                                      _np.arange(extent[2], extent[3] + 1))
            X, Y = get_coors_matrix(rows, cols)  # get coordinates
        return(X, Y)  # get_grid_coordinates()

    # Get raster extent
    # OUTPUTS
    #  extent    [np.ndarray] [xmin, xmax, ymin, ymax] array
    def get_extent(self):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get extent
        cols = _np.array([0, self.xsize])
        rows = _np.array([0, self.ysize])
        gt = _deepcopy(self.geotransform)  # get geotransform
        # get coordinates
        X = gt[0] + cols * gt[1] + rows * gt[2]
        Y = gt[3] + cols * gt[4] + rows * gt[5]
        # Output extent
        X = [X.min(), X.max()]
        Y = [Y.min(), Y.max()]
        extent = _np.hstack((X, Y))  # extent
        return(extent)  # get_extent()

    # Return pixel width and height
    # OUTPUTS
    #  width, height    [float] pixel size
    def get_resolution(self):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get geotransform
        gt = _deepcopy(self.geotransform)
        width, height = gt[1], abs(gt[5])
        return(width, height)

    # Return raster origin (upper left corner)
    # OUTPUT
    #  origin     [np.ndarray] [x_left, y_upper] array
    def get_origin(self):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get geotransform
        gt = _deepcopy(self.geotransform)
        origin = (gt[0], gt[3])
        return(origin)

    # Get no data values from selected bands
    # INPUTS
    #  bands     [int, list] band number or list of band numbers
    # OUTPUTS
    #  nodata     If bands is an integer, nodata is an int or float
    #             If bands is a list, nodata is a numpy array
    def get_nodata(self, bands=1):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster gdal dataset!')
            # Check inputs
        if type(bands) == int:
            bands = _np.array([bands], dtype=int)
        elif type(bands) in [list, tuple, _np.ndarray]:
            bands = _np.array(bands, dtype=int)
        else:
            raise TypeError('Bad bands type {}!'.format(str(type(bands))))
        # Iterate over bands
        bands = _np.unique(_np.sort(bands))
        data = []
        for i in range(len(bands)):
            band_data = self.driver.GetRasterBand(i + 1)
            data.append(band_data.GetNoDataValue())
            band_data = None
        # Output data
        if len(data) == 1:
            nodata = data[0]
        else:
            nodata = _np.array(data)
        return(nodata)  # get_nodata()

    # Extract coordinates from a pixel
    # INPUTS
    #  pixels     [list, tuple, np.ndarray] [row, col] array. For multiple pixels use
    #              [[row1, col1], [row2, col2],...]
    # OUTPUT
    #  coors      [np.ndarray] output coordinate arrays [[x1, y1], [x2, y2]]
    def pixel2coor(self, pixels):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check pixels
        if type(pixels) in [list, tuple, _np.ndarray]:
            pixels = _np.array(pixels, dtype=_np.int32)
            if pixels.ndim == 1 and pixels.shape[-1] == 2:
                pixels = pixels.reshape(1, 2)
            elif pixels.ndim > 2 or pixels.shape[-1] != 2:
                raise TypeError('Bad pixels dimenssions {}'.format(str(pixels.shape)))
        else:
            raise TypeError('Wrong pixels parameter type {}'.format(str(type(pixels))))
        # Check errors
        assert 0 <= pixels[:, 0].min()
        assert pixels[:, 0].max() < self.ysize
        assert 0 <= pixels[:, 1].min()
        assert pixels[:, 1].max() < self.xsize

        # Get coordinates
        rows, cols = pixels[:, 0], pixels[:, 1]
        gt = _deepcopy(self.geotransform)  # get geotransform
        X = gt[0] + cols * gt[1] + rows * gt[2]
        Y = gt[3] + cols * gt[4] + rows * gt[5]
        # Return object
        coors = _np.array(zip(X, Y), dtype=float)
        return(coors)  # pixel2coor()

    # Convert coordinates from a pixel to row and col
    # INPUTS
    #  points      [list, tuple, np.ndarray] [x, y] coordinates.
    #               For multiple points use [[x1, y1], [x2, y2], ...]
    # OUPUTS
    #  pixels   [np.ndarray] rows and cols of coordinates [[row1, col1], [row2, col2]]
    def coor2pixel(self, points):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check pixels
        if type(points) in [list, tuple, _np.ndarray]:
            points = _np.array(points, dtype=_np.int32)
            if points.ndim == 1 and points.shape[-1] == 2:
                points = points.reshape(1, 2)
            elif points.ndim > 2 or points.shape[-1] != 2:
                raise TypeError('Bad points dimenssions {}'.format(str(points.shape)))
        else:
            raise TypeError('Wrong points parameter type {}'.format(str(type(points))))
        # Get raster extent and geotransformation
        gt = _deepcopy(self.geotransform)
        extent = self.get_extent()  # extent
        origin = self.get_origin()  # raster origin
        width, height = self.get_resolution()  # resolution
        # Check errors
        assert extent[0] - width / 2.0 <= points[:, 0].min()
        assert points[:, 0].max() <= extent[1] + width / 2.0
        assert extent[2] - height / 2.0 <= points[:, 1].min()
        assert points[:, 1].max() <= extent[3] + height / 2.0
        # Get rows and cols
        cols = _np.array(_np.round((points[:, 0] - origin[0]) / width), dtype=int)
        rows = _np.array(_np.round((origin[1] - points[:, 1]) / height), dtype=int)
        # Return pixels
        pixels = _np.array(zip(rows, cols), dtype=int)
        return(pixels)  # coor2pixel()

    # Get pixel values using rows and cols
    # INPUTS
    #  pixels    [list, tuple, np.ndarray] [row, col] pixel. For multiple pixels use
    #             [[row1, col1], [row2, col2]]
    #  bands     [int, list] band number or list of band numbers
    #  dtype     [int, float, np.int, np.float] data type
    # OUTPUTS
    #  data     If one pixel of a single band is called, data is a int or float value
    #           If multiple pixels are called of a single band, data is a pandas DataFrame
    #            with pixel number as index
    #           If multiple pixels of multiple bands are called, data is a pandas DataFrame
    #            where row index is the band number (starting from 1) and columns index as
    #            pixel number
    def get_pixel_value(self, pixels, bands=1, dtype=float):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check inputs
        if type(bands) == int:
            bands = _np.array([bands], dtype=int)
        elif type(bands) in [list, tuple, _np.ndarray]:
            bands = _np.array(bands, dtype=int)
        else:
            raise TypeError('Bad bands type {}!'.format(str(type(bands))))

        if type(pixels) in [list, tuple, _np.ndarray]:
            pixels = _np.array(pixels, dtype=int)
            # create a 2 dimensional array
            if pixels.ndim == 1:
                pixels = pixels.reshape((1, 2))
            elif pixels.ndim > 2:
                raise TypeError('Bad pixels dimenssion {}'.format(str(pixels.ndim)))
        else:
            raise TypeError('Bad pixels type variable {}'.format(str(type(pixels))))

        # Get raster data
        # array to save data
        data = _np.zeros((len(bands), pixels.shape[0]), dtype=dtype)
        # main loop
        bands = _np.unique(_np.sort(bands))
        for i in range(len(bands)):
            # verify raster band
            assert 1 <= bands[i] <= self.bands
            # get raster band
            band = self.driver.GetRasterBand(bands[i])
            nodata = band.GetNoDataValue()  # no data values
            # get values for each pixel
            for j in range(pixels.shape[0]):
                row, col = pixels[j]  # get row and col
                value = band.ReadAsArray(col, row, 1, 1).astype(dtype)[0][0]
                # save pixel value
                if value == nodata:
                    data[i, j] = _np.nan
                else:
                    data[i, j] = value
            # close band connections
            band = None

        # Output data frame
        if len(bands) == 1 and pixels.shape[0] == 1:  # single value
            data = data[0][0]
        elif len(bands) == 1:  # single band
            data = data.reshape(pixels.shape[0])
            data = _pd.DataFrame(data=data, columns=['band'])
            data.index.name = 'pixel'
        else:
            names = ['pixel' + str(index) for index in range(pixels.shape[0])]
            data = _pd.DataFrame(data=data, columns=names, index=bands)
            data.index.name = 'band'
        return(data)  # get_pixel_values()

    # Get raster values from bands
    # INPUTS
    #  bands     [int, list] band number or list of band numbers
    #  extent    [list] grid extent with cells number or coordinates [xmin, xmax, ymin, ymax]
    #  index     [boolean] if index is True, extent uses row and col indexes. If index
    #             is False (default) extent uses coordinates
    #  dtype     [int, float, np.int, np.float] data type
    # OUTPUT
    #  data      [np.array] output numpy array. If bands is a list, output data is
    #             an 3 dimensional array
    def get_data(self, bands=1, extent=None, index=False, dtype=_np.float64):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check inputs
        if type(bands) == int:
            bands = _np.array([bands], dtype=int)
        elif type(bands) in [list, tuple, _np.ndarray]:
            bands = _np.array(bands, dtype=int)
        else:
            raise TypeError('Bad bands type {}!'.format(str(type(bands))))

        # Check input extent
        if extent is None:
            # Default range of pixels
            xoff, yoff = 0, 0
            xsize = int(self.xsize)
            ysize = int(self.ysize)

        elif len(extent) == 4:
            if index:  # use extent as cell index
                extent = [int(value) for value in extent]
                xoff, xsize, yoff, ysize = extent

            else:      # use extent as coordinates
                # convert extent to rows and cols
                points = [[extent[0], extent[2]], [extent[1], extent[3]]]
                pixels = self.coor2pixel(points)
                # convert pixel to extent
                extent = [pixels[0, 1], pixels[1, 1],
                          pixels[1, 0], pixels[0, 0]]
                # Get a range of pixels
                xoff = int(pixels[0, 1])
                yoff = int(pixels[1, 0])
                xsize = int(pixels[1, 1] - pixels[0, 1] + 1)
                ysize = int(pixels[0, 0] - pixels[1, 0] + 1)

        else:
            raise TypeError('Bad extent type {}!'.format(str(type(extent))))
        # Get raster data
        bands = _np.unique(_np.sort(bands))
        data = []
        for i in range(len(bands)):
            # verify raster band
            assert 1 <= bands[i] <= self.bands
            # get raster band
            band = self.driver.GetRasterBand(bands[i])
            values = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(dtype)
            nodata = band.GetNoDataValue()  # no data values
            values[values == nodata] = _np.nan  # change nan values
            # save data
            data.append(values)
        # Output array
        if len(data) == 1:
            data = data[0]
        elif len(data) > 1:
            data = _np.array(data, dtype=data[0].dtype)
        return(data)  # get_data()

    # Change band values
    # If GridObj is connected with a raster file, file is overwrited
    # INPUTS
    #  data      [np.ndarray] data array
    #  band      [int] band number
    #  row, col  [int] upper left pixel row and col to set new data
    def set_data(self, data, band=1, row=0, col=0):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check data
        row, col, band = int(row), int(col), int(band)
        if type(data) is not _np.ndarray:
            data = _np.array(data)
        # check dimenssions
        if data.ndims != 2:
            raise TypeError('Parameter data must be a bidimenssional numpy array.')
        # check extent
        nr, nc = data.shape
        assert 1 <= band <= self.bands
        assert 0 <= row <= self.ysize - nr
        assert 0 <= col <= self.xsize - nc

        # Write data in band
        band_data = self.driver.GetRasterBand(band)  # get band
        band_data.WriteArray(data, col, row)  # write data
        # Try to flush data to disk
        if self.drivername not in ['MEM', 'VRT']:
            band_data.FlushCache()
        band_data = None  # close
        # set_data()

    # Change no data values for all bands
    # INPUTS
    #  nodata     [int, float] new no data value
    def set_nodatavalue(self, nodata=-99999):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Change no data value of all bands
        for i in range(self.bands):
            band_data = self.driver.GetRasterBand(i + 1)  # get band
            band_data.SetNoDataValue(nodata)  # change no data value

            # Try to flush data to disk
            if self.drivername not in ['MEM', 'VRT']:
                band_data.FlushCache()
            band_data = None
        # set_nodatavalue()

    # Write GridObj in a raster file
    # INPUTS
    #  filename      [string] output raster file name
    #  driver        [string] gdal driver name. SAGA as default
    def to_file(self, filename=None, driver='SAGA'):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check inputs
        if not self.filename and filename is None:
            raise ValueError('Parameter filename must be input!')
        elif filename is None:
            filename = self.filename
        if not self.drivername and driver is None:
            raise ValueError('Parameter driver must be input!')
        elif driver is None or driver in ['MEM', 'VRT']:
            driver = self.drivername
        if filename is None:
            raise ValueError('Parameter filename cant be None')
        if driver is None:
            raise ValueError('Parameter driver cant be None')

        # Check filename
        drivers = gdal_driver_list()  # get driver list
        if not drivers.has_key(driver):
            raise TypeError('Wrong driver name {}'.format(driver))
        ext = '.' + drivers[driver]
        if not filename.endswith(ext):
            filename = _os.path.splitext(filename)[0]
            filename += ext

        # Create a copy
        dest = _gdal.GetDriverByName(driver)  # create driver
        outfile = dest.CreateCopy(filename, self.driver)  # copy raster
        outfile.SetProjection(self.projectionref)  # set same projection

        # Close connections
        dest = None
        outfile = None
        self.close()

        # Connection with new
        self.read_file(filename)
        # to_file()

    # Make a virtual copy of GridObj
    # OUTPUT
    # OUTPUTS
    #  newobj       [GridObj] virtual GridObj
    def to_memory(self):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster gdal dataset!')

        # Create a virtual copy
        driver = _gdal.GetDriverByName('MEM')  # create driver
        dest = driver.CreateCopy('', self.driver)  # copy raster
        dest.FlushCache()

        # Create new grid obj
        newobj = GridObj()  # create new grid object
        newobj.driver = dest  # set connection
        newobj.filename = ''
        newobj.divername = 'MEM'
        newobj.bands = dest.RasterCount
        newobj.geotransform = dest.GetGeoTransform()
        newobj.projectionref = dest.GetProjectionRef()
        newobj.xsize = dest.RasterXSize
        newobj.ysize = dest.RasterYSize

        # Close files
        del (dest)
        band_data = None
        return(newobj)  # resample()

    # Copy resolution and projection from other GridObj
    # Results are saved in a virtual raster
    # INPUTS
    #  refobj        [instance] reference GridObj
    #  resampling    [int] resampling method
    #                 [0] Nearest Neighbour
    #                 [1] Averange
    #                 [2] Bilinear
    #                 [3] Cubic (default)
    #                 [4] Cubic Spline
    # OUTPUTS
    #  newobj       [GridObj] resampled and reprojected new GridObj
    def copy_resolution(self, refobj, resampling=3):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        if type(refobj.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('refobj is not a gdal Dataset!')

        # Default resampling methods
        methods = [
            _gdalconst.GRA_NearestNeighbour,
            _gdalconst.GRA_Average,
            _gdalconst.GRA_Bilinear,
            _gdalconst.GRA_Cubic,
            _gdalconst.GRA_CubicSpline
        ]
        resampling = int(resampling)
        assert 0 <= resampling <= 4
        method = methods[resampling]

        # Create memory layer
        band_data = self.driver.GetRasterBand(1)
        driver = _gdal.GetDriverByName('MEM')
        dest = driver.Create('', refobj.xsize, refobj.ysize, self.bands,
                             band_data.DataType)
        dest.SetGeoTransform(refobj.geotransform)
        dest.SetProjection(refobj.projectionref)

        # Reproject image
        _gdal.ReprojectImage(self.driver, dest, self.projectionref,
                            refobj.projectionref, method)

        # Create new grid obj
        newobj = GridObj()  # create new grid object
        newobj.driver = dest  # set connection
        newobj.filename = ''
        newobj.divername = 'MEM'
        newobj.bands = dest.RasterCount
        newobj.geotransform = dest.GetGeoTransform()
        newobj.projectionref = dest.GetProjectionRef()
        newobj.xsize = dest.RasterXSize
        newobj.ysize = dest.RasterYSize

        # Close files
        del (dest)
        band_data = None
        return(newobj)  # copy_resolution()

    # Resample raster and save in a virtual raster
    # INPUTS
    #  cellsize      [int] new cellsize. If cellsize is 0, cellsize is ignored
    #  extent        [list, tuple, np.ndarray] new raster extent [xmin, xmax, ymin, ymax].
    #                 If extent is None, actual extent is used
    #  ncols, nrows  [int] new number of cols and rows. It can be used instead of cellsize parameter.
    #                 If ncols or nrows is 0, then they are ignored
    #  resampling    [int] resampling method
    #                 [0] Nearest Neighbour
    #                 [1] Averange
    #                 [2] Bilinear
    #                 [3] Cubic (default)
    #                 [4] Cubic Spline
    # OUTPUTS
    #  newobj       [GridObj] resampled virtual GridObj
    def resample(self, cellsize=0, extent=None, ncols=0, nrows=0, resampling=3):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check inputs
        if type(extent) in [list, tuple, _np.ndarray]:
            if len(extent) != 4:
                raise TypeError('Bad extent lenght {}'.format(len(extent)))

        # Default resampling methods
        methods = [
            _gdalconst.GRA_NearestNeighbour,
            _gdalconst.GRA_Average,
            _gdalconst.GRA_Bilinear,
            _gdalconst.GRA_Cubic,
            _gdalconst.GRA_CubicSpline
        ]
        resampling = int(resampling)
        assert 0 <= resampling <= 4
        method = methods[resampling]

        # Get extent
        if extent is None:
            extent = self.get_extent()

        # Compute new cellsize
        width, height = self.get_resolution()
        if cellsize > 0:
            width, height = cellsize, cellsize
            # Get number of rows and cols
            ncols = int(_np.round(abs(extent[1] - extent[0]) / float(width)))
            nrows = int(_np.round(abs(extent[3] - extent[2]) / float(height)))
            # Fix extent limits
            # extent[1] = extent[0] + (ncols - 1) * width
            # extent[3] = extent[2] + (nrows - 1) * height
        elif ncols > 0 and nrows > 0:
            width = abs(extent[1] - extent[0]) / float(ncols)
            height = abs(extent[3] - extent[2]) / float(nrows)

        # Create new geotrandform
        gt = _deepcopy(self.geotransform)
        gt = list(gt)
        gt[0] = extent[0]  # left x
        gt[1] = width      # pixel width
        gt[3] = extent[3]  # upper y
        gt[5] = -height    # pixel height (negative)

        # Create memory layer
        band_data = self.driver.GetRasterBand(1)
        driver = _gdal.GetDriverByName('MEM')
        dest = driver.Create('', ncols, nrows, self.bands,
                             band_data.DataType)
        dest.SetGeoTransform(gt)
        dest.SetProjection(self.projectionref)

        # Resample image
        _gdal.ReprojectImage(self.driver, dest, self.projectionref,
                            self.projectionref, method)

        # Create new grid obj
        newobj = GridObj()  # create new grid object
        newobj.driver = dest  # set connection
        newobj.filename = ''
        newobj.divername = 'MEM'
        newobj.bands = dest.RasterCount
        newobj.geotransform = dest.GetGeoTransform()
        newobj.projectionref = dest.GetProjectionRef()
        newobj.xsize = dest.RasterXSize
        newobj.ysize = dest.RasterYSize

        # Close files
        del(dest)
        band_data = None
        return(newobj)  # resample()

    # Reproject raster and save in a virtual raster
    # INPUTS
    #  proj          [int, str] proj can be a EPSG integer code or
    #                 a Well Known string projection
    #  resampling    [int] resampling method
    #                 [0] Nearest Neighbour
    #                 [1] Averange
    #                 [2] Bilinear
    #                 [3] Cubic (default)
    #                 [4] Cubic Spline
    # OUTPUTS
    #  newobj       [GridObj] reprojected virtual GridObj
    def reproject(self, proj, resampling=3, threshold=0.125):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Default resampling methods
        methods = [
            _gdalconst.GRA_NearestNeighbour,
            _gdalconst.GRA_Average,
            _gdalconst.GRA_Bilinear,
            _gdalconst.GRA_Cubic,
            _gdalconst.GRA_CubicSpline
        ]
        resampling = int(resampling)
        assert 0 <= resampling <= 4
        method = methods[resampling]

        # Get projection
        if type(proj) is int:
            proj = _crs_from_epsg(proj)
        elif type(proj) is not str:
            raise TypeError('Bad proj parameter type {}. proj must be an EPSG'
                            'integer code or a Well Known string'.format(str(type(proj))))

        # Fetch default values for target raster dimensions and geotransform
        tmp_ds = _gdal.AutoCreateWarpedVRT(self.driver,
                                          None,  # use self.projectionref
                                          proj,
                                          method,
                                          threshold)
        # Project raster
        dest = _gdal.GetDriverByName('MEM').CreateCopy('', tmp_ds)

        # Create new grid obj
        newobj = GridObj()  # create new grid object
        newobj.driver = dest  # set connection
        newobj.filename = ''
        newobj.divername = 'MEM'
        newobj.bands = dest.RasterCount
        newobj.geotransform = dest.GetGeoTransform()
        newobj.projectionref = dest.GetProjectionRef()
        newobj.xsize = dest.RasterXSize
        newobj.ysize = dest.RasterYSize

        # Close files
        del(dest)
        band_data = None
        return(newobj)  # resample()

    # Check if GridObj is compatible for direct map algebra
    # with a reference GridObj
    # INPUTS
    #  refobj       [GridObj] reference GridObj
    # OUTPUTS
    #  flag         [bool] flag is True if GridObj is compatible
    #                with reference GridObj
    def is_compatible(self, refobj):
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster gdal dataset!')
        # Check
        if self.geotransform != refobj.geotransform:
            return(False)
        if (self.xsize != refobj.xsize) or (self.ysize != refobj.ysize):
            return(False)
        # if self.projectionref != refobj.projectionref:
        #    return(False)
        return(True)


#==============================================================================
# Grid Object functions
#==============================================================================

# Get _gdal drivers dictionary
# OUTPUT
#  drivers    [dict] drivers and file extension
def gdal_driver_list():
    drivers = {}
    for i in range(_gdal.GetDriverCount()):
        drv = _gdal.GetDriver(i)
        if drv.GetMetadataItem(_gdal.DCAP_RASTER):
            drivers[drv.ShortName] = drv.GetMetadataItem(_gdal.DMD_EXTENSIONS)
    return(drivers)  # gdal_driver_list()


# Create virtual layer from a numpy array
# INPUTS
#  data            [np.ndarray] input data. If data is a bi-dimensional array
#                   only one band is created. For tree-dimensional array, number of bands
#                   is extracted from the first dimension
#  geotransform    [list, tuple, np.ndarray] geo transformation array
#                   [x_left, width, x_rotation, y_upper, y_rotation, height]
#  proj            [int, str] projection reference. If proj is an integer
#                   takes as the EPSG code. If proj is a string is considering
#                   as a WKT projection
#  noval           [int, float] no data value
#  dtype           [type] data type (int or float)
# OUTPUTS
#  gridobj         Memory GridObj
def create_virtualraster(data, geotransform=[0, 100, 0, 0, 0, -100],
                        proj=4326, noval=-99999, dtype=float):
    # Get data type
    if dtype == float:
        dtype = _gdal.GDT_Float32
    elif dtype == int:
        dtype = _gdal.GDT_Int32
    else:  # default data type
        dtype = _gdal.GDT_Float32
    # Get bands number
    if _np.ndim(data) == 2:
        nbands = 1
        ysize, xsize = _np.shape(data)
    elif _np.ndim(data) == 3:
        nbands, ysize, xsize = _np.shape(data)
    else:
        raise TypeError('Wrong data type {}!'.format(str(type(data))))
    nbands, xsize, ysize = int(nbands), int(xsize), int(ysize)

    # Check parameters
    # check geotransform
    if type(geotransform) in [list, tuple, _np.ndarray]:
        if len(geotransform) != 6:
            raise TypeError('Wrong geotransform dimensions!')
    else:
        raise TypeError('Wrong geotransform type {}!'.format(str(type(geotransform))))

    # Get projection
    if type(proj) in [int, float]:
        proj = _crs_from_epsg(int(proj))
    elif type(proj) != str:
        raise TypeError('Wrong proj type {}!'.format(str(type(proj))))

    # Write raster data
    driver = _gdal.GetDriverByName('MEM')
    raster = driver.Create('', xsize, ysize, nbands, dtype)
    raster.SetGeoTransform(geotransform)
    raster.SetProjection(proj)
    if nbands == 1:
        band = raster.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(noval)
    else:
        for i in range(nbands):
            band = raster.GetRasterBand(i + 1)
            band.WriteArray(data[i, :, :])
            band.SetNoDataValue(noval)
    band = raster.GetRasterBand(1)
    # Create GridObj
    gridobj = GridObj()
    gridobj.driver = raster
    gridobj.bands = nbands
    gridobj.divername = 'MEM'
    gridobj.geotransform = geotransform
    gridobj.projectionref = proj
    gridobj.xsize = xsize
    gridobj.ysize = ysize
    # Return object
    return(gridobj)  #   create_virtualraster()


# Create dataset from a numpy array
# INPUTS
#  filename        [string] output raster filename
#  data            [np.ndarray] 2 or 3 dimensional numpy array. Number of
#                   bands is equal to data.shape[0]
#  geotransform    [list, tuple, np.ndarray] geo transform list
#                   [origin_x, cellsize_x, rotation_x, origin_y, rotation_y, cellsize_y]
#  proj            [int, string] crs from epsg code or wellknown text
#  noval           [int, float] no data values
#  dtype           [int, float] define type of raster data: int or float
#  driver          [string] driver name of raster file
def array2raster(filename, data, geotransform=[0, 100, 0, 0, 0, -100],
                 proj=4326, noval=-99999, dtype=float, driver='SAGA'):
    # Raster file name
    drivers = gdal_driver_list()  # get driver list
    if not drivers.has_key(driver):
        raise TypeError('Wrong driver name {}'.format(driver))

    # Check filename
    ext = '.' + drivers[driver]
    if not filename.endswith(ext):
        filename = _os.path.splitext(filename)[0]
        filename += ext

    # Get data type
    if dtype == float:
        dtype = _gdal.GDT_Float32
    elif dtype == int:
        dtype = _gdal.GDT_Int32
    else:  # default data type
        dtype = _gdal.GDT_Float32

    # Get bands number
    if _np.ndim(data) == 2:
        nbands = 1
        ysize, xsize = _np.shape(data)
    elif _np.ndim(data) == 3:
        nbands, ysize, xsize = _np.shape(data)
    else:
        raise TypeError('Wrong data type {}!'.format(str(type(data))))
    nbands, xsize, ysize = int(nbands), int(xsize), int(ysize)

    # Check parameters
    # check geotransform
    if type(geotransform) in [list, tuple, _np.ndarray]:
        if len(geotransform) != 6:
            raise TypeError('Wrong geotransform dimensions!')
    else:
        raise TypeError('Wrong geotransform type {}!'.format(str(type(geotransform))))

    # Get projection
    if type(proj) in [int, float]:
        proj = _crs_from_epsg(int(proj))
    elif type(proj) != str:
        raise TypeError('Wrong proj type {}!'.format(str(type(proj))))

    # Write raster data
    driver = _gdal.GetDriverByName(driver)
    rasterout = driver.Create(filename, xsize, ysize, nbands, dtype)
    rasterout.SetGeoTransform(geotransform)
    rasterout.SetProjection(proj)
    if nbands == 1:
        band = rasterout.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(noval)
    else:
        for i in range(nbands):
            band = rasterout.GetRasterBand(i + 1)
            band.WriteArray(data[i, :, :])
            band.SetNoDataValue(noval)

    # Save to disk
    rasterout.FlushCache()
    # close raster
    rasterout = None
    band = None
    driver = None


# Create or read a grid system
# INPUTS
#  grid      [string] grid system file (.sgrd). By default grid is None.
#             When grid is None an empty grid system is created
# OUTPUTS
#  gridsys   [dict] grid system
def grid_system(grid=None):
    if grid is None:  # deifine an empty gridsystem
        gridsys = _OrderedDict()  # empty orderdict
        gridsys['NAME'] = 'New'
        gridsys['DESCRIPTION'] = ''
        gridsys['UNIT'] = ''
        gridsys['DATAFORMAT'] = float
        gridsys['DATAFILE_OFFSET'] = 0
        gridsys['BYTEORDER_BIG'] = False
        gridsys['TOPTOBOTTOM'] = False
        gridsys['POSITION_XMIN'] = 0.0
        gridsys['POSITION_YMIN'] = 0.0
        gridsys['CELLCOUNT_X'] = 10
        gridsys['CELLCOUNT_Y'] = 10
        gridsys['CELLSIZE'] = 10.0
        gridsys['Z_FACTOR'] = 1.0
        gridsys['Z_OFFSET'] = 0
        gridsys['NODATA_VALUE'] = -99999.0

    else:  # Grid system from file
        # Check input
        grid = _files.default_file_ext(grid, 'grid')

        # check if grid exist
        if not _os.path.exists(grid):
            raise IOError('Grid does not exist!')

        # Open File
        gridsys = _OrderedDict()
        with open(grid, 'r') as fid:
            while True:
                l = fid.readline()
                if not bool(l):  # End of file
                    break
                row = l.replace('\t', '').replace('\n', '').split('=')
                value = row[1].strip()
                gridsys[row[0].strip()] = value

        # Convert some values
        # BYTEORDER_BIG key
        if gridsys.has_key('BYTEORDER_BIG'):
            if gridsys['BYTEORDER_BIG'].lower() == 'true':
               gridsys['BYTEORDER_BIG'] = True
            elif gridsys['BYTEORDER_BIG'].lower() == 'false':
               gridsys['BYTEORDER_BIG'] = False

        # TOPTOBOTTOM key
        if gridsys.has_key('TOPTOBOTTOM'):
            if gridsys['TOPTOBOTTOM'].lower() == 'true':
               gridsys['TOPTOBOTTOM'] = True
            elif gridsys['TOPTOBOTTOM'].lower() == 'false':
               gridsys['TOPTOBOTTOM'] = False

        # DATAFORMAT key
        if gridsys.has_key('DATAFORMAT'):
            if gridsys['DATAFORMAT'].upper() == 'FLOAT':
               gridsys['DATAFORMAT'] = float
            elif gridsys['DATAFORMAT'].upper() == 'BYTE_UNSIGNED':
               gridsys['DATAFORMAT'] = int

        # Integer values
        keys = ['DATAFILE_OFFSET', 'CELLCOUNT_X',
                'CELLCOUNT_Y', 'Z_OFFSET']
        for key in keys:
            if gridsys.has_key(key):
                gridsys[key] = int(gridsys[key])

        # Float values
        keys = ['POSITION_XMIN', 'POSITION_YMIN', 'CELLSIZE',
                'Z_FACTOR', 'NODATA_VALUE']
        for key in keys:
            if gridsys.has_key(key):
                gridsys[key] = float(gridsys[key])

    return(gridsys)  # grid_sys()


# Get grid extent
# INPUTS
#  grid       [str] grid system file name (.sgrd)
# OUTPUTS
#  extent     [np.ndarray] grid extent [xmin, xmax, ymin, ymax]
def get_grid_extent(grid):
    if type(grid) is _OrderedDict:  # grid is a gridsystem
        gs = _deepcopy(grid)
    else:
        # Check inputs
        grid = _files.default_file_ext(grid, 'grid')
        # Get grid system and values
        gs = grid_system(grid)
    xmin = gs['POSITION_XMIN']
    ymin = gs['POSITION_YMIN']
    dxy = gs['CELLSIZE']
    nx = gs['CELLCOUNT_X']
    ny = gs['CELLCOUNT_Y']
    # Create extent
    extent = _np.array([xmin, xmin + dxy * (nx - 1), ymin, ymin + dxy * (ny - 1)])
    return(extent)  # get_grid_extent()


# Get rows and cols of pixels with values in an array
# INPUTS
#  data    [np.ndarray, GridObj] input array or GridObj
# OUTPUTS
#  pixels  [np.ndarray] output array with [[row1, col1], [row2, col2],...]
def cell_with_value(data):
    if type(data) is GridObj:
        data = data.get_data()
    elif type(data) is not _np.ndarray:
        raise TypeError('Bad pixels parameter type {}'.format(str(type(data))))

    # Get rows and cols with data
    rows, cols = _np.where(_np.isnan(data) == False)
    pixels = _np.array(zip(rows, cols), dtype=int)
    return(pixels)  # cell_with_value()

