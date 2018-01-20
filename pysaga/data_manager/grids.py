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
        """
        Create a Grid Object to work as a virtual raster
        When a file is input, the raster is read as a gdal

        ATTRIBUTES:
            filename       [string] file name associated to the data.
                             If None, the GridObj is a virtual raster
            drivername     [string] gdal driver short name for rasters
            geotransform   [list] geo transformation elements
                             [xoff, xsize, rtnx, yoff, rtny, ysize]
                             where xoff and yoff are the coordinates of the
                             upper-left corner, xsize and y ysize are the pixel
                             size, and rtnx and rtny are the rotation
            projectionref  [string] reference projection as WKt text
            xsize          [int] number of columns
            ysize          [int] number of rows
            bands          [int] number of raster bands
            driver         [gdal.Dataset] gdal raster class, that can be used
                            for access to raster properties
        """

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

    def __getitem__(self, key):
        return(self.__dict__.get(key, None))

    # Text special function
    def __str__(self):
        text = 'Grid Object'
        for key, value in self.__dict__.iteritems():
            text += '\n' + key + ':'
            text += '\n' + str(value)
        return(text)

    def __repr__(self):
        return('GridObj()')

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

    def read_file(self, filename):
        """
        Connection with a raster file using gdal driver

        INPUTS:
          filename      [string] valid raster file compatible with gdal
        """
        # Check grids file
        if filename.endswith('.sgrd'):  # change sgrd to sdat
            filename = _files.default_file_ext(filename, 'sdat', True)

        # Read raster with gdal
        raster = _gdal.Open(filename, _gdal.GA_Update)
        if raster is None:
            raise IOError('{} can not be read!'.format(filename))

        # save properties
        self.driver = raster  # gdal dataset
        self.filename = filename  # save filename
        rinfo = self.get_grid_info()  # get raster properties
        self.__dict__.update(rinfo)  # update values
        del(raster)
        # read_raster()

    def close(self):
        """
        Close connection with actual raster file. It must be done for
        avoid conflicts with raster file
        """
        if type(self.driver) is _gdal.Dataset:
            self._reset_attributes()

    def get_grid_info(self):
        """
        Use gdal library for extract raster information as a dictionary
        OUTPUT
         rinfo     [dict] raster information dictionary
        """
        # Check grid connection
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get raster properties
        geotrans = _deepcopy(self.driver.GetGeoTransform())  # get transformation
        projref = self.driver.GetProjectionRef()  # get projection
        drivername = self.driver.GetDriver().ShortName  # get file driver
        xsize = self.driver.RasterXSize  # get x size
        ysize = self.driver.RasterYSize  # get y size
        bands = self.driver.RasterCount  # count number of layers
        # Output data
        rinfo = {'geotransform': geotrans,
                 'projectionref': projref,
                 'drivername': drivername,
                 'xsize': xsize,
                 'ysize': ysize,
                 'bands': bands}
        return(rinfo)  # grid_info()

    def get_coordinates(self, extent=None, asindex=False, center=True):
        """
        Gets grid coordinates as arrays of whole data or from a selected extension

        INPUTS
         extent    [list, tuple, np.ndarray] optional extension [xmin, xmax, ymin, ymax]
         asindex   [bool] if True, extension is used as [xoff, cols, yoff, rows] where
                    the xoff and yoff are the indexes of the upper left pixel, and
                    rows and cols are the number of cells that will be considered
                    moving to the down-right direction
         center    [bool] if True, centered pixel coordinates are returned, in other case
                    upper left corner is returned
        OUTPUT
         X, Y        [np.array] X and Y coordinate matrix
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Get geo transformation
        gt = self.geotransform

        # Define get coordinates sub-function
        def get_coors_matrix(rows, cols):
            # Get coordinates matrix
            X = gt[0] + cols * gt[1] + rows * gt[2]
            Y = gt[3] + cols * gt[4] + rows * gt[5]
            if center:
                X += gt[1] / 2.0
                Y += gt[5] / 2.0
            return(X, Y)

        # Extract by extent or by pixels
        if extent is None:
            cols, rows = _np.meshgrid(_np.arange(self.xsize),
                                      _np.arange(self.ysize))
            x, y = get_coors_matrix(rows, cols)  # get coordinates
            return(x, y)

        elif type(extent) in [list, tuple, _np.ndarray]:  # use extent
            if len(extent) != 4:
                raise TypeError('extent must be a 4-elements array!')

            if asindex:  # use [xoff, cols, yoff, rows]
                cid = _np.array([extent[0], sum(extent[:2]) + 1],
                                dtype=int)
                rid = _np.array([extent[2], sum(extent[2:]) + 1],
                                dtype=int)
            else:
                # convert extent to rows and cols
                points = [[extent[0], extent[2]],
                          [extent[1], extent[3]]]
                pixels = self.coor2pixel(points)
                # convert pixel to extent
                rid, cid = pixels[:, 0], pixels[:, 1]

            # Get pixel indexes
            cols, rows = _np.meshgrid(_np.arange(cid.min(),
                                                 cid.max() + 1,
                                                 dtype=int),
                                      _np.arange(rid.min(),
                                                 rid.min() + 1,
                                                 dtype=int))
            # Get coordinates
            x, y = get_coors_matrix(rows, cols)
            return(x, y)  # get_grid_coordinates()

    def get_extent(self, center=True):
        """
        Gets the raster extent as [xmin, xmax, ymin, ymax]
        INPUTS
          center    [bool] if True, pixel centroids are used, in other case
                     upper left corner is used. By default True
        OUTPUTS
         extent     [np.ndarray] extent
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get extent
        cols = _np.array([0., self.xsize - 1.])
        rows = _np.array([0., self.ysize - 1.])
        gt = self.geotransform  # get geotransform
        # get coordinates
        x = gt[0] + cols * gt[1] + rows * gt[2]
        y = gt[3] + cols * gt[4] + rows * gt[5]
        if center:
            x += gt[1] / 2.0
            y += gt[5] / 2.0
        # Output extent
        extent = _np.array([x.min(), x.max(), y.min(), y.max()])
        return(extent)  # get_extent()

    def get_resolution(self):
        """
        Returns pixel width and height (as absolute values)

        OUTPUTS
         width, height    [float] pixel size
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get geotransform
        gt = _deepcopy(self.geotransform)
        width, height = gt[1], abs(gt[5])
        return(width, height)

    def get_origin(self):
        """
        Returns raster upper-left corner coordinates

        OUTPUT
         origin     [np.ndarray] [x_left, y_upper] array
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')
        # Get geotransform
        gt = self.geotransform
        origin = (gt[0], gt[3])
        return(origin)

    def get_nodata(self, bands=1):
        """
        Get no data values from selected bands

        INPUTS
         bands     [int, list] band number or list of band numbers
        OUTPUTS
         nodata     If bands is an integer, nodata is an int or float
                    If bands is a list, nodata is a numpy array
        """
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
            if band_data is None:
                data.append(None)
            else:
                data.append(band_data.GetNoDataValue())
            band_data = None
        # Output data
        if len(data) == 1:
            nodata = data[0]
        else:
            nodata = _np.array(data)
        return(nodata)  # get_nodata()

    def pixel2coor(self, pixels, center=True, remove=False):
        """
        Extract coordinates x,y given the pixels rows and cols indexes

        INPUTS
         pixels     [list, tuple, np.ndarray] array with [row, col] indexes.
                      For multiple pixels use [[row1, col1], [row2, col2],...]
        center      [bool] if center is True, pixel centroid is input as pixels,
                      in other case, upper-left corner is used
        remove      [bool] if True, pixels out of the raster size are ignored
                     In other case, pixels are forced to no exceed the xsize and ysize
        OUTPUT
         coors      [np.ndarray] output coordinate arrays [[x1, y1], [x2, y2]]

         NOTE: note that input is row, col and the output is x, y
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check pixels
        if type(pixels) in [list, tuple, _np.ndarray]:
            pixels = _np.array(pixels, dtype=_np.int32)
            if pixels.ndim == 1 and pixels.shape[-1] == 2:
                pixels = pixels.reshape(1, 2)
            elif pixels.ndim > 2 or pixels.shape[-1] != 2:
                raise TypeError('Bad pixels dimensions {}'.
                                format(str(pixels.shape)))
        else:
            raise TypeError('Wrong pixels parameter type {}'.
                            format(str(type(pixels))))

        # Get coordinates
        rows, cols = pixels[:, 0], pixels[:, 1]
        if remove:
            pos = _np.where((rows < 0) | (rows > (self.ysize - 1)) |
                            (cols < 0) | (cols > (self.xsize - 1)))[0]
            rows = _np.delete(rows, pos)
            cols = _np.delete(cols, pos)
            print('Some points where deleted because their were found out of raster size!')
        else:
            rows[rows < 0] = 0
            rows[rows > (self.ysize - 1)] = self.ysize - 1
            rows[cols < 0] = 0
            rows[cols > (self.xsize - 1)] = self.xsize - 1

        gt = self.geotransform  # get geotransform
        x = gt[0] + cols * gt[1] + rows * gt[2]
        y = gt[3] + cols * gt[4] + rows * gt[5]
        if center:  # get center coordinates
            x += gt[1] / 2.0
            y += gt[5] / 2.0
        # Return object
        coors = _np.hstack((x.reshape((len(x), 1)), y.reshape(len(y), 1)))
        return(coors)  # pixel2coor()

    def coor2pixel(self, points, center=True, remove=False):
        """
        Convert coordinates X,Y from a pixel to row and col indexes

        INPUTS
         points      [list, tuple, np.ndarray] [x, y] coordinates.
                      For multiple points use [[x1, y1], [x2, y2], ...]
         center      [bool] if True, input points corresponds to the pixel centroid
                      In other case, points corresponds to the upper-left corner
         remove      [bool] if True, pixels out of the raster size are ignored
                      In other case, pixels are forced to no exceed the xsize and ysize
        OUTPUTS
         pixels   [np.ndarray] rows and cols of coordinates [[row1, col1], [row2, col2]]

         NOTE: note that input is x, y and the output is row, col
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check pixels
        if type(points) in [list, tuple, _np.ndarray]:
            if type(points) is _np.ndarray:
                points = points.astype(_np.float32)
            else:
                points = _np.array(points, dtype=_np.float32)
            if points.ndim == 1 and points.shape[-1] == 2:
                points = points.reshape(1, 2)
            elif points.ndim > 2 or points.shape[-1] != 2:
                raise TypeError('Bad points dimenssions {}'.format(str(points.shape)))
        else:
            raise TypeError('Wrong points parameter type {}'.format(str(type(points))))
        # Get raster extent and geotransformation
        origin = self.get_origin()  # raster origin
        width, height = self.get_resolution()  # resolution
        # Get rows and cols
        if center:
            points[:, 0] -= width / 2.0
            points[:, 1] += height / 2.0
        
        print(points)
        
        cols = _np.array(_np.round((points[:, 0] - origin[0]) / width), dtype=int)
        rows = _np.array(_np.round((origin[1] - points[:, 1]) / height), dtype=int)
        
        if remove:
            pos = _np.where((rows < 0) | (rows > (self.ysize - 1)) |
                            (cols < 0) | (cols > (self.xsize - 1)))[0]
            rows = _np.delete(rows, pos)
            cols = _np.delete(cols, pos)
            print('Some points where deleted because their were found out of raster size!')
        else:
            rows[rows < 0] = 0
            rows[rows > (self.ysize - 1)] = self.ysize - 1
            cols[cols < 0] = 0
            cols[cols > (self.xsize - 1)] = self.xsize - 1
        
        # Return pixels
        pixels = _np.array(zip(rows, cols), dtype=int)
        return(pixels)  # coor2pixel()

    def get_pixel_value(self, pixels, bands=1, dtype=float):
        """
        Get pixel values using rows and cols

        INPUTS
         pixels    [list, tuple, np.ndarray] [row, col] pixel. For multiple pixels use
                    [[row1, col1], [row2, col2]]
         bands     [int, list] band number or list of band numbers
         dtype     [int, float, np.int, np.float] data type
        OUTPUTS
         data     If one pixel of a single band is called, data is a int or float value
                  If multiple pixels are called of a single band, data is a pandas DataFrame
                   with pixel number as index
                  If multiple pixels of multiple bands are called, data is a pandas DataFrame
                   where row index is the band number (starting from 1) and columns index as
                   pixel number
        """
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
                raise TypeError('Bad pixels dimension {}'.format(str(pixels.ndim)))
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

    def get_data(self, bands=1, extent=None, asindex=False, dtype=float):
        """
        Get raster values as arrays from one bands or multiple bands

        INPUTS
         bands     [int, list] band number or list of band numbers
         extent    [list] grid extent with cells number or coordinates [xmin, xmax, ymin, ymax]
         asindex   [bool] if True, extension is used as [xoff, cols, yoff, rows] where
                    the xoff and yoff are the indexes of the upper left pixel, and
                    rows and cols are the number of cells that will be considered
                    moving to the down-right direction
         dtype     [type] data type. By default float
        OUTPUT
         data      [np.array] output numpy array. If bands is a list, output data is
                    a 3 dimensional array
        """
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

        elif type(extent) in (list, tuple, _np.ndarray):
            if len(extent) != 4:
                raise TypeError('Bad extent type {}!'.format(str(type(extent))))

            if asindex:  # use [xoff, cols, yoff, rows]
                extent = _np.array(extent, dtype=int)
                xoff, xsize, yoff, ysize = extent

            else:       # use [xmin, xmax, ymin, ymax]
                # convert extent to rows and cols
                points = [[extent[0], extent[2]],
                          [extent[1], extent[3]]]
                pixels = self.coor2pixel(points)
                # Get a range of pixels
                xoff = int(pixels[:, 1].min())
                yoff = int(pixels[:, 0].min())
                xsize = int(pixels[:, 1].max() - xoff + 1)
                ysize = int(pixels[:, 0].max() - yoff + 1)

        else:
            raise TypeError('Bad extent type {}!'.format(str(type(extent))))
        # Get raster data
        bands = _np.unique(_np.sort(bands))
        data = _np.full((max(bands), ysize, xsize), _np.nan, dtype=dtype)

        for i in range(len(bands)):
            # verify raster band
            assert 1 <= bands[i] <= self.bands
            # get raster band
            band = self.driver.GetRasterBand(bands[i])
            if band is None:  # band is empty
                continue
            values = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(dtype)
            nodata = band.GetNoDataValue()  # no data values
            values[values == nodata] = _np.nan  # change nan values
            # save data
            data[i, :, :] = values
        # Output array
        if len(bands) == 1:
            return(data[0, :, :])
        else:
            return(data)
        # get_data()

    def set_data(self, data, band=1, row=0, col=0):
        """
        Change band values given an array
        If GridObj is connected with a raster file, file is overwrited

        INPUTS
         data      [np.ndarray] data array
         band      [int] band number
         row, col  [int] upper left pixel row and col to set new data
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check data
        row, col, band = int(row), int(col), int(band)
        if type(data) is not _np.ndarray:
            data = _np.array(data)
        # check dimensions
        if data.ndim != 2:
            raise TypeError('Parameter data must be a bidimenssional numpy array.')
        # check extent
        nr, nc = data.shape
        assert 1 <= band <= self.bands, 'Wrong band number'
        assert 0 <= row <= self.ysize - nr, 'row is out of the array dimension'
        assert 0 <= col <= self.xsize - nc, 'col is out of the array dimension'

        # Write data in band
        band_data = self.driver.GetRasterBand(band)  # get band
        band_data.WriteArray(data, col, row)  # write data
        # Try to flush data to disk
        if self.drivername not in ['MEM', 'VRT']:
            band_data.FlushCache()
        band_data = None  # close
        # set_data()

    def set_nodatavalue(self, nodata=-99999):
        """
        Change no data values for all bands

        INPUTS
         nodata     [int, float] new no data value
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Change no data value of all bands
        for i in range(self.bands):
            band_data = self.driver.GetRasterBand(i + 1)  # get band
            if band_data is None:
                continue
            band_data.SetNoDataValue(nodata)  # change no data value

            # Try to flush data to disk
            if self.drivername not in ['MEM', 'VRT']:
                band_data.FlushCache()
            band_data = None
        # set_nodatavalue()

    def to_file(self, filename=None, driver='SAGA'):
        """
        Write GridObj in a raster file
        
        INPUTS
         filename      [string] output raster file name
         driver        [string] gdal driver name. SAGA as default
        """
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
        if driver not in drivers:
            raise TypeError('Wrong driver name {}'.format(driver))
        ext = drivers[driver]
        filename = _validation.output_file(filename, ext)

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

    def to_memory(self):
        """
        Create a virtual copy of the actual GridObj
        
        OUTPUTS
         newobj       [GridObj] virtual GridObj
        """
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

    def copy_resolution(self, refobj, resampling=3):
        """
        Copy resolution and projection from other GridObj
        Results are saved in a virtual raster
        
        INPUTS
         refobj        [instance] reference GridObj
         resampling    [int] resampling method
                        [0] Nearest Neighbour
                        [1] Averange
                        [2] Bilinear
                        [3] Cubic (default)
                        [4] Cubic Spline
        OUTPUTS
         newobj       [GridObj] resampled and reprojected new GridObj
        """
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

    def resample(self, cellsize=0, extent=None, ncols=0, nrows=0, resampling=3):
        """
        Resample raster and save in a virtual raster

        INPUTS
         cellsize      [int] new cellsize. If cellsize is 0, cellsize is ignored
         extent        [list, tuple, np.ndarray] new raster extent [xmin, xmax, ymin, ymax].
                        If extent is None, actual extent is used
         ncols, nrows  [int] new number of cols and rows. It can be used instead of cellsize parameter.
                        If ncols or nrows is 0, then they are ignored
         resampling    [int] resampling method
                        [0] Nearest Neighbour
                        [1] Average
                        [2] Bilinear
                        [3] Cubic (default)
                        [4] Cubic Spline
        OUTPUTS
         newobj       [GridObj] resampled virtual GridObj
        """
        if type(self.driver) is not _gdal.Dataset:  # it is a valid gdal dataset?
            raise TypeError('You must connect with a raster file!')

        # Check inputs
        if type(extent) in [list, tuple, _np.ndarray]:
            if len(extent) != 4:
                raise TypeError('Bad extent length {}'.format(len(extent)))

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

        # Create new geotransform
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

    def reproject(self, proj, resampling=3, threshold=0.125):
        """
        Reproject raster and save in a virtual raster
        
        INPUTS
         proj          [int, str] proj can be a EPSG integer code or
                        a Well Known string projection
         resampling    [int] resampling method
                        [0] Nearest Neighbour
                        [1] Averange
                        [2] Bilinear
                        [3] Cubic (default)
                        [4] Cubic Spline
        OUTPUTS
         newobj       [GridObj] reprojected virtual GridObj
        """
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

    def is_compatible(self, refobj):
        """
        Check if GridObj is compatible for direct map algebra
        
        with a reference GridObj
        INPUTS
         refobj       [GridObj] reference GridObj
        OUTPUTS
         flag         [bool] flag is True if GridObj is compatible
                       with reference GridObj
        """
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


# ==============================================================================
# Grid Object functions
# ==============================================================================


def gdal_driver_list():
    """
    Get _gdal drivers dictionary

    OUTPUT
     drivers    [dict] drivers and file extension
    """
    drivers = {}
    for i in range(_gdal.GetDriverCount()):
        drv = _gdal.GetDriver(i)
        if drv.GetMetadataItem(_gdal.DCAP_RASTER):
            drivers[drv.ShortName] = drv.GetMetadataItem(_gdal.DMD_EXTENSIONS)
    return(drivers)  # gdal_driver_list()


def create_virtualraster(data, geotransform=None, proj=4326,
                         noval=-99999, dtype=float):
    """
    Create virtual layer from a numpy array

    INPUTS
     data            [np.ndarray] input data. If data is a bi-dimensional array
                      only one band is created. For tree-dimensional array, number of bands
                      is extracted from the first dimension
     geotransform    [list, tuple, np.ndarray] geo transformation array
                      [x_left, width, x_rotation, y_upper, y_rotation, height]
                      If geotransform is None, default value is set as [0, 100, 0, 0, 0, -100]
     proj            [int, str] projection reference. If proj is an integer, proj is
                      used as EPSG code. If proj is string, WKT string projection is used
     noval           [int, float] no data value
     dtype           [type] data type (int, float, np.types)
    OUTPUTS
     gridobj         Memory GridObj
    """
    # Check geotransform
    if geotransform is None:
        geotransform = [0, 100, 0, 0, 0, -100]
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
    return(gridobj)  # create_virtualraster()


def array2raster(filename, data, geotransform=None,
                 proj=4326, noval=-99999, dtype=float, driver='SAGA'):
    """
    Create dataset from a numpy array
    INPUTS
     filename        [string] output raster filename
     data            [np.ndarray] 2 or 3 dimensional numpy array. Number of
                      bands is equal to data.shape[0]
     geotransform    [list, tuple, np.ndarray] geo transform list
                      [origin_x, cellsize_x, rotation_x, origin_y, rotation_y, cellsize_y]
                      If geotransform is None, default value is set as [0, 100, 0, 0, 0, -100]
     proj            [int, string] crs from epsg code or wellknown text
     noval           [int, float] no data values
     dtype           [int, float] define type of raster data: int or float
     driver          [string] driver name of raster file
    """
    # Raster file name
    drivers = gdal_driver_list()  # get driver list
    if driver not in drivers:
        raise TypeError('Wrong driver name {}'.format(driver))

    # Check filename
    ext = drivers[driver]
    filename = _validation.output_file(filename, ext)

    # Check geotransform
    if geotransform is None:
        geotransform = [0, 100, 0, 0, 0, -100]

    # Get data type
    if dtype == float:
        dtype = _gdal.GDT_Float32
    elif dtype == int:
        dtype = _gdal.GDT_Int32
    else:  # default data type
        dtype = _gdal.GDT_Float32

    # Get bands number
    if type(data) != _np.ndarray:
        data = _np.array(data)
    if data.ndim == 2:
        nbands = 1
        ysize, xsize = _np.shape(data)
    elif data.ndim == 3:
        nbands, ysize, xsize = _np.shape(data)
    else:
        raise TypeError('Wrong data type {}!'.format(str(type(data))))
    nbands, xsize, ysize = int(nbands), int(xsize), int(ysize)

    # Check parameters
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


def grid_system(grid=None):
    """
    Create an empty grid system or read a grid system from a .sgrd file

    INPUTS
     grid      [string] grid system file (.sgrd). By default grid is None.
                When grid is None an empty grid system is created
    OUTPUTS
     gridsys   [dict] grid system
    """
    if grid is None:  # define an empty gridsystem
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
        grid = _validation.input_file(grid, 'grid', True)

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
        if 'BYTEORDER_BIG' in gridsys:
            if gridsys['BYTEORDER_BIG'].lower() == 'true':
               gridsys['BYTEORDER_BIG'] = True
            elif gridsys['BYTEORDER_BIG'].lower() == 'false':
               gridsys['BYTEORDER_BIG'] = False

        # TOPTOBOTTOM key
        if "TOPTOBOTTOM" in gridsys:
            if gridsys['TOPTOBOTTOM'].lower() == 'true':
               gridsys['TOPTOBOTTOM'] = True
            elif gridsys['TOPTOBOTTOM'].lower() == 'false':
               gridsys['TOPTOBOTTOM'] = False

        # DATAFORMAT key
        if 'DATAFORMAT' in gridsys:
            if gridsys['DATAFORMAT'].upper() == 'FLOAT':
               gridsys['DATAFORMAT'] = float
            elif gridsys['DATAFORMAT'].upper() == 'BYTE_UNSIGNED':
               gridsys['DATAFORMAT'] = int

        # Integer values
        keys = ['DATAFILE_OFFSET', 'CELLCOUNT_X',
                'CELLCOUNT_Y', 'Z_OFFSET']
        for key in keys:
            if key in gridsys:
                gridsys[key] = int(gridsys[key])

        # Float values
        keys = ['POSITION_XMIN', 'POSITION_YMIN', 'CELLSIZE',
                'Z_FACTOR', 'NODATA_VALUE']
        for key in keys:
            if key in gridsys:
                gridsys[key] = float(gridsys[key])
    return(gridsys)  # grid_sys()


def get_grid_extent(grid):
    """
    Get grid extent from a .sgrd file

    INPUTS
     grid       [str] grid system file name (.sgrd)
    OUTPUTS
     extent     [np.ndarray] grid extent [xmin, xmax, ymin, ymax]
    """
    if type(grid) is _OrderedDict:  # grid is a gridsystem
        gs = _deepcopy(grid)
    else:
        # Check inputs
        grid = _validation.input_file(grid, 'grid', True)
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


def cell_with_value(data):
    """
    Get rows and cols of pixels with values in an array. That means that
    pixels with nans are ignored

    INPUTS
     data    [np.ndarray, GridObj] input array or GridObj
    OUTPUTS
     pixels  [np.ndarray (int)] output array with [[row1, col1], [row2, col2],...]
    """
    if type(data) is GridObj:
        data = data.get_data()
    elif type(data) is not _np.ndarray:
        raise TypeError('Bad pixels parameter type {}'.format(str(type(data))))

    # Get rows and cols with data
    rows, cols = _np.where(~_np.isnan(data))
    pixels = _np.array(zip(rows, cols), dtype=int)
    return(pixels)  # cell_with_value()

