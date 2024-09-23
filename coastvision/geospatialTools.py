"""
This script contains helper geospatial functions for CoastVision. Some of the code used was addapted from CoastSat_ps.

Author: Joel Nicolow, Climate Resiliance Collaborative, School of Ocean and Earth Science and Technology (August, 03 2022)

"""

import os
import rasterio
import numpy as np
from osgeo import gdal, osr
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
from coastvision import supportingFunctions
import geojson


def convert_coor_arr_epsg(coordArr, epsgOld, epsgNew):
    """
    This function takes in a 2d array of coordinates in a given epsg and convert it to the epsg that it should be converted to and converts it

    :param coordArr: 2d array [[x1, y1], [x2, y2], ...]
    :param epsgOld: the current epsg
    :param epsgNew: int the epsg that the coordinates will be transformed to
    """
    nameList = []
    PointList = []
    for i in range(0, len(coordArr)):
        point = Point(coordArr[i][0], coordArr[i][1])
        PointList.append(point)
        nameList.append(i)


    d = {'name': nameList, 'geometry': PointList}
    gdf = geopandas.GeoDataFrame(d, crs=int(epsgOld)) # this EPSG is for lon/lat in degrees
    gdf = gdf.to_crs(int(epsgNew)) # just incase a string or float was passed we cast to an int
    retuArray = []
    for row in range(0, gdf.shape[0], 1):
        x = gdf.loc[row, 'geometry'].x
        y = gdf.loc[row, 'geometry'].y
        retuArray.append(np.array([x, y])) # notice adding it as an np array
    return(np.array(retuArray)) # also needs to be a np array


import geopandas
from shapely.geometry import Point
def convert_lonlat_coord_arr_to_WGS84(coordArr, epsgNew):
    """
    This function takes in a 2d array of coordinates (longitude and lattitude in degrees EPSG 4326) and the epsg that it should be converted to and converts it

    :param coordArr: 2d array [[longitude1, lattitude1], [long2, lat2], ...]
    :param epsgNew: int the epsg that the coordinates will be transformed to

    :return: 2d array same format as input array but now transformed into the provided EPSG
    """
    return convert_coor_arr_epsg(coordArr, 4326, epsgNew)


def get_im_ms(fn):
    # Generate im_ms    
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im_ms = np.stack(bands, 2)
    return im_ms


def get_georef(tif_im_path):
    """
    returns georef for specified tif image
    """
    data = gdal.Open(tif_im_path, gdal.GA_ReadOnly)
    return(np.array(data.GetGeoTransform()))

import rasterio
def get_epsg(tiffFilePath):
    """
    """
    data = rasterio.open(tiffFilePath, 'r')
    epsgStr = str(data.crs)
    return epsgStr.split(':')[-1]



def get_udm1_mask_from_udm2(fn):
    """
    This function gets the boolean array of clear pixels and cloudy/no pixels (black) from the UDM 1 band of the UDM2 file

    :param fn: fn for udm2 file
    :return: UDM 1 band array
    """
    with rasterio.open(fn) as src:
        fullUdm2 = src.read()
    udmBand8 = fullUdm2[7,:,:] # 0=clear img, 1=space around AOI, 2=cloud
    boolCloudMask = udmBand8 != 0 # this is because the cloud mask is true for where there are clouds
    return(boolCloudMask)


def get_cloud_mask_from_udm2_band8(fn):
    """
    The eigth band of the udm has 0=clear img, 1=space around satalite img, 2=cloud
    We are converting this to a boolean for if its is clear or not.
    NOTE: This gets the cloud mask band from the UDM2 file this does NOT show the part of the AOI that has data/dont have data

    :param fn: fn for udm2 file
    """
    with rasterio.open(fn) as src:
        fullUdm2 = src.read()
    udmBand8 = fullUdm2[7,:,:] # 0=clear img, 1=space around AOI, 2=cloud
    boolCloudMask = udmBand8 != 0 # this is because the cloud mask is true for where there are clouds
    return(boolCloudMask)


def get_ps_and_mask(toa_path):
    """
    gets both the multispectral image (4-band im_ms) and the udm union of space around image and cloud mask

    :param toa_path: file path to tif image
    :return: im_ms, im_mask
    """
    # get filepath of udm
    print(toa_path)
    file_name = os.path.basename(toa_path) 
    print(file_name)
    file_dir = os.path.dirname(toa_path)
    print(file_dir)
    udm2_file_name = file_name.replace("AnalyticMS_toar_clip.tif", "udm2_clip.tif")
    udm2_file_path = os.path.join(file_dir, udm2_file_name)
    print(udm2_file_path)
    return get_im_ms(toa_path), get_cloud_mask_from_udm2_band8(udm2_file_path) # im_ms, im_mask



def get_ps_no_mask(toa_path):
    
    # Generate im_ms
    im_ms = get_im_ms(toa_path)
    
    # Create fake im_mask
    with rasterio.open(toa_path, 'r') as src1:
        im_band = src1.read(1)
    im_mask = np.zeros((im_band.shape[0], im_band.shape[1]), dtype = int)
    
    # for each band, find zero values and add to nan_mask
    for i in [1,2,3,4]:
        # open band
        with rasterio.open(toa_path, 'r') as src1:
            im_band = src1.read(i)
            
        # boolean of band pixels with zero value
        im_zero = im_band == 0
        
        # update mask_out
        im_mask += im_zero
    # Convert mask_out to boolean
    # im_mask = im_mask>0
    im_mask = im_mask<-1# TEMP TEMP (this makes it so there is no clouds marked in the cloud mask)
    
    return im_ms, im_mask

def pad_misscropped_image(referenceImagePath, toaPath, im_ms, im_mask=None):
    """
    makes sure the im_ms from toaPath is fromatted the same as the reference image and if not it padds it
    """
    with rasterio.open(referenceImagePath, 'r') as src: 
        refmeta = src.meta
        refbounds = src.bounds
    with rasterio.open(toaPath, 'r') as src: 
        meta = src.meta
        bounds = src.bounds

    if refmeta['height'] != meta['height'] or refmeta['width'] != meta['width']:
        heightDif = refmeta['height'] - meta['height']
        widthDif = refmeta['width'] - meta['width']
        offsets = {}
        offsets['left'] = abs(refbounds.left - bounds.left)
        offsets['bottom'] = abs(refbounds.bottom - bounds.bottom)
        offsets['right'] = abs(refbounds.right - bounds.right)
        offsets['top'] = abs(refbounds.top - bounds.top)
        totalHorizantalOffset = offsets['left'] + offsets['right']
        totalVerticalOffset = offsets['bottom'] + offsets['top']
        left = 0
        bottom=0
        right = 0
        top =0
        for key, offset in offsets.items():
            if offset != 0:
                if 'left' == key: 
                    percentOffset = offset / totalHorizantalOffset
                    left = int(percentOffset*widthDif)
                elif 'bottom' == key: 
                    percentOffset = offset / totalVerticalOffset
                    bottom = int(percentOffset*heightDif)
                elif 'right' == key: 
                    percentOffset = offset / totalHorizantalOffset
                    right = int(percentOffset*widthDif)
                else: 
                    percentOffset = offset / totalVerticalOffset
                    top = int(percentOffset*heightDif)
            
        padding = ((top, bottom), (left, right), (0, 0)) # have to say no need padd 3rd dimension (0, 0)
        padded_array = np.pad(im_ms, padding, mode='constant', constant_values=0)
        retu = padded_array
        if not im_mask is None: 
            padded_mask = np.pad(im_mask, padding[0:2], mode='constant', constant_values=0) # only 2d image so no need 
            retu = (retu, padded_mask)
        return(retu)
    else:
        # no need padd image its not misscropped
        retu = im_ms
        if not im_mask is None: retu = (retu, im_mask)
        return(retu)

# convert contours to world coord
import skimage.transform as transform
def convert_pix2world(points, georef):
    """
    Converts pixel coordinates (pixel row and column) to world projected 
    coordinates performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates, first columns with X and second column with Y
        
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            tmp = arr[:,[1,0]]
            points_converted.append(tform(tmp))
          
    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:,[1,0]]
        points_converted = tform(tmp)
        
    else:
        raise Exception('invalid input type')
        
    return points_converted


def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates 
    (pixel row and column) performing an affine transformation.
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates (pixel row and column)
    
    """
    
    # make affine transformation matrix
    aff_mat = np.array([[georef[1], georef[2], georef[0]],
                       [georef[4], georef[5], georef[3]],
                       [0, 0, 1]])
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)
    
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(tform.inverse(points))
            
    # if single array    
    elif type(points) is np.ndarray:
        #print(points)
        points_converted = tform.inverse(points)
        
    else:
        print('invalid input type')
        print(points)
        raise
    
    if isinstance(points_converted, list):
        points_converted = points_converted[0] # sometimes it returns a list with multipole copies of the coordinates
    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes
    
    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be            
                
    Returns:    
    -----------
    points_converted: np.array or list of np.array 
        converted coordinates from epsg_in to epsg_out
        
    """
    
    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points): 
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))  
    else:
        raise Exception('invalid input type')

    return points_converted


def save_geojson_linestring(coords, epsg, savePath, sitename="unknown"):
    """
    This function saves a linestring in a geojson file

    :param coords: array of coordinates [[x1, y1], [x2, y2],...]
    :param epsg: String epsg for linestring
    :param savePath: where to save the file
    :param sitename: sitename/id used in geojson (default="unknown")

    written by CRI
    """
    geojsonDict = {
                    "type": "FeatureCollection",
                    "name": sitename,
                    "crs": {
                        "type": "name",
                        "properties": {
                            "name": f"urn:ogc:def:crs:EPSG::{epsg}"
                        }
                    },
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                    "type": "LineString",
                                    "coordinates": coords,
                            }#,
                            # "properties": {
                            #     "id": sitename,
                            #     "easting": 0,
                            #     "northing": 0
                            # }
                        }
                    ]
                }
    with open(savePath,'w') as f:
        geojson.dump(geojsonDict, f)
    

def get_epsg_from_geojson(geojsonPath):
    """
    This function gets the epsg from a geojson file. Semi contingent on format
    
    :param geojsonPath: path to geojson file
    
    :return: string epsg (it does not return the epsg as an int)
    """
    with open(geojsonPath,'r') as f:
        data = geojson.load(f)
    if not 'crs' in data.keys():
        return(None)
    epsgString = data['crs']['properties']['name']
    return(epsgString.split('::')[-1])


def get_coordinates_from_geojson(filePath):
    with open(filePath,'rb') as f:
        data = geojson.load(f)
    return np.array(data["features"][0]['geometry']['coordinates']) # down stream code requieres np array format
