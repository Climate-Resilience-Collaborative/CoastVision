"""
This script classifies pixels in images as sand, whitewater, or water. Then the shore line is extracted form this which can be used to get shoreline transect intersection. Some of the code used was addapted from CoastSat_ps.

Author: Joel Nicolow, Anna Mikkelsen Climate Resiliance Collaborative, School of Ocean and Earth Science and Technology (June, 25 2022)

"""


import os
import rasterio
import numpy as np
import numpy.ma as ma
from osgeo import gdal#, osr
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
# ORIGINAL:
from astropy.convolution import convolve
import skimage.morphology as morphology
from scipy.ndimage import gaussian_filter1d
import geojson
from shapely.geometry import Point, shape, LineString
from geopandas import GeoDataFrame
import glob
import math
from skimage import measure
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from coastvision import supportingFunctions
from coastvision import geospatialTools
from coastvision.classifier import pixelclassifier
max_dist=40

#### image classification ####
def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of 
    specified radius. Uses astropy's convolution library'
    
    Copied with permission from CoastSat (KV, 2020) 
        https://github.com/kvos/CoastSat
    
    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation. 
        For example, radius = 1 will produce a 3x3 moving window.
        
    Returns:    
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image
        
    """  
    
    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, 'reflect')
    # window size
    win_rows, win_cols = radius*2 + 1, radius*2 + 1
    # calculate std with uniform filters
    win_mean = convolve(image_padded, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_sqr_mean = convolve(image_padded**2, np.ones((win_rows,win_cols)), boundary='extend',
                        normalize_kernel=True, nan_treatment='interpolate', preserve_nan=True)
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std


def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification. 
    The features include spectral normalized-difference indices and standard 
    deviation of the image for all the bands and indices.

    KV WRL 2018
    
    Modified for PS data by YD 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:    
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool
        
    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)
        
    # NIR-G
    im_NIRG = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    
    # NIR-B
    im_NIRB = nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRB[im_bool],axis=1), axis=-1)
    
    # NIR-R
    im_NIRR = nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)

    # NOTE: ADD NDVI (veg index)
    #NOTE: add MNDWI
        
    # B-R
    im_BR = nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)
    
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std =  image_std(im_ms[:,:,k], 2)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
        
    # calculate standard deviation of the spectral indices
    im_std = image_std(im_NIRG, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_NIRB, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_NIRR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = image_std(im_BR, 2)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    return features

def classify_image(im_ms, clf, im_mask=None, nonbinary=False):
    """
    This returns a binary image classifier (land and water)
    :return: binary mask of image shape
    """
    # featureArr = pixelclassifier.calculate_features(im_ms, None, None, None)
    featureArr = pixelclassifier.calculate_features(im_ms, booleanMask=None, stdRadius=2)

    # print(featureArr.shape)
    featureArr2D = featureArr.reshape((im_ms.shape[0]*im_ms.shape[1]), featureArr.shape[-1])  # Reshape to (width*height, 17)
    featureArr2D[np.isnan(featureArr2D)] = 0  #make to 0 or 255
    predictions = clf.predict(featureArr2D)
    predictions2D = np.reshape(predictions, (featureArr.shape[0], featureArr.shape[1]))
    predictionsBinary = predictions2D.astype(np.uint8)
    #predictionsBinary[~im_mask] = 1 # set everything masked to a certain class

    return predictionsBinary

def classify_image_NN(im_ms, cloud_mask, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018
    
    Modified YD 2020

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    clf: joblib object
        pre-trained classifier

    Returns:    
    -----------
    im_classif: np.array
        2D image containing labels
            Pixel Values:
                0) Other, 1) Sand, 2) Whitewater, 3) Water
    
    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, vec_nan)
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    if vec_features.shape[0] < 1:
        # this means that after removing cloud cover there are no eledgable pixels so we cannot preform classification
        im_classif = None
    else:
        # There are valid features (pixels) in the image
        labels = clf.predict(vec_features)

        # recompose image
        vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
        vec_classif[~vec_mask] = labels
        im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    return im_classif


def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:    
    -----------
    im_nd: np.array
        Image (2D) containing the ND index
        
    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(vec1[~vec_mask] - vec2[~vec_mask],
                     vec1[~vec_mask] + vec2[~vec_mask])
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd


def process_img_classif(im_classif, minBeachArea=1000, minWaterSize=1000, erosion=True, dilation=True):
    """
    proccess image classification array. Smooth out edges and remove small objects from water and land 
    NOTE:this is not smoothing the shoreline its self (that is done later with gausian smoothing algo)

    :param im_classif: binary np.array of land and water
    :param minBeachArea: smallest area a beach will be, used to remoce small land sections in water (default=1000) 
    :param minWaterSize: smallest area water will be, used to remoce small water sections in land (default=1000)
    :param erosion: boolean true if binary erosion should be applied

    :return: binary np.array of land and water (processed, int not boolean)
    """

    processedImClassif = im_classif.astype(dtype=bool)


    # binary opening (erosion -> dilation)
    if erosion:
        processedImClassif = morphology.binary_erosion(processedImClassif) # https://medium.com/swlh/image-processing-with-python-morphological-operations-26b7006c0359
    if dilation:
        processedImClassif = morphology.binary_dilation(processedImClassif, morphology.square(5))
    
    # remove small objects
    processedImClassif = morphology.remove_small_objects(np.invert(processedImClassif), 
                            min_size=minBeachArea, 
                            connectivity=1) # invert it so that we are removing small water objects on the land

    processedImClassif = morphology.remove_small_objects(np.invert(processedImClassif), 
                                min_size=minWaterSize, 
                                connectivity=1) # invert back so that we remove small objects in the water and are back in orriginal format

    processedImClassif = processedImClassif.astype(int) # convert back to int array from boolean array

    return(processedImClassif)


def classify_single_image(toaPath, modelPklName, imMaskPath=None, beachType=None, saveTif=False):
    """

    """
    im_fn = os.path.basename(toaPath)

    if imMaskPath is None:
        im_ms, im_mask = geospatialTools.get_ps_no_mask(toaPath)
    else:
        try:
            im_ms = geospatialTools.get_im_ms(toaPath)
            im_mask = geospatialTools.get_udm1_mask_from_udm2(imMaskPath)
        except np.AxisError:
            print('corrup image, Returning None', im_fn)
            return(None) 
        except:
            im_ms, im_mask = geospatialTools.get_ps_no_mask(toaPath)


    dir1 = os.path.dirname(toaPath) # this code ASSUMES that the file structure of the image is region/sitename/image
    sitename = os.path.basename(dir1)
    region = os.path.basename(os.path.dirname(dir1))
    # referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, '*.tif'))[0] # glb returns list
    # referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, f'{sitename}_reference_*.tif'))[0] # glb returns list

    # if beachType is None:
    #     modelPklName = "default.pkl"
    # elif not os.path.exists(os.path.join( os.getcwd(), 'coastvision', 'classifier', 'models', (beachType + ".pkl") )):
    #     modelPklName = "default.pkl"
    # else:
    #     modelPklName = beachType + ".pkl"
    # modelPklName = classifier
    # modelPklName = 'HI_logisticregression.pkl' #'logistic.pkl' #mlperceptron #logistic_seaward

    clf = joblib.load(os.path.join(os.getcwd(), 'coastvision', 'classifier', 'models', modelPklName))

    if 'logistic' in modelPklName or 'mlperceptron' in modelPklName: 
        im_classif = classify_image(im_ms, clf, im_mask, nonbinary=False)
    else:
        im_classif = classify_image_NN(im_ms, im_mask, clf)

    if im_classif is None:
        # this happens when there are no features availible in the image (i.e totally cloudy)
        return(None)

    # Copy geo info from TOA file
    with rasterio.open(toaPath, 'r') as src:
        kwargs = src.meta
    # Update band info /type
    kwargs.update(
        dtype=rasterio.uint8,
        count = 1)
    # Save im_classif in TOA folder
    # im_fn = os.path.basename(toaPath)
    if saveTif:
        shortPath = os.path.split(toaPath)[0] 
        # shortPath = os.path.split(shortPath)[0] # this was when there was a 'files' folder inside of the 
        siteName = os.path.split(shortPath)[1]
        shortPath = os.path.split(shortPath)[0]
        region = os.path.split(shortPath)[1]
        supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs', region, siteName, 'classification', 'tifs'))
        classPath = os.path.join(os.getcwd(), 'outputs', region, siteName, 'classification', 'tifs', ("classif_" + im_fn))
        with rasterio.open(classPath, 'w', **kwargs) as dst:
            dst.write_band(1, im_classif.astype(rasterio.uint8))

    return(im_classif)


#### contours/shoreline extraction ####
import re
def get_timeperiod_from_ref_sl(refSlFn):
    """
    gets and returns timeperiod of reference shoreline (assuming the default reference shorelines will not be passed in).
    NOTE: assumes this format site_shoreline_year-month_year-range_month-range.geojson, example: kailua_shoreline_2017-01_2015-2020_01-06.geojson
    """
    regex = r'^\w+\d*_\w+_\d{4}-\d{2}_\d{4}-\d{4}_\d{2}-\d{2}\.geojson$'
    if re.match(regex, refSlFn):
        endRemoved = refSlFn.split('.geojson')[0]
        parts = endRemoved.split('_')
        yearrange_parts = parts[3].split('-')
        yearrange_start = int(yearrange_parts[0])
        yearrange_end = int(yearrange_parts[1])
        monthrange_parts = parts[4].split('-')
        monthrange_start = int(monthrange_parts[0])
        monthrange_end = int(monthrange_parts[1])
        return [[int(yearrange_start), int(yearrange_end)], [int(monthrange_start), int(monthrange_end)]]
    else: 
        return None


def get_ref_sl_fn(region, sitename, yearMonth=None, timeperiod=None):
    """
    This function searches for 
    
    :param timeperiod: if you are also going by month (seasonality) insert
    """
    rangeMatchFound = False # used event if year=None
    defaultSlRef = None
    if not yearMonth is None or not timeperiod is None:
    
        for fn in os.listdir(os.path.join(os.getcwd(), 'user_inputs', region, sitename)):
            if fn.startswith((sitename +'_shoreline')):
                fn_name = fn.split('.')[0]
                # naming: site_shoreline_year-month_yearrange_monthrange.geojson
                # example: kailua_shoreline_2017-01_2015-2020_01-06.geojson, example no month range: kailua_shoreline_2017-01_2015-2020_01-12.geojson
                # NOTE: of there is no months given in specific we just say 01-12 (all months), if we want a time period that goes through the end of a year (nov-feb) we would put november first 11-04 and this would get 01,02,11,12 of that year              
                if not timeperiod is None and not isinstance(timeperiod, int):
                    # if timeperiod is an int then the user just gave a year
                    # this means the timeperiod was entered with months
                    years = timeperiod[0]
                    months = timeperiod[1]
                    valid = True
                    slRefTP = get_timeperiod_from_ref_sl(fn)
                    if not slRefTP is None:
                        if not slRefTP[0][0] <= int(years[0]) or not slRefTP[0][1] >= int(years[1]):
                            valid = False
                        if valid and slRefTP[1][0] == int(months[0]) and slRefTP[0][1] == int(months[1]):
                            retuFn = fn # at this point we found our match
                            rangeMatchFound = True
                            break   
                    else:
                        defaultSlRef = fn
                else:
                    if not timeperiod is None:
                        year = timeperiod # if the user passed a year into the timeperiod field
                    elif not yearMonth is None:
                        year = yearMonth[0]
                    year = int(year) # incase year was passed as a str
                    # if the time period is just by year just one year is sufficient
                    slRefTP = get_timeperiod_from_ref_sl(fn)
                    if not slRefTP is None:
                        if slRefTP[0][0] <= year and slRefTP[0][1] >= year and slRefTP[1][0] <= int(yearMonth[1]) and slRefTP[1][1] >= int(yearMonth[1]):
                            rangeMatchFound = True
                            retuFn = fn
                            break    
                    else:
                        defaultSlRef = fn
    else:
        for fn in os.listdir(os.path.join(os.getcwd(), 'user_inputs', region, sitename)):
            if fn.endswith('shoreline.geojson'):
                if not fn[4] == '-':
                    defaultSlRef = fn
                    break
                
    if rangeMatchFound:
        return(retuFn)
    else:
        if not defaultSlRef is None:
            return(defaultSlRef) # assuming if there is no
        else:
            return(None) # there are no reference shorelines


def sort_item_ids_by_timeperiod(region, sitename, itemIds):
    """
    Given a list of item ids this will sort the ids into their corresponding timeperiods (deduced by reference shoreline windows for site)

    :param region: String region that the time is in
    :param sitename: String name of site
    :param itemIds: List of strings item ids ex: "20221115_204452_79_248"
    """
    # get all possible timeperiods
    timeperiodsDict = {} # for each timeperiod a list of files will be included
    namepreamble = (sitename +'_shoreline')
    for fn in os.listdir(os.path.join(os.getcwd(), 'user_inputs', region, sitename)):
        if fn.startswith(namepreamble):
            fnShort = fn.split('.')[0]
            timeperiod = fnShort.split(namepreamble)[1]
            if not timeperiod:
                # this is the default time period shoreline
                timeperiodsDict[(namepreamble + '.geojson')] = []
            else:
                timeperiodsDict[fn] = []
    for itemId in itemIds:
        year = itemId[0:4]
        month = itemId[4:6]
        refSlFile = get_ref_sl_fn(region, sitename, yearMonth=[year, month])
        if refSlFile == f'{namepreamble}.geojson':
            timeperiodsDict[(namepreamble + '.geojson')].append(itemId)
        else:
            timeperiodsDict[refSlFile].append(itemId)
       
    return(timeperiodsDict)


# from shapely.geometry import LineString, MultiPolygon
def create_shoreline_buffer(region, sitename, im_shape, georef, pixel_size, max_dist=max_dist, timeperiod=None):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is 
    

    KV WRL 2018
    
    Modified for PS by YD 2020
    Modified for CoastVision by Joel Nicolow 2022

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres
    

    Returns:    
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)
    # convert reference shoreline to pixel coordinates
    slFn = get_ref_sl_fn(region, sitename, timeperiod=timeperiod)
    if slFn is None:
        return(None) # there is no reference shoreline that we can use
    slBuffPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, slFn)
    # if not os.path.exists(slBuffPath):
    #     return(None) # there is no reference shoreline file so we cannot get the reference shoreline

    with open(slBuffPath, 'r') as f:
        ref_sl_info = geojson.load(f)
    if len(ref_sl_info['features'][0]['geometry']['coordinates']) == 1:
        # this means its prolly saved as a multi string not linestring so just reduce the extra dimension
        ref_sl = ref_sl_info['features'][0]['geometry']['coordinates'][0]
    else:
        ref_sl = np.array(ref_sl_info['features'][0]['geometry']['coordinates'])
    # fig, ax = plt.subplots(figsize=(10,10))
    
    ref_sl_pix = geospatialTools.convert_world2pix(ref_sl, georef)
    ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)


    # # assume that "points" is a numpy array containing the coordinates of the shoreline points
    # line = LineString(ref_sl_pix_rounded)
    # # create a buffer around the shoreline
    # buffer_width = 10 # specify the buffer width in units of the input coordinates
    # buffer_poly = line.buffer(buffer_width)
    # # convert the buffer polygon to a binary mask
    # mask = np.ones(im_shape).astype(bool)
    # # plot the shoreline and the buffer
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.plot(ref_sl_pix_rounded[:,0], ref_sl_pix_rounded[:,1], linewidth=3) 
    # ax.imshow(mask, cmap=plt.cm.gray)
    # # ax.plot(ref_sl_pix[:, 1], ref_sl_pix[:, 0], linewidth=3) 
    # create binary image of the reference shoreline (1 where the shoreline is 0 otherwise)

    im_binary = np.zeros(im_shape)
    # print(im_shape)
    im_valid = False
    ref_sl_pix_rounded = supportingFunctions.add_points(ref_sl_pix_rounded, max_dist)
    # ref_sl_pix_rounded = supportingFunctions.add_points(ref_sl_pix, max_dist)
    for j in range(len(ref_sl_pix_rounded)):
        # print(f"{ref_sl_pix_rounded[j,1]}, {ref_sl_pix_rounded[j,0]}")
        if ref_sl_pix_rounded[j,1] + 1 <= im_shape[0] and ref_sl_pix_rounded[j,0] + 1 <= im_shape[1] and ref_sl_pix_rounded[j,1] > 0 and ref_sl_pix_rounded[j,0] > 0 :
            im_valid = True
            # if the image is a partial image we need only make the sl buffer for that.
            im_binary[ref_sl_pix_rounded[j,1], ref_sl_pix_rounded[j,0]] = 1
    im_binary = im_binary.astype(bool)
    
    # dilate the binary image to create a buffer around the reference shoreline
    max_dist_ref_pixels = np.ceil(max_dist/pixel_size)
    se = morphology.disk(max_dist_ref_pixels)
    im_buffer = morphology.binary_dilation(im_binary, se)

    # Invert boolean (True is outside region)
    im_buffer = im_buffer == False
    # ax.imshow(im_buffer, cmap=plt.cm.gray)
    if im_valid:
        return(im_buffer)
    else:
        print('ref sl buffer is invalid')
        return(None)


def crop_shoreline_with_im_mask(contour, imMaskPath, buff=3):
    """
    this cuts out the contour points that are on the edge of the image or on the could
    """
    if imMaskPath is None:
        print('faulty udm mask')
        return contour
    try:
        im_mask = geospatialTools.get_udm1_mask_from_udm2(imMaskPath) # NOTE SWITCH TO USING UDM1 MASK from udm2
    except:
        print('faulty udm mask, or mask containing only nans')
        return contour
    cleanedContour = []
    for point in range(contour.shape[0]):
        intPoint = contour[point].astype(np.int64)
        pointValList = np.unique(im_mask[intPoint[0]-(buff-1):intPoint[0]+buff, intPoint[1]-(buff-1):intPoint[1]+buff])
        # NOTE NEED TO ALSO LOOK AT VALUE OF IM_MS BECAUSE IF IT IS [0 0 0 0] THIS MEANS THIS POINT IS PART OF THE BLACK BACKGROUND
        if len(pointValList) == 1 and not pointValList[0]: 
            cleanedContour.append([contour[point][0], contour[point][1]])
    return(np.array(cleanedContour))


def get_contour_real_world_length(contour, minSlLength=5):
    """
    takes in a contour and gets the real world length of that contour (list of coordinates making up a line)

    :param contour: numpy array of coordinates (x, y)(WGS84 UTM) needs to be in UTM so that distance between contour points represents real world distance (in meters)
    :param minSlLength: Int, if the contour is shorter than the minSlLength (in real world meters)
    """
    totalLength = 0
    for i in range(0, contour.shape[0]-1, 1):
        totalLength = totalLength + math.sqrt((contour[i][0]-contour[i+1][0])**2 + (contour[i][1]-contour[i+1][1])**2)

    return(totalLength)


def remove_small_shorelines_and_smooth_contour(contour, sigma=0.1, smoothFilterWindow=None, minSlLength=5, maxDistBetweenpoints=10):
    """
    This function removes small contours and uses the gaussian smoothing algorithm to make the contour smooze (if sigma is not None)

    :param contour: nd array (#,2) of coordinates that the contour is
    :param sigma: Int or None, sigma value for gausian smoothing. if sigma=None then don't runn smoothing (default=1)
    :param smoothFilterWindow: int if not None then we use that num as the num points to smooth together
    :param minSlLength: Int, min length (in real world meters) that a shoreline can be 
    :param maxDistBetweenpoints: int, if two contour coordinates (next too each other in list) are farther than this away from each other treat this as a multiple sections of a contour default=2
    """
    if contour.shape[0] == 0:
        # there  are no valid shoreline points so we just return the empty contour that was given and the framework will sort it out
        return(contour)
   
    contiguousSectionsList = list()
    start = 0
    for i in range(len(contour)-1):
        x1 = contour[i][0]
        x2 = contour[i+1][0]
        y1 = contour[i][1]
        y2 = contour[i+1][1]
        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        if dist > maxDistBetweenpoints:
            if start != i:
                contiguousSectionsList.append((start, i))
            start = i+1 # need to increment start either way because it means we are skipping a contig section of two points
            
    if start == 0 or len(contiguousSectionsList) < 1:
        # this means there is just one shoreline section and it is in order (starts at one end and goes to other end)
        contiguousSectionsList = list()
        contiguousSectionsList.append((0, len(contour)-1)) # this is just to add it as a touple (prolly bettah way for do this)
    else:
        if contiguousSectionsList[-1][1]+1 != len(contour)-1:
            # this means there is a section at the end that we need to add
            contiguousSectionsList.append((contiguousSectionsList[-1][1]+1, len(contour)-1))
            
    outputContour=None
    for contigSLSection in contiguousSectionsList:
        slSection = contour[contigSLSection[0]:contigSLSection[1],:]
        if get_contour_real_world_length(slSection, minSlLength) < minSlLength:
            # print(slSection)
            continue # skip sls that are too short
        if not sigma is None and len(slSection) >= 1:
            # if the sigma value is None that means smoothing is turned off
            x, y = slSection.T
            t = np.linspace(0, 1, len(x))  # Create an array of t values with the same length as x
            t2 = np.linspace(0, 1, 100)  # Create an array of t values with 100 points

            x2 = np.interp(t2, t, x)
            y2 = np.interp(t2, t, y)

            if not smoothFilterWindow is None:
                for i in range(1, len(x2)-(smoothFilterWindow-1), smoothFilterWindow):
                    # filter (looks at three points at a time) NOTE: stepsite 3 so points aren't revisited
                    xi = x2[i-1:i+2]
                    yi = y2[i-1:i+2]
                    xiSmooth = gaussian_filter1d(xi, sigma)# Smooth the x2 array using Gaussian filtering
                    yiSmooth = gaussian_filter1d(yi, sigma)
                    if outputContour is None:
                        outputContour = np.array(list(zip(xiSmooth, yiSmooth)))
                    else:
                        outputContour = np.concatenate((outputContour, np.array(list(zip(xiSmooth, yiSmooth)))), axis=0)
            else:
                # smooth sections of contour together
                x3 = gaussian_filter1d(x2, sigma)# Smooth the x2 array using Gaussian filtering
                y3 = gaussian_filter1d(y2, sigma)  # Smooth the y2 array using Gaussian filtering

                # Interpolate the smoothed x3 and y3 arrays onto the original t array
                x4 = np.interp(t, t2, x3)
                y4 = np.interp(t, t2, y3)

                if outputContour is None:
                    outputContour = np.array(list(zip(x4, y4))) # If there is no existing output contour array, create a new one
                else:
                    outputContour = np.concatenate((outputContour, np.array(list(zip(x4, y4)))), axis=0)
        else:
            # this is if smoothing is turned off
            if outputContour is None:
                outputContour = np.array(slSection)
            else:
                outputContour = np.concatenate((outputContour, slSection), axis=0)

    return(outputContour)


def shoreline_contour_extract_single(im_classif, region, sitename, georef, shorelinebuff=None, min_sl_len=20, pixel_size=3, max_dist=max_dist, imMaskPath=None, saveContour=False, timestamp=None, smoothShorelineSigma=None, smoothWindow=None, year=None):
    """
    This function takes the pixel classification matrix and and creates a contour of the shoreline

    :param im_classif: numpy array where each value represents the value of a pixel (sand, white water, water)
    :param region: region that the time is from
    :param sitename: name of site
    :param georef: vector of 6 elements used to scale to real world coordinates [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    :param shorelinebuff: because we may run multiple images for the same site at the same time we can save time but just creating the sl ref buff once and passing it to all images for that site
    :param pixel_size: resolution of pixels in meters (e.g. 3 for each pixel is 3 by 3 meters) default=3
    :param max_dist: max distance the extracted shoreline can be from the reference shoreline (if shorelinebuff is entered) default=75
    :param imMaskPath: path to image udm2 file used to cut out shorelines on the edge of clouds or boarder of the image default=None
    :param saveContour: true if the contour should be saved in a geojson file default=False
    :param timestamp: timestamp of image (raw) used to name contour geojson default=None
    :param smoothShorelineSigma: True if the contours (shoreline) should be smoothed using a smoothing algorithm default=False
    :param smoothWindow: int passed to remove_small_shorelines_and_smooth_contour() to say if there should be a filter or not and if there should be how big
    :param year: year can be passed so the user can get data from a specific timeframe (default=None)

    written by CRC
    """
    isoValue = 0.5 # because water pixels are 0 and other pixels are 1
    im_classif = 1-im_classif # INVERT ARRAY TO GET MORE LANDWARD PREDICTION
    contours = measure.find_contours(im_classif, isoValue, fully_connected='low', positive_orientation='low') # second two params just set to there default (low) https://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html

    ############ ABM TEST ##########
    # print(im_classif.shape)
    # shape_polygon = gpd.read_file(os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_polygon.geojson')))
    # print(shape_polygon.geometry)



    if not len(contours) > 0:
        return(None) # means there is no shoreline in the image
    
    mainContour = contours[0] # so we can contactinate 
    for i in range(1, len(contours), 1):
        mainContour = np.concatenate([mainContour, contours[i]])
    
    if not imMaskPath is None:
        # if there is a mask file use it to cut out contours on the edge of the image and around clouds
        mainContour = crop_shoreline_with_im_mask(mainContour, imMaskPath)
        if len(mainContour) == 0:
            return(None) # all of the contour was either on the edge of the image or on clouds


    contourConvert = geospatialTools.convert_pix2world(mainContour, georef)

    # just get contour in the shoreline buffer
    if shorelinebuff is None:
        shorelinebuff = create_shoreline_buffer(region=region, sitename=sitename, im_shape=im_classif.shape, georef=georef, pixel_size=pixel_size, max_dist=max_dist, timeperiod=year)
    if not shorelinebuff is None:
        # shoreline buffer
        validShorelinePoints = []
        for row in range(0, shorelinebuff.shape[0], 1):
            for col in range(0, shorelinebuff.shape[1]):
                if not shorelinebuff[row,col]:
                    validShorelinePoints.append([row, col])
            
        validShorelinePoints = np.array(validShorelinePoints)
        slConvert = geospatialTools.convert_pix2world(validShorelinePoints, georef)

        perimeterPointsFirsts = []
        perimeterPointsLasts = []
        for xval in np.unique(slConvert[:,1]):

            points = slConvert[np.where((xval == slConvert[:,1]))]
            first = points[0]
            last = points[-1]
            if (first == last).all():
                # we don't need to add the same point twice
                perimeterPointsFirsts.append({"x": first[0], "y": first[1]})
            else:
                perimeterPointsFirsts.append({"x": first[0], "y": first[1]})
                perimeterPointsLasts.append({"x": last[0], "y": last[1]})

        perimeterPointsLasts.reverse() # this is so that the end of first points runs into the begining of last points
            
        perimeterPoints = perimeterPointsFirsts + perimeterPointsLasts

        poly = geometry.Polygon([[p['x'], p['y']] for p in perimeterPoints])

        validContour = []

        for i in range(0, contourConvert.shape[0], 1):
            if poly.contains(Point(contourConvert[i][0], contourConvert[i][1])):
                validContour.append([contourConvert[i][0], contourConvert[i][1]])
                
        validContour = np.array(validContour)
    else:
        validContour = geospatialTools.convert_pix2world(mainContour, georef)

    validContour = remove_small_shorelines_and_smooth_contour(validContour, sigma=smoothShorelineSigma, smoothFilterWindow=smoothWindow, minSlLength=min_sl_len)

    if saveContour:
        refSlPth = os.path.join('user_inputs', region, sitename, f"{sitename}_shoreline.geojson")
        if os.path.exists(refSlPth):
            epsg = geospatialTools.get_epsg_from_geojson(refSlPth)
        else:
            infoJson = supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_info.json')))
            epsg = infoJson['epsg']

        basepth = os.path.join('outputs', region, sitename, 'shoreline', 'contours')
        supportingFunctions.create_dir(basepth)
        pth = os.path.join(basepth, f"{sitename}_{timestamp}_contour.geojson")
        if validContour is None:
            return(None)
        geospatialTools.save_geojson_linestring(supportingFunctions.ndarray_to_string(validContour), epsg=epsg, savePath=pth, sitename=sitename)


    return(validContour)


import datetime as dt
def add_all_contours_to_one_site_geojson(reg, sites = None, qaqc=False):  ## ABM added sites = None so you can add a list of sites to process if working w specific sites (faster!)
    '''
    sites: list of sites to process. if none is passed all sites are processed #ABM
    qaqc: if True, will save with '_qaqc' extension, else without
    '''

    dataDir = os.path.join(os.getcwd(), 'outputs', reg)
    region =reg
    # sites = os.listdir(dataDir)
    for sitename in sites:
        # if sites != None and sitename not in sitelist:
        #     print('skipping')
        #     continue

        print(f'Compiling all shorelines into one geojson for site {sitename}\r', end="")
        shorelineDir = os.path.join(os.getcwd(), 'outputs', region, sitename, 'shoreline')
        contoursDir = os.path.join(shorelineDir, 'contours')
        contourGeojsons = os.listdir(contoursDir)
        
        ## ABM ADDED 
        if len(contourGeojsons) == 0:
            continue
        #### combine shorelines into one geojson
        epsg = geospatialTools.get_epsg_from_geojson(os.path.join(contoursDir, contourGeojsons[0])) # get epsg from first contour file (all should have the same epsg, also assuming that the first file will not be some other filetype)
        allContoursGJ = {"type":"FeatureCollection",
                        "crs": { "type":"name", "properties":{ "name": f"urn:ogc:def:crs:EPSG::{epsg}" } },
                        "features":[]}
        for fn in contourGeojsons:
            if '_contour.geojson' in fn:
                # get values needed for geojson
                rawTimestamp = fn.split('_contour.geojson')[0].split(sitename + '_')[1]
                if len(rawTimestamp) == 18:
                    rawtimestampPsFormat = rawTimestamp[0:-3] # remove miliseconds just incase this file doesnt have it
                elif len(rawTimestamp) == 15:
                    rawtimestampPsFormat = rawTimestamp
                    rawTimestamp = rawTimestamp + '_00'
                else:
                    continue # format does not match the two possibilities
                timestamp = dt.datetime.strptime(rawTimestamp, "%Y%m%d_%H%M%S_%f")
                pth = os.path.join(os.getcwd(), 'data', region, sitename)
                imgMetaDataFile = supportingFunctions.get_fn_in_dir_with_substrs(pth, [rawtimestampPsFormat, '_metadata.json'])# we are relying on the assumtion that there is only one meta data file for each timestamp
                if len(imgMetaDataFile) == 2:
                    imgMetaDataFile = str(imgMetaDataFile[0])
                else:
                    imgMetaDataFile = str(imgMetaDataFile)
                satType = imgMetaDataFile.split(rawtimestampPsFormat + '_')[1].split('_metadata.json')[0]
                metadata = supportingFunctions.read_json_file(os.path.join(pth, imgMetaDataFile))
                if metadata is None:
                    continue # if there isnt goof meta data
                cloudCover = metadata['properties']['cloud_cover']

                with open(os.path.join(contoursDir, fn), 'r') as f:
                    data = geojson.load(f)
                contourCoordinates = data['features'][0]['geometry']['coordinates']
                stringFormCoords = supportingFunctions.ndarray_to_string(np.array(contourCoordinates))
                contourDict = {"type":"Feature", "properties":{"date":str(timestamp), 
                                "satname":satType, "geoaccuracy":999, "cloud_cover":cloudCover},
                            'geometry':{"type":"LineString", "coordinates":stringFormCoords}}

                allContoursGJ['features'].append(contourDict)

        # NOTE: this dones not append it overwrites the file if there was already one
        if qaqc == True:
            path = os.path.join(shorelineDir, f"{sitename}_all_contours_qaqc.geojson")
        else:
            path = os.path.join(shorelineDir, f"{sitename}_all_contours.geojson")
        with open(path, "w") as f:
            geojson.dump(allContoursGJ, f, indent=4)
        # NOTE if you want to save these all in one folder uncomment the code bellow
        # supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs_region', region))
        # with open(os.path.join('outputs_region', region, f"{sitename}_all_contours.geojson"), "w") as f:
        #     geojson.dump(allContoursGJ, f)


#### Transects ####
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.
    
    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded
        
    Returns:    
    -----------
    transects: dict
        contains the X and Y coordinates of each transect
        
    Source:
        addapted from https://github.com/kvos/CoastSat
        
    """  
    
    gdf = gpd.read_file(filename)
    transects = dict([])
    # for i in gdf.index:
    #     transects[gdf.loc[i,'name']] = np.array(gdf.loc[i,'geometry'].coords)
    for i in gdf.index:
        geom = gdf.loc[i, 'geometry']
        if isinstance(geom, LineString):
            transects[gdf.loc[i, 'name']] = np.array(geom.coords)
        elif isinstance(geom, MultiLineString):
            # Ensure the MultiLineString has exactly one LineString with two coordinates
            if len(geom.geoms) == 1 and len(geom.geoms[0].coords) == 2:
                transects[gdf.loc[i, 'id']] = np.array(geom.geoms[0].coords)
            else:
                raise ValueError(f"MultiLineString at index {i} does not have exactly one LineString with two coordinates.")
        else:
            raise TypeError(f"Unsupported geometry type at index {i}: {type(geom)}")

    # print('%d transects have been loaded' % len(transects.keys()))

    return transects


def transect_intersect_single(along_dist, contour, transects):
    """
    Finds length along transect from origin (of transect) to where the shoreline is.
    This function converts the elements to real world coordinates using georef.

    :param along_dist: int distance along shoreline
    :param contour: numpy array (each value represents a point on the contour)
    :param georef: vector of 6 elements used to scale to real world coordinates [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    :param transects: transects loaded using interactive.transects_from_geojson

    :return cross_dist: dictionary with transects and the distance along them that the shoreline is at
    example output: 
    {'PF1': array([254.05505181]),
    'PF2': array([453.96602023]),
    'PF4': array([225.81277343]),
    'PF6': array([199.30647753]),
    'PF8': array([186.3904818])}
    """
    # sl = geospatialTools.convert_pix2world(contour, georef) 
    sl = contour # now contour is in real world coordinates
    intersections = np.zeros((1,len(transects)))
    i = 0
    for j,key in enumerate(list(transects.keys())): 

        # compute rotation matrix
        X0 = transects[key][0,0]
        Y0 = transects[key][0,1]
        #X0 = transects[key][-1,0] ### ABM ###: change point from origin to end of transect
        #Y0 = transects[key][-1,1] #ABM
        temp = np.array(transects[key][-1,:]) - np.array(transects[key][0,:])
        #temp = np.array(transects[key][0,:]) - np.array(transects[key][1,:]) #ABM
        phi = np.arctan2(temp[1], temp[0])
        Mrot = np.array([[np.cos(phi), np.sin(phi)],[-np.sin(phi), np.cos(phi)]])

        # calculate point to line distance between shoreline points and the transect
        p1 = np.array([X0,Y0])
        p2 = transects[key][-1,:] 
        #p2 = transects[key][0,:] ### ABM
        d_line = np.abs(np.cross(p2-p1,sl-p1)/np.linalg.norm(p2-p1))
        # calculate the distance between shoreline points and the origin of the transect
        d_origin = np.array([np.linalg.norm(sl[k,:] - p1) for k in range(len(sl))])
        # find the shoreline points that are close to the transects and to the origin
        # the distance to the origin is hard-coded here to 1 km 
        idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
        # find the shoreline points that are in the direction of the transect (within 90 degrees)
        temp_sl = sl - np.array(transects[key][0,:]) 
        #temp_sl = sl - np.array(transects[key][-1,:]) #ABM
        phi_sl = np.array([np.arctan2(temp_sl[k,1], temp_sl[k,0]) for k in range(len(temp_sl))])
        diff_angle = (phi - phi_sl)
        idx_angle = np.abs(diff_angle) < np.pi/2
        # combine the transects that are close in distance and close in orientation
        idx_close = np.where(np.logical_and(idx_dist,idx_angle))[0]     

        # in case there are no shoreline points close to the transect 
        if len(idx_close) == 0:
            intersections[i,j] = np.nan
        else:
            # change of base to shore-normal coordinate system
            xy_close = np.array([sl[idx_close,0],sl[idx_close,1]]) - np.tile(np.array([[X0],
                            [Y0]]), (1,len(sl[idx_close])))
            xy_rot = np.matmul(Mrot, xy_close)
            # compute the median of the intersections along the transect
            intersections[i,j] = np.nanmedian(xy_rot[0,:])
    
    # fill the a dictionnary
    cross_dist = dict([])
    for j,key in enumerate(list(transects.keys())): 
        cross_dist[key] = intersections[:,j]   
    # print(cross_dist)
    return(cross_dist)


###############################################

###  HELPER FUNCTION  to extent transect line further inland for compute_intersection_CRC function below
## if transect and line dont intersect, results in a nan
def extend_linestring(line, distance):
    # Get the start and end points of the LineString
    start_point = Point(line.coords[0])
    end_point = Point(line.coords[1])
    # Calculate the direction vector
    direction_vector = (end_point.x - start_point.x, end_point.y - start_point.y)
    # Normalize the direction vector
    norm = (direction_vector[0]**2 + direction_vector[1]**2)**0.5
    normalized_direction = (direction_vector[0] / norm, direction_vector[1] / norm)
    # Calculate the new end point after backward extension
    new_end_point = (end_point.x, end_point.y)
    # Calculate the new start point after backward extension
    new_start_point = (start_point.x - normalized_direction[0] * distance,
                      start_point.y - normalized_direction[1] * distance)
    # Create a new LineString with the extended start point and fixed end point
    extended_line = LineString([new_start_point, new_end_point])
    return extended_line

def transect_intersect_single_crc(along_dist, contour, region, sitename, timestamp, extend_line=20):
    """
    Finds length along transect from origin (of transect) to where the shoreline is.
    """
    # print('\n', timestamp)
    ### LOAD transect
    transects_path  = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_transects.geojson'))
    transects = gpd.read_file(transects_path )

    ### LOAD contour
    contour_path = glob.glob(os.path.join(os.getcwd(), 'outputs', region, sitename, 'shoreline', 'contours', ('*'+timestamp+'*')))[0]
    contour_gpd = gpd.read_file(contour_path)

    #initiate cross_dist
    cross_dist = dict()
    x,y = [],[]

    #iterate over each transect
    for n, transect in enumerate(transects.geometry):
        # Extend transect and calculate intersection with contour
        transect_extended = extend_linestring(transect, extend_line)
        intersect_point = contour_gpd.intersection(transect_extended)

        # Get landward point of transect and calculate distance
        t1 = Point(transect_extended.coords[0])
        distance = intersect_point.distance(t1)[0] - extend_line

        # Check distances and overlapping transects
        bad = False
        dist_to_contour_coords = [intersect_point.distance(Point(contour[i]))[0] for i in range(len(contour))]

        if min(dist_to_contour_coords) > along_dist or sum(transects.intersects(transect)) > 1:
            distance = np.nan
            bad = True

        # Update cross distance and coordinates
        cross_dist[str(n)] = distance
        if not bad:
            try:
                x.append(intersect_point.x[0])
                y.append(intersect_point.y[0])
            except IndexError:
                pass  # Handle missing points

    # Create GeoDataFrame for intersections
    geometry = [Point(xy) for xy in zip(x, y)]
    intersections = GeoDataFrame(np.arange(len(x)), crs="EPSG:32604", geometry=geometry)

    return cross_dist, intersections


#### misc ####
def get_all_pixel_resolutions(region, sitename):
    """
    Looks at all of the metadata.json files for a stite and returns a list of the puxel size (resolution)
    """
    retuList = list()
    for file in os.listdir(os.path.join(os.getcwd(), 'data', region, sitename)):
            if file.endswith('metadata.json'):
                try:
                    metadataJson = os.path.join(os.getcwd(), 'data', region, sitename, file)
                    with open(metadataJson, 'r') as f:
                        data = json.load(f)
                    retuList.append(data['properties']['pixel_resolution'])
                except ValueError:
                    dir = os.path.join(os.getcwd(), 'data', region, sitename, file)
                    print(f'coastvision.get_all_pixel_resolutions() json decoder error {dir}')
                    retuList.append(3) # giving a 3m pixel default
    return(retuList)


def is_sat_img_valid(region, sitename, tifPath, udmTifPath, maxDist=30, validPixThreshold=20):
    """
    This function checks if a tif file is valid:
        does it contain any of the reference shoreline buffer (at least validPixThreshold pixels)
        is there too much white wash (in the reference shoreline buffer) (UNDER CONTRUCTION)

    :param region: String, region name (used to find reference shoreline along with year)
    :param sitename: String, name of site (beach) (used to find reference shoreline along with year)
    :param tifPath: full path to tif file (assuming that the first 4 digits of the basename are the year the photo was taken)
    :param udmTifPath: full path to udm2 tif file
    :param maxDist: NOTE: NOT BEING USED int, number of meters for shoreline buffer distance from reference shoreline (default=75)
    :param validPixThreshold: int, number of pixels that the sat image has inside of the reference shoreline buffer (default=20)

    """
    
    year = os.path.basename(tifPath)[0:4]
    # data = gdal.Open(tifPath, gdal.GA_ReadOnly)
    # georef = np.array(data.GetGeoTransform())
    georef = geospatialTools.get_georef(tifPath)
    im_ms = geospatialTools.get_im_ms(tifPath)

    pixelSize = supportingFunctions.get_pixel_size(os.path.dirname(tifPath), os.path.basename(tifPath))
    slBuff = create_shoreline_buffer(region, sitename, im_shape=im_ms.shape[0:2], georef=georef, pixel_size=pixelSize, max_dist=max_dist, timeperiod=year)
    udm_mask = geospatialTools.get_udm1_mask_from_udm2(udmTifPath) # USE UDM 1 not cloud band

    #print(f'udm mask shape: {udm_mask.shape}, sl buff shape: {slBuff.shape}')
    #print(udm_mask.shape == slBuff.shape)
    validPixelCounter = 0
    for row in range(0, udm_mask.shape[0], 1):
        for col in range(0, udm_mask.shape[1], 1):
            if not slBuff[row, col]:
                # this means we are inside the shoreline buffer
                if udm_mask[row, col] == 0:
                    # this means this pixel is a real pixel that is not part of a cloud mask (or part of the black backdrop)
                    validPixelCounter +=  1
                    if validPixelCounter == validPixThreshold:
                        break 
        if validPixelCounter == validPixThreshold:
            break # then break out of outter loop


    if validPixelCounter < validPixThreshold:
        return(False) # the image is not valid because it doesn't contain enough pixels in the ref shoreline buffer

    # add "too much white water" section here
    
    return(True)


def check_tif_name_valid(rawTimestamp):
    """
    This is used to quickly test if a fn is in the format of a planet scope API image file. This means the it starts with a timestamp in this format "%Y%m%d_%H%M%S"

    :param rawtimestamp: timestamp string in filename

    :return: boolean True if valid False if not
    """

    valid = True
    if len(rawTimestamp) != 18:
        return(False)
    for i in range(0, len(rawTimestamp), 1):
        if i < 8:
            if not rawTimestamp[i].isdigit():
                valid = False
        elif i == 8:
            if not rawTimestamp[i] == '_':
                valid == False
        elif i < 15 and i > 8:
            if not rawTimestamp[i].isdigit():
                valid = False
        elif i == 15:
            if not rawTimestamp[i] == '_':
                valid == False
        else:
            if not rawTimestamp[i].isdigit():
                valid = False
    return(valid)


#### run all ####
import json
from coastvision import coastvisionPlots
def run_coastvision_single(region, sitename, itemID, siteInputs=None, justShorelineExtract=False, smoothShoreline=0.1, smoothWindow=None, min_sl_len=30, intersect_func='cs', modelPklName='HI_logisticregression.pkl', dataProducts=False):
    """

    :param siteInputs: dictonary that contains transects and infojson for the site. This allows use to save time when running multiple items from the same site (defualt=None)
    NOTE siteInputs needs to have 'infoJSON', 'shorelinebuff', and 'transects
    :param itemID: its the timestamp and often some satalite info
    """

    toaPath = os.path.join(os.getcwd(), 'data', region, sitename, f'{itemID}_3B_AnalyticMS_toar_clip.tif')
    maskFileName = os.path.join(os.getcwd(), 'data', region, sitename, f'{itemID}_3B_udm2_clip.tif')
    metadataJson = os.path.join(os.getcwd(), 'data', region, sitename, f'{itemID}_metadata.json')
    timestamp = '_'.join(itemID.split('_')[:-1])
    #print(sitename, timestamp)
    #print(f'\r{sitename}; {timestamp}', end='', flush=True)
    
    try:  ##ABM Dummy check when udm file is faulty, make path=None  
        rasterio.open(maskFileName)
    except:
        maskFileName = None


    if siteInputs is None:
        infoJson = supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_info.json')))
        shorelinebuff = None
    else:
        infoJson = siteInputs['infoJson']
        shorelinebuff = siteInputs['shorelinebuff']

    if not infoJson is None:
        beachType = infoJson['beach_type']
        maxDistSlRef = infoJson['max_dist_from_sl_ref']
    else:
        beachType = None
        #maxDistSlRef = 10
        #maxDistSlRef = 25
        maxDistSlRef = max_dist ###ABM
        infoJson = {'minBeachArea':500, 'minWaterSize':500}
    # along_dist = 25


    if not metadataJson is None:
        try:
            with open(metadataJson, 'r') as f:
                data = json.load(f)
            try:
                infoJson['pixel_size'] = data['properties']['pixel_resolution']
            except TypeError:
                infoJson['pixel_size'] = 3
        except ValueError:
            print(f'coastvision.run_coastvision_single() {f}')
            infoJson['pixel_size'] = 3 # defualt pixel size is 3m


    data = gdal.Open(toaPath, gdal.GA_ReadOnly)
    if data is None:
        return(None)
    georef = np.array(data.GetGeoTransform())

    # supportingFunctions.show_rgb(toaPath) #### ABM: COMMENT OUT/ UNCOMMENT to show image
    ### image classification
    im_classif_raw = classify_single_image(toaPath, modelPklName=modelPklName, imMaskPath=maskFileName, beachType=beachType, saveTif=False)
    if im_classif_raw is None:
        # this happens when there are no features availible in the image (i.e totally cloudy)
        return(None)
    im_classif = process_img_classif(im_classif_raw, minBeachArea=infoJson['minBeachArea'], minWaterSize=infoJson['minWaterSize']) # use sklearn.morphology to smooth and clean classification


    ### contour extraction
    contour = shoreline_contour_extract_single(im_classif, region, sitename, 
                                                        georef=georef, shorelinebuff=shorelinebuff, min_sl_len=min_sl_len, pixel_size=int(infoJson['pixel_size']), 
                                                        max_dist=int(maxDistSlRef), imMaskPath=maskFileName, 
                                                        saveContour=True, timestamp=timestamp,
                                                        smoothShorelineSigma=smoothShoreline, smoothWindow=smoothWindow,
                                                        year=int(itemID[0:4]))
    # is conour valid
    if contour is None or contour.shape[0] == 0 or len(contour) == 0:
        # this means the image didnt include any of the reference shoreline (no contours extracted)
        # print(f'skipping this image because it doesn\'t contain any of the sl ref: {toaPath}')
        print('  contour outside reference')
    else:
        if dataProducts:
            # plot contour and classif
            pixCoordContour = geospatialTools.convert_world2pix(contour, georef)
            coastvisionPlots.plot_classif_and_intersection(toaPath, im_classif, transects=siteInputs['transects'], sl=pixCoordContour, slBuff=shorelinebuff)
            # supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'classification'))
            # supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'contours'))
            # savePath = os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'classification', f"{sitename}_{timestamp}_rgb_classif_sl_plot.png")
            # if shorelinebuff is None:
            #     # NOTE this is slightly redondent but would only really happen if you are running one image at a time and at that point the time is negligable
            #     shorelinebuff = create_shoreline_buffer(region, sitename, im_shape=im_classif.shape, georef=georef, pixel_size=int(infoJson['pixel_size']), max_dist=maxDistSlRef, timeperiod=int(itemID[0:4]))
            # coastvisionPlots.rgb_classif_plot_single(toaPath, im_classif, sl=pixCoordContour, slBuff=shorelinebuff, savePath=savePath)
            # supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'final'))
            # savePath = os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'final', f"{sitename}_{timestamp}_data_product.png")            
            # coastvisionPlots.master_data_products_plot(toaPath, im_classif, sl=pixCoordContour, slBuff=shorelinebuff, savePath=savePath)
            # savePath = os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'contours', f"{sitename}_{timestamp}_rgb_sl_plot.png")
            # coastvisionPlots.rgb_contour_plot_single(toaPath, pixCoordContour, savePath, loudShoreline=False)

        if not justShorelineExtract:
            if siteInputs is None:
                transectPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + "_transects.geojson"))
                transects = transects_from_geojson(transectPath)
            else:
                transects = siteInputs['transects']

            ########## ABM ADDED ###########
            if intersect_func == 'crc':
                cross_dist, intersections = transect_intersect_single_crc(35, contour, region, sitename, timestamp) 
            else:
                cross_dist = transect_intersect_single(35, contour, transects)
            
            if dataProducts:
                supportingFunctions.create_dir(os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'transects'))
                #import geopandas as gpd
                transectPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + "_transects.geojson"))
                transects_gpd = gpd.read_file(transectPath)
                savePath = os.path.join(os.getcwd(), 'outputs', region, sitename, 'data_products', 'transects', f"{sitename}_{timestamp}_transect_int.png")
                coastvisionPlots.rgb_plot_tif_transect_intersection(toaPath, transects_gpd, transects, region, sitename, cross_dist, savePath, intersections=None)
                
            return(cross_dist)

            # return(transect_intersect_single(25, contour, transects))
    # only need to return things if we are looking at transect intersection. Otherwise every output we need is saved somewhere duringthe run