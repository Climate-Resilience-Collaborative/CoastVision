"""
This script contains general purpose functions that can be used in varius applications

Author: Joel Nicolow, Climate Resiliance Initiative, School of Ocean and Earth Science and Technology (July, 15 2022)

"""

#### Misc

def clean_timestamp(rawTimestamp):
    underscoreCount = rawTimestamp.count('_')
    if underscoreCount == 2:
        rawStr = '_'.join(rawTimestamp.split('_')[0:2])
        rawStr = rawStr + '_00'
        timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    else:
        rawStr = '_'.join(rawTimestamp.split('_')[:-1])
        timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    cleanTimestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    return cleanTimestamp


def yearmonth_inYearmonth_range(year, month, yearMonthRange):
    """
    This function returns true if date (made by year and month) is within the year-month range
    """
    print(yearMonthRange)
    if yearMonthRange[0][0] <= year <= yearMonthRange[0][1]:
        monthRange = yearMonthRange[1]
        return(month_in_month_range(month, monthRange))


def month_in_month_range(month, monthRange):
    """
    Checks if month is inbetween month range

    :param month: int 1-12 (if bigger than 12 or less than 1 invalid)
    :param monthRange: array of two ints (like month) if the first int is bigger that means its from that month to the rest of the year and then the start of the year until the end of the month passed
    """
    print(monthRange)
    if not 1 <= month <= 12:
        raise KeyError('Month must be 01-12, supportingFunctions.month_in_month_range()')
    if len(monthRange) == 2:
        for mon in monthRange:
            if not 1 <= mon <= 12:
                raise KeyError('Month must be 01-12, supportingFunctions.month_in_month_range() monthRange falty')

        if monthRange[0] < monthRange[1]:
            if monthRange[0] <= month <= monthRange[1]:
                return(True)
        elif monthRange[0] > monthRange[1]:
            if monthRange[0] <= month <= 12 or 1 <= month <= monthRange[1]:
                return(True)
        elif monthRange[0] == monthRange[1]:
            if month == monthRange[0]:
                return(True)
    else:
        raise KeyError('monthRange must be of length 2, supportingFunctions.month_in_month_range()')

    return(False)

#### String manipulation
def str_contains_list_elements(str, lst, all=True):
    """
    """
    retu = False            
    for el in lst:
        if el in str:
            if not all:
                return(True)
            retu = True
        else:
            if all:
                return(False) # at leas one of the else was not founr
    return(retu)    


#### os opperations
import os
def create_dir(path):
    split = os.path.split(path)

    endingList = [split[1]]
    while not split[1] == '':
        split = os.path.split(split[0])
        if split[1] == '':
            endingList.append(split[0])
        else:
            endingList.append(split[1])


    rootPath = endingList[len(endingList)-1]
    for i in range(len(endingList)-2, -1, -1):
        if not os.path.exists(os.path.join(rootPath, endingList[i])):
            os.mkdir(os.path.join(rootPath, endingList[i]))
        rootPath = os.path.join(rootPath, endingList[i])
    return(True)


def get_fn_in_dir_with_substrs(dir, substrs, containsAllSubstrs=True):
    """
    This function loops through a directory and finds all files that contain some or all of the sub strings given in substrs

    :param dir: path to a directory that we are searching
    :param substrs: list or 1d array of substrs that we are looking for
    :param containsAllSubstrs: if True there needs to be all of the substrs present in the fn if not at leas one needs to be default=True
    """
    retuList = list() # if multiple filenames match they will all be returned
    if not containsAllSubstrs:
        for fn in os.listdir(dir):
            validFn = False
            for substr in substrs:
                if substr in fn:
                    validFn = True
            if validFn:
                retuList.append(fn)
    else:
        for fn in os.listdir(dir):
            validFn = True
            for substr in substrs:
                if not substr in fn:
                    validFn = False
            if validFn:
                # this will only be true if all of the substrs are found
                retuList.append(fn)
    
    if len(retuList) == 1:
        retuList = retuList[0] # so we are not returning a list with 1 element
    
    return(retuList)


#### working with data structures
def only_keep_these_dict_elements(dictionary, keep):
    """
    Takes in a dictionary and a list of elements you want to keep. Then removes the elements you do not wish to keep

    :param dictionary: a dictionary
    :param keep: a list ([]) of the keys for the elements that you wish to keep

    :return: the updated dictionary
    """
    removes = []
    for key in dictionary.keys():
        if key not in keep:
            removes.append(key)
    for remove in removes:
        dictionary.pop(remove)
    return(dictionary)


def mad(arr, equalToStd=False):
    """
    Calculates the median absolute deviation (MAD) for a NumPy array
    :param arr: numpy array
    :param equalToStd: if true use correction term to make MAD equal to STD (for normally distributed data)
    """
    if equalToStd:
        multiplier = 1.482 # paul wessels book geological data analysis
    else:
        multiplier = 1

    median = np.median(arr)
    abs_deviations = np.abs(arr - median)
    mad = multiplier*np.median(abs_deviations)
    return mad


def df2_is_subset_df1(df1, df2):
    if df1.columns.tolist() != df2.columns.tolist():
        raise KeyError(f'the data frames do not have the same columns \n {df1.columns}\n{df2.columns}')
    results = []
    for col in df1.columns:
        s1 = df1[col].dropna()
        s2 = df2[col].dropna()
        results.append(s2.isin(s1).all())
    results = list(set(results)) # unique list now
    if len(results) and results[0]:
        return True
    return False


#### working with json
import json
def read_json_file(filepath):
    """
    loads a json file and returns a dictionary with the info

    :param filepath: full filepath to the json file
    :return: python dictionary of the json data
    """
    try: ## ABM Added to faulty metadatafiles
        with open(filepath, 'r') as f:
            data = json.load(f)
        return(data)
    except ValueError:
        print('bad metadata')
        return(None)

def get_info_from_readme(readmePath):
    """
    This reads in the readme file and creates a dictionary
    """
    
    with open(readmePath) as f:
        readme = json.load(f)
    
    return(readme)


def get_info_from_info_json(infoJsonPath):
    """
    This function returns the dictionary inside of said json file
    """
    if os.path.exists(infoJsonPath):
        with open(infoJsonPath,'r') as file:
            # First we load existing data into a dict.
            fileData = json.load(file)
        return(fileData)
    return(None)


def append_json(newData, filename):
    """
    append a new dictionary onto the end of a json file
    
    :param newData: dictionary that needs to be added
    :param filename: String path to file that we are appending
    """
    if os.path.exists(filename):
        with open(filename,'r+') as file:
            # First we load existing data into a dict.
            print(file)
            fileData = json.load(file)
            print(fileData)
            # Join new_data with file_data inside emp_details
            fileData = {**fileData, **newData}
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(fileData, file, indent = 4)
            file.close()
    else:
        with open(filename, 'w') as file:
            jsonObj = json.dumps(newData)
            file.write(jsonObj)


def ndarray_to_string(ndarray):
    """
    this converts an n-dimensional array to a list

    :param ndarray: n-dimentional numpy array
    :return: list form of nd array
    """
    listRetu = list()
    for subArrIndex in range(0, ndarray.shape[0], 1):
        listRetu.append(list(ndarray[subArrIndex]))
    return(listRetu)


def convert_array_to_string_representation(array):
    """
    converts array to a string representation of the string

    :param array: the array to be converted

    :return: string representing array
    """
    return(" ,".join(str(x) for x in array))
    

#### working with TIFF files
import rasterio
from rasterio.plot import show
import numpy as np
def show_rgb(img_file):
    """
    Displays a satalite image saved as a tif file in visual light (red green blue)
    
    :param img_file: full file name of tif satalite image (may be relative to cwd)
    """
    with rasterio.open(img_file) as src:
        b,g,r,n = src.read()
        
    rgb = np.stack((r,g,b), axis=0)
    show(rgb/rgb.max())


import matplotlib.pyplot as plt
def show_udm_mask(udm2_filename):
    """
    this displays the mask that is housed in the eigth band of the udm2 file

    :param udm2_filename: full file name of udm2 file
    """
    with rasterio.open(udm2_filename) as src:
        fullUdm2 = src.read()
    udmBand8 = fullUdm2[7,:,:] # 0=clear img, 1=space around satalite img, 2=cloud
    plt.imshow(udmBand8, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()


def get_pixel_size(pth, tifFn):
    rawtimestampPsFormat = tifFn[0:15]
    imgMetaDataFile = get_fn_in_dir_with_substrs(pth, [rawtimestampPsFormat, '_metadata.json']) # we are relying on the assumtion that there is only one meta data file for each timestamp
    metadata = read_json_file(os.path.join(pth, imgMetaDataFile))
    return(metadata['properties']['pixel_resolution'])

##### geometry, trig, and matrices
import math
def calculate_intersect_point(dist, transect):
    """
    calculate a point on a line using a distance a long the line and the line
    Equation can be found here: https://www.geeksforgeeks.org/find-points-at-a-given-distance-on-a-line-of-given-slope/
    Note no code from the webpage above was used
    
    :param dist: float distance along the transect
    :param transect: [x, y], [x, y]
    """
    
#     print(f" {transect[1][1]} - {transect[0][1]} / {transect[1][0]} - {transect[0][0]}")
    my = transect[1][1] - transect[0][1] # change in y
    mx = transect[1][0] - transect[0][0] # change in x
    # special cases for horizontal and vertical lines
    if my == 0:
        return({'x':transect[0][0] + dist, 'y':transect[0][1]})
    if mx == 0:
        return({'x':transect[0][0], 'y':transect[0][1]+dist})
    m = (my) / (mx) # slope
#     print(m)
    r = math.sqrt(  (1/ (1+m**2) ) )
    
    # if slope is positive but both components are negative:
#     if mx < 0 and my < 0:
#         x = transect[0][0] + ( dist*r)
#         y = transect[0][1] + (( dist * m)*r)
#     else:
    mxpos = mx > 0
    mypos = my > 0
    if (mxpos and mypos) or (mxpos and not mypos):
        x = transect[0][0] + ( dist*r)
        y = transect[0][1] + (( dist * m)*r)
    elif not mxpos and not mypos:
        x = transect[0][0] - ( dist*r)
        y = transect[0][1] - (( dist * m)*r)
    elif not mxpos and mypos:
        x = transect[0][0] - ( dist*r)
        y = transect[0][1] - (( dist * m)*r)

    points = {'x':x, 'y':y}

    return(points)



def add_points(points, min_distance):
    """
    Given a list of x and y coordinates and a min_distance function adds more points directly in between
    """
    min_distance /= 2
    new_points = [points[0]]
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        dist = np.linalg.norm(p2-p1)
        if dist > min_distance:
            num_points = int(np.ceil(dist / min_distance)) - 1
            for j in range(1, num_points+1):
                t = j / (num_points+1)
                new_point = (1-t)*p1 + t*p2
                new_points.append(new_point.round().astype(int))
        new_points.append(p2)
    return np.array(new_points)


def prune_points_dist_from_origin(points, fraction):
    """
    takes a list of points and calculates their euclidian distance from (0,0) and removes the farthest points to make data set size of fraction

    :param oints: 2d np array of coordinates
    :param fraction: how many points we should keep (1/3) would be take half (0.3333333 also would work for that)
    :param returnRemove: if true then we return the removed not the kept

    :return: the pruned points and also the index of them in the original array
    """
    distances = np.sqrt(np.sum(points**2, axis=1))
    num_pruned = int(fraction * len(points))
    sorted_indices = np.argsort(distances)
    pruned_indices = sorted_indices[num_pruned:]
    pruned_points = points[pruned_indices]
    return pruned_points, pruned_indices


def prune_points_return_clippings(points, fraction):
    numPruned = int(fraction*len(points))
    distances = np.sqrt(np.sum(points**2, axis=1))
    sorted_indices = np.argsort(-distances)  # sort in descending order
    pruned_indices = sorted_indices[:numPruned]
    return points[pruned_indices], pruned_indices
    
    
def closest_point(points, refPoint, returnIndex = False):
    """
    This function finds the closest point in a list of points ([x, y]) and returns it or it and the index at which it was found

    :param points: list of points [[x1, y1], [x2, y2]]
    :param refPoint: point that all of the points in the list of points are being compared to [x, y]
    :param returnIndex: boolean if True the function returns the index at which the closest point is in the list as well as the point

    :return: point in points that is closest to refPoint, and possibly the index at which is was found
    """
    closestPoint = None
    closestDistance = None
    index = 0
    for point in points:
        distance = ((point[0] - refPoint[0])**2 + (point[1] - refPoint[1])**2)**0.5
        if closestDistance is None or distance < closestDistance:
            closestPoint = point
            closestDistance = distance
            retuIndex = index
        index += 1
    if returnIndex:
        return closestPoint, retuIndex
    return closestPoint


#### datetime
import datetime as dt

def convert_dt_to_raw_planet_timestamp(ts):
    """
    """
    year = str(ts.year).zfill(4)
    month = str(ts.month).zfill(2)
    day = str(ts.day).zfill(2)
    hour = str(ts.hour).zfill(2)
    minute = str(ts.minute).zfill(2)
    second = str(ts.second).zfill(2)
    
    rawTimestamp = f'{year}{month}{day}_{hour}{minute}{second}'
    return(rawTimestamp)


def convert_planet_raw_timestamp_to_dt(rawTs):
    """
    Converts planet timestamp to datetime instance 

    :param rawTs: string YYYMMDD_hhmmss example 20170101_001349
    """

    return(dt.datetime.strptime(rawTs, '%Y%m%d_%H%M%S'))


def clean_timestamp(rawTimestamp):
    underscoreCount = rawTimestamp.count('_')
    if underscoreCount == 2:
        rawStr = '_'.join(rawTimestamp.split('_')[0:2])
        rawStr = rawStr + '_00'
        timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    else:
        rawStr = '_'.join(rawTimestamp.split('_')[:-1])
        timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    cleanTimestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    return cleanTimestamp








