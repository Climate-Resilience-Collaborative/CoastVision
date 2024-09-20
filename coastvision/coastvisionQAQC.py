"""
This script facilitates QAQC (automated and manual) of CoastVision outputs

Author: Joel Nicolow Climate Resiliance Collabrative, School of Ocean and Earth Science and Technology (January, 30 2023)

"""

from coastvision import coastvision
from coastvision import supportingFunctions
from coastvision import geospatialTools
from coastvision.classifier import data_annotation_tool

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt




#### helper functions
def remove_points_outside_of_MAD(df, cleaned=True):
    """
    this function removes points that are outside of one MAD from the median

    :param df: df with timestamp column and column for each transect where each cell represents the distance from the start of the transect that the intersection is at
    :param cleaned: boolean if true the returned df will have points outside the MAD removed if false they will just be marked as invalid
    """

    if not cleaned:
        # just to save some space because these data frames could be really big
        resultsDf = df.copy()
    cleanedDf = df.copy()

    for colI in range(1, len(df.columns), 1):
        # starting from one because we skip the index(timestamp) col
        col = df.columns[colI]
        if col != 'timestamp':
            # just a double check
            mad = df[col].mad() # median absolute deviation (less affected by outliers than std and variance)
            # mean = df[col].mean()
            median = df[col].median()

            results = []
            for row in range(0, len(df), 1):
                if df.iat[row, colI] < median - mad or df.iat[row, colI] > median + mad:
                    results.append(0)
                    cleanedDf.iat[row, colI] = None # remove points that are outside of one std for the cleanedDf 
                else:
                    results.append(1)

            if not cleaned:
                resultsDf[f'results_{col}'] = results

    if cleaned:
        return cleanedDf
    else:
        return resultsDf


import geojson
def save_sl_ref_intersection_with_transects(filePath=None, region=None, sitename=None, year=None, timeperiod=None, save=True):
    """
    This function computes the intersection the reference shoreline has with the transects and then saves the along distance for each transect as a csv.
    The along distance is how far along (starting from the first coordinate of the transect) the reference shoreline intersects it at

    :param region: String the region name
    :param sitename: String the site name
    :param year: int what year you want a reference shoreline from (they are in timeperiod, default None)
    :param timeperiod: list [[2017, 2023], [1, 6]] example that looks for timestamp with range 2017-2023 with months jan-June (if you want whole year do [1, 6], default None)
    """
    if filePath is None and not region is None and not sitename is None:
        slFn = coastvision.get_ref_sl_fn(region, sitename, year=year, timeperiod=timeperiod) 
        slBuffPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, slFn)
        slBuffPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, slFn)
    else:
        slBuffPath = filePath
        *_, region, sitename, _ = filePath.split(os.path.sep) # extract regiuon and sitename from filepath

    # if not os.path.exists(slBuffPath):
    #     return(None) # there is no reference shoreline file so we cannot get the reference shoreline

    with open(slBuffPath, 'r') as f:
        ref_sl_info = geojson.load(f)
    ref_sl_world = np.array(ref_sl_info['features'][0]['geometry']['coordinates'])

    # get transects
    transFn = os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_transects.geojson')
    transects = coastvision.transects_from_geojson(transFn)

    # for some reason, sometimes ref_sl_world shape is 1d, but below code expect 2d. make 2d and pivot
    if ref_sl_world.ndim == 1:
        ref_sl_world = ref_sl_world.reshape(1, -1)

    # get transect intersections
    intersections = coastvision.transect_intersect_single(along_dist=25, contour=ref_sl_world, transects=transects)
    intersectionsDf = pd.DataFrame(intersections)
    if not timeperiod is None and type(timeperiod) != int:
        fName = f'reference_shoreline_transect_intersections_{timeperiod[0][0]}-{timeperiod[0][1]}_{timeperiod[1][0]:02d}-{timeperiod[1][1]:02d}.csv'
    else:
        if year is None:
            year = timeperiod
            fName = f'reference_shoreline_transect_intersections_{year}.csv'
    intersectionsDf.insert(0, 'name', os.path.basename(slBuffPath))
    if save:
        intersectionsDf.to_csv(os.path.join(os.getcwd(), 'user_inputs', region, sitename, fName))
    
    return intersectionsDf



#### unsupervised (ML)
# from coastvision import dataSelectionTools
from sklearn import linear_model
from tqdm import tqdm

# def feature_extraction_ref_sl(cleanedDf, region, sitename, furthestPoint=40, acceptableChange=5, acceptableChangeDailyRate=0.5):
#     """
#     this function extracts features for each data point which are used when performing QAQC (unsupervised QAQC) 
#     This specifically uses the reference shoreline (unlike feature_extraction() which does not)
#     """
#     dfdict = {} # implemented with a heap cuz dis python
#     for row in tqdm(range(0, len(cleanedDf), 1)):
#         # check timestamp and see if you need to create a new 



def feature_extraction(cleanedDf, furthestPoint=40, acceptableChange=5, acceptableChangeDailyRate=0.5):
    """
    :param cleanedDf: 

    :return: dictionary of dimensionality df for each column (key column)
    """
    dfDict = {}
    # dataSelectionTools.get_reference_sl() THIS IS WHERE WE GET THE REFERENCE SHORELONE
    for col in cleanedDf.columns:
        if col != 'timestamp':
            # print(col)
            dimensionsDf = pd.DataFrame(columns=['timestamp', 'distFromLinearModel', 'distFromMedian', 'distFromLocalMedian'])
            median = cleanedDf[str(col)].median()

            for row in tqdm(range(0, len(cleanedDf), 1)):
                if not pd.isnull(cleanedDf[col].iloc[row]):
                    # ncheck = 5 # how many other data points should be looked at in each direction
                    rowDict = {}
                    df = pd.DataFrame(columns=['timestamp', 'val', 'thisRow'])
                    currTimestamp = cleanedDf['timestamp'].iloc[row]

                    compRow = row
                    dayDifference = 0
                    while compRow > 0 and dayDifference <= furthestPoint:
                        compRow -= 1 # decrement untill you get more than 40 days away
                        val = cleanedDf[col].iloc[compRow]
                        if not pd.isnull(val):
                            dayDifference = (cleanedDf['timestamp'].iloc[compRow] - currTimestamp).days
                            if dayDifference <= furthestPoint:
                                rowDict = {'timestamp':[cleanedDf['timestamp'].iloc[compRow]], 'val':[val], 'thisRow':[False]}
                                df = pd.concat([df, pd.DataFrame.from_dict(rowDict)], axis=0)


                    val = cleanedDf[col].iloc[row]
                    rowDict = {'timestamp':[cleanedDf['timestamp'].iloc[row]], 'val':[val], 'thisRow':[True]}
                    df = pd.concat([df, pd.DataFrame.from_dict(rowDict)], axis=0)


                    compRow = row
                    dayDifference = 0
                    while compRow+1 < len(cleanedDf) and dayDifference <= furthestPoint:
                        compRow += 1 # increment untill you get more than 40 days away
                        val = cleanedDf[col].iloc[compRow]
                        if not pd.isnull(val):
                            dayDifference = (cleanedDf['timestamp'].iloc[compRow] - currTimestamp).days
                            if dayDifference <= furthestPoint:
                                rowDict = {'timestamp':[cleanedDf['timestamp'].iloc[compRow]], 'val':[val], 'thisRow':[False]}
                                df = pd.concat([df, pd.DataFrame.from_dict(rowDict)], axis=0)



                    if len(df) <=1:
                        # this means that there is no close point chronologically for this data point
                        df['distanceFromLinearModel'] = 0 # squared vs abs because it penalizes outliers more
                        distFromLM = 0

                        localPoints = 0
                        localMedian = 0
                        #localStd = localPoints['val'].std()

                        distFromLocalMedian = 0
                    else:
                        X = df.timestamp.values
                        y = df.val.values
                        X = X.reshape(len(df), 1)
                        y = y.reshape(len(df), 1)

                        regr = linear_model.LinearRegression()
                        regr.fit(X, y)

                        df['distanceFromLinearModel'] = ( y - regr.predict(X) )**2 # squared vs abs because it penalizes outliers more
                        distFromLM = df.loc[df['thisRow'] == True]['distanceFromLinearModel'][0]

                        localPoints = df.loc[df['thisRow'] == False]
                        localMedian = localPoints['val'].median()
                        #localStd = localPoints['val'].std()

                        distFromLocalMedian = (cleanedDf[col].iloc[row] - localMedian)**2

                    # change from adjacent points
                    df.reset_index(inplace = True) # before this all of the indeces are zeros
                    currRowIndex = df.index[df['thisRow'] == True].tolist()[0]
                    if currRowIndex - 1 >= 0:
                        change = df['val'].iloc[currRowIndex - 1] -  df['val'].iloc[currRowIndex]

                        delta = df['timestamp'].iloc[currRowIndex] - df['timestamp'].iloc[currRowIndex - 1]
                        dayDiff = delta.days # STUB!!!!! NEED TO GET DAYS INCLUDING PARTIAL DAYS LIKE 3.4 NOT 3
                        maxChange = acceptableChange + (acceptableChangeDailyRate*dayDiff)
                        changeValidPre = 0 if abs(change) > maxChange else 1

                        squaredChangeFromPrev = ( change )**2 # using squared to penalize higher changes more strongly
                    else:
                        changeValidPre = 1
                        squaredChangeFromPrev = 0
                    if currRowIndex + 1 < len(df):
                        change = df['val'].iloc[currRowIndex] -  df['val'].iloc[currRowIndex + 1]

                        delta = df['timestamp'].iloc[currRowIndex + 1] - df['timestamp'].iloc[currRowIndex]
                        dayDiff = delta.days # STUB!!!!! NEED TO GET DAYS INCLUDING PARTIAL DAYS LIKE 3.4 NOT 3
                        maxChange = acceptableChange + (acceptableChangeDailyRate*dayDiff)
                        changeValidPost = 0 if abs(change) > maxChange else 1

                        squaredChangeFromPost = ( change )**2 # using squared to penalize higher changes more strongly
                    else: 
                        changeValidPost = 1
                        squaredChangeFromPost = 0



                    distFromMedian = (cleanedDf[col].iloc[row] - median)**2


                    dimensions = {'timestamp':[cleanedDf['timestamp'].iloc[row]],
                                'distFromLinearModel':[distFromLM], 'distFromMedian':[distFromMedian], 
                                'distFromLocalMedian':[distFromLocalMedian],
                                'squaredChangeFromPrev':[squaredChangeFromPrev],
                                'squaredChangeFromPost':[squaredChangeFromPost],
                                'changeValidPre':[changeValidPre],
                                'changeValidPost':[changeValidPost]}
                    dimensionsDf = pd.concat([dimensionsDf, pd.DataFrame.from_dict(dimensions)], axis=0)
                    
            dfDict[col] = dimensionsDf

        
    return(dfDict)


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
def dimensionality_reduction(dimensionsDf, n_components=2):

    X = np.array([ [dimensionsDf['distFromLinearModel'].iloc[0],
                          dimensionsDf['distFromMedian'].iloc[0],
                          dimensionsDf['distFromLocalMedian'].iloc[0],
                          dimensionsDf['squaredChangeFromPrev'].iloc[0],
                          dimensionsDf['squaredChangeFromPost'].iloc[0]
                   ] ])
    for row in range(1, len(dimensionsDf), 1):
        X1 = np.array([ [dimensionsDf['distFromLinearModel'].iloc[row],
                          dimensionsDf['distFromMedian'].iloc[row],
                          dimensionsDf['distFromLocalMedian'].iloc[row],
                          dimensionsDf['squaredChangeFromPrev'].iloc[row],
                          dimensionsDf['squaredChangeFromPost'].iloc[row]
                   ] ])

        X = np.concatenate((X,X1))

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=n_components)
    nD = pca.fit_transform(X)
    return(nD)


# def dimensionality_reduction_transect(dimensionsDf, transect, n_components=2):

#     # drop nas for this specific column

#     X = np.array([ [dimensionsDf[f'{transect}_distance_from_local_median'].iloc[0],
#                           dimensionsDf[f'{transect}_distance_from_local_linear_model'].iloc[0],
#                           dimensionsDf[f'{transect}_z_score_local'].iloc[0],
#                           dimensionsDf[f'{transect}_total_difference_from_local_points'].iloc[0],
#                           dimensionsDf[f'{transect}_difference_from_sl_ref'].iloc[0],
#                    ] ])
#     for row in range(1, len(dimensionsDf), 1):
#         X1 = np.array([ [dimensionsDf[f'{transect}_distance_from_local_median'].iloc[row],
#                           dimensionsDf[f'{transect}_distance_from_local_linear_model'].iloc[row],
#                           dimensionsDf[f'{transect}_z_score_local'].iloc[row],
#                           dimensionsDf[f'{transect}_total_difference_from_local_points'].iloc[row],
#                           dimensionsDf[f'{transect}_difference_from_sl_ref'].iloc[row],
#                    ] ])

#         X = np.concatenate((X,X1))

#     scaler = MinMaxScaler()
#     scaler.fit(X)
#     X = scaler.transform(X)

#     pca = PCA(n_components=n_components)
#     nD = pca.fit_transform(X)
#     return(nD)


def dimensionality_reduction_transect(dimensionsDf, transect, columns, n_components=2):

    # drop nas for this specific column

    X = np.array([ [dimensionsDf[f'{transect}_{col}'].iloc[0] for col in columns] ])
    for row in range(1, len(dimensionsDf), 1):
        X1 = np.array([ [dimensionsDf[f'{transect}_{col}'].iloc[row] for col in columns] ])

        X = np.concatenate((X,X1))

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(n_components=n_components)
    nD = pca.fit_transform(X)
    return(nD)



def unsupervised_QAQC(df, dfDict, removePercent=0.20):
    """
    This function itterates through a df dictionary which includes the dimesions dfs for each column created by feature_extraction()

    :param df: pandas df 
    :param dfDict: dictionary of pandas data frames (need to satisfy the dimensionality_reduction() function)
    :param removeFraction: percent of points that should be "pruned"

    :return: pandas df like the cleanedDf returned by feature_extraction() but with a column for each transect _valid that is true if it passed the QAQC
    """
    cleanedDf2 = df.copy()
    for transectName, dimensionsDf in dfDict.items():
        twod = dimensionality_reduction(dimensionsDf, 2)
        # medx = np.median(twod[:,0])
        # medy = np.median(twod[:,1])

        # madx = supportingFunctions.mad(twod[:,0]) * 2 # multiply by 2 to make the model less restrictive
        # mady = supportingFunctions.mad(twod[:,1]) * 2
        # stdx = np.std(twod[:,0])
        # stdy = np.std(twod[:,1])


        # plt.scatter(twod[:,0], twod[:,1])
        # x1, y1 = [medx+stdx, medx+stdx], [medy-stdy, medy+stdy]
        # x2, y2 = [medx-stdx, medx-stdx], [medy-stdy, medy+stdy]
        # x3, y3 = [medx+stdx, medx-stdx], [medy-stdy, medy-stdy]
        # x4, y4 = [medx+stdx, medx-stdx], [medy+stdy, medy+stdy]
        # plt.plot(x1, y1, x2, y2, x3, y3, x4, y4)

        _, badIndeces = supportingFunctions.prune_points_return_clippings(twod, removePercent) # trim off 20 percent
        print('=====================================================================')
        # print(twod.shape)
        # print(len(badIndeces))


        # goodPoints = []
        # badIndeces = []
        # for i in range(0, len(twod), 1):
            
        #     if dimensionsDf['changeValidPre'].iloc[i] == 0 or dimensionsDf['changeValidPost'].iloc[i] == 0:
        #         # if points break max change guidlines then they are automatically invalid
        #         badIndeces.append(i)
        #     else:

        #         x = twod[i,0]
        #         y = twod[i,1]   
        #         # also add to bad if it breaks our max change rate (changeValidPre or post)
        #         if x <= medx + madx and x >= medx - madx and y <= medy + mady and y >= medy -mady:
        #             goodPoints.append([x, y])
        #         else:
        #             # bad points
        #             badIndeces.append(i)
        # npGoodPoints = np.array(goodPoints)
        # plt.scatter(npGoodPoints[:,0], npGoodPoints[:,1])
        # plt.scatter(np.median(twod[:,0]), np.median(twod[:,1]) 
        # plt.show()

        if not f'{transectName}_valid' in df.columns:  
            cleanedDf2.insert(cleanedDf2.columns.tolist().index(transectName)+1, f'{transectName}_valid', True)

        for j in range(0, len(badIndeces), 1):
            idx = cleanedDf2[cleanedDf2.timestamp == dimensionsDf.iloc[badIndeces].iloc[j]['timestamp']].index.tolist()[0]
            cleanedDf2[f'{transectName}_valid'].iat[idx] = False

    return(cleanedDf2)



def remove_invalid_observations(cleanedDf):
    """
    This function gives a value NaN to all of the cells where the _valid column equals False

    :param cleanedDf: pd df that has a timestamp column and then two columns per transect (one with the transect name and intersection data and one with <transectName>_valid and a boolean for if it is valid according to QAQC)
    
    :return: pd df timestamp column and columns for each transect (with the distance along the transect at which the intersection was; same as input)
    """
    cleanedDf = cleanedDf.reset_index(drop=True)
    # cleanedDf1 = cleanedDf.copy()
    # filter col names to get name of the 
    transectNames = [i for i in cleanedDf.columns if 'timestamp' not in i and '_valid' not in i]
    for transectName in transectNames:
        for row in range(0, cleanedDf.shape[0], 1):
            if not cleanedDf[f'{transectName}_valid'].iloc[row]:
                # if transectName == 'NA5':
                #     print(cleanedDf[transectName].iat[row])
                #     print(f'transect {transectName}, row {row}')
                #     cleanedDf[transectName].iat[row] = np.NaN
                #     print(cleanedDf.loc[row-2:row+6])
                #     raise KeyError('just testing tings')
                cleanedDf[transectName].iat[row] = np.NaN

    cleanedDf = cleanedDf[[col for col in cleanedDf.columns if '_valid' not in col]] # drop all of the _valid columns
    # cleanedDf = cleanedDf.dropna(subset=transectNames, how='all') # if all of the observations are now null there is no reason to keep the column
    return(cleanedDf)

    

#### hand classification
def hand_QAQC_transect_intersections(tiffImage, transectFn, df, onlyReturnRow = False):
    """
    Displays a planet scope image with the extracted transect intersecs from coastvision. Then allows user to select points as invalid (non selected points will be considered valid)

    """
    transects = coastvision.transects_from_geojson(transectFn)
    georef = geospatialTools.get_georef(tiffImage)
    rawTimestamp = os.path.basename(tiffImage)[0:15]
    # timestamp = dt.datetime.strptime(rawTimestamp, '%Y%m%d_%H%M%S')
    timestamp = supportingFunctions.convert_planet_raw_timestamp_to_dt(rawTimestamp)

    df['timestamp'] =  pd.to_datetime(df['timestamp'])



    if len(df.index[df['timestamp'] == timestamp].tolist()) < 1:
        # this means we dont have data for this timestamp
        return None
    thisImageIndex = df.index[df['timestamp'] == timestamp].tolist()[0]
    thisImageDf = df.loc[df['timestamp'] == timestamp]

    combinedPointsList = []
    intersects = {}
    for transectKey in [item for item in thisImageDf.columns if '_valid' not in item and 'manual_QAQC' not in item]:
        if not transectKey == 'timestamp':
            dist = thisImageDf[transectKey].iloc[0]
            if not np.isnan(dist):
                intersects[transectKey] = supportingFunctions.calculate_intersect_point(dist, transects[transectKey])
                combinedPointsList.append( [intersects[transectKey]['x'], intersects[transectKey]['y']] )

    if len(combinedPointsList) < 1:
        return None
    combinedPointsListPix = geospatialTools.convert_world2pix(combinedPointsList, georef)
    
    pixArr = np.array(combinedPointsListPix[0])
        

    matplotlib.use("Qt5Agg") # needed for popup window
    fig, ax = plt.subplots(figsize=(10,15))
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()

    # Plot classes over RGB
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tiffImage)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    ax.imshow(im_RGB)

    ax.scatter(pixArr[:,0], pixArr[:,1], s=3, color='red')

    # axis limit
    min_x, max_x = min(pixArr[:,0]), max(pixArr[:,0])
    min_y, max_y = min(pixArr[:,1]), max(pixArr[:,1])
    plt.xlim(min_x-40, max_x+40), plt.ylim(max_y+40, min_y-40)
    
    
    ### INTERACTIVE SECCTION
    ax.set_title('Click on or close to the red transect intersections to mark them as invalid\nClick delete to undo last point and enter or escape (exits tool when looping as well) to exit the tool',
                      fontsize=14)
    plt.draw()
    
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            eventsDict['point'] = (event.xdata, event.ydata)
            
    def press(event):
        # store what key was pressed in the dictionary
        eventsDict['pressed'] = event.key

    selected = []
    invalidTransects = []
    while True:
            
        eventsDict = {}

        fig.canvas.draw_idle()      
        
        fig.canvas.mpl_connect('button_press_event', onclick) # check for where user clicks
        fig.canvas.mpl_connect('key_press_event', press) # check for if user pushes escape or unde
        plt.waitforbuttonpress()

        
        if 'point' in eventsDict:
            firstPoint = eventsDict['point']
            selectedPoint, selectedIndex = supportingFunctions.closest_point(pixArr, firstPoint, returnIndex=True)
            invalidTransects.append(selectedIndex)
            selected.append(selectedPoint)
            ax.scatter(selectedPoint[0], selectedPoint[1], s=3, color='#00ff00')
        elif 'pressed' in eventsDict:
            if eventsDict['pressed'] == 'enter':
                ax.clear()
                plt.close(fig)
                break
            elif eventsDict['pressed'] == 'delete':
                invalidTransects.pop()
                ax.scatter(selected[-1][0], selected[-1][1], s=3, color='red') # would be better to remove green plot but just plotting back over it is fine for now
                selected.pop()
            elif eventsDict['pressed'] == 'escape':
                ax.clear()
                plt.close(fig)
                retuDone = True
                break

    # df add columns for if the point is valid
    transectNames =  [item for item in df.columns if '_valid' not in item and 'timestamp' not in item and 'manual_QAQC_check' not in item] # remove timestamp col name so it's just transect
    # print(transectNames)
    i = 0
    j = 0
    while True:
        if not f'{transectNames[i]}_valid' in df.columns:
            # this should control for if the df is already in the correct format
            j += 2
            df.insert(j, f'{transectNames[i]}_valid', True)
        i += 1 # need to add two because we are adding a new column to the df each itteration
        if not i < len(transectNames):
            break
    for tran in invalidTransects:
        df[f'{transectNames[tran]}_valid'].iat[thisImageIndex] = False


    # print(not 'manual_QAQC_check' in df.columns)
    if not 'manual_QAQC_check' in df.columns:
        df['manual_QAQC_check'] = [False for i in range(df.shape[0])]
        
    df[f'manual_QAQC_check'].iat[thisImageIndex] = True # tells us that this image has been QAQC'd
    
    if 'retuDone' in locals():
        # this means that the user wants to specify that they want to completly exit the view (if they are in a forloop it will not continue)
        if onlyReturnRow:
            return(df.loc[df['timestamp'] == timestamp], 'Done') # just return the row for the image we are looking at
        else:
            return(df, 'Done')
    else:
        if onlyReturnRow:
            return(df.loc[df['timestamp'] == timestamp]) # just return the row for the image we are looking at
        else:
            return(df)


#### Vectorized
import glob
def QAQC_tool(df, region, siteName, removeFraction=0.2, windowDays=50):
    fileList = glob.glob(os.path.join(os.getcwd(), 'user_inputs', region, siteName) + '/*shoreline*.geojson')

    # Define the function to be called on each file
    def load_slref_intersections(filePath):
        return save_sl_ref_intersection_with_transects(filePath=filePath, timeperiod=None, save=False)

    dfList = [load_slref_intersections(filePath) for filePath in fileList]

    # Concatenate the DataFrames into a single DataFrame
    refSlIntersectionDf = pd.concat(dfList, ignore_index=True)
    df = df.copy() # so you are not making inplace changes to the df passed in
    df['timestamp'] =  pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df.set_index('timestamp', inplace=True)

    # define window size
    window_size = pd.Timedelta(days=windowDays)

    # define function to compute difference between center value and mean of surrounding values
    def distance_from_local_median(arr):
        """
        This function is built to be use in the pandas apply() function while rolling() through a df column
        
        :param arr: window elements (notice last element in widow is the one that the window was set on)
        """    
        return ( arr[ arr.index.tolist()[(len(arr)-1)] ] - np.median(arr[np.logical_not(np.isnan(arr))]) )**2 # difference between center point and local median (squared)


    def z_score_local(arr):
        """
        Calculates the z score for a point 

        :param arr: window elements (notice last element in widow is the one that the window was set on)
        :return; z score of last element in arr (the current element)
        """    
        return ( arr[ arr.index.tolist()[(len(arr)-1)] ] - np.median(arr[np.logical_not(np.isnan(arr))]) ) / np.std(arr[np.logical_not(np.isnan(arr))]) 


    from sklearn import linear_model
    import matplotlib.pyplot as plt
    def distance_from_local_linear_model(arr):
        """
        This function computes a linear model for the points in arr and sees how far away the current point is from that

        :param arr: window elements (notice last element in widow is the one that the window was set on)
        """
        if len(arr) > 2:

            df = pd.DataFrame({'timestamp': arr.index.tolist(), 'val':arr.values})
            df.dropna(inplace=True)
            if len(df) > 2:
                X = df.timestamp.values
                y = df.val.values
                X = X.reshape(len(df), 1)
                y = y.reshape(len(df), 1)

                regr = linear_model.LinearRegression()
                regr.fit(X, y)

                # plt.scatter(X, y)
                # plt.plot(X, regr.predict(X.reshape(-1, 1)), color='red')
                # plt.show()

                # not sure if this is working correctly, but need to convert datetime of interest to float
                x_pred = X[-1].astype('datetime64[s]').view('int64') / 1e9
                x_pred = float(x_pred)
                return (regr.predict(np.array(x_pred).reshape(-1,1)) - y[-1] )**2 # could also use absolute value but we are using squared to penalize more severely
                ## LOCAL RETURN FUNCTION
                # return (regr.predict((np.array([X[-1]]).reshape(-1, 1))) - y[-1] )**2 # could also use absolute value but we are using squared to penalize more severely
            return 0 # there are only 1 or 2 points

        # if length is 2 then the linear model would just be a str8 line between the points this the error would be 0 anyway.
        # also cannot do it for just one point
        return 0


    def total_difference_from_local_points(arr):
        """
        This function calculates the differences between each point and the current point and then returns the average distance

        :param arr: window elements (notice last element in widow is the one that the window was set on)
        """
        return (arr[:arr.index.tolist()[(len(arr)-1)]] - arr[ arr.index.tolist()[(len(arr)-1)] ]).mean() / (len(arr)-1)



    def difference_from_sl_ref(arr, col):
        """
        This function takes the value for the [appropriate] reference shoreline (at the transect) and calculates the difference.
        """
        timestamp = arr.index.tolist()[(len(arr)-1)]
        refSlName = coastvision.get_ref_sl_fn(region=region, sitename=siteName, yearMonth =[timestamp.year]) # for now just searching by year this is not the best
        refValue = refSlIntersectionDf.loc[refSlIntersectionDf['name'] == refSlName, col].iloc[0]
        return arr[arr.index.tolist()[(len(arr)-1)]] - refValue



    QAQCFunctions = [distance_from_local_median, distance_from_local_linear_model, z_score_local, total_difference_from_local_points, difference_from_sl_ref]

    df.sort_index(inplace=True) # make sure that df is sorted so can use apply and rolling from pd

    # apply function using rolling window
    pd.set_option('display.max_rows', None)
    def col_QAQC_caculator(df, col, calcFunction):
        if len(inspect.signature(calcFunction).parameters) == 1:
            return df[col].rolling(window=window_size, center=False, closed='both').apply(lambda x: calcFunction(x), raw=False)#, result_type='expand') # need center to be true or you will get offset    
        return df[col].rolling(window=window_size, center=False, closed='both').apply(lambda x: calcFunction(x, col), raw=False)#, result_type='expand') # need center to be true or you will get offset

    import inspect
    resultDfDict = {}
    for func in QAQCFunctions:
        print(func)
        result = df.apply(lambda col: col_QAQC_caculator(df, col.name, func) if col.dtype == 'float64' else col)
        resultDfDict[func.__name__] = result

    combinedDf = pd.concat([df.add_suffix(f'_{key}') for key, df in resultDfDict.items()], axis=1)

    for transect in df.columns:
        transectCols = combinedDf.filter(like=transect).columns
        # Subset the dataframe with selected columns and drop rows containing NaN values
        df1 = combinedDf[transectCols].dropna() # if there is an NA it should be for every column for that transect so this is okay
        if df1.empty: ## abm add, so ignore if all data from this date are nan
            continue
        if df1.shape[0] < 2: ## abm add, so ignore if all data from this date are nan
            continue
        # print(df1)

        twoD = dimensionality_reduction_transect(df1, transect, columns=[func.__name__ for func in QAQCFunctions])
        _, badIndeces = supportingFunctions.prune_points_return_clippings(twoD, removeFraction) # trim off 20 percent
        badPositions = df1.iloc[badIndeces].index.values

        # Create a new column with boolean values indicating validity
        df[f'isvalid_{transect}'] = True
        df.loc[badPositions, f'isvalid_{transect}'] = False

    for col in df.columns:
        if col.startswith('isvalid_'):
            transect = col.split('_')[1] # get the transect number from column name
            invalid_rows = df.loc[df[col]==False].index # get index of invalid rows
            df.loc[invalid_rows, transect] = np.nan # set corresponding values to NaN
    df.drop([col for col in df.columns if col.startswith('isvalid_')], axis=1, inplace=True)
    return df
