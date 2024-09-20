######### Script to bulk run cstsi ##########


import os
import datetime as dt
import glob
from osgeo import gdal, osr
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pandas as pd        
import geopandas as gpd
from coastvision import coastvision
from coastvision import supportingFunctions
from coastvision import coastvisionSiteSetup
from coastvision import coastvisionPlots
from coastvision import geospatialTools
from coastvision import coastvisionTides
from coastvision import coastvisionQAQC as QAQC

import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)

### #extra function. put in tools
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

##############################################################################
################                   SETUP                    ##################
##############################################################################


### SITES ################
# region = 'waikiki'
region = 'kauai'
siteids = [8] #old intersect function, BIG buffer
sites = [region+str(id).zfill(4) for id in siteids]
# sites = ['waikikicoregnomask', 'waikikicoregoceanmask']
# sites = os.listdir(os.path.join(os.getcwd(), 'data', region))[1:]

### SHORELINE SETTINGS #################
justShorelineExtract = False # don't look at transects, just extract shoreline
smoothShorelineSigma = 0.1 #0.1
max_dist = 100
plotIntermediateSteps = True
median_limit = 40
intersect_func = 'cs' # crc or cs: ## decide which intersect function to use: coastsats or CRC's (crc:shapely, direct intersect good for embayments, cs: general intersect considers surounding shoreline pts)
modelPklName = 'HI_logisticregression.pkl'

#### TIDAL SETTINGS  #################
path_to_aviso = os.path.join(os.getcwd(), 'aviso-fes-main', 'data', 'fes2014')
path_to_buffer = os.path.join(os.getcwd(), 'historical_data', "coastline_buffer_1mi.geojson")
reference_elevation = 0 #-0.251  #0 #### MLLW: -0.251  0 
slope = 0.147  #DEFUALT   North:0.147 West:0.149 South:0.150 East:0.095
#beach_slope = [0.11,0.11,0.11,0.13,0.12] #for one site w 5 transects 
plot_tide = False
### QAQC SETTINGS #################
do_QAQC = True
removeFraction=0.12
windowDays=50
median_limit = 40 ### distance in meteres from median to cutoff any shorelines points
#########################
## TEST MODE SETTINGS ###
dataProducts = True
testMode = False  
test_run = False ## an option to run a site and save output in a seperate folder (sitename+name)
name = 'name'
#########################



## Slope
##################################################################
known_slope = {'NARRABEEN': 0.1, 'DUCK': 0.1, 'TORREYPINES': 0.035, 'TRUCVERT': 0.05} 
#for Oahu:
reg_oahu = 'oahu'
north = np.arange(1,22)
east = np.arange(22,58)
south = np.arange(58,90) 
west = np.arange(90,113) 
n_sites = 112
for s in np.arange(1,n_sites+1,1):
    if s in south:
        known_slope[reg_oahu+str(s).zfill(4)] = 0.150
    elif s in east:
        known_slope[reg_oahu+str(s).zfill(4)] = 0.096
    elif s in north:
        known_slope[reg_oahu+str(s).zfill(4)] = 0.147
    elif s in west:
        known_slope[reg_oahu+str(s).zfill(4)] = 0.148

##############################################################################
################                  PROCESS                    #################
##############################################################################
#sites_missing = ['oahu0026', 'oahu0029', 'oahu0037', 'oahu0047', 'oahu0051', 'oahu0054']


for sitename in sites:
    start_time = dt.datetime.now()
    #if sitename in sites_missing: # not sitename=='oahu0009':
        #continue
    # if sitename in ['oahu0014', 'oahu0017']:
    #     continue

    ####################################################################
    ####################################################################
    ## check if output folder exists, if so rename it to last timestamp it was edited
    output_folder = os.path.join(os.getcwd(), 'outputs', region, sitename)
    output_path = os.path.join(os.getcwd(), 'outputs', region, sitename, (sitename + "_intersects_outliersremoved.csv"))
    output_path1 = os.path.join(os.getcwd(), 'outputs', region, sitename, (sitename + "_QAQC_transect_interesections.csv"))
    if os.path.exists(output_path) or os.path.exists(output_path1):
        last_edit_time = os.path.getmtime(output_folder)
        # Convert the timestamp to a readable format
        last_edit_time = dt.datetime.fromtimestamp(last_edit_time)
        datestamp = str(last_edit_time).replace(' ', '_').replace(':','-')[:-10]
        new_folder_name = os.path.join(os.getcwd(), 'outputs', region, f'{sitename}_{datestamp}')
        print(f'outputpath already exists. renaming existing outputfolder to {sitename}_{datestamp}')
        os.rename(output_folder, new_folder_name)

    ####################################################################
    ####################################################################
    
    print(f"{sitename} ***********************")
    siteInputs = dict()
    if testMode:
        testModeDict = {} # we want one for each site

    if sitename in known_slope.keys():
        beach_slope = known_slope[sitename]
    else:
        beach_slope = slope

    # ###########################################################
    # ###########             SET UP RUN              ###########
    # ###########################################################
    df = None
    if not justShorelineExtract:
        transectPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, 
                                    (sitename + "_transects.geojson"))
        transects = coastvision.transects_from_geojson(transectPath)
        siteInputs['transects'] = transects
    
    infoJson = supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 
                                                                        'user_inputs', 
                                                                        region, sitename, 
                                                                        (sitename + '_info.json')))
    siteInputs['infoJson'] = infoJson
    
    # get reference shoreline availible timeperiods
    itemIdsByTimeperiod = dict()
    pixelResolutions = coastvision.get_all_pixel_resolutions(region, sitename) 
    # this isn't perfict because it gets all of the different pixel sizes 
    # and not just the ones for the timeperiod we are on
    
    user_inputsFiles = os.listdir(os.path.join(os.getcwd(), 'user_inputs', region, sitename))
    for fn in user_inputsFiles:
        if fn.startswith(sitename + '_shoreline') and not fn.endswith('shoreline.geojson'):
            # we don't care about the default reference shoreline (starts with sitemane) 
            # we just want the time period ones
            yearRange = fn.split('.geojson')[0].split('_')[-1]
            for pixelResolution in pixelResolutions:
                itemIdsByTimeperiod[yearRange + '_' + str(pixelResolution)] = []
    itemIds = set()
    for file in os.listdir(os.path.join(os.getcwd(), 'data', region, sitename)):
        if file.endswith('metadata.json'):
            # print(file)
            itemId = file.split('_metadata')[0]
            metadataJson = os.path.join(os.getcwd(), 'data', region, sitename, file)
            
            try:
                with open(metadataJson, 'r') as f:
                    data = json.load(f)
                pixel_size = data['properties']['pixel_resolution']
            except ValueError:
                print("JSONDecodeError Occurred and Handled")
                pixel_size = 3
            # NOTE: waiting on pixel size for now
            itemIds.add(itemId)
    n_total = len(itemIds)
    # print('itemId lists')
    itemIdsByTimeperiod = coastvision.sort_item_ids_by_timeperiod(region, sitename, itemIds)
    # print(itemIdsByTimeperiod)

    
    ###########################################################
    ###########           RUN COASTVISION           ###########
    ###########################################################
    intersectionDict = {}
    
    # run CoastVision on items that fall into timeperiods of timeperiod specific shorelines 
    # (e.g. 2020-2029_<sitename>_shoreline.geojson)
    n=0
    for refSlFn, itemIds in itemIdsByTimeperiod.items():
        if len(itemIds) > 0:
            # print(refSlFn)
            timeperiod = coastvision.get_timeperiod_from_ref_sl(refSlFn) # if the timeperiod is none because this is the default ref sl create-shoreline_buffer will handle it
            # reference shoreline buffer for site (for timeperiod)
            randomFileUsedToGenerateSlBuff = os.path.join(os.getcwd(), 'data', 
                                                          region, sitename,
                                                          f'{itemIds[0]}_3B_AnalyticMS_toar_clip.tif')
            im_ms = geospatialTools.get_im_ms(randomFileUsedToGenerateSlBuff)
            georef = geospatialTools.get_georef(randomFileUsedToGenerateSlBuff)
            siteInputs['shorelinebuff'] = coastvision.create_shoreline_buffer(region, sitename, im_shape=im_ms.shape[0:2], 
                                                                              georef=georef, pixel_size=3,
                                                                              max_dist=max_dist, timeperiod=timeperiod)
            # fig, ax = plt.subplots(figsize=(10,10))
            # ax.imshow(siteInputs['shorelinebuff'], cmap=plt.cm.gray)
            
            for itemID in itemIds:
                n+=1
                print('\r', sitename, round(n/n_total*100,2), 'percent progress', itemID, end='', flush=True)
                # run coastvision
                intersectionDict[itemID] = coastvision.run_coastvision_single(region, sitename, itemID, siteInputs, justShorelineExtract, 
                                                                                smoothShorelineSigma, smoothWindow=None, min_sl_len=30, intersect_func=intersect_func, modelPklName=modelPklName,
                                                                                dataProducts=dataProducts, plotIntermediateSteps=plotIntermediateSteps, testMode=testMode)


    # print(intersectionDict)
    ##########################################################
    ##########      CREATE & SAVE DATAFRAME        ###########
    ##########################################################
    ## CREATE DATAFRAME
    dfCreated = False
    if dfCreated==False and len(intersectionDict) > 0:
        #create df
        df = pd.DataFrame(intersectionDict).T.astype(float)
        #fix timestamp from raw time to datetime
        df.rename(index=lambda x: pd.to_datetime(clean_timestamp(x)), inplace=True) 
        #remove rows with only nan
        df = df.dropna(axis=0, how='all')
        dfCreated = True


    # ###remove overlapping transects
    # transects_gpd = gpd.read_file(transectPath)
    # for n,t in enumerate(transects_gpd.geometry): #for each transect for that site
    #     #### Dummy check if any transects overlap with more than itself. if so, make distance nan
    #     if sum(transects_gpd.intersects(t)) > 1:
    #         df.iloc[:,n] = np.nan
    #print(df)
    # dfCreated = False
    # for rawTimestamp, intersections in intersectionDict.items():
    #     underscoreCount = rawTimestamp.count('_')
    #     if underscoreCount == 2:
    #         rawStr = '_'.join(rawTimestamp.split('_')[0:2])
    #         rawStr = rawStr + '_00'
    #         timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    #     else:
    #         rawStr = '_'.join(rawTimestamp.split('_')[:-1])
    #         timestamp = dt.datetime.strptime(rawStr, "%Y%m%d_%H%M%S_%f")
    #     cleanTimestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    #     print(timestamp)
    #     time2 = '_'.join(rawTimestamp.split('_')[:-1])
    #     print('timestamp2', rawTimestamp, time2)
    #     if not intersections is None:
    #         print(intersections)
    #         if not dfCreated:
    #             df = pd.DataFrame(columns=['timestamp'] + list(intersections.keys()))
    #             list1 = list(intersections.values())
    #             for i in range(len(list1)):
    #                 print(i)
    #                 list1[i] = list1[i][0]
    #             df.loc[len(df)] = [cleanTimestamp] + list1
    #             dfCreated = True
    #         else:
    #             list1 = list(intersections.values())
    #             for i in range(len(list1)):
    #                 list1[i] = list1[i][0]
    #             df.loc[len(df)] = [cleanTimestamp] + list1

    ## CHANGE filename if test run
    if test_run:
        outdir = os.path.join(os.getcwd(), 'outputs', region, f'{sitename}_{name}')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            print('path already exists, adding number')
            randon_n = np.random.randint(low=0,high=100)
            outdir = os.path.join(os.getcwd(), 'outputs', region, f'{sitename}_{name}_{str(randon_n)}')
            os.mkdir(outdir)
    else:
        outdir = os.path.join(os.getcwd(), 'outputs', region, sitename)
    ## WRITE OUTPUT
    if dfCreated:
        df = df.sort_index() # sort index
        if not df.index.is_unique: 
            df = df.groupby(df.index).first()
        df.to_csv(os.path.join(outdir, f'{sitename}_transect_intersections.csv'))
 



    # ###########################################################
    # #########           POSTPROCESS DATA             ##########
    # ###########################################################

    ### MERGE CONTOURS INTO ONE GEOJSON
    coastvision.add_all_contours_to_one_site_geojson(reg=region, sitelist=sitename)

    ### TIDAL CORRECTION   
    coastvisionTides.tidal_correction_site(sitename, region, path_to_buffer, path_to_aviso, 
                                   reference_elevation, plot_tide, beach_slope)
    
  

    #######################################
    ### outlier removal 
    # load data
    outputs = pd.read_csv(os.path.join(os.getcwd(), 'outputs', region, sitename, 
                                       (sitename + '_intersections_tidally_corrected_'+str(reference_elevation)+'m.csv')), index_col=0, parse_dates=True)
    # outputs['timestamp'] = pd.to_datetime(outputs['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # use tool
    qcDf = outputs.copy() 

    ### remove points more than x away from mean
    limit = median_limit
    for col in qcDf.columns:
        median = np.nanmedian(qcDf[col].values)
        qcDf[col][(qcDf[col] < (median - limit)) | (qcDf[col] > (median + limit))] = np.nan

    ##########
    print('before QAQC: shape', outputs.shape, 'number of nans', outputs.iloc[:,1:].isna().sum().sum())
    print('after QAQC: shape', qcDf.shape, 'number of nans', qcDf.iloc[:,1:].isna().sum().sum())

    #plot preliminary qaqc
    fig, ax = plt.subplots(figsize=(20,7), tight_layout=True)
    qcDf.boxplot(ax=ax)
    ax.grid(axis='x', visible=None)
    ax.grid(axis='y', color='black', alpha=0.3, lw =0.3)
    ax.set_title('boxplot of transect variability; %s' %sitename)
    ax.set_ylabel('variability, $m$')
    ax.set_xlabel('transect number')
    plt.xticks(rotation=30)
    path = os.path.join(os.getcwd(), outdir, (sitename + '_boxplot_AfterQAQC.png'))
    plt.savefig(path)

    # save qaqc'ed data
    path = os.path.join(outdir, (sitename + '_intersects_outliersremoved.csv'))
    qcDf.to_csv(path)


    # ### QAQC 
    # # load data
    # outputs = pd.read_csv(os.path.join(os.getcwd(), 'outputs', region, sitename, 
    #                                    (sitename + '_intersections_tidally_corrected_'+str(reference_elevation)+'m.csv')))
    # outputs['timestamp'] = pd.to_datetime(outputs['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # # use tool
    # qcDf = QAQC.QAQC_tool(outputs, region, sitename, removeFraction=removeFraction, windowDays=windowDays)
    # ## print changes:
    # print('before QAQC: shape', outputs.shape, 'number of nans', outputs.iloc[:,1:].isna().sum().sum())
    # print('after QAQC: shape', qcDf.shape, 'number of nans', qcDf.iloc[:,1:].isna().sum().sum())

    # #plot preliminary qaqc
    # fig, ax = plt.subplots(figsize=(20,7), tight_layout=True)
    # qcDf.iloc[:,1:].boxplot(ax=ax)
    # ax.grid(axis='x', visible=None)
    # ax.grid(axis='y', color='black', alpha=0.3, lw =0.3)
    # ax.set_title('boxplot of transect variability; %s' %sitename)
    # ax.set_ylabel('variability, $m$')
    # ax.set_xlabel('transect number')
    # plt.xticks(rotation=30)
    # path = os.path.join(os.getcwd(), outdir, (sitename + '_boxplot_AfterQAQC.png'))
    # plt.savefig(path)

    # ### remove points more than x away from mean
    # limit = median_limit
    # for col in qcDf.columns[1:]:
    #     median = np.nanmedian(qcDf[col].values)
    #     qcDf[col][qcDf[col] < (median-limit)] = np.nan
    #     qcDf[col][qcDf[col] > (median+limit)] = np.nan

    # # save qaqc'ed data
    # path = os.path.join(outdir, (sitename + '_QAQC_transect_interesections.csv'))
    # qcDf.to_csv(path)

    ##################################################################
    #save plots
    fig, ax = plt.subplots(figsize=(20,7), tight_layout=True)
    qcDf.iloc[:,1:].boxplot(ax=ax)
    ax.grid(axis='x', visible=None)
    ax.grid(axis='y', color='black', alpha=0.3, lw =0.3)
    ax.set_title('boxplot of transect variability; %s' %sitename)
    ax.set_ylabel('variability, $m$')
    ax.set_xlabel('transect number')
    plt.xticks(rotation=30)
    path = os.path.join(os.getcwd(), outdir, (sitename + '_boxplot_AfterQAQC_after.png'))
    plt.savefig(path)


    ###################################################################
    ##########  IF YOU'RE *ONLY* SAVING PLOTS, NO PROCESSING      #####
    ###################################################################
    # outputs = pd.read_csv(os.path.join(os.getcwd(), 'outputs', region, sitename, 
    #                                    (sitename + '_intersections_tidally_corrected_'+str(reference_elevation)+'m.csv')))
    # outputs['timestamp'] = pd.to_datetime(outputs['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

    # qcDf_path = os.path.join(outdir, (sitename + '_QAQC_transect_interesections.csv'))
    # qcDf = pd.read_csv(qcDf_path, parse_dates=True, index_col=0)
    # # qcDf.index=qcDf.iloc[:,0]
    # print('before QAQC: shape', outputs.shape, 'number of nans', outputs.iloc[:,1:].isna().sum().sum())
    # print('after QAQC: shape', qcDf.shape, 'number of nans', qcDf.iloc[:,1:].isna().sum().sum())


    ##########################################################
    ##########            GENERATE PLOTS           ###########
    ##########################################################

    # Before/after QAQC
    transect = 4
    if len(df.columns)-1 <= transect:
        continue

    fig, (ax1) = plt.subplots(1,1, figsize=(10,5))
    ax1.scatter(outputs.index, outputs.iloc[:,transect], s=15, facecolor='None', edgecolor='black', label = 'raw')
    ax1.scatter(qcDf.index, qcDf.iloc[:,transect], s=3, facecolor='None', edgecolor='red', label='after QAQC')
    ax1.set_title('Before/After QAQC; tranect %s' %transect)
    ax1.set_ylabel('shoreline position, $m$')
    ax1.set_xlabel('time, year')
    # plt.legend()
    path = os.path.join(outdir, (sitename+"_before_after_QAQC.png"))
    plt.savefig(path)
    plt.close()

    #plot transect variability
    fig, ax = plt.subplots(figsize=(20,7), tight_layout=True)
    try:
        xaxis = (qcDf.columns).astype(int)
    except:
        xaxis = np.arange(1,len(qcDf.columns)+1)
    # plot all lines 
    for i in range(len(qcDf)):
        ax.plot(xaxis, qcDf.iloc[i,:], lw=0.15,  ls='--', color = 'black', alpha=0.3)
    # plot main line
    ax.plot(xaxis, qcDf.iloc[:,:].mean(), lw = 2, ls='dotted', color='red')
    ax.scatter(xaxis, qcDf.iloc[:,:].mean(), s=20, color='red')
    # plt.xticks(np.arange(0,len(xaxis),5))
    plt.xticks(range(len(qcDf.columns)), qcDf.columns, size='small')
    plt.ylim((0,250))
    ax.set_title('transect variability; %s; transect %s' %(sitename,str(transect)))
    ax.set_ylabel('shoreline position, $m$')
    ax.set_xlabel('transect number')
    plt.xticks(rotation=30)
    path = os.path.join(outdir, (sitename+"transect_variability.png"))
    plt.savefig(path)


    ###########################################################
    ###########      SAVE SETTINGS IN TXT FILE       ##########
    ###########################################################  
    ########## Get info and settings 
    date = siteInputs['infoJson']['date_range']
    beach = siteInputs['infoJson']['beach_type']
    epsg = siteInputs['infoJson']['epsg']
    with open(os.path.join(outdir, 'settings.txt'), 'w') as f:
        f.write(f'\n Settings \n dataProducts: {dataProducts} \n plotIntermediateSteps: {plotIntermediateSteps} \n testMode: {testMode}  \n justShorelineExtract: {justShorelineExtract} \n smoothShorelineSigma: {smoothShorelineSigma} \n max_dist: {max_dist} \n intersect_funcion: {intersect_func} \n dates: {date} \n epsg: {epsg} \n beach type: {beach}  \n  modelPklName: {modelPklName}')
    
    elapsed = dt.datetime.now() - start_time
    print('DONE with', sitename, 'TOOK:', elapsed, 'to complete')

print('\n\n done \n\n')