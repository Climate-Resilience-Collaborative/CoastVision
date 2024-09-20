##########
### Add tidal correction to shoreline intersections using global AVISO FES tidal model
### CRC Anna Mikkelsen February 2023
##########

import os
from datetime import datetime, timedelta
import pytz
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pyfes
from shapely.geometry import Point, shape
from shapely.ops import nearest_points
import math
import geopandas as gpd


## FUNCTION FROM COASTSAT_SLOPE ###
def compute_tide_dates(coords,dates,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates (using a dates vector)'
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
    load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100

    return tide_level

####### next 2 functions ONLY FOR PLOTTING NOT USED FOR TIDAL CORRECTION #######
# from coastsat slope
def compute_tide(coords,date_range,time_step,ocean_tide,load_tide):
    'compute time-series of water level for a location and dates using a time_step'
    # list of datetimes (every timestep)
    dates = []
    date = date_range[0]
    while date <= date_range[1]:
        dates.append(date)
        date = date + timedelta(seconds=time_step)
    # convert list of datetimes to numpy dates
    dates_np = np.empty((len(dates),), dtype='datetime64[us]')
    for i,date in enumerate(dates):
        dates_np[i] = datetime(date.year,date.month,date.day,date.hour,date.minute,date.second)
    lons = coords[0]*np.ones(len(dates))
    lats = coords[1]*np.ones(len(dates))
    # compute heights for ocean tide and loadings
    ocean_short, ocean_long, min_points = ocean_tide.calculate(lons, lats, dates_np)
    load_short, load_long, min_points = load_tide.calculate(lons, lats, dates_np)
    # sum up all components and convert from cm to m
    tide_level = (ocean_short + ocean_long + load_short + load_long)/100

    return dates, tide_level

### CRC plotting function ####
def plot_tides(sitename, coords, dates_sat, tide_sat, ocean_tide,load_tide, daterange=[2000,2023], time_step= (15*60)):
    """
    daterange: range for plotting, default 2000-present
    time_step: interval to plot in seconds; defualt 15min (15*60)
    """
    # get tide time-series with 15 minutes intervals (only for plotting purposes)
    daterange = [pytz.utc.localize(datetime(daterange[0],1,1)),
                pytz.utc.localize(datetime(daterange[1],1,1))]

    dates_fes, tide_fes = compute_tide(coords,daterange,time_step,ocean_tide,load_tide)

    ##### PLOT TIDES ######### 
    # plot tide time-series
    fig, ax = plt.subplots(1,1,figsize=(12,3), tight_layout=True)
    ax.set_title('Sub-sampled tide levels')
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.plot(dates_fes, tide_fes, '-', color='black', lw=0.5, alpha=0.6)
    ax.plot(dates_sat, tide_sat, '-o', color='k', ms=4, mfc='w',lw=1)
    ax.set_ylabel('tide level [m]')
    path_save = os.path.join(os.getcwd(), 'outputs', region, sitename, (sitename + '_tides.png'))
    plt.savefig(path_save)


###########################
## CRC Tidal correction ###
def tidal_correction_site(sitename, region, path_to_buffer, path_to_aviso, intersection_df=None, reference_elevation=0, plot_tide=False, beach_slope=0.1):
    """
    Inputs
        sitename: 
            str. site to correct
        region: 
            str. region of site
        path_to_aviso: 
            str (path) point to the directory where aviso .nc model is saved (you will also have to change directories within the model after download when setting up)
        path_to_buffer: 
            str (path) point to the directory where the offshore shoreline buffer exists  (e.g. shapefile line ~ 1 mile offshore)
        plot_tide: 
            if true, saves a plot of tide level and tides at time of each image
        reference_elevation: 
            int. the reference elevation (Mean sea level=0) to correct intersection to. default 0m
        beach_slope: 
            int or list. choose a generic beach slope. defualt 0.1. if a list is passed and len(list) = number of transects, specific beach slope with be used for each transect
    outputs
        saves a csv with corrected shorelines intersections
        saves a csv with tide level at the time of each image
    """

    # Load generic shoreline buffer (buffered about 1mi offshore - min location offshore where to extract tide data )
    #filepath = os.path.join(os.getcwd(), "cosmos", "coastline_buffer_1mi.geojson")
    filepath = path_to_buffer
    with open(filepath, 'r', encoding="utf8") as f:
        coastline = json.load(f)
    l = shape(coastline["features"][0]["geometry"])

    # Read in polygon
    filepath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename+'_polygon.geojson'))
    with open(filepath, 'r', encoding="utf8") as f:
        data = json.load(f)
        
    # calculate coordinates for offshore point where you extract tide information 
    s = shape(data['features'][0]['geometry'])
    centroid = Point(s.centroid.x, s.centroid.y)
    p1, p2 = nearest_points(l, centroid)
    # coordinates of the location (always select a point 1-2km offshore from the beach)
    # if the model returns NaNs, change the location of your point further offshore.
    coords = [p1.x, p1.y]

    ###### LOAD SHORELINE INTERSECTIONS ########
    if intersection_df is None:
        csv = os.path.join(os.getcwd(), 'outputs', region, sitename, (sitename + '_transect_intersections.csv'))
        try:
            intersections = pd.read_csv(csv, index_col=0, parse_dates=True)
            dates_sat = intersections.index
        except FileNotFoundError: 
            print('no transect_intersections found for site', sitename)
            return
    else:
        intersections = intersection_df # it can be passed or can be loaded

    ###### GET TIDE DATA  ########
    # point to the folder where you downloaded the .nc files
    filepath = path_to_aviso 
    config_ocean = os.path.join(filepath, 'ocean_tide.ini') # change to ocean_tide.ini
    config_load =  os.path.join(filepath, 'load_tide.ini')  # change to load_tide.ini
    ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
    load_tide = pyfes.Handler("radial", "io", config_load)
    tide_sat = compute_tide_dates(coords, dates_sat, ocean_tide, load_tide)

    ###### SAVE TIDE DATA ########
    #save tide data 
    tides = pd.DataFrame(tide_sat, columns=['tide'])
    tides.index = dates_sat
    tides.index.name = 'dates'
    path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_tides.csv'))
    tides.to_csv(path) 

    ###### TIDAL CORRECTION ########
    corrected = intersections.copy()
    # print(corrected)
    #if there are specific sloes for each transect
    if isinstance(beach_slope, list) and len(beach_slope) == len(intersections.columns):
        print('individual beach slopes detected')
        for i in range(len(beach_slope)):
            tr = intersections.columns[i]
            print('correcting transect', tr)
            tides['correction'] = (tides['tide']-reference_elevation)/beach_slope[i]
            # correct for each day
            for date in tides.index:
                corrected.loc[date,tr] = corrected.loc[date,tr] + tides.loc[date,'correction']

    else: #use one slope for all
        tides['correction'] = (tides['tide']-reference_elevation)/beach_slope
        # correct for each day
        for date in tides.index:
            corrected.loc[date] = corrected.loc[date] + tides.loc[date,'correction']
    
    #### save output ####
    # print('saving tidally corrected csv')
    corrected.insert(0, 'timestamp', corrected.index)
    corrected.index = np.arange(0,len(corrected.index),1)

    path_corrected = os.path.join(os.getcwd(), 'outputs', region, sitename,
        (f'{sitename}_intersections_tidally_corrected_{reference_elevation}m.csv'))
    corrected.to_csv(path_corrected, index= False)

    if plot_tide:
        tide_sat = tides['tide']
        plot_tides(sitename, coords, dates_sat, tide_sat, ocean_tide,load_tide, daterange=[2016,2023], time_step= (15*60))

    return corrected


############# Tidally correct for a whole region ################
# ------------------------------


def tidal_correction_region(region, path_to_buffer, path_to_aviso, reference_elevation=0, plot_tide=False, beach_slope=0.1):
    
    sites = os.listdir(os.path.join(os.getcwd(), 'outputs', region))
    for n, site in enumerate(sites):
        p = (n+1)/len(sites)*100
        print(site, '  -  ', p, '%', flush=True, end="\r")
        tidal_correction_site(site, region, path_to_buffer, path_to_aviso, reference_elevation, plot_tide, beach_slope)

# -----------------------------




#####################################################################
#################             WAVE CORRECTIONS            ###########
#####################################################################

def wave_correction(sitename, region, waves, wave_coord, beach_slope, reference_elevation=0, default_slope=0.15):
    ##### LOAD WAVE DATA ##########
    # wave_path = os.path.join(os.path.dirname(os.getcwd()), 'inputs', 'waves_all.csv')
    # waves = pd.read_csv(wave_path, parse_dates=True, index_col=0)

    # wave_coord_path = os.path.join(os.path.dirname(os.getcwd()), 'inputs', 'wave_coordinate.geojson')
    # wave_coord = gpd.read_file(wave_coord_path)

    ###### CALCULATE WAVE SETUP ###########
    waves['L'] = (9.81 * waves['T']**2) / (2*math.pi)
    waves['setup'] = 0.016 * (waves['Hs']*waves['L'])**0.5
    # waves['setup'] = 0.016 * (waves['Hs']*waves['L'])**0.5 + 0.15
    ## setup = 0.35 * slope * (Hs*L)**0.5 ### if using this do it under wave_site
    waves = waves.dropna()

    #### LOAD TIDALLY CORRECTED SHORELINE DATA #############
    csv = os.path.join(os.getcwd(), 'outputs', region, sitename, f'{sitename}_intersections_tidally_corrected_{reference_elevation}m.csv')
    try:
        tidal_corrected_df = pd.read_csv(csv, index_col=0, parse_dates=True)
        dates_sat = tidal_corrected_df.index
    except FileNotFoundError: 
        print('no transect_intersections found for site', sitename)
        return
    tidal_corrected_df.index.name = 'dates'
    #### spatial DATA #############
    ### load polygon, get centroid
    poly_path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename+'_polygon.geojson'))
    polygon = gpd.read_file(poly_path)
    centroid_x, centroid_y = polygon.geometry.centroid.values.x[0], polygon.geometry.centroid.values.y[0]
    centroid = Point(centroid_x, centroid_y)

    #### find closest point from the geojson file of points
    wave_coord['distance'] = wave_coord['geometry'].apply(lambda geom: centroid.distance(geom))
    # Find the index of the row with the smallest distance
    nearest_index = wave_coord['distance'].idxmin()
    nearest_point_row = wave_coord.loc[nearest_index]
    # wave_coord = wave_coord.drop(columns=['distance'])
    wave_x, wave_y = nearest_point_row[0].x, nearest_point_row[0].y
    print('identified closest wave data coordinate at', wave_x, wave_y)

    ### find closest lat/long from dataframe and subset wave dataframe ##########
    waves_site = waves[(waves['longitude']==wave_x) & (waves['latitude']==wave_y)]
    print('found %d wave observations for this location' %(len(waves_site)))

    # Merge the DataFrames based on the closest timestamps
    waves_site.index = waves_site.index.tz_convert(None)
    waves_sat = pd.merge_asof(tidal_corrected_df, waves_site, on='dates', direction='nearest', tolerance=timedelta(days=1))
    # Set the dates as the index
    waves_sat.set_index('dates', inplace=True)
    waves_sat = waves_sat.iloc[:,len(tidal_corrected_df.columns):]


    ### do horizontal correction 
    corrected = tidal_corrected_df.copy()
     
    ###### SETUP CORRECTION ########
    corrected = tidal_corrected_df.copy()
    #if there are specific sloes for each transect
    if isinstance(beach_slope, list) and len(beach_slope) == len(tidal_corrected_df.columns):
        print('individual beach slopes detected')
        for i in range(len(beach_slope)):
            tr = tidal_corrected_df.columns[i]
            print('wave setup correction for transect %s with slope %.3f' %(tr, beach_slope[i]))
            waves_sat['correction'] = waves_sat['setup']/beach_slope[i]
            # correct for each day
            for date in waves_sat.index:
                corrected.loc[date,tr] = corrected.loc[date,tr] + waves_sat.loc[date,'correction']

    else: #use one slope for all
        waves_sat['correction'] = waves_sat['setup']/beach_slope
        # correct for each day
        for date in waves_sat.index:
            corrected.loc[date] = corrected.loc[date] + waves_sat.loc[date,'correction']
    
    #### save output ####
    # print('saving tidally corrected csv')
    corrected.insert(0, 'timestamp', corrected.index)
    corrected.index = np.arange(0,len(corrected.index),1)

    out_path = os.path.join(os.getcwd(), 'outputs', region, sitename,
        (f'{sitename}_timeseries_tide_wave_corrected.csv'))
    corrected.to_csv(out_path, index= False)



####################################################
####################################################
######## SINGLE SITE ################
####################################################
# path_to_aviso = os.path.join(os.getcwd(), 'aviso-fes-main', 'data', 'fes2014')
# path_to_buffer = os.path.join(os.getcwd(), 'historical_data', "coastline_buffer_1mi.geojson")
# region = 'narrabeen'
# reference_elevation = 0.7 #-0.329 #### MLLW: -0.251 
# # beach_slope = 0.12
# beach_slope = [0.11,0.11,0.11,0.13,0.12]
# plot_tide = False

# sites = os.listdir(os.path.join(os.getcwd(), 'outputs', region))
# for n, site in enumerate(sites):
#     # print(site)
#     if not site == 'NARRA':
#         continue
#     p = (n+1)/len(sites)*100
#     print(site, '  -  ', p, '%', flush=True, end="\r")
#     tidal_correction_site(site, region, path_to_buffer, path_to_aviso, reference_elevation, plot_tide, beach_slope)

#sitename = 'HawaiiKai_2020'
#tidal_correction(sitename, region, path_to_buffer, path_to_aviso, plot_tide=False, reference_elevation=0, beach_slope=0.15)
################################






