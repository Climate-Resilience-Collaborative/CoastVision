##########
### Add tidal correction to shoreline intersections using global AVISO FES tidal model
### Built based on CoastSat.slope https://doi.org/10.5281/zenodo.2779293
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
    print(dates)
    print(tide_level)
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





# def get_tides_from_avisoFES():


def get_tides_from_tidegauge(sat_intersections, tidegauge):
    # Interpolate tidesgauge to match timestamps in df
    interpolated_tides = pd.DataFrame(index=sat_intersections.index, columns=tidegauge.columns)
    for idx in sat_intersections.index:
        # Find the nearest timestamps in dataframe2
        prev_time = tidegauge.index[tidegauge.index <= idx].max()
        next_time = tidegauge.index[tidegauge.index >= idx].min()
        # print(prev_time, next_time)
        if prev_time == next_time == idx:
            interpolated_tides.loc[idx] = tidegauge.loc[idx]
        # Linear interpolation
        else:
            if prev_time is not pd.NaT and next_time is not pd.NaT:
                weight_prev = (next_time - idx) / (next_time - prev_time)
                weight_next = (idx - prev_time) / (next_time - prev_time)
                interpolated_value = tidegauge.loc[prev_time] * weight_prev + tidegauge.loc[next_time] * weight_next
                interpolated_tides.loc[idx] = interpolated_value
            else:
                interpolated_tides.loc[idx] = pd.NaT  # If no nearest timestamps found, insert NaT

    return interpolated_tides


def tidal_correction_tidegauge(sat_intersections, tidegauge, reference_elevation=0, beach_slope=0.12):
    """
    Inputs
        sitename: site to correct
        region: region of site
        reference_elevation: the reference elevation (Mean sea level=0) to correct intersection to. default 0m
        beach_slope: choose a generic beach slope (**need to add ability to pass list of variable beach slopes**) defualt 0.15
    outputs
        saves a csv with corrected shorelines intersections
        saves a csv with tide level at the time of each image
    """
    tides = get_tides_from_tidegauge(sat_intersections, tidegauge)
    tides = tides.dropna()
    dates_sat = sat_intersections.index
        
    ###### TIDAL CORRECTION ########
    corrected = sat_intersections.copy()
    tides['correction'] = (tides['sl']-reference_elevation) / beach_slope
    # correct for each day
    for date in tides.index:
        corrected.loc[date] = corrected.loc[date] + tides.loc[date,'correction']

    ## save ## 
    path_corrected = os.path.join(os.getcwd(), 'outputs', 'hawaii', 'waikiki', f'waikiki_tidally_corrected_{reference_elevation}m.csv')
    corrected.to_csv(path_corrected)

    return corrected


def tidal_correction_FES2014(sat_intersections, fes2014_path, coords, reference_elevation=0, beach_slope=0.12):
    """
    Inputs
        sitename: site to correct
        region: region of site
        reference_elevation: the reference elevation (Mean sea level=0) to correct intersection to. default 0m
        beach_slope: choose a generic beach slope (**need to add ability to pass list of variable beach slopes**) defualt 0.15
    outputs
        saves a csv with corrected shorelines intersections
    """
    dates_sat = sat_intersections.index

    ###### GET TIDE DATA  ########
    # point to the folder where you downloaded the .nc files
    config_ocean = os.path.join(fes2014_path, 'ocean_tide.ini') # change to ocean_tide.ini
    config_load =  os.path.join(fes2014_path, 'load_tide.ini')  # change to load_tide.ini
    ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
    load_tide = pyfes.Handler("radial", "io", config_load)
    tide_sat = compute_tide_dates(coords, dates_sat, ocean_tide, load_tide)
    ###### SAVE TIDE DATA ########
    tides = pd.DataFrame(tide_sat, columns=['tide'], index=dates_sat)
    tides.index.name = 'dates'
    # path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_tides.csv'))
    # tides.to_csv(path) 

    ###### TIDAL CORRECTION ########
    corrected = sat_intersections.copy()
    tides['correction'] = (tides['tide']-reference_elevation)/beach_slope
    # correct for each day
    for date in tides.index:
        corrected.loc[date] = corrected.loc[date] + tides.loc[date,'correction']
    ## save ## 
    path_corrected = os.path.join(os.getcwd(), 'outputs', 'hawaii', 'waikiki', f'waikiki_tidally_corrected_{reference_elevation}m.csv')
    corrected.to_csv(path_corrected)

    return corrected


