"""
This

Author: Joel Nicolow, Climate Resiliance Initiative, School of Ocean and Earth Science and Technology (September 2024)
"""
import os
import pandas as pd
import numpy as np
from glob import glob
import pyfes

from coastvision import coastvision
from coastvision import supportingFunctions
from coastvision import geospatialTools
from coastvision import coastvisionTides

class CoastVisionRun(object):
    """
    Each class instance represents running coastvision for one site
    """

    def __init__(self, region, sitename, just_extract_shoreline = False, data_products = False, smooth_window = None, smooth_shoreline_sigma = 0.1, min_sl_len=30, max_dist_from_sl_ref = 50):

        self.region = region
        self.sitename = sitename
        self.outdir = os.path.join(os.getcwd(), 'outputs', self.region, self.sitename)

        ##### user inputs #####
        self.data_products = data_products # boolean weather to save data product plots or not
        self.smooth_window = smooth_window # None or 
        self.smooth_shoreline_sigma = smooth_shoreline_sigma # float value for shoreline smoothing algorithm
        self.just_shoreline_extract = just_extract_shoreline # Shorelines are extracted but intersection with transects is not computed
        self.min_sl_len = min_sl_len # minumum length of shoreline segment (meters)
        self.max_dist_from_sl_ref = max_dist_from_sl_ref # max distance extracted shoreline can be from reference shoreline (in meters)

        ##### get site inputs #####
        self.site_inputs = {}
        transectPath = os.path.join(os.getcwd(), 'user_inputs', self.region, self.sitename, (self.sitename + "_transects.geojson"))
        self.transects = coastvision.transects_from_geojson(transectPath)
        self.site_inputs['transects'] = self.transects
        infoJson = supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', self.region, self.sitename,(self.sitename + '_info.json')))
        self.site_inputs['infoJson'] = infoJson
        # pixelResolutions = coastvision.get_all_pixel_resolutions(self.region, self.sitename)
        # user_inputsFiles = os.listdir(os.path.join(os.getcwd(), 'user_inputs', self.region, self.sitename))

        ##### get item ids #####
        self.item_ids = set()
        for file in os.listdir(os.path.join(os.getcwd(), 'data', region, sitename)):
            if file.endswith('metadata.json'):
                itemId = file.split('_metadata')[0]
                # metadataJson = os.path.join(os.getcwd(), 'data', region, sitename, file)
                # try:
                #     with open(metadataJson, 'r') as f:
                #         data = json.load(f)
                #     pixel_size = data['properties']['pixel_resolution']
                # except ValueError:
                #     print("JSONDecodeError Occurred and Handled")
                #     pixel_size = 3
                # NOTE: waiting on pixel size for now
                self.item_ids.add(itemId)


        self.intersection_dict = None # this will be filled out by run_shoreline_extraction()


    def __len__(self):
        return len(self.item_ids)
    

    def run_shoreline_extraction(self):
        self.intersection_dict = {}
        n=0
        tifs = glob(os.path.join('data', self.region, self.sitename, '*_3B_AnalyticMS_toar_clip.tif'))
        if len(tifs) < 1: raise ValueError(f"There are no images for the site: {self.sitename}. Use PlanetScope API to download imagery")
        ref_tif = tifs[0]
        im_ms = geospatialTools.get_im_ms(ref_tif)
        georef = geospatialTools.get_georef(ref_tif)
        self.site_inputs['shorelinebuff'] = coastvision.create_shoreline_buffer(self.region, self.sitename, im_shape=im_ms.shape[0:2], 
                                                        georef=georef, pixel_size=3,
                                                        max_dist=self.max_dist_from_sl_ref)
        for itemID in self.item_ids:
            n+=1
            print('\r', self.sitename, round(n/len(self.item_ids)*100,2), 'percent progress', itemID, end='', flush=True)
            # run coastvision
            self.intersection_dict[itemID] = coastvision.run_coastvision_single(
                self.region, 
                self.sitename,
                itemID, 
                siteInputs = self.site_inputs, 
                justShorelineExtract = self.just_shoreline_extract, 
                smoothShoreline = self.smooth_shoreline_sigma, 
                smoothWindow = self.smooth_window, 
                min_sl_len= self.min_sl_len, 
                dataProducts=self.data_products
            )

        
        return self.create_intersection_df()
    

    def create_intersection_df(self):
        if not self.intersection_dict is None:
            #create df
            self.intersection_df = pd.DataFrame(self.intersection_dict).T.astype(float)
            #fix timestamp from raw time to datetime
            self.intersection_df.rename(index=lambda x: pd.to_datetime(supportingFunctions.clean_timestamp(x)), inplace=True) 
            #remove rows with only nan
            self.intersection_df = self.intersection_df.dropna(axis=0, how='all')
            self.intersection_df = self.intersection_df.sort_index() # sort index
            if not self.intersection_df.index.is_unique: 
                self.intersection_df = self.intersection_df.groupby(self.intersection_df.index).first()
            self.intersection_df.to_csv(os.path.join(self.outdir, f'{self.sitename}_transect_intersections.csv'))

            return self.intersection_df
        return None
    

    def tidal_correction_FES2014(self, fes2014_path, offshore_coord, reference_elevation=0, beach_slope=0.12):
        if not os.path.exists(fes2014_path):
            print('path to FES2014 is not set up correct')
            return
        dates_sat = self.intersection_df.index
        ###### GET TIDE DATA  ########
        # point to the folder where you downloaded the .nc files
        config_ocean = os.path.join(fes2014_path, 'ocean_tide.ini') # change to ocean_tide.ini
        config_load =  os.path.join(fes2014_path, 'load_tide.ini')  # change to load_tide.ini
        ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
        load_tide = pyfes.Handler("radial", "io", config_load)
        tide_sat = coastvisionTides.compute_tide_dates(offshore_coord, dates_sat, ocean_tide, load_tide)
        ###### SAVE TIDE DATA ########
        tides = pd.DataFrame(tide_sat, columns=['tide'], index=dates_sat)
        tides.index.name = 'dates'

        ###### TIDAL CORRECTION ########
        corrected = self.intersection_df.copy()
        tides['correction'] = (tides['tide']-reference_elevation)/beach_slope
        # correct for each day
        for date in tides.index:
            corrected.loc[date] = corrected.loc[date] + tides.loc[date,'correction']
        ## save ## 
        path_corrected = os.path.join(self.outdir, f'{self.sitename}_tidally_corrected_{reference_elevation}m.csv')
        corrected.to_csv(path_corrected)

        return corrected

    def tidal_correction_tidegauge(self, tidegauge, reference_elevation=0, beach_slope=0.12):
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
        tides = coastvisionTides.get_tides_from_tidegauge(self.intersection_df, tidegauge)
        tides = tides.dropna()
        ###### TIDAL CORRECTION ########
        corrected = self.intersection_df.copy()
        tides['correction'] = (tides['sl']-reference_elevation) / beach_slope
        # correct for each day
        for date in tides.index:
            corrected.loc[date] = corrected.loc[date] + tides.loc[date,'correction']
        ## save ## 
        path_corrected = os.path.join(self.outdir, f'{self.sitename}_tidally_corrected_{reference_elevation}m.csv')
        corrected.to_csv(path_corrected)

        return corrected


    def intersection_QAQC(self, remove_fraction = 0.15, window_days = 50, limit = 30):
        """

        """
        qcDf = self.intersection_df.copy()    
        for col in qcDf.columns[1:]:
            median = np.nanmedian(self.intersection_df[col].values)
            qcDf.loc[qcDf[col] < (median - limit), col] = np.nan
            qcDf.loc[qcDf[col] > (median + limit), col] = np.nan
        print('before QAQC: shape', self.intersection_df.shape, 'number of nans', self.intersection_df.iloc[:,1:].isna().sum().sum())
        print('after QAQC: shape', qcDf.shape, 'number of nans', qcDf.iloc[:,1:].isna().sum().sum())
        path = os.path.join(self.outdir, (self.sitename + '_QAQC_transect_interesections.csv'))
        qcDf.to_csv(path)
        return qcDf




