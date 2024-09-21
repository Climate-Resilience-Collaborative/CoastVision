##### Scripts for co-registration using arosics #####
## Anna Mikkelsen March 2023


import os
from datetime import datetime, timedelta
import time
import glob
import numpy as np
# import matplotlib.pyplot as plt
import shutil
import rasterio
# from osgeo import gdal
import logging
import sys 
# import sklearn
# if sklearn.__version__[:4] == '0.20':
#     from sklearn.externals import joblib
# else:
#     import joblib
# import skimage.morphology as morphology
import signal

from coastvision import geospatialTools
from arosics import COREG_LOCAL, COREG, DESHIFTER



###### FUNCTIONS ##########
## REFERENCE 
def select_reference_im(region, site, ref_im_path = None):
    """
    Select a reference image to register all other images to
    1) lets the user add a path to a specific reference image 
    2) If None is passed, it searches for reference image in user_inputs CSTSI/user_inputs/region/site
    3) If none exists, prompts user to select a reference image, and saves that in CSTSI/user-inputs/region/site

    Arguments:
    ----------
    region: str
        Region name of site (e.g. oahu)
    site: str
        site name (e.g. oahu0021)
    ref_im_path: str
        Path to reference image. Default: None.

    Returns
    ----------
    ref_im_path: str
        path to saved reference image
    """
    
    if ref_im_path == None:
        
        ## first search in user_inputs
        site_inputs_dir = os.path.join(os.getcwd(), 'user_inputs', region, site)
        potential_ref = glob.glob(os.path.join(site_inputs_dir, ('*.tif')))
        if len(potential_ref) > 0:
            print(potential_ref[0])
            ref_im_path = potential_ref[0] # use that as reference
        else: 
            ref_im_path = glob.glob(os.path.join(os.getcwd(), 'data', region, site, '*_3B_AnalyticMS_toar_clip.tif'))[0]

    return ref_im_path

# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred")

def coreg_single(im_ref, im_tgt, grid_res):
    """
    Function to perform co-registration on a single image given settings and a reference image

    Arguments:
    ----------
    im_ref: str
        path to reference images
    im_tgt: str
        path to image to perform co-registration on
    im_ref_mask: 2d masked array 
        Array that shows where there is water/clouds (ignore for coregistation). (True=bad_data; False=clear_data)
    im_ref_mask_path: str
        path to im_ref_mask
    grid_res: int
        grid resolution for coregistration (see Arosics)
    temp_dir: str
        temporary directory to store temporary masks for co-registration
    failed: list
        list of failed images 

    Returns nothing
    """
    
    #### Setup run: Get unique image ID, datadirectory and a list metadatafiles
    im_id = os.path.basename(im_tgt)[0:20] 
    datadir = os.path.dirname(im_tgt) 
    meta_files = glob.glob(os.path.join(datadir, (im_id+'*'))) 
    
    # Create path for a temporary image mask file
    # temp_mask_path = os.path.join(temp_dir, 'im_tgt_mask_temp.tif')
    
    level = 0 

    # Identify and load the image cloud mask if the udm file exists
    if all(['udm2_clip' not in f for f in meta_files]):
        print('no cloud mask found. skipping this image')
        return str(im_id), level
    cloud_mask_path = [udm for udm in meta_files if 'udm2_clip' in udm][0]
    
    try:
        #try statement here because corrupt udm files break loop
        cloud_mask = geospatialTools.get_cloud_mask_from_udm2_band8(cloud_mask_path)
        # and corrupt tiff files
        with rasterio.open(im_tgt, 'r') as src:
            kwargs = src.meta
    except:
        print('cloudmask or image faulty. skipping this image')
        logging.exception("%s cloudmask or image faulty. skipping this image" %(str(im_id)))
        return str(im_id), level
    
   # If the clear part of the image is None or with little overlap, skip
    if np.sum(~cloud_mask) < (5*grid_res**2):
        print('not enough image overlap: keeping raw image', im_id)
        logging.exception("%s not enough image overlap: keeping raw image" %(str(im_id)))
        return str(im_id), level
    
    # if loop makes it to here, any errors past this is in coregistation
    level = 1 

    #######################################
    ########  COREGISTRATION #############
    #######################################
    CRL= 0
    try: 
        print('performing coregistration')
        CRL = COREG_LOCAL(im_ref, im_tgt, path_out=im_tgt, 
                        grid_res=grid_res,
                        # mask_baddata_ref = im_ref_mask_path,
                        # mask_baddata_tgt = temp_mask_path,
                        fmt_out='GTIFF',
                        out_crea_options = ["COMPRESS=LZW"],
                        progress=False,
                        nodata = (0,0),
                        min_reliability = 50,
                        tieP_filter_level = 2 #adjust filter settings
                        )
        
        print('correcting shifts')
        ## Add TIMEOUT FUNCTION for CRL.correct_shifts
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(20)
        # try:
        CRL.correct_shifts(min_points_local_corr = 3)
        print('finished coregistration')
    # except TimeoutError:
    #     print("Timeout occurred. Skipping to the next item.")
    #     logging.exception("%s Timeout occurred. Skipping to the next item" %(str(im_id)))
    #     return str(im_id), level
    # signal.alarm(0) # Reset the alarm
    except Exception as e: #if for some reason COREG_LOCAL breaks
        print('Failed at coregistration; Keeping raw image')
        CRL= 0
        logging.exception("%s Error at coregistration: %s" %(str(im_id), str(e)))
        #time.sleep(2)
    except KeyboardInterrupt as k:
        CRL= 0
        logging.exception("KeyboardInterrupt at correcting shifts for im %s: %s" %(str(im_id), str(k)))
        sys.exit()

    ########
    if CRL != 0:
        print('Coregistration looks successful. SHIFTING cloud masks')
        #### IF SUCCESSFULL, ALSO SHIFT CLOUD MASK ######
        cr_param = CRL.coreg_info
        try:
            #try statement because cloud mask shift occasionally fails
            DESHIFTER(cloud_mask_path, cr_param, 
              path_out =  cloud_mask_path, 
              fmt_out = 'GTIFF',
              out_crea_options = ["COMPRESS=LZW"],
              nodata = 255,
              progress=False, 
              min_points_local_corr = 3
              ).correct_shifts()
            print('done shifting cloudmask')
        except TypeError as e:
            print('shifting cloud mask failed. keeping current mask')
            logging.exception("%s Error at shifing cloudmask: %s" %(str(im_id), str(e)))
    else:
        print('CRL = 0, no coreg performed')
        return str(im_id), level
    level = 2 #success
    return None, level

# def save_bool_mask(file_to_save, output_path, path_to_get_metadata):
#     """
#     Saves a boolean mask 
#     Arguments:
#     ----------
#     file_to_save: 2d masked array
#         MAsked array to save
#     output_path: str
#         path+name to save files to
#     path_to_get_metadata: str
#         path to reference file to get metadata kwargs from; e.g. UDM cloud mask

#     Returns nothing

#     """
#     with rasterio.open(path_to_get_metadata, 'r') as src:
#         kwargs = src.meta
#     kwargs.update(
#         driver='GTiff',
#         dtype='uint8',
#         nodata=0,  
#         count = 1,
#         compress='lzw')
#     # save mask
#     with rasterio.open(output_path, 'w', **kwargs) as dst:
#         dst.write_band(1, file_to_save.astype(rasterio.uint8))

# def get_raster_bounds(file):
    
#     ''' Find raster bounding parameters '''
    
#     dataset = rasterio.open(file)
    
#     bounds = [
#         dataset.bounds.left,
#         dataset.bounds.top,
#         dataset.bounds.right,
#         dataset.bounds.bottom
#         ]
    
#     return bounds

def coreg_site(region, site, grid_res, start=0):
    # Perform coregistration for a whole site by looping through all images of that site

    """
    Arguments:
    ----------
    region: str
        Region name of site (e.g. oahu)
    site: str
        site name (e.g. oahu0021)
    grid_res: int
        grid resolution for coregistration (see Arosics)
    start: int
        n-image to start process at (if process fails, to avoid re-registering the same images). Default: 0

    Returns nothing
    """
    ##############################################################################
    ########  Set up variables    ################################################
    ##############################################################################
    datapath = os.path.join(os.getcwd(), 'data', region, site)
    if not os.path.exists(datapath):
        return
    ##### Dummy check if site is already coregistered #######
    txt_path = os.path.join(datapath, (f"coregistration_{site}.txt"))
    if os.path.exists(txt_path):
        if "This folder was coregistered on" in open(txt_path,"r").read(): 
            print(site, 'already coregistered. Skipping') #if statement "This folder was coregistered" in txt file, already co-registered
            return
    # Configure error logger
    logging.basicConfig(filename=os.path.join(datapath,'errors.log'), level=logging.ERROR)
    # get all images to co-register
    files_tif = sorted(glob.glob(os.path.join(datapath, '*_toar_clip.tif')))
    n_total = len(files_tif) 
    #set up temporary folder for storage of masks
    # temp_dir = os.path.join(os.getcwd(), 'data_temp', region)
    # supportingFunctions.create_dir(temp_dir)

    ########################################################
    # Make a copy of raw folder before processing. TEMPORARILY SKIPPING
    #region_rawfolder = os.path.join(os.getcwd(), 'data', (region+'_raw'))
    #if not os.path.exists(region_rawfolder):
    #    os.mkdir(region_rawfolder)
    # site_rawfolder = os.path.join(region, (site+'_raw'))
    # if not os.path.exists(site_rawfolder):
    #     print('copying raw data, this might take a minute')
    #     shutil.copytree(datapath, site_rawfolder)
    ########################################################

    #######################################
    ### Format image mask + ref image #####
    #######################################
    # get reference image and create mask for reference image
    ref_im_path = select_reference_im(region, site, ref_im_path = None)
    # # load udm cloud mask based on ref image id
    # im_ref_id = os.path.basename(ref_im_path)[0:20]
    # im_ref_cloud_path = glob.glob(os.path.join(datapath, (im_ref_id+'*udm2_clip.tif')))[0] 
    # im_ref_cloud_mask = geospatialTools.get_cloud_mask_from_udm2_band8(im_ref_cloud_path)
    # # combine water and cloud mask, and save output
    # # im_ref_mask = (water_mask + im_ref_cloud_mask) >0
    # im_ref_mask = im_ref_cloud_mask > 0 # dont use water mask for now

    # SAVE reference image mask ######
    # im_ref_mask_path = os.path.join(temp_dir, 'im_ref_mask.tif')
    #save mask
    # save_bool_mask(im_ref_mask, im_ref_mask_path, ref_im_path)

    #dummy var.. to note if coreg was started
    f= open(os.path.join(datapath, (f"coregistration_{site}.txt")),"w+")
    f.write(f'\n coregistration started on {datetime.now()}')
    f.close()

    #dummy var.. to save failed image ids
    f= open(os.path.join(datapath, (f"failed_{site}.txt")),"w+")
    f.write(f'\n These images were not coregistered:')
    f.close()

    ##############################################################################
    #####     RUN COREGISTRATION    ##############################################
    ##############################################################################

    time_start = datetime.now()
    for n,file in enumerate(files_tif[start:]): ## in case it stops, pick up where it left off by changing 'start' value
        name = os.path.basename(file)[0:20]
        print('%s n %d; %.2f percent; time elapsed: %s; %s'%(site, (n+start), (100*(n+start)/n_total), str(datetime.now() - time_start), name))
        
        #### run coregistration
        im_failed, level = coreg_single(im_ref=ref_im_path, im_tgt=file, grid_res=grid_res)
        if im_failed is None:
            continue
        f= open(os.path.join(datapath, (f"failed_{site}.txt")),"a")
        f.write(f'\n{im_failed}')
        f.close()

    ### Add text file to folder ###
    f = open(os.path.join(datapath, (f"coregistration_{site}.txt")),"a")
    f.write(f'\n This folder was coregistered on {datetime.now()} \n with settings: \n    grid_res: {grid_res} \n    reference image: {os.path.basename(ref_im_path)}')
    f.close()

    print('\n \n FINISHED!! \n \n ')
    return
