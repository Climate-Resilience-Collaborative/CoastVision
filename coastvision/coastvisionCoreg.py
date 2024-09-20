##### Scripts for co-registration using arosics #####
## Anna Mikkelsen March 2023


import os
from datetime import datetime, timedelta
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import shutil
import rasterio
from osgeo import gdal
import logging
import sys 
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
import skimage.morphology as morphology
import signal

from coastvision import geospatialTools
from coastvision import coastvision
from coastvision import dataSelectionTools
from coastvision import supportingFunctions
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
        #if os.path.exists(os.path.join(site_inputs_dir, (site+'reference_im.tif'))):
        potential_ref = glob.glob(os.path.join(site_inputs_dir, ('*.tif')))
        if len(potential_ref) > 0:
            print(potential_ref[0])
            ref_im_path = potential_ref[0] # use that as reference
        else: # no files found, select a reference
            # ref_im = dataSelectionTools.select_tif_images(datapath, timeperiod=None, onlySelectOne=True)
            # ref_im_path = os.path.join(os.getcwd(), 'data', region, site, ref_im)
            # site_inputs_dir = os.path.join(os.getcwd(), 'user_inputs', region, site)
            # shutil.copy(ref_im_path, site_inputs_dir)
            # print('copied reference image')

            ref_im_path = glob.glob(os.path.join(os.getcwd(), 'data', region, site, '*_3B_AnalyticMS_toar_clip.tif'))[0]

    return ref_im_path

## WATER MASK
def create_water_mask(region, site, ref_im_path, water_mask_path=None):
    """
    Make a generic water mask to use to mask water pixels for coregistration
    1) Let's the user specify a path to a water mask,
    2) If not will classify one (water/land) using pixel classification on the reference image
    
    Arguments:
    ----------
    region: str
        Region name of site (e.g. oahu)
    site: str
        site name (e.g. oahu0021)
    ref_im_path: str
        Path to reference image.
    water_mask: str
        path to existing water mask. Default: None

    Returns
    ----------
    water_mask: masked array
        Masked array where water pixels=True/1, land pixels=False/0
    """

    files_udm = glob.glob(os.path.join(os.getcwd(), 'data', region, site, '*_udm2_clip.tif'))
    
    ## if none provided, then calculate from reference image 
    if water_mask_path == None:
        # define classifier
        clf = joblib.load(os.path.join(os.getcwd(), 'coastvision', 'classifier', 'models', 'HI_logisticregression.pkl'))
        # load image and define cloudmask
        im_ms = geospatialTools.get_im_ms(ref_im_path)
        # classify image ##
        im_class = coastvision.classify_single_image(ref_im_path)
        water_mask = im_class == 0
        #### SAVE MASK #####
        # need a way to save the water mask

    return water_mask

# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred")


def coreg_single(im_ref, im_tgt, im_ref_mask, im_ref_mask_path, grid_res, temp_dir, failed):
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
    temp_mask_path = os.path.join(temp_dir, 'im_tgt_mask_temp.tif')
    
    level = 0 ## anna's odd way of assigning type of error to failed img. errors here are image related (faulty, too small overlap)

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
    ########  Format image masks  #########
    #######################################
    # #then check if extent of generic water mask and image cloud mask match
    # #if not, crop generic water mask, open it, combine with cloud mask and save a temporary tgt mask
    # if (cloud_mask.shape == im_ref_mask.shape) & (get_raster_bounds(im_ref_mask_path)== get_raster_bounds(cloud_mask_path)):
    #     #if extent match, combine mask and save output
    #     im_tgt_mask = (im_ref_mask + cloud_mask) > 0
    #     save_bool_mask(im_tgt_mask, temp_mask_path, im_tgt)
    # else:
    #     # if extents don't match, crop generic reference mask to shape of target image
    #     # open reference mask with gdal
    #     im_ref_mask_open = gdal.Open(im_ref_mask_path) 
    #     # get bounds of target image  
    #     bounds=get_raster_bounds(cloud_mask_path)
    #     # specify name of cropped mask to save it as
    #     im_ref_mask_crop_path = os.path.join(temp_dir, 'im_ref_mask_crop.tif') # save to location
    #     #crop reference mask to bounds; save as im_ref_mask_crop_path
    #     crop_gdal = gdal.Translate(im_ref_mask_crop_path, im_ref_mask_open, projWin = bounds)
    #     #open new cropped mask
    #     with rasterio.open(im_ref_mask_crop_path) as src:
    #         mask = src.read()
    #     im_ref_mask_crop = (mask[0,:,:]).astype(bool)
    #     # combine with cloud mask and save output
    #     im_tgt_mask = (im_ref_mask_crop + cloud_mask) > 0 
    #     save_bool_mask(im_tgt_mask, temp_mask_path, im_tgt)
    #     #close masks
    #     crop_gdal = None
    #     src.close()
    #     os.remove(im_ref_mask_crop_path)

    
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
                        progress=False, #q = True, progress = True,
                        nodata = (0,0),
                        #CPUs = None,
                        min_reliability = 50,
                        tieP_filter_level = 2 #adjust filter settings
                        )
        
        print('correcting shifts')
        ## Add TIMEOUT FUNCTION for CRL.correct_shifts
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(20)
        # try:
        CRL.correct_shifts(min_points_local_corr = 3) #this is where it stalls
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
              nodata = 255, # if 0 or 1 doesn't shoft properly
              progress=False, #q = True, progress = True,
              #CPUs = None,
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


def save_bool_mask(file_to_save, output_path, path_to_get_metadata):
    """
    Saves a boolean mask 
    Arguments:
    ----------
    file_to_save: 2d masked array
        MAsked array to save
    output_path: str
        path+name to save files to
    path_to_get_metadata: str
        path to reference file to get metadata kwargs from; e.g. UDM cloud mask

    Returns nothing

    """
    with rasterio.open(path_to_get_metadata, 'r') as src:
        kwargs = src.meta
    kwargs.update(
        driver='GTiff',
        dtype='uint8',
        nodata=0,  ##ABM tryign to reduce size of UDM file
        count = 1,
        compress='lzw')
    # save mask
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write_band(1, file_to_save.astype(rasterio.uint8))

def get_raster_bounds(file):
    
    ''' Find raster bounding parameters '''
    
    dataset = rasterio.open(file)
    
    bounds = [
        dataset.bounds.left,
        dataset.bounds.top,
        dataset.bounds.right,
        dataset.bounds.bottom
        ]
    
    return bounds

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
    temp_dir = os.path.join(os.getcwd(), 'data_temp', region)
    supportingFunctions.create_dir(temp_dir)

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
    # water mask; pixels to ignore during coregistration
    # print('calculating water mask')
    # water_mask = create_water_mask(region, site, ref_im_path, water_mask_path=None)
    # #process water mask to remove holes in land pixels
    # water_mask = morphology.binary_erosion(water_mask,  morphology.square(50)) # https://medium.com/swlh/image-processing-with-python-morphological-operations-26b7006c0359
    # water_mask = morphology.binary_dilation(water_mask, morphology.square(40))

    # load udm cloud mask based on ref image id
    im_ref_id = os.path.basename(ref_im_path)[0:20]
    im_ref_cloud_path = glob.glob(os.path.join(datapath, (im_ref_id+'*udm2_clip.tif')))[0] 
    im_ref_cloud_mask = geospatialTools.get_cloud_mask_from_udm2_band8(im_ref_cloud_path)
    # combine water and cloud mask, and save output
    # im_ref_mask = (water_mask + im_ref_cloud_mask) >0
    im_ref_mask = im_ref_cloud_mask > 0 # dont use water mask for now

    # SAVE reference image mask ######
    im_ref_mask_path = os.path.join(temp_dir, 'im_ref_mask.tif')
    #save mask
    # save_bool_mask(im_ref_mask, im_ref_mask_path, ref_im_path)

    #dummy.. to note if coreg was started
    f= open(os.path.join(datapath, (f"coregistration_{site}.txt")),"w+")
    f.write(f'\n coregistration started on {datetime.now()}')
    f.close()

    #dummy.. to save failed image ids
    f= open(os.path.join(datapath, (f"failed_{site}.txt")),"w+")
    f.write(f'\n These images were not coregistered:')
    f.close()
    f= open(os.path.join(datapath, (f"failed_coreg_{site}.txt")),"w+")
    f.write(f'\n These images failed coregistration:')
    f.close()
    ##############################################################################
    #####     RUN COREGISTRATION    ##############################################
    ##############################################################################

    failed=[] #start empty list
    time_start = datetime.now()

    for n,file in enumerate(files_tif[start:]): ## in case it stops, pick up where it left off by changing 'start' value
        name = os.path.basename(file)[0:20]
        print('%s n %d; %.2f percent; time elapsed: %s; %s'%(site, (n+start), (100*(n+start)/n_total), str(datetime.now() - time_start), name))
        
        #### run coregistration
        im_failed, level = coreg_single(im_ref=ref_im_path, im_tgt=file, im_ref_mask=im_ref_mask, im_ref_mask_path=im_ref_mask_path, grid_res=grid_res, temp_dir=temp_dir, failed=failed)
        ####

        if im_failed is None:
            continue
        # all failed images (level 0 and 1)
        f= open(os.path.join(datapath, (f"failed_{site}.txt")),"a")
        f.write(f'\n{im_failed}')
        f.close()
        # images that failed in coreg to rerun locally
        if level == 1: 
            f= open(os.path.join(datapath, (f"failed_coreg_{site}.txt")),"a")
            f.write(f'\n{im_failed}')
            f.close()



    ### Add text file to folder ###
    f = open(os.path.join(datapath, (f"coregistration_{site}.txt")),"a")
    f.write(f'\n This folder was coregistered on {datetime.now()} \n with settings: \n    grid_res: {grid_res} \n    reference image: {os.path.basename(ref_im_path)}')
    f.close()

    print('\n \n FINISHED!! \n \n ')

    return
