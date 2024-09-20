
## py. script to perfrom coregistration (image alignment) for all images for a specified site ##
## Anna Mikkelsen 03/14/2023, CRC
## AROSICS source documentation: https://danschef.git-pages.gfz-potsdam.de/arosics/doc/

### imports ##
import numpy as np
from coastvision import coastvisionCoregKoa as coastvisionCoreg
from coastvision import geospatialTools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os

#### SELECT INPUT VARIABLES ####
region = 'oahu'
# siteids = [19]
# siteids = np.arange(19,20,1)
# sites = [region+str(id).zfill(4) for id in siteids]
sites =['oahu0029','oahu0026', 'oahu0037', 'oahu0047', 'oahu0051', 'oahu0054']
# sites = sorted(os.listdir(os.path.join('data', region)))

grid_res = 50

#######################################################
###########      RUN coregistration       #############
#######################################################
if __name__ == '__main__':
	for site in sites:
		print(site)
		coastvisionCoreg.coreg_site(region, site, grid_res, start=0)
# coastvisionCoreg.coreg_site(region, site, grid_res, start=0)


####################################################################
#######      SELECT reference image for a number of sites      #####
####################################################################
# for site in sites:
    # print(site)
    # coastvisionCoreg.select_reference_im(region, site, ref_im_path = None)