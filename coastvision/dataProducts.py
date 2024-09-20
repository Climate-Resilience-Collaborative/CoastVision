"""
This script contains functions used to create data products for coastvision

Author: Joel Nicolow, Climate Resiliance Initiative, School of Ocean and Earth Science and Technology (July, 20 2022)

"""



#### TIFF image display ####
import rasterio
from rasterio.plot import show
import numpy as np
def show_rgb(img_file):
    """
    Displays a satalite image saved as a tif file in visual light (red green blue)
    
    :img_file: full file name of tif satalite image (may be relative to cwd)
    """
    with rasterio.open(img_file) as src:
        b,g,r,n = src.read()
        
    rgb = np.stack((r,g,b), axis=0)
    show(rgb/rgb.max())