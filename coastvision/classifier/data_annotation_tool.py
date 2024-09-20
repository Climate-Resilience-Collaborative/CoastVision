"""
This is a tool that can be used to hand classify pixels in a satalite image (originally for 4band PlanetScope .tif files)
CURRENTLY ADDAPTED FROM CoastSat_PS

Author: Joel Nicolow, Climate Resilience Initiative, SOESTClimate Resiliance Initiative, School of Ocean and Earth Science and Technology (July, 22 2022)
"""
import os
import pytz
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from datetime import datetime
from pylab import ginput
from matplotlib.widgets import LassoSelector
from matplotlib import path
from sklearn.metrics import confusion_matrix
from skimage.segmentation import flood
import skimage.morphology as morphology

from coastvision import geospatialTools
from coastvision import coastvision
from coastvision import supportingFunctions


class SelectFromImage(object):
    """
    Class used to draw the lassos on the images with two methods:
        - onselect: save the pixels inside the selection
        - disconnect: stop drawing lassos on the image
    
    Copied with permission from CoastSat (KV, 2020) 
        https://github.com/kvos/CoastSat
        
    """
    # initialize lasso selection class
    def __init__(self, ax, implot, color=[1,1,1]):
        self.canvas = ax.figure.canvas
        self.implot = implot
        self.array = implot.get_array()
        xv, yv = np.meshgrid(np.arange(self.array.shape[1]),np.arange(self.array.shape[0]))
        self.pix = np.vstack( (xv.flatten(), yv.flatten()) ).T
        self.ind = []
        self.im_bool = np.zeros((self.array.shape[0], self.array.shape[1]))
        self.color = color
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        # find pixels contained in the lasso
        p = path.Path(verts)
        self.ind = p.contains_points(self.pix, radius=1)
        # color selected pixels
        array_list = []
        for k in range(self.array.shape[2]):
            array2d = self.array[:,:,k]    
            lin = np.arange(array2d.size)
            new_array2d = array2d.flatten()
            new_array2d[lin[self.ind]] = self.color[k]
            array_list.append(new_array2d.reshape(array2d.shape))
        self.array = np.stack(array_list,axis=2)
        self.implot.set_data(self.array)
        self.canvas.draw_idle()
        # update boolean image with selected pixels
        vec_bool = self.im_bool.flatten()
        vec_bool[lin[self.ind]] = 1
        self.im_bool = vec_bool.reshape(self.im_bool.shape)

    def disconnect(self):
        self.lasso.disconnect_events()


import skimage.exposure as exposure
def rescale_image_intensity(im, cloud_mask, prob_high=99.9):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])

    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec =  im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(vec[~vec_mask, i],
                                                      in_range=(prc_low, prc_high),
                                                      out_range = (0,1))  # YD
            vec_adj[~vec_mask,i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec =  im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(vec[~vec_mask], in_range=(prc_low, prc_high))
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj



import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg") # NEEDS TO BE CALLED IN THE NOTEBOOKS
from coastvision.classifier import dataAnnotationTool
def data_annotation_one_type(modelName, traningDataDir, labeledSavedData, cloundThresh=0.9, floodFillTolerance=0.02, augmentationScale=None):
    """
    This function pulls up a data annotation tool that can be used to hand classify pixels. The saved results can be used for training models.
    
    """
    classInfo = dataAnnotationTool.get_class_info()

    settings ={'filepath_train':labeledSavedData, # folder where the labelled images will be stored
           'cloud_thresh':cloundThresh, # percentage of cloudy pixels accepted on the image
           'inputs':{'filepath':traningDataDir, 'modelname': modelName}, # folder where the images are stored
           'labels':classInfo['labels'], # labels for the classifier
           'colors':classInfo['colors'],
           'tolerance':floodFillTolerance, # this is the pixel intensity tolerance, when using flood fill
                             # set to 0 to select one pixel at a time
            }
    matplotlib.use("Qt5Agg")
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                          sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    
    
    for imgFn in os.listdir(traningDataDir):
        print(imgFn)
        im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(traningDataDir, imgFn))
        
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
        im_NDVI = coastvision.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)  # Nir - Red
        #im_NDWI = nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)  # Nir - Green
        im_NDWI = coastvision.nd_index(im_ms[:,:,3], im_ms[:,:,0], cloud_mask)  # Nir - Blue
        
        im_viz = im_RGB.copy()
        im_labels = np.zeros([im_RGB.shape[0],im_RGB.shape[1]])
        
        ax.axis('off')  
        ax.imshow(im_RGB)
        implot = ax.imshow(im_viz, alpha=0.6)            
        ax.set_title(imgFn)
        ##############################################################
        # select image to label
        ##############################################################           
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        key_event = {}
        def press(event):
            # store what key was pressed in the dictionary
            key_event['pressed'] = event.key
        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            fig.canvas.draw_idle()                         
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled labelling images')
            else:
                plt.waitforbuttonpress()

        # if user decided to skip show the next image
        if skip_image:
            ax.clear()
            continue
        # otherwise label this image
        else:
        
            ##############################################################
            # digitize sand pixels
            ##############################################################
            ax.set_title('Click on SAND pixels (flood fill activated, tolerance = %.2f)\nwhen finished press <Enter>'%settings['tolerance'])
            # create erase button, if you click there it delets the last selection
            btn_erase = ax.text(im_ms.shape[1], 0, 'Erase', size=20, ha='right', va='top',
                                bbox=dict(boxstyle="square", ec='k',fc='w'))                
            fig.canvas.draw_idle()
            color_sand = settings['colors']['sand']
            sand_pixels = []
            while 1:
                seed = ginput(n=1, timeout=0, show_clicks=True)
                # if empty break the loop and go to next label
                if len(seed) == 0:
                    break
                else:
                    # round to pixel location
                    seed = np.round(seed[0]).astype(int)     
                # if user clicks on erase, delete the last selection
                if seed[0] > 0.95*im_ms.shape[1] and seed[1] < 0.05*im_ms.shape[0]:
                    if len(sand_pixels) > 0:
                        im_labels[sand_pixels[-1]] = 0
                        for k in range(im_viz.shape[2]):                              
                            im_viz[sand_pixels[-1],k] = im_RGB[sand_pixels[-1],k]
                        implot.set_data(im_viz)
                        fig.canvas.draw_idle() 
                        del sand_pixels[-1]

                # otherwise label the selected sand pixels
                else:
                    # flood fill the NDVI and the NDWI
                    fill_NDVI = flood(im_NDVI, (seed[1],seed[0]), tolerance=settings['tolerance'])
                    fill_NDWI = flood(im_NDWI, (seed[1],seed[0]), tolerance=settings['tolerance'])
                    # compute the intersection of the two masks
                    fill_sand = np.logical_and(fill_NDVI, fill_NDWI)
                    im_labels[fill_sand] = settings['labels']['sand'] 
                    sand_pixels.append(fill_sand)
                    # show the labelled pixels
                    for k in range(im_viz.shape[2]):                              
                        im_viz[im_labels==settings['labels']['sand'],k] = color_sand[k]
                    implot.set_data(im_viz)
                    fig.canvas.draw_idle() 


            ##############################################################
            # digitize white-water pixels
            ##############################################################
            color_ww = settings['colors']['white-water']
            ax.set_title('Click on individual WHITE-WATER pixels (no flood fill)\nwhen finished press <Enter>')
            fig.canvas.draw_idle() 
            ww_pixels = []                        
            while 1:
                seed = ginput(n=1, timeout=0, show_clicks=True)
                # if empty break the loop and go to next label
                if len(seed) == 0:
                    break
                else:
                    # round to pixel location
                    seed = np.round(seed[0]).astype(int)     
                # if user clicks on erase, delete the last labelled pixels
                if seed[0] > 0.95*im_ms.shape[1] and seed[1] < 0.05*im_ms.shape[0]:
                    if len(ww_pixels) > 0:
                        im_labels[ww_pixels[-1][1],ww_pixels[-1][0]] = 0
                        for k in range(im_viz.shape[2]):
                            im_viz[ww_pixels[-1][1],ww_pixels[-1][0],k] = im_RGB[ww_pixels[-1][1],ww_pixels[-1][0],k]
                        implot.set_data(im_viz)
                        fig.canvas.draw_idle()
                        del ww_pixels[-1]
                else:
                    im_labels[seed[1],seed[0]] = settings['labels']['white-water']  
                    for k in range(im_viz.shape[2]):                              
                        im_viz[seed[1],seed[0],k] = color_ww[k]
                    implot.set_data(im_viz)
                    fig.canvas.draw_idle()
                    ww_pixels.append(seed)
                    
            im_sand_ww = im_viz.copy()
            btn_erase.set(text='<Esc> to Erase', fontsize=12)

            ##############################################################
            # digitize water pixels (with lassos)
            ##############################################################
            color_water = settings['colors']['water']
            ax.set_title('Click and hold to draw lassos and select WATER pixels\nwhen finished press <Enter>')
            fig.canvas.draw_idle() 
            selector_water = SelectFromImage(ax, implot, color_water)
            key_event = {}
            while True:
                fig.canvas.draw_idle()                         
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                if key_event.get('pressed') == 'enter':
                    selector_water.disconnect()
                    break
                elif key_event.get('pressed') == 'escape':
                    selector_water.array = im_sand_ww
                    implot.set_data(selector_water.array)
                    fig.canvas.draw_idle()                         
                    selector_water.implot = implot
                    selector_water.im_bool = np.zeros((selector_water.array.shape[0], selector_water.array.shape[1])) 
                    selector_water.ind=[]          
            # update im_viz and im_labels
            im_viz = selector_water.array
            selector_water.im_bool = selector_water.im_bool.astype(bool)
            im_labels[selector_water.im_bool] = settings['labels']['water']

            im_sand_ww_water = im_viz.copy()

            ##############################################################
            # digitize land pixels (with lassos)
            ##############################################################
            color_land = settings['colors']['other land features']
            ax.set_title('Click and hold to draw lassos and select OTHER LAND pixels\nwhen finished press <Enter>')
            fig.canvas.draw_idle() 
            selector_land = SelectFromImage(ax, implot, color_land)
            key_event = {}
            while True:
                fig.canvas.draw_idle()                         
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                if key_event.get('pressed') == 'enter':
                    selector_land.disconnect()
                    break
                elif key_event.get('pressed') == 'escape':
                    selector_land.array = im_sand_ww_water
                    implot.set_data(selector_land.array)
                    fig.canvas.draw_idle()                         
                    selector_land.implot = implot
                    selector_land.im_bool = np.zeros((selector_land.array.shape[0], selector_land.array.shape[1])) 
                    selector_land.ind=[]
            # update im_viz and im_labels
            im_viz = selector_land.array
            selector_land.im_bool = selector_land.im_bool.astype(bool)
            im_labels[selector_land.im_bool] = settings['labels']['other land features']  

            # save labelled image
            ax.set_title(imgFn)
            fig.canvas.draw_idle()   
            supportingFunctions.create_dir(labeledSavedData)                      
            fig.savefig(os.path.join(labeledSavedData,imgFn+'.jpg'), dpi=150)
            ax.clear()
            # save labels and features
            features = dict([])
            for key in settings['labels'].keys():
                im_bool = im_labels == settings['labels'][key]
                if not augmentationScale is None:
                    im_ms_pluss = im_ms + 0
                    im_ms_pluss[im_ms_pluss > 0] = im_ms_pluss[im_ms_pluss > 0] + augmentationScale
                    scaledUp = coastvision.calculate_features(im_ms_pluss, cloud_mask, im_bool) 
                    im_ms_minus = im_ms + 0
                    im_ms_minus[im_ms_minus > 0] = im_ms_minus[im_ms_minus > 0] - 300 # 300 is a known value that works
                    scaledDown = coastvision.calculate_features(im_ms_minus, cloud_mask, im_bool)
                    regular = coastvision.calculate_features(im_ms, cloud_mask, im_bool) 
                    features[key] = np.concatenate((regular, scaledUp, scaledDown), axis=0)
                else:
                    features[key] = coastvision.calculate_features(im_ms, cloud_mask, im_bool) 

                if features[key].shape [1] != 16:
                    print('what the what the what the')
                    return(features)
            training_data = {'labels':im_labels, 'features':features, 'label_ids':settings['labels']}
            with open(os.path.join(labeledSavedData, imgFn + '.pkl'), 'wb') as f:
                pickle.dump(training_data,f)
            ax.clear()
                
    # close figure when finished
    plt.close(fig)



from matplotlib.widgets import LassoSelector
from matplotlib.widgets import Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from skimage import io
from scipy import ndimage
#### Flexible hand pixel classification tool (not geared to the specific )
def binary_data_annotation_tool(modelName, traningDataDir, labeledSavedData, classes=None, cloundThresh=0.9, floodFillTolerance=0.02, augmentationScale=None):
    """
    This function pulls up a data annotation tool that can be used to hand classify pixels. The saved results can be used for training models.
    
    """
    classInfo = dataAnnotationTool.get_class_info()

    matplotlib.use("Qt5Agg")
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                          sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()
    

    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    for imgFn in os.listdir(traningDataDir):
        print(imgFn)
        im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(traningDataDir, imgFn))
        
        im_RGB = rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)    
        im_viz = im_RGB.copy()
        # im_labels = np.zeros([im_RGB.shape[0],im_RGB.shape[1]])
        
        ax.axis('off')  
        ax.imshow(im_RGB)
        implot = ax.imshow(im_viz, alpha=0.6)            
        ax.set_title(imgFn)

        while True:
            btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
            fig.canvas.draw_idle()                         
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()

            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get('pressed') == 'right':
                skip_image = False
                break
            elif key_event.get('pressed') == 'left':
                skip_image = True
                break
            elif key_event.get('pressed') == 'escape':
                plt.close()
                raise StopIteration('User cancelled labelling images')
            else:
                plt.waitforbuttonpress()

        # if user decided to skip show the next image
        if skip_image:
            ax.clear()
            continue
        # otherwise label this image
        else:
            ax.set_title('loading...')
            fig.canvas.draw_idle()
            if classes is None:
                className = class_text_field(plt, fig, ax)
            else:
                className = classes[0]
            print(className)
            # plt.waitforbuttonpress()
            # ax.imshow(im_RGB)
            # fig.canvas.draw_idle()

            # annotate_image_with_class(im_RGB, className, fig, ax)     


            # Create the buttons
            gs = gridspec.GridSpec(2, 2, top=0.2, left=0.1, right=0.3, bottom=0.1, wspace=0.1, hspace=0.1)
            ax_button_singlePoint = fig.add_subplot(gs[0, 0])
            button_singlePoint = Button(ax_button_singlePoint, 'Single point')
            button_singlePoint.on_clicked(single_point)

            ax_button_rectangle_tool = fig.add_subplot(gs[0, 1])
            button_rectangle_tool = Button(ax_button_rectangle_tool, 'Rectangle tool')
            button_rectangle_tool.on_clicked(lambda event: rectangle_tool(event, fig, ax))

            ax_button_flood_fill = fig.add_subplot(gs[1, 0]) #plt.axes([0.25, 0.9, 0.1, 0.075])
            button_flood_fill = Button(ax_button_flood_fill, 'Flood fill')
            button_flood_fill.on_clicked(flood_fill)

            ax_button_lasso_tool = fig.add_subplot(gs[1, 1]) #plt.axes([0.25, 0.9, 0.1, 0.075])
            button_lasso_tool = Button(ax_button_lasso_tool, 'Lasso tool')
            button_lasso_tool.on_clicked(lambda event: lasso_tool(event, fig, ax))

            plt.show()
            plt.waitforbuttonpress()
            ax.imshow(im_RGB)
            fig.canvas.draw_idle()

            plt.waitforbuttonpress()
            ax.clear()
            plt.close()
            return 0     


            # plt.waitforbuttonpress()
            # ax.clear()
            # plt.close()
            # return 0

def single_point(event):
    print('single point stub')


import matplotlib.patches as patches
from PIL import Image
def rectangle_tool(event, fig, ax):
    ax.set_title('Click two points to create a Region of Interest (ROI)',
                      fontsize=14)
    plt.draw()
    
    click_event = {}
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            click_event['point'] = (event.xdata, event.ydata)

    fig.canvas.draw_idle()                         
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.waitforbuttonpress()

    firstPoint = click_event['point']
    x1, y1 = firstPoint

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.waitforbuttonpress()

    secondPoint = click_event['point']
    x2, y2, = secondPoint


    # get bottom left corner of rectangle
    xLow = int(min([x1, x2]))
    ylow = int(max([y1, y2])) # although it is high because y starts counting from the top of the image
    width = int(max([x1, x2]) - xLow) # this is to get distance between
    height = -int(ylow - min([y1, y2])) # because we are using x and ylow which are already cast to ints this could be slightly more percise but right now it doesnt matter and reusing the variable saves some computation
    # NOTE: remember everything as ints because we are looking at individual pixels


    rect = patches.Rectangle((xLow, ylow), width, height, linewidth=4, edgecolor='b', facecolor='none')

    # Add the patch to the Axes        
    ax.add_patch(rect)
    ax.plot([x1, x2], [y1, y2], '--r')
    ax.set_title('Press on <delete> to digitize another ROI or on <enter> to finish and save the ROI',
              fontsize=14)
    plt.draw() # to update changes
    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key
        
    fig.canvas.draw_idle()                         
    fig.canvas.mpl_connect('key_press_event', press)
    plt.waitforbuttonpress()
    
    print(key_event['pressed'])
    
    if key_event['pressed'] == 'enter':
        ax.clear()
        plt.close(fig)

        print(fig.canvas.get_width_height())
        canvasWidth, canvasHeight = fig.canvas.get_width_height()
        # Create an array of zeros with the same size as the canvas
        mask = np.zeros((canvasHeight, canvasWidth), dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1 # Set the values inside the box to 1
        mask = Image.fromarray(np.uint8(mask*255)) # Convert the numpy array to a PIL image
        return mask
    elif key_event['pressed'] == 'delete':
        print("better luck next time boy")

def flood_fill(event):
    # Code for flood fill tool
    print('yeah ium bavk to hea')

from matplotlib.widgets import LassoSelector
def lasso_tool(event, fig, ax):
    print('Lasso tool activated')
    plt.gca().set_rasterization_zorder(1)
    plt.gca().set_rasterization_zorder(1)
    plt.gcf().canvas.draw_idle()
    plt.waitforbuttonpress()
    print('Lasso tool deactivated')
    # print('Lasso tool activated')
    # ax.set_title('Select region with Lasso tool')

    # # Create a LassoSelector instance
    # lasso = LassoSelector(ax, onselect=lasso_callback)
    # cid = fig.canvas.mpl_connect('button_press_event', lasso_tool)
    # fig.canvas.mpl_disconnect(cid)

    # LEAVING OFF HEA NEED FA FINISH DIS

def lasso_callback(vertices):
    # Code to handle selected region goes here
    print('Selected region:', vertices)



def class_text_field(plt, fig, ax):
    plt.subplots_adjust(bottom=0.2)
    ax_box = plt.axes([0.1, 0.9, 0.1, 0.075])
    textbox = TextBox(ax_box, 'enter class name:', initial='land')
    orig_title = ax.get_title()
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()

    def update_title(text):
        ax.set_title(f'digitize pixels for {text} class')
        fig.canvas.draw_idle()
        global textbox_text
        textbox_text = text

    global textbox_text 
    textbox_text = None

    def textbox_submit():
        global textbox
        global textbox_text
        textbox_text = textbox.text
        # textbox.remove()
        textbox.visible = False # Hide the textbox
        ax_box.visible = False # Hide the ax_box
        ax.set_title(orig_title)
        ax.set_xlim(orig_xlim)
        ax.set_ylim(orig_ylim)
        # fig.canvas.draw_idle()

    textbox.on_submit(update_title)
    fig.canvas.draw_idle()
    def on_key_press(event):
        if event.key == 'enter':
            textbox_submit()

    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event))

    while textbox_text is None:
        plt.waitforbuttonpress()

    ax_box.set_visible(False)
    # del textbox
    # del ax_box
    return(textbox_text)