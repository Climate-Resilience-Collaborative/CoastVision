"""
This module houses pixel annotation tools allowing the user to classify pixels in tiff images into multiple categories
This script includes components dirived from https://github.com/jnicolow/Python-pixel-annotation-tool

Author: Joel Nicolow, Climate Resiliance Colaborative, School of Ocean and Earth Science and Technology (March, 28 2023)
"""

import os
import json
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from coastvision.classifier import data_annotation_tool
from coastvision import geospatialTools


from matplotlib.widgets import LassoSelector
from matplotlib.widgets import Button, TextBox
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from skimage import io
from scipy import ndimage

def get_class_info(modelName=None):
    """
    This function creates a dictionary from a json that contains info about the model classes (sand, water, etc.)

    :return: Dictionary with the class labels and colors
    """
    if not modelName is None:
        classInfoPath = os.path.join(os.getcwd(), 'coastvision', 'classifier', f'class_info_{modelName}.json')
    else:
        classInfoPath = os.path.join(os.getcwd(), 'coastvision', 'classifier', 'class_info.json')
    with open(classInfoPath) as f:
        data = json.load(f)
        f.close()
    
    return(data)



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
    ax_box.remove()
    return(textbox_text)


def pixel_annotation_tool(modelName, traningDataDir, labeledSavedData):
    """
    This function pulls up a data annotation tool that can be used to hand classify pixels. The saved results can be used for training models.
    
    :param modelName: String name of the mode (e.g. snowClassifier)
    :param traningDataDir: String filepath to where the images to annotate are
    :param labeledSavedData: String filepath to where annotations should be saved

    :return: dictionary where each image file name has a dictionary containing masks for each class
    """
    try:
        classInfo = get_class_info(modelName)
        classIdx = 0
    except FileNotFoundError:
        try:
            classInfo = get_class_info(None) # check for default class info file
            classIdx = 0
        except FileNotFoundError:
            colors = [[1, 0.65, 0], [1,0,1], [0.1,0.1,0.7], [0.8,0.8,0.1]]
            colorIdx = 0

    # matplotlib.use('TkAgg')  # tkinter back end
    matplotlib.use("Qt5Agg")
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                          sharey=True)
    mng = plt.get_current_fig_manager()  
    # mng.window.showMaximized()
    

    key_event = {}
    def press(event):
        # store what key was pressed in the dictionary
        key_event['pressed'] = event.key

    # let the user press a key, right arrow to keep the image, left arrow to skip it
    # to break the loop the user can press 'escape'
    imgAnnotations = {}
    for imgFn in os.listdir(traningDataDir):
        print(imgFn)
        im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(traningDataDir, imgFn))
        
        im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)    
        im_viz = im_RGB.copy()
        # im_labels = np.zeros([im_RGB.shape[0],im_RGB.shape[1]])
        
        ax.axis('off')  
        ax.imshow(im_RGB)
        implot = ax.imshow(im_viz, alpha=0.3)            
        ax.set_title(imgFn)
        classIdx = 0 # only one of these is going to be used
        colorIdx = 0

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
                ax.clear()
                plt.close()
                print('User exited annotation tool')
                return(imgAnnotations)
                # raise StopIteration('User cancelled labelling images')
            else:
                plt.waitforbuttonpress()

        # if user decided to skip show the next image
        if skip_image:
            ax.clear()
            continue
        # otherwise label this image
        else:
            # classifiedPixelsMask = np.zeros_like(ax.get_images()[0].get_array()[:,:,0], dtype=np.uint8) # Initialize mask with all zeros
            maskDict = {}
            categorizedPixelsMask = np.zeros(im_viz.shape[:2], dtype=bool)
            while True:
                ax.set_title('loading...')
                fig.canvas.draw_idle()
                nextClass = True # set to false if there is no next class
                # get class
                try:
                    classInfo
                except NameError:
                    # variable is not defined
                    className = class_text_field(plt, fig, ax)
                    color = colors[colorIdx]
                    classLabel = colorIdx
                    colorIdx += 1
                else:
                    # variable is defined
                    classes = list(classInfo['labels'].keys())
                    className =  classes[classIdx]
                    classIdx += 1
                    if classIdx == len(classes):
                        nextClass = False
                    classLabel = classInfo['labels'][className]
                    color = classInfo['colors'][className]
                ax.set_title(f'classifying pixels for {className}')
                classTitle = ax.get_title()
                fig.canvas.draw_idle()
                print(className)

                # annotate!
                mask = fig_annotate(fig, ax, implot, im_viz, im_RGB, categorizedPixelsMask, color)
                categorizedPixelsMask = np.logical_or(categorizedPixelsMask, mask)
                maskDict[className] = np.array(mask, dtype=bool)
                # maskLabel = mask*int(classLabel) # example if class label is 2 then now the true values are 2
                # print(maskLabel)
                # print(classifiedPixelsMask)
                # classifiedPixelsMask = classifiedPixelsMask | maskLabel


                btn_next_class = ax.text(1.1, 0.9, 'next class ⇨', size=12, ha="right", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
                btn_esc = ax.text(0.5, 0, '<esc> select next image', size=12, ha="center", va="top",
                                transform=ax.transAxes,
                                bbox=dict(boxstyle="square", ec='k',fc='w'))
                fig.canvas.draw_idle()                         
                fig.canvas.mpl_connect('key_press_event', press)
                plt.waitforbuttonpress()
                btn_next_class.remove()
                btn_esc.remove()

                if key_event.get('pressed') == 'right':
                    ax.set_title(classTitle)
                    if not nextClass:
                        break
                    continue # go back to
                elif key_event.get('pressed') == 'escape':
                    break
                else:
                    plt.waitforbuttonpress()
        # save class masks to labeledSavedData directory
        with open(os.path.join(labeledSavedData, f'{imgFn}_annotated.pkl'), 'wb') as f:
            pkl.dump(maskDict, f)
        fig.savefig(os.path.join(labeledSavedData, f'{imgFn}_annotated.jpg')) # save a visual of the categorizations
        # combine to final output (note we save individually per image because we are saving annotations as image masks)
        imgAnnotations[imgFn] = maskDict
    ax.clear()
    plt.close()
    return(imgAnnotations)

           

def fig_annotate(fig, ax, implot, im_viz, img_RGB, categorizedPixelsMask=None, color=[1, 0, 1]):
    
    ax.set_title(f'{ax.get_title()}\nAnnotation tool, to select a tool click the number key corresponding to the tool')
    fig.canvas.draw_idle()
    tool_selection = {}
    def on_tool_select(event):
        tool_selection['pressed'] = event.key
    mask = np.zeros_like(im_viz[:,:,0], dtype=np.uint8)
    if categorizedPixelsMask is None:
        categorizedPixelsMask = mask
    # mask any pixel that is completely black in im_viz
    black_pixels = (im_viz[:, :, 0] == 0) & (im_viz[:, :, 1] == 0) & (im_viz[:, :, 2] == 0)
    categorizedPixelsMask[black_pixels] = 1
    while True:
        btn_lasso = ax.text(-0.1, 0.80, 'Lasso: 3', size=12, ha="right", va="top",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_floodfill = ax.text(-0.1, 0.85, 'flood fill: 2', size=12, ha="right", va="top",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_single = ax.text(-0.1, 0.9, 'single pixel: 1', size=12, ha="right", va="top",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="square", ec='k',fc='w'))
        btn_quit = ax.text(0.5, 0, '<esc> to finish this class', size=12, ha="center", va="top",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="square", ec='k',fc='w'))
        fig.canvas.draw_idle()                         
        fig.canvas.mpl_connect('key_press_event', on_tool_select)
        plt.waitforbuttonpress()
        # after button is pressed, remove the buttons
        btn_lasso.remove()
        btn_floodfill.remove()
        btn_single.remove()
        btn_quit.remove()

        classTitle = ax.get_title()
        # keep/skip image according to the pressed key, 'escape' to break the loop
        if tool_selection.get('pressed') == '1':
            newMask = single_points(fig, ax, implot, im_viz, color)
        elif tool_selection.get('pressed') == '3':
            newMask = lasso_tool(fig, ax, implot, im_viz, categorizedPixelsMask, color)
        elif tool_selection.get('pressed') == '2':
            newMask = flood_fill(fig, ax, implot, img_RGB, im_viz, color)
            pass
        elif tool_selection.get('pressed') == 'enter':
            break # go to next class
        elif tool_selection.get('pressed') == 'escape':
            # ax.clear() # need so little window is not left
            # plt.close()

            return mask
        else:
            plt.waitforbuttonpress()
            newMask = mask # no changes were made

        # reset title (after it was changed by tools)
        ax.set_title(classTitle)
        fig.canvas.draw_idle()        

        # oldMask = mask
        mask = np.logical_or(mask, newMask)



def single_points(fig, ax, implot, im_viz, color=[1, 0, 1]):
    print('select points tool')
    ax.set_title('select individual pixels. Press q or enter to quit')
    fig.canvas.draw_idle()

    points = []  # list to store selected points
    
    mask = np.zeros_like(im_viz[:,:,0], dtype=np.uint8) # Initialize mask with all zeros (using im_viz for the mask because it updates)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            point = (event.xdata, event.ydata)
            ix, iy = int(event.xdata + 0.5), int(event.ydata + 0.5)
            mask[iy, ix] = 1
            points.append(point)
            
            im_viz[np.array(mask, dtype=bool)] = color
            implot.set_data(im_viz)
            # implot.set_data(convert_bool_mask_to_color_mask(mask, color, invert=False)) # this will overwrite older points
            fig.canvas.draw_idle()



    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    key_events = {}
    def finishtool(event):
        key_events['pressed'] = event.key

    while True:
        fig.canvas.mpl_connect('key_press_event', finishtool)
        plt.waitforbuttonpress()
        if key_events.get('pressed') == 'q' or key_events.get('pressed') == 'enter':
            fig.canvas.mpl_disconnect(cid)
            break

    return mask


from skimage import morphology, color
from matplotlib.widgets import Slider
def flood_fill(fig, ax, implot, im_RGB, im_viz, maskColor=[1, 0, 1], tolerance = 0.009):

    im_viz_old = im_viz.copy()
    ax.set_title('Click to begin flood filling!\nPress enter to finish, delete to undo the last flood fill',
        fontsize=14)
    
    hsv = color.rgb2hsv(im_RGB)
    print(hsv.shape)
    hue = hsv[:, :, 0] # used for flood
     
    def onclick(event):
        if event.xdata != None and event.ydata != None:
            eventsDict['point'] = (event.xdata, event.ydata)
    def press(event):
            # store what key was pressed in the dictionary
            eventsDict['pressed'] = event.key

    def update(val):
        # update the tolerance value
        tolerance = slider.val
        update.tolerance = tolerance
    
    # create the slider
    slider_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
    slider = Slider(slider_ax, 'Flood fill tolerance', 0, 1, valinit=0.02)
    slider.on_changed(update)
    update.tolerance = tolerance

    mask = np.ones_like(im_viz[:,:,0], dtype=np.uint8)
    # mask = np.ones_like(ax.get_images()[0].get_array()[:,:,0], dtype=np.uint8) # Initialize mask with all zeros
    oldMasks = []
    while True:
        eventsDict = {}
        fig.canvas.draw_idle()      
        fig.canvas.mpl_connect('button_press_event', onclick) # check for where user clicks
        fig.canvas.mpl_connect('key_press_event', press) # check for if user pushes escape or unde
        plt.waitforbuttonpress()

        if 'point' in eventsDict:
            x, y = eventsDict['point']
            x, y = int(x), int(y) # convert to closes pixel (not 100% perfect)
            filled = morphology.flood(hue, (y, x), tolerance=update.tolerance)
            mask[filled] = False
            # Update the plot
            # pltMask = ~mask
            im_viz[~np.array(mask, dtype=bool)] = maskColor
            implot.set_data(im_viz)
            # implot.set_data(convert_bool_mask_to_color_mask(pltMask, maskColor, invert=True))
            plt.draw()
            maskClean = np.ones_like(im_viz[:,:,0], dtype=np.uint8)
            # maskClean = np.ones_like(ax.get_images()[0].get_array()[:,:,0], dtype=bool) # all True
            maskClean[filled] = False
            oldMasks.append(maskClean)

        elif 'pressed' in eventsDict:
            if eventsDict['pressed'] == 'delete':
                # undo floods
                if len(oldMasks) > 1:
                    print(mask.shape)
                    print(oldMasks[-1].shape)
                    oldMasks.pop()
                    mask = oldMasks[-1].copy() # reset it to the mask before last
                    # mask[~oldMasks.pop()] = True                     
                else:
                    # mask[filled] = True
                    mask = np.ones_like(im_viz[:,:,0], dtype=bool)
                im_viz[np.array(mask, dtype=bool)] = im_viz_old[np.array(mask, dtype=bool)] # revert pixels to original
                implot.set_data(im_viz)
                # implot.set_data(convert_bool_mask_to_color_mask(mask, maskColor, invert=True)) # shows plot missing last flood
            elif eventsDict['pressed'] == 'enter':
                slider_ax.remove()
                return np.logical_not(mask) # our mask is inverted to must invert again


def convert_bool_mask_to_color_mask(boolMask, color=[1, 0, 1], invert=False):
    """
    This takes a boolean mask and converts to color mask

    :param boolMask: 2d np array type boolean
    :param color: array of length three (value for each color channel ex: [1, 0, 1])
    """
    # set color
    if invert:
        boolMask = ~np.array(boolMask, dtype=bool) # just double check a bool arr
    else:
        boolMask = np.array(boolMask, dtype=bool) # just double check a bool arr
    colorMask = np.zeros(boolMask.shape + (4,), dtype=np.float32) # empty 3d image
    # Set the color value for all True values in bool_mask
    colorMask[boolMask] = tuple(color + [255]) # add alpha channel so that there isnt a tint to the non=selected pixels
    return colorMask



from matplotlib.widgets import LassoSelector
from matplotlib import path
class LassoPixels(object):
    def __init__(self, fig, ax, implot, im_viz, previousMask=None, color=[1, 0, 1]):
        self.fig = fig
        self.ax = ax
        self.im = implot
        self.im_viz = im_viz
        self.im_viz_old = im_viz
        self.previousMask = previousMask
        self.color = color
        xv, yv = np.meshgrid(np.arange(self.im_viz.shape[1]),np.arange(self.im_viz.shape[0]))
        self.pix = np.vstack( (xv.flatten(), yv.flatten()) ).T
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.mask = np.zeros((self.im_viz.shape[0], self.im_viz.shape[1]), dtype=int)
        self.thisMask = self.mask
        self.oldMasks = []

    def onselect(self, vertices):
        lassoPath = path.Path(vertices=vertices)
        inLasso = lassoPath.contains_points(self.pix, radius=1)
        thisMask = np.zeros(self.im_viz.shape[:2], dtype=bool)
        # self.previousMask = None
        if not self.previousMask is None:
            # mask_filter = np.logical_not(self.previousMask)
            inLasso = inLasso.reshape(thisMask.shape)
            # print(np.unique(self.previousMask))
            thisMask[np.logical_and(~self.previousMask, inLasso)] = True
        else:
            thisMask.flat[self.pix[:,1]*self.im_viz.shape[1]+self.pix[:,0]] = inLasso
        self.im_viz[np.array(thisMask, dtype=bool)] = self.color
        self.im.set_data(self.im_viz)
        self.fig.canvas.draw_idle()
        self.thisMask = thisMask
        self.oldMasks.append(self.mask)
        self.mask = self.mask | thisMask
       
    def disconnect(self):
        self.lasso.disconnect_events()

def lasso_tool(fig, ax, implot, im_viz, previousMask=None, color=[1, 1, 1]):

    ax.set_title('Lasso tool')
    fig.canvas.draw_idle()
    print('lasso tool')
    im_viz_old = im_viz.copy()
    lasso = LassoPixels(fig, ax, implot, im_viz, previousMask, color=color)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            eventsDict['point'] = (event.xdata, event.ydata)
    
    def press(event):
            # store what key was pressed in the dictionary
            eventsDict['pressed'] = event.key

    while True:
        eventsDict = {}
        fig.canvas.draw_idle()      
        fig.canvas.mpl_connect('button_release_event', onclick)  # call onclick only when Lasso tool is active
        fig.canvas.mpl_connect('key_press_event', press)
        plt.waitforbuttonpress()


        # if 'point' in eventsDict:
        #     thisMask = lasso.get_mask(thisMask=True)
        #     oldMasks.append(thisMask)
        #     print(len(oldMasks))
        #     mask = mask | thisMask
        if 'pressed' in eventsDict:
            if eventsDict['pressed'] == 'delete':
                # undo all lasso grabs
                lasso.mask = np.zeros_like(ax.get_images()[0].get_array()[:,:,0], dtype=np.uint8) # Initialize mask with all zeros

                lasso.im_viz = im_viz_old.copy()
                implot.set_data(lasso.im_viz)
                fig.canvas.draw_idle()
                # implot.set_data(convert_bool_mask_to_color_mask(mask, maskColor, invert=True)) # shows plot missing last flood
            elif eventsDict['pressed'] == 'enter':
                lasso.disconnect()
                return lasso.mask 

   



