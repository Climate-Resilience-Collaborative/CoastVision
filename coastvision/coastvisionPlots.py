"""
This script contains functions used to plot outputs of coastvision.

Author: Joel Nicolow, Climate Resiliance Initiative, School of Ocean and Earth Science and Technology (August, 03 2022)

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import plotting_extent
import rasterio
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry import LineString
import glob

from coastvision import supportingFunctions

def add_transects_to_ax(ax, transects, toaPath, transect_labels=True, color='black'):
    """
    pass transects in real world coordinates (same format they come out of the intersection function)
    """
    # Plot transects
    georef = geospatialTools.get_georef(toaPath)
    for key, points in transects.items():
        points = geospatialTools.convert_world2pix(points, georef)
        ax.plot(points[:, 0], points[:, 1], color=color, label='transect ' + str(key))#, zorder=10)  # Modify the label if needed
        if transect_labels:
            end_point = points[-1, :]  # Get the last point of the transect
            ax.text(end_point[0], end_point[1], key, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'), zorder=11)  # Add text box with transect name


def add_shoreline_to_ax(ax, sl, color='black', ls='dashed'):
    ax.plot(sl[:,0], sl[:,1], color = color, ls = ls, linewidth = 1.5, alpha=1)


def add_transect_intersections_to_ax(ax, transects, sl, toaPath):
    georef = geospatialTools.get_georef(toaPath)
    shapely_sl = LineString(sl)
    for _, points in transects.items():
        # Create a LineString from the dictionary points
        points = geospatialTools.convert_world2pix(points, georef) # convert to pixel coordinates (would be slightly faster to do once with all transects)
        transect = LineString(points)
        
        # Check if the lines intersect
        if shapely_sl.intersects(transect):
            # Find the intersection point(s)
            intersection_point = shapely_sl.intersection(transect)
            
            # Plot the intersection point
            if intersection_point.geom_type == 'Point':
                ax.plot(intersection_point.x, intersection_point.y, 'ro', markersize=3)  # Plot as red dot
            elif intersection_point.geom_type == 'MultiPoint':
                for pt in intersection_point:
                    ax.plot(pt.x, pt.y, 'ro')



def plot_classif_and_intersection(toaPath, im_classif, transects, sl, slBuff):
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(toaPath)
    dir1 = os.path.dirname(toaPath) # this code ASSUMES that the file structure of the image is region/sitename/image
    sitename = os.path.basename(dir1)
    region = os.path.basename(os.path.dirname(dir1))
    # referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, '*.tif'))[0] # glb returns list


    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_class = np.copy(im_RGB)
    water = 1
    # Apply colors
    classInfo = dataAnnotationTool.get_class_info()
    colors = classInfo['colorsForBinaryClassif']
    im_class[im_classif != water] = colors['land'] # just showing water
    # im_class[im_classif == land] = colors['land']

    # Set nan color
    im_class = np.where(np.isnan(im_class), 0.3, im_class)

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))  # 1 row, 2 columns

    # Display the first image (im_RGB) in the first subplot
    axs[0].imshow(im_RGB)
    axs[0].set_title("RGB Image")
    axs[0].axis('off')  # Hide axis for better visualization

    # Display the second image (im_class) in the second subplot
    axs[1].imshow(im_class)
    axs[1].set_title("Image Segmentation")
    axs[1].axis('off')

    axs[2].imshow(im_RGB)
    axs[2].set_title("Extracted Shoreline")
    axs[2].axis('off')  # Hide axis for better visualization

    # add_shoreline_to_ax(axs[0], sl=sl)
    # add_transects_to_ax(axs[2], transects=transects, toaPath=toaPath, transect_labels=False, color='black')

    add_shoreline_to_ax(axs[2], sl=sl, color='red', ls='solid')

    # get transect intersections

    axs[3].imshow(im_RGB)
    axs[3].set_title("Transect Intersections")
    axs[3].axis('off')  # Hide axis for better visualization
    add_transects_to_ax(axs[3], transects=transects, toaPath=toaPath, transect_labels=False, color='black')
    add_transect_intersections_to_ax(axs[3], transects=transects, sl=sl, toaPath=toaPath)

    plt.tight_layout()


    supportingFunctions.create_dir(os.path.join('outputs', region, sitename, 'stages'))
    fig.savefig(os.path.join('outputs', region, sitename, 'stages', f'stages_plot_{os.path.basename(toaPath)}.jpg'), dpi=300, bbox_inches='tight')
    # Show the figure with both images
    plt.show()


def plot_im_classif_single(im_classif, im_classif_raw=None, im_classif_dict=None, title="unknown", savePath=None, toaPath=None):
    """
    plot the gray scale image classifications

    :param im_classif: np.array each value represents a pixel value (value based on class)
    :param savePath: path where plot should be saved (default=None)
    """
    if not im_classif_raw is None:
        figure, axis = plt.subplots(1, 2, figsize=(20,60))
        im = axis[0].imshow(im_classif_raw, cmap=plt.cm.gray)
        axis[0].set_title(f"{title} raw classification") 
        im = axis[1].imshow(im_classif, cmap=plt.cm.gray)
        axis[1].set_title(f"{title} classification - binary dilation and small object remove")
    elif not im_classif_dict is None:
        figure, axis = plt.subplots(1, len(im_classif_dict), figsize=(30,10))
        counter = 0
        for classifType, classif in im_classif_dict.items():
            im = axis[counter].imshow(classif, cmap=plt.cm.gray)
            axis[counter].set_title(f"{title} - {classifType}")

            counter += 1        
    else:
        if not toaPath is None:
            figure, axis = plt.subplots(1, 2, figsize=(10,10))

            im_ms, cloud_mask = geospatialTools.get_ps_no_mask(toaPath)
            im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            axis[0].imshow(im_RGB)

            im = axis[1].imshow(im_classif, cmap=plt.cm.gray)
            axis[1].set_title(f"{title} classification - binary dilation and small object remove")
        else:
            figure, axis = plt.subplots( figsize=(10,10))
            im = axis.imshow(im_classif, cmap=plt.cm.gray)
            axis.set_title(f"{title} classification - binary dilation and small object remove")

    if not savePath is None:
        figure.savefig(savePath)
    #plt.close()
    #return(figure)


def plot_shoreline_contour_single(contour, savePath=None, title="no title provided", im_classif=None):
    """
    plot shoreline contour for one satalite image

    :param contour: np array (x, 2) of x,y coordinates for contour in real world cordinates
    :param savePath: String file path to where the plot should be saved, if None plot is not saved (default=None)
    :param title: String plot title (default="no title provided")
    :param im_classif: array of image classification (currently not using) (default=None)
    """

    fig, ax = plt.subplots(figsize=(10,10))
    # if not im_classif is None:
    #     ax.imshow(im_classif, cmap=plt.cm.gray)

    # contour = geospatialTools.convert_world2pix(contour, georef)

    ax.plot(contour[:, 1], contour[:, 0], linewidth=3)


    ax.axis('image')

    fig.suptitle(title, fontsize=13)
    if not savePath is None:
        fig.savefig(savePath)
    plt.close()

    #return(fig)


def plot_transect_intersections_single(df, transects, baseSavePath=None):
    """
    Plots points where shoreline contour intersects transects using distance along the transect that the shoreine crossed it

    :param df: Data.Frame colnames are transect names, rownames are timestamps of when the image was taken, cells are the distance along transect where intersection occured (from first point of transect)
    :param transects: Dictionary of numpy arrays (one for each transect) of (2, 2) [[x1, y1],[x2, y2]]

    :return: intersection points between contour an transects
    """

    for row in range(0, df.shape[0], 1):
        ts = df.iloc[row]['timestamp']
        thisImage = df[df['timestamp'] == ts].reset_index()
        thisImage.drop('index', axis=1, inplace=True)

        # transectPath = os.path.join(os.getcwd(), 'user_inputs', 'lanai', 'lanai0001', ('lanai0001' + "_transects.geojson"))
        # transects = coastvision.transects_from_geojson(transectPath)


        fig, ax = plt.subplots(figsize=(10,10))

        intersects = {}
        for transectKey in thisImage.columns:
            if not transectKey == 'timestamp':
                dist = thisImage.loc[0, transectKey]
                if not np.isnan(dist):
                    intersects[transectKey] = supportingFunctions.calculate_intersect_point(dist, transects[transectKey])
                    ax.plot([transects[transectKey][0][1], transects[transectKey][1][1]], [transects[transectKey][0][0], transects[transectKey][1][0]], linewidth=1, color='red')
                    ax.plot(intersects[transectKey]['y'], intersects[transectKey]['x'], 'bo')
                else:
                    # this is so we plot even the transects that weren't itersected (would save time to not plot them)
                    ax.plot([transects[transectKey][0][1], transects[transectKey][1][1]], [transects[transectKey][0][0], transects[transectKey][1][0]], linewidth=1, color='red')
        if not baseSavePath is None:
            rawTimestamp = supportingFunctions.convert_dt_to_raw_planet_timestamp(ts)
            savePath = os.path.join(baseSavePath, f'{rawTimestamp}_transect_intersections.png')
            fig.savefig(savePath)
            plt.close()
    plt.close()
    #return(fig,ax)


from coastvision import coastvision
from coastvision import geospatialTools
from coastvision.classifier import data_annotation_tool 
def rgb_plot_tif(tifPath):
    """
    Plots a tif image in matplotlib
    """
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tifPath)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    fig, ax = plt.subplots(figsize=(10,10))

    # Plot classes over RGB
    ax.imshow(im_RGB)
    plt.close()
    #return(fig)


def rgb_plot_tif_transect_intersection(tifPath, transects_gpd, transects, region, sitename, cross_dist, savePath, intersections=None):
    """
    Plots a tif image in matplotlib
    """
    #get tif file and extent
    tif = rasterio.open(tifPath) 
    extent =plotting_extent(tif)

    #get tif image as rgb
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tifPath)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    
    #create figure and plot RGB
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im_RGB, extent =extent)

    #get intersecting points
    if cross_dist is None:
        return None
    intersects = {}
    for key in cross_dist.keys():
        cross_dist_n = cross_dist[key]
        #print(key, cross_dist_n)
        if not np.isnan(cross_dist_n):
            intersects[key] = supportingFunctions.calculate_intersect_point(cross_dist_n, transects[key])

    # from intersecting points, create a geodataframe (a bit redundant, but works to plot transects and points)
    x,y =[],[]
    for key in intersects.keys():
        x.append(intersects[key]['x'])
        y.append(intersects[key]['y'])
    geometry = [Point(xy) for xy in zip(x, y)]
    intersections = GeoDataFrame(np.arange(len(x)), crs="EPSG:32604", geometry=geometry)

    # add transects and intersects to plot
    intersections.plot(color = 'red', ax=ax)
    transects_gpd.plot(color = 'black', lw= 1, ax=ax)
    #save fig
    fig.savefig(savePath)
    plt.close()
    #return(fig,ax)


from coastvision.classifier import dataAnnotationTool 
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
def rgb_classif_plot_single(toaPath, im_classif, sl=None, slBuff=None, transects=None, savePath=None):
    """

    :param toaPath: path to rgb image
    :param im_classif: image classification in pixel (not real world) coordinates
    :param sl: image sl in pixel coordinates (default=None)
    :param slBuff: boolean array where True=area outside the slbuffer (default=None)
    :param transects: NOTE under construction (default=None)
    :param savePath: where to save the plot too (default=None)
    """

    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(toaPath)
    #################
    dir1 = os.path.dirname(toaPath) # this code ASSUMES that the file structure of the image is region/sitename/image
    sitename = os.path.basename(dir1)
    region = os.path.basename(os.path.dirname(dir1))
    referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, '*.tif'))[0] # glb returns list
    # referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, f'{sitename}_reference_*.tif'))[0] # glb returns list
    # im_ms, cloud_mask = geospatialTools.pad_misscropped_image(referenceImagePath, toaPath, im_ms, cloud_mask)
    ##########

    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_class = np.copy(im_RGB)
    # print('\n shape', im_ms.shape)
    # print('shape', im_RGB.shape)
    # print('class', im_classif.shape)

    water = 0
    # land = 1 # NOTE for now only plotting water (because if we plot both it just covers the entire image)

    # im_water = im_classif == water
    # im_land = im_classif == land

    # Apply colors
    classInfo = dataAnnotationTool.get_class_info()
    colors = classInfo['colorsForBinaryClassif']
    im_class[im_classif == water] = colors['water']
    # im_class[im_classif == land] = colors['land']
    
    # Set nan color
    im_class = np.where(np.isnan(im_class), 0.3, im_class)

    fig, ax = plt.subplots(1, 2, figsize=(20,10))

    # Plot classes over RGB
    # ax[0].imshow(im_RGB)
    # ax[1].imshow(im_class)
    
    # plot colors
    # orange_patch = mpatches.Patch(color=colors['land'], label='land')
    # land_patch = mpatches.Patch(color=[0.95,0.98,1.00], label='land shown as satalite image')
    blue_patch = mpatches.Patch(color=colors['water'], label='water')
    # pltHandles = [land_patch, blue_patch]
    pltHandles = [blue_patch]
    if not sl is None:
        # Plot shoreline
        ax[1].plot(sl[:,0], sl[:,1], 'k.', linewidth = 3, markersize = 0.7)     
        black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
        pltHandles.append(black_line)
    if not slBuff is None:
        masked = np.ma.masked_where(slBuff == 0, slBuff)
        ax[1].imshow(masked,alpha=.2,cmap = 'Reds')
        outsideSLBuff_patch = mpatches.Patch(color=[0.89,0.86,0.86], label='area outside reference shoreline buffer')
        pltHandles.append(outsideSLBuff_patch)

    # Add legend
    ax[1].legend(handles=pltHandles,
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)

    if not savePath is None:
        fig.savefig(savePath)
    plt.close()
    #return(fig,ax)

def rgb_contour_plot_single(toaPath, sl, savePath=None, loudShoreline=False):

    """
    """
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(toaPath)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im_RGB)

    # Plot shoreline
    if loudShoreline:
        ax.plot(sl[:,0], sl[:,1], 'ro', linewidth=4, markersize = 0.7) 
        shore_line = mlines.Line2D([],[],color='ro',linestyle='-', label='shoreline')
    else:
        # ax.plot(sl[:,0], sl[:,1], 'k.', linewidth = 3, markersize = 0.7)     
        ax.plot(sl[:,0], sl[:,1], color ='black', ls ='dashed', linewidth = 1.5, alpha=0.6)     
        shore_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    pltHandles = []
    pltHandles.append(shore_line)
    ax.legend(handles=pltHandles,
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)

    if not savePath is None:
        fig.savefig(savePath)

    plt.close()
    #return(fig,ax)


import geojson
def plot_all_site_shorelines(region, sitename):
    """
    plot all of the contours 
    """
    fig, ax = plt.subplots(figsize=(12,7))
    handles = []
    parentPth = os.path.join(os.getcwd(), 'outputs', region, sitename, 'shoreline', 'contours')
    for fn in os.listdir(parentPth):
        if 'contour.geojson' in fn:
            pth = os.path.join(parentPth, fn)
            with open(pth, 'r') as f:
                sl = geojson.load(f)
            refSl = np.array(sl['features'][0]['geometry']['coordinates'])
            if len(refSl):
                handles.append(fn.split('_contour')[0])
                ax.plot(refSl[:, 1], refSl[:, 0], linewidth=3)
    ax.legend(handles)
    fig.savefig(os.path.join(os.getcwd(), 'outputs', region, sitename, 'shoreline', f"{sitename}_all_contours.png"))
    plt.close()
    #return(fig,ax)


from matplotlib.backends.backend_pdf import PdfPages
def test_mode_create_plot_dict_single(toaPath, im_classif, im_classif_dict, sl, savePath=None):
    """
    creates classif, contoorvrbg, and classif and contourvrbg plots. Then if savePath is passed it will save them into one pdf
    """
    retuDict = {}
    retuDict['classif_plots'] = plot_im_classif_single(im_classif, im_classif_raw=None, im_classif_dict=im_classif_dict) # not saving this indicidually

    retuDict['rgb_contour_plot'] = rgb_contour_plot_single(toaPath, sl=sl)

    retuDict['rgb_classif_contour_plot'] = rgb_classif_plot_single(toaPath, im_classif=im_classif, sl=sl)

    if not savePath is None:
        with PdfPages(savePath) as pdf:
            pdf.savefig(retuDict['classif_plots'])
            pdf.savefig(retuDict['rgb_contour_plot'])
            pdf.savefig(retuDict['rgb_classif_contour_plot'])
    #return(fig,ax)
    #return(retuDict)

   
def test_mode_plot_all(testModeDict, savePath, imgByImg=False):
    """
    plots test_mode_create_plot_dict_single plots for every image to the same pdf

    currently there are issues with rgb_classif_contour_plot (Joel Nicolow, 2022-08-05)
    """

    if imgByImg:
        classif_plots = []
        contour_rgb_plots = []
        classif_contour_rgb_plots = []

        for _, plotsDict in testModeDict.items():
            classif_plots.append(plotsDict['classif_plots'])
            contour_rgb_plots.append(plotsDict['rgb_contour_plot'])
            classif_contour_rgb_plots.append(plotsDict['rgb_classif_contour_plot'])


        with PdfPages(savePath) as pdf:
            for figure in classif_plots:
                pdf.savefig(figure)
            for figure in contour_rgb_plots:
                pdf.savefig(figure)
            for figure in classif_contour_rgb_plots:
                pdf.savefig(figure)
    else:
        with PdfPages(savePath) as pdf:
            for _, plotsDict in testModeDict.items():
                for _, fig in plotsDict.items():
                    pdf.savefig(fig)
            

def get_smallest_rectangle_that_fits_slBuffer(slBuff):
    #print('shape of buffer is', np.shape(slBuff))
    if slBuff is None:
        return None
    i, j = np.where(slBuff==False)
    return([[np.min(i), np.min(j)], [np.max(i), np.max(j)]])


def master_data_products_plot(toaPath, im_classif, sl, slBuff, savePath=None):
    """

    :param toaPath: path to rgb image
    :param im_classif: image classification in pixel (not real world) coordinates
    :param sl: image sl in pixel coordinates (default=None)
    :param slBuff: boolean array where True=area outside the slbuffer (default=None)
    :param transects: NOTE under construction (default=None)
    :param savePath: where to save the plot too (default=None)
    """

    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(toaPath)
    
    
    #################
    dir1 = os.path.dirname(toaPath) # this code ASSUMES that the file structure of the image is region/sitename/image
    sitename = os.path.basename(dir1)
    region = os.path.basename(os.path.dirname(dir1))
    referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, '*.tif'))[0] # glb returns list
    # referenceImagePath = glob.glob(os.path.join('user_inputs', region, sitename, f'{sitename}_reference_*.tif'))[0] # glb returns list
    # im_ms, cloud_mask = geospatialTools.pad_misscropped_image(referenceImagePath, toaPath, im_ms, cloud_mask)
    ##########

    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    im_class = np.copy(im_RGB)
    water = 0

    # Apply colors
    classInfo = dataAnnotationTool.get_class_info()
    colors = classInfo['colorsForBinaryClassif']
    im_class[im_classif == water] = colors['water'] # just showing water
    # im_class[im_classif == land] = colors['land']
    
    # Set nan color
    im_class = np.where(np.isnan(im_class), 0.3, im_class)

    fig, ax = plt.subplots(2, 2, figsize=(20,10))

    # Plot classes over RGB (on all plots)
    ax[0, 0].imshow(im_RGB) # just the image
    ax[0, 1].imshow(im_RGB) # classification, shoreline, shoreline

    bb = get_smallest_rectangle_that_fits_slBuffer(slBuff)
    if bb is None: ###ABM if slBuff is none
        return
    #print(f'{bb[0][0]}:{bb[1][0]}, {bb[0][1]}:{bb[1][1]}')
    #print('----------------------------------')
    ax[1,0].imshow(im_RGB)#[ bb[0][0]:bb[1][0], bb[0][1]:bb[1][1] ]) # shoreline and slbuff # fitted to smallest rectangle that fits sl buffer
    ax[1,0].imshow(im_RGB)
    
    ax[1, 1].imshow(im_RGB)

    ax[0, 1].imshow(im_class)
    blue_patch = mpatches.Patch(color=colors['water'], label='water')
    pltHandles2 = [blue_patch]

    pltHandles3 = []
    pltHandles4 = []
    # Plot shoreline
    ax[0, 1].plot(sl[:,0], sl[:,1], 'k.', linewidth = 3, markersize = 0.7)     
    black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
    pltHandles2.append(black_line)
    # ax[1, 0].plot(sl[:,0], sl[:,1], 'k.', linewidth = 3, markersize = 0.7)     
    # pltHandles3.append(black_line)
    ax[1, 1].plot(sl[:,0], sl[:,1], 'k.', linewidth = 3, markersize = 0.7)     
    pltHandles4.append(black_line)
    # sl buffer
    outsideSLBuff_patch = mpatches.Patch(color=[0.89,0.86,0.86], label='area outside reference shoreline buffer')
    slBuffSmall = slBuff[bb[0][0]:bb[1][0], bb[0][1]:bb[1][1]]  # filtered down to the smallest rectangle that firs the entire reference shoreline
    maskedSmall = np.ma.masked_where(slBuffSmall == 0, slBuffSmall)
    ax[1, 0].imshow(maskedSmall,alpha=.6,cmap = 'Reds')
    pltHandles3.append(outsideSLBuff_patch)

    # Add legend
    ax[0, 1].legend(handles=pltHandles2,
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)

    ax[1, 0].legend(handles=pltHandles3,
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)

    ax[1, 0].legend(handles=pltHandles4,
                bbox_to_anchor=(0.5, 0), loc='upper center', fontsize=9,
                ncol = 6)
    if not savePath is None:
        
        fig.savefig(savePath)
    plt.close()
    #return(fig,ax)            

#### QAQC ####

def plot_rgb_img_with_extracted_sl_points(transectFn, tiffForBackgroundFn, df, transectName, savePath=None):
    """
    This function plots all of the points given in the df along the specified transect. Note the df just gives distance allong transect and then this is used to compute the point coordinates
    """
    transects = coastvision.transects_from_geojson(transectFn)
    georef = geospatialTools.get_georef(tiffForBackgroundFn)

    combinedPointsList = []

    for row in range(0, df.shape[0], 1):
        # each itteration represents a timestamp
        ts = df.iloc[row]['timestamp']
        thisImage = df[df['timestamp'] == ts].reset_index()
        thisImage.drop('index', axis=1, inplace=True)

        intersects = {}
        for transectKey in thisImage.columns:
            if not transectKey == 'timestamp' and transectKey == transectName:
                dist = thisImage.loc[0, transectKey]
                if not np.isnan(dist):
                    intersects[transectKey] = supportingFunctions.calculate_intersect_point(dist, transects[transectKey])
                    combinedPointsList.append( [intersects[transectKey]['x'], intersects[transectKey]['y']] )

    combinedPointsListPix = geospatialTools.convert_world2pix(combinedPointsList, georef)
    pixArr = np.array(combinedPointsListPix[0])

    fig, ax = plt.subplots(figsize=(10,15))

    ax.scatter(pixArr[:,0], pixArr[:,1], s=.5, color='red')


    # create limits
    transectConverted = geospatialTools.convert_world2pix(transects[transectName], georef)
    farthestLeft,farthestright  = max(transectConverted[:,0]), min(transectConverted[:,0])
    farthestup, farthestdown = min(transectConverted[:,1]), max(transectConverted[:,1])
    plt.xlim(farthestLeft-40, farthestright+40), plt.ylim(farthestdown, farthestup)


    # Plot classes over RGB
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tiffForBackgroundFn)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    ax.imshow(im_RGB)

    if not savePath is None:
        fig.savefig(savePath)


import datetime as dt
import pandas as pd
def plot_rgb_im_with_sl_points(tiffImage, transectFn, df, savePath = None):
    """
    This plots a tiff image with the corresponding transect intersections
    """
    transects = coastvision.transects_from_geojson(transectFn)
    georef = geospatialTools.get_georef(tiffImage)
    rawTimestamp = os.path.basename(tiffImage)[0:15]
    timestamp = dt.datetime.strptime(rawTimestamp, '%Y%m%d_%H%M%S')
    # timestamp = supportingFunctions.convert_planet_raw_timestamp_to_dt(rawTimestamp)

    df['timestamp'] =  pd.to_datetime(df['timestamp'])



    thisImageDf = df.loc[df['timestamp'] == timestamp]

    combinedPointsList = []
    intersects = {}
    for transectKey in thisImageDf.columns:
        if not transectKey == 'timestamp':
            dist = thisImageDf[transectKey].iloc[0]
            if not np.isnan(dist):
                intersects[transectKey] = supportingFunctions.calculate_intersect_point(dist, transects[transectKey])
                combinedPointsList.append( [intersects[transectKey]['x'], intersects[transectKey]['y']] )

    combinedPointsListPix = geospatialTools.convert_world2pix(combinedPointsList, georef)
    pixArr = np.array(combinedPointsListPix[0])
    
        

        
    fig, ax = plt.subplots(figsize=(10,15))

    # Plot classes over RGB
    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tiffImage)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
    ax.imshow(im_RGB)

    ax.scatter(pixArr[:,0], pixArr[:,1], s=3, color='red')

    # axis limit
    min_x, max_x = min(pixArr[:,0]), max(pixArr[:,0])
    min_y, max_y = min(pixArr[:,1]), max(pixArr[:,1])
    plt.xlim(min_x-40, max_x+40), plt.ylim(max_y+40, min_y-40)

    if not savePath is None:
        fig.savefig(savePath)

    
from matplotlib.backends.backend_pdf import PdfPages
def plot_rgb_im_with_sl_point_single(transectKey, df, regionName, siteName, savePathPDF=None):
    """
    This plots each point for a transect onto the image that it comes from (cropped around that area) and can save the results as a pdf
    This is good for qualitative testing of removed points by QAQC
    """
    transects = coastvision.transects_from_geojson(os.path.join(os.getcwd(), 'user_inputs', regionName, siteName, f'{siteName}_transects.geojson'))
    if not savePathPDF is None:
        pp = PdfPages(os.path.join(os.getcwd(), 'outputs', regionName, siteName, f'singletransectIntersections{transectKey}.pdf'))

    georef = None
    intersects = {}
    for row in range(0, df.shape[0], 1):
        # each itteration represents a timestamp
        ts = df.iloc[row]['timestamp']
        thisImage = df[df['timestamp'] == ts].reset_index()
        thisImage.drop('index', axis=1, inplace=True)
        
        # get tiff file
        tsRaw = supportingFunctions.convert_dt_to_raw_planet_timestamp(ts)
        fnList = os.listdir(os.path.join(os.getcwd(), 'data', regionName, siteName))
        filesForTs = [i for i in fnList if tsRaw in i]
        tiffImage = [i for i in filesForTs if 'toar_clip.tif' in i][0]
        if georef is None:
            print(tiffImage)
            georef = geospatialTools.get_georef( os.path.join(os.getcwd(), 'data', regionName, siteName, tiffImage) ) # assume its stays the same for all the images from one site
            print(georef)
        
        dist = thisImage.loc[0, transectKey]
        if not np.isnan(dist):
            intersects[transectKey] = supportingFunctions.calculate_intersect_point(dist, transects[transectKey])
            point = [intersects[transectKey]['x'], intersects[transectKey]['y']]     
            point = geospatialTools.convert_world2pix(point, georef)[0]
    
        im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(os.getcwd(), 'data', regionName, siteName, tiffImage))
        im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
        fig, ax = plt.subplots(figsize=(10,15))

        ax.scatter(point[:,0], point[:,1], s=10, color='red')


        # create limits
        transectConverted = geospatialTools.convert_world2pix(transects[transectKey], georef)
        farthestLeft,farthestright  = max(transectConverted[:,0]), min(transectConverted[:,0])
        farthestup, farthestdown = min(transectConverted[:,1]), max(transectConverted[:,1])
        plt.xlim(farthestLeft-40, farthestright+40), plt.ylim(farthestdown, farthestup)

        # Plot classes over RGB
        ax.imshow(im_RGB)
        if not savePathPDF is None:
            pp.savefig(fig)
    if not savePathPDF is None:
        pp.close()   