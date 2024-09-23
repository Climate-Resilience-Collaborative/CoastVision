"""
This script contains functions used to plot outputs of coastvision.

Author: Joel Nicolow, Climate Resiliance Collaborative, School of Ocean and Earth Science and Technology, University of Hawaiʻi (August, 03 2022)

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
import glob
from coastvision import supportingFunctions, geospatialTools
from coastvision.classifier import data_annotation_tool, dataAnnotationTool


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
        failed =[]
        # Check if the lines intersect
        if shapely_sl.intersects(transect):
            # Find the intersection point(s)
            intersection_point = shapely_sl.intersection(transect)
            
            # Plot the intersection point
            if intersection_point.geom_type == 'Point':
                ax.plot(intersection_point.x, intersection_point.y, 'ro', markersize=3)  # Plot as red dot
            elif intersection_point.geom_type == 'MultiPoint':
                try:
                    for pt in intersection_point:
                        ax.plot(pt.x, pt.y, 'ro')
                except:
                    # print('\n could not plot', toaPath)
                    failed.append(toaPath)



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
