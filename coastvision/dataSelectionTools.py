"""
This module creates a data annotation tool. Giving the user a GUI to sort uncategprozed images.
NOTE: when using hand_classify_images() you must have %matplotlib qt in the line before (for notebooks)

Author: Joel Nicolow, Climate Resilience Initiative, University of Hawaii at Manoa
"""

# import modules
import os
import pandas as pd
import matplotlib.pyplot as plt # figure for interactive annotation tool
import matplotlib.pyplot
import PIL
from PIL import Image                # working with images
from PIL.ExifTags import TAGS        # getting image meta data
import shutil # moving files
from osgeo import gdal, osr


# coastvision modules
from coastvision import coastvision
from coastvision import geospatialTools
from coastvision.classifier import data_annotation_tool 


def select_tif_images(satImgFolder, timeperiod=None, onlySelectOne=False):
    """
    Shows user satalite images and askes them to choose to keep or skip; then the kept files' names are returned in a list.

    :param satImgFolder: path to where the satalite images are
    :param timeperiod: [[year1, year2], [month1, month2]] # both are inclusive
    :param onlySelectOne: if true once user selects keep the function exits and returns the filename the user selected (default=False)

    :return: list of the short file names of the satalite images
    """
    matplotlib.use("Qt5Agg")
    validImages = list()

    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                            sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()

    for fn in sorted(os.listdir(satImgFolder), reverse=True):
        if fn.endswith('toar_clip.tif'):
            print(fn)
            year = int(fn[0:4])
            month = int(fn[4:6])
            if timeperiod is None: #if no timeperiod defined, just read im_ms and RGB
                im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(satImgFolder, fn))
                im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
            else: #if timeperiod is defined, check criteria
                if supportingFunctions.yearmonth_inYearmonth_range(year, month, yearMonthRange=timeperiod):
                    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(os.path.join(satImgFolder, fn))
                    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)
                

            # Plot classes over RGB
            ax.imshow(im_RGB)
            ax.set_title(f'image: {fn}\n' +
                    'Press <right arrow> if you would like to use image.\n' +
                    'If the image does not look cherraih <left arrow> to skip.\n' +
                    f'Number of selected images: {len(validImages)}', fontsize=14)
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
                    print('User canceled tif image selection')
                    return(validImages)
                    # raise StopIteration('User cancelled image selection') # we do not want to raise an error we just want to exit
                else:
                    plt.waitforbuttonpress()

            # if user decided to skip show the next image
            if skip_image:
                ax.clear()
                continue
            # otherwise label this image
            else:
                validImages.append(fn)
                if onlySelectOne:
                    ax.clear()
                    plt.close(fig)
                    return(validImages[0])
    ax.clear()
    plt.close(fig)
    return(validImages)


from coastvision import supportingFunctions
def select_images(imgFolder, timeperiod=None, filterList=None, onlySelectOne=False, returnNonSelected=False):
    """

    :param returnNonSelected: if true we return a dictionary with the selecte dand non selected files
    """
    matplotlib.use("Qt5Agg")
    

    validImages = list()
    if returnNonSelected:
        invalidImages = list()

    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                            sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()

    for fn in os.listdir(imgFolder):
        if fn.endswith('rgb_sl_plot.png'):
            if not filterList is None:
                if not supportingFunctions.str_contains_list_elements(fn, filterList, all=False):
                    print(fn)
                    continue

            if '-' in fn:
                year = int(fn.split('-')[1][0:4])
            else:
                year = int(fn.split('_')[1][0:4])
            if timeperiod is None or (not timeperiod is None and year >= timeperiod[0] and year <= timeperiod[1]):
                # weather there is no timeperiod or there is and this is in it
                image = PIL.Image.open(os.path.join(imgFolder, fn))
                ax.imshow(image)
                ax.set_title(f'image: {fn}\n' +
                        'Press <right arrow> to select image (it will be added to an image fn list returned at the end.\n' +
                        'To skip image press <left arrow>.\n' +
                        f'Number of Images selected: {len(validImages)}', fontsize=14)
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
                        print('User canceled image selection')
                        if returnNonSelected:
                            return({'selected':validImages, 'notselected':invalidImages})
                        return(validImages)
                        # raise StopIteration('User cancelled image selection') # we do not want to raise an error we just want to exit
                    else:
                        plt.waitforbuttonpress()

                # if user decided to skip show the next image
                if skip_image:
                    if returnNonSelected:
                        invalidImages.append(fn)
                    ax.clear()
                    continue
                # otherwise label this image
                else:
                    validImages.append(fn)
                    if onlySelectOne:
                        ax.clear()
                        plt.close(fig)
                        return(validImages[0])
    ax.clear()
    plt.close(fig)
    if returnNonSelected:
        return({'selected':validImages, 'notselected':invalidImages})
    return(validImages)


def show_tif_image_window(tifPath):

    matplotlib.use("Qt5Agg")

    # fn = tifPath.basename()

    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                            sharey=True)
    # mng = plt.get_current_fig_manager()                                         
    # mng.window.showMaximized()
    # # year = int(fn[0:4])
    # # weather there is no timeperiod or there is and this is in it
    # im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tifPath)
    # im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # # Plot classes over RGB
    # ax.imshow(im_RGB)
    # ax.set_title(f'image: {18}\n' +
    #         'Press <right arrow> if you would like to use image.\n' +
    #         'If the image does not look cherraih <left arrow> to skip.\n' +
    #         f'Number of selected images: {999}', fontsize=14)

    # key_event = {}
    # def press(event):
    #     # store what key was pressed in the dictionary
    #     key_event['pressed'] = event.key
    # # let the user press a key, right arrow to keep the image, left arrow to skip it
    # # to break the loop the user can press 'escape'
    # while True:
    #     btn_keep = ax.text(1.1, 0.9, 'keep ⇨', size=12, ha="right", va="top",
    #                         transform=ax.transAxes,
    #                         bbox=dict(boxstyle="square", ec='k',fc='w'))
    #     btn_skip = ax.text(-0.1, 0.9, '⇦ skip', size=12, ha="left", va="top",
    #                         transform=ax.transAxes,
    #                         bbox=dict(boxstyle="square", ec='k',fc='w'))
    #     btn_esc = ax.text(0.5, 0, '<esc> to quit', size=12, ha="center", va="top",
    #                         transform=ax.transAxes,
    #                         bbox=dict(boxstyle="square", ec='k',fc='w'))
    #     fig.canvas.draw_idle()                         
    #     fig.canvas.mpl_connect('key_press_event', press)
    #     plt.waitforbuttonpress()
    #     # after button is pressed, remove the buttons
    #     btn_skip.remove()
    #     btn_keep.remove()
    #     btn_esc.remove()

    #     # keep/skip image according to the pressed key, 'escape' to break the loop
    #     if key_event.get('pressed') == 'right':
    #         skip_image = False
    #         break
    #     elif key_event.get('pressed') == 'left':
    #         skip_image = True
    #         break
    #     elif key_event.get('pressed') == 'escape':
    #         plt.close()
    #         print('User canceled tif image selection')
    #         # raise StopIteration('User cancelled image selection') # we do not want to raise an error we just want to exit
    #     else:
    #         plt.waitforbuttonpress()


import geojson
import numpy as np
from pylab import ginput
import skimage.transform as transform
from shapely import geometry
def get_reference_sl(region, sitename, timeperiod=None, referenceImagePath=None):
    matplotlib.use("Qt5Agg")
    # datapth = os.path.join( os.getcwd(), 'data', region, sitename)
    # shortFn = select_tif_images(datapth, timeperiod=timeperiod, onlySelectOne=True)
    # if type(shortFn) == list and len(shortFn) == 0:
    #     raise Exception('There are no tif files for the selected site for the timeperiod (if time period given)')
    # elif type(shortFn) == list and len(shortFn) > 0:
    #     shortFn = shortFn[0]
    # print(type(shortFn))
    # fn = os.path.join(datapth, shortFn)

    # matplotlib.use("Qt5Agg")
    # validImages = list()

    # fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
    #                         sharey=True)
    # mng = plt.get_current_fig_manager()                                         
    # mng.window.showMaximized()

    # fnList = list(fn)
    # for fn in fnList:
    #     if fn.endswith('toar_clip.tif'):
    #         year = int(fn[0:4])
    #         if timeperiod is None or (not timeperiod is None and year >= timeperiod[0] and year <= timeperiod[1]):
    #             # weather there is no timeperiod or there is and this is in it
    #             im_ms, cloud_mask = geospatialTools.get_ps_no_mask(fn)
    #             im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    #             # Plot classes over RGB
    #             ax.imshow(im_RGB)
    #             ax.set_title(f'image: {fn}\n' +
    #                     'Press <right arrow> if you would like to use image.\n' +
    #                     'If the image does not look cherraih <left arrow> to skip.\n' +
    #                     f'Number of selected images: {len(validImages)}', fontsize=14)

    
    # # fig.show()
    # import time
    # time.sleep(4)

    # ax.clear()
    # plt.close(fig)

    pts_coords = []
    
    # check if reference shoreline already exists in the corresponding folder
    geojson_sl = None# coastvision.get_ref_sl_fn(region, sitename, timeperiod)

    digitizeNewSL = True
    # if it exist, ask user if they want to over write it
    if geojson_sl == None:
        digitizeNewSL = True
    elif os.path.exists(geojson_sl):
        print('There is already a reference shoreline. Enter \'yes\' if you would like to overwrite it:')
        overwrite = input("region(s): ")
        if overwrite == 'yes':
            digitizeNewSL = True
        else:
            digitizeNewSL == False

    # otherwise get the user to manually digitise a shoreline
    if digitizeNewSL:
        # Digitise shoreline on im_ref
        # fn = settings['ref_merge_im']
        if referenceImagePath is None:
            datapth = os.path.join( os.getcwd(), 'data', region, sitename)
            fnShort = select_tif_images(datapth, timeperiod=timeperiod, onlySelectOne=True)
           
            fn = os.path.join(datapth, fnShort)
            # # fn = os.path.join(os.getcwd(), 'data', 'oahu', 'hawaiiloa', '20201025_201712_24_2212_3B_AnalyticMS_toar_clip.tif')
        else:
            fn = referenceImagePath
            fnShort = os.path.basename(fn)
        print(f'{sitename}:{fnShort}')
        year = int(fnShort[0:4])
        month = int(fnShort[4:6])


        # create figure
        fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                            sharey=True)
        mng = plt.get_current_fig_manager()                                         
        mng.window.showMaximized()


        im_ms, cloud_mask = geospatialTools.get_ps_no_mask(fn)
        im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

        # Plot classes over RGB
        ax.clear()
        ax.imshow(im_RGB)
        ax.set_title(f'image: {fn}\n' +
                'Press <right arrow> if you would like to use image.\n' +
                'If the image does not look cherraih <left arrow> to skip.\n' +
                f'Number of selected images:', fontsize=14)


        # create buttons
        add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                               transform=plt.gca().transAxes,
                               bbox=dict(boxstyle="square", ec='k',fc='w'))
        end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                               transform=plt.gca().transAxes,
                               bbox=dict(boxstyle="square", ec='k',fc='w'))
        # add multiple reference shorelines (until user clicks on <end> button)
        pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
        geoms = []
        while 1:
            add_button.set_visible(False)
            end_button.set_visible(False)
            # update title (instructions)
            ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                      'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                      fontsize=14)
            plt.draw()

            # let user click on the shoreline
            pts = ginput(n=50000, timeout=-1, show_clicks=True)
            pts_pix = np.array(pts)
            # get georef val
            data = gdal.Open(fn, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            # convert pixel coordinates to world coordinates
            pts_world = geospatialTools.convert_pix2world(pts_pix[:,[1,0]], georef)

            # interpolate between points clicked by the user (1m resolution)
            pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
            for k in range(len(pts_world)-1):
                pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
                xvals = np.arange(0,pt_dist)
                yvals = np.zeros(len(xvals))
                pt_coords = np.zeros((len(xvals),2))
                pt_coords[:,0] = xvals
                pt_coords[:,1] = yvals
                phi = 0
                deltax = pts_world[k+1,0] - pts_world[k,0]
                deltay = pts_world[k+1,1] - pts_world[k,1]
                phi = np.pi/2 - np.math.atan2(deltax, deltay)
                tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
                pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
            pts_world_interp = np.delete(pts_world_interp,0,axis=0)

            # save as geometry (to create .geojson file later)
            geoms.append(geometry.LineString(pts_world_interp))

            # convert to pixel coordinates and plot
            pts_pix_interp = geospatialTools.convert_world2pix(pts_world_interp, georef)
            pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
            ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
            ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
            ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

            # update title and buttons
            add_button.set_visible(True)
            end_button.set_visible(True)
            ax.set_title('click on <add> to digitize another shoreline or <end> to finish and save the shoreline(s)',
                      fontsize=14)
            plt.draw()

            # let the user click again (<add> another shoreline or <end>)
            pt_input = ginput(n=1, timeout=-1, show_clicks=False)
            pt_input = np.array(pt_input)

            # if user clicks on <end>, save the points and break the loop
            if pt_input[0][0] > im_ms.shape[1]/2:
                add_button.set_visible(False)
                end_button.set_visible(False)
                plt.title('Reference shoreline saved as ' + sitename + '_reference_shoreline.pkl and ' + sitename + '_reference_shoreline.geojson')
                plt.draw()
                ginput(n=1, timeout=3, show_clicks=False)
                plt.close()
                break


        pts_sl = np.delete(pts_sl,0,axis=0) # this is NAs thats why
        if pts_sl.shape[0] == 0:
            raise Exception('Shoreline digitisation failed') 

        # we will assume that we do not have any geojsons with the epsg yet for this sight (which could be the case if there is not already a shoreline)
        # epsg =supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_info.json'))['epsg']
        epsg = geospatialTools.get_epsg(fn)
        geojson_sl = {"type":"FeatureCollection",
                        "crs": { "type":"name", "properties":{ "name": f"urn:ogc:def:crs:EPSG::{epsg}" } },
                        "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": supportingFunctions.ndarray_to_string(pts_sl)}}]}
            


        if timeperiod is None:
            folderDir = os.path.join(os.getcwd(), 'user_inputs', region, sitename)
            supportingFunctions.create_dir(folderDir)
            ref_sl_path = os.path.join(folderDir, f"{sitename}_shoreline.geojson")
            with open(ref_sl_path, "w") as f:
                geojson.dump(geojson_sl, f)
            # NOTE: I am not exactly sure why this was here
            # ref_sl_path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_shoreline_{year}_{timeperiod[0]}-{timeperiod[1]}.geojson')
            # with open(ref_sl_path, "w") as f:
            #     geojson.dump(geojson_sl, f)
        elif len(timeperiod[0]) == 1:
            ref_sl_path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_shoreline_{year}-{month:02}_{timeperiod[0]}-{timeperiod[1]}_01-12.geojson') # because 01-12 is all months
            with open(ref_sl_path, "w") as f:
                geojson.dump(geojson_sl, f)
        else:
            ref_sl_path = os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_shoreline_{year}-{month:02}_{timeperiod[0][0]}-{timeperiod[0][1]}_{timeperiod[1][0]:02}-{timeperiod[1][1]:02}.geojson')
            with open(ref_sl_path, "w") as f:
                geojson.dump(geojson_sl, f)


def hand_digitize_shoreline(tiffImagePath, sl_path):
    matplotlib.use("Qt5Agg")
    # pts_coords = []
    
    # Digitise shoreline on im_ref
    
    print(f'{tiffImagePath}')

    # create figure
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                        sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()


    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(tiffImagePath)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # Plot classes over RGB
    ax.clear()
    ax.imshow(im_RGB)
    ax.set_title(f'image: {tiffImagePath}\n' +
            'Press <right arrow> if you would like to use image.\n' +
            'If the image does not look cherraih <left arrow> to skip.\n' +
            f'Number of selected images:', fontsize=14)


    # create buttons
    add_button = plt.text(0, 0.9, 'add', size=16, ha="left", va="top",
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
    end_button = plt.text(1, 0.9, 'end', size=16, ha="right", va="top",
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle="square", ec='k',fc='w'))
    # add multiple reference shorelines (until user clicks on <end> button)
    pts_sl = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
    geoms = []
    while 1:
        add_button.set_visible(False)
        end_button.set_visible(False)
        # update title (instructions)
        ax.set_title('Click points along the shoreline (enough points to capture the beach curvature).\n' +
                    'Start at one end of the beach.\n' + 'When finished digitizing, click <ENTER>',
                    fontsize=14)
        plt.draw()

        # let user click on the shoreline
        pts = ginput(n=50000, timeout=-1, show_clicks=True)
        pts_pix = np.array(pts)
        # get georef val
        data = gdal.Open(tiffImagePath, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        # convert pixel coordinates to world coordinates
        pts_world = geospatialTools.convert_pix2world(pts_pix[:,[1,0]], georef)

        # interpolate between points clicked by the user (1m resolution)
        pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]),axis=0)
        for k in range(len(pts_world)-1):
            pt_dist = np.linalg.norm(pts_world[k,:]-pts_world[k+1,:])
            xvals = np.arange(0,pt_dist)
            yvals = np.zeros(len(xvals))
            pt_coords = np.zeros((len(xvals),2))
            pt_coords[:,0] = xvals
            pt_coords[:,1] = yvals
            phi = 0
            deltax = pts_world[k+1,0] - pts_world[k,0]
            deltay = pts_world[k+1,1] - pts_world[k,1]
            phi = np.pi/2 - np.math.atan2(deltax, deltay)
            tf = transform.EuclideanTransform(rotation=phi, translation=pts_world[k,:])
            pts_world_interp = np.append(pts_world_interp,tf(pt_coords), axis=0)
        pts_world_interp = np.delete(pts_world_interp,0,axis=0)

        # save as geometry (to create .geojson file later)
        geoms.append(geometry.LineString(pts_world_interp))

        # convert to pixel coordinates and plot
        pts_pix_interp = geospatialTools.convert_world2pix(pts_world_interp, georef)
        pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
        ax.plot(pts_pix_interp[:,0], pts_pix_interp[:,1], 'r--')
        ax.plot(pts_pix_interp[0,0], pts_pix_interp[0,1],'ko')
        ax.plot(pts_pix_interp[-1,0], pts_pix_interp[-1,1],'ko')

        # update title and buttons
        add_button.set_visible(True)
        end_button.set_visible(True)
        ax.set_title('click on <add> to digitize another shoreline or <end> to finish and save the shoreline(s)',
                    fontsize=14)
        plt.draw()

        # let the user click again (<add> another shoreline or <end>)
        pt_input = ginput(n=1, timeout=-1, show_clicks=False)
        pt_input = np.array(pt_input)

        # if user clicks on <end>, save the points and break the loop
        if pt_input[0][0] > im_ms.shape[1]/2:
            add_button.set_visible(False)
            end_button.set_visible(False)
            plt.title('Shoreline saved as ' + sl_path)
            plt.draw()
            ginput(n=1, timeout=3, show_clicks=False)
            plt.close()
            break


    pts_sl = np.delete(pts_sl,0,axis=0) # this is NAs thats why
    if pts_sl.shape[0] == 0:
        raise Exception('Shoreline digitisation failed') 

    # we will assume that we do not have any geojsons with the epsg yet for this sight (which could be the case if there is not already a shoreline)
    # epsg =supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', region, sitename, f'{sitename}_info.json'))['epsg']
    epsg = geospatialTools.get_epsg(tiffImagePath)
    geojson_sl = {"type":"FeatureCollection",
                    "crs": { "type":"name", "properties":{ "name": f"urn:ogc:def:crs:EPSG::{epsg}" } },
                    "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": supportingFunctions.ndarray_to_string(pts_sl)}}]}
    
    with open(sl_path, "w") as f:
        geojson.dump(geojson_sl, f)


# from coastvision import supportingFunctions
def get_transects(region, sitename, timeperiod=None):
    """
    This function allows the user to digitize transects for a shoreline
    transect file is stored in user_inputs/<region>/<sitename>/<sitename>_transects.geojson

    :param region: String region name
    :param sitename: String site
    :param timeperiod: 2d array year range and month range used to only select reference images from that timeperiod (ex. [[2010, 2025], [1, 12]])
    """
    matplotlib.use("Qt5Agg")
    datapth = os.path.join('data', region, sitename)
    fnShort = select_tif_images(datapth, timeperiod=timeperiod, onlySelectOne=True)
    if len(fnShort) == 0:
        print('no reference images')
        return
    fn = os.path.join(datapth, fnShort)
    georef = geospatialTools.get_georef(fn)
    # create figure
    fig,ax = plt.subplots(1,1,figsize=[17,10], tight_layout=True,sharex=True,
                        sharey=True)
    mng = plt.get_current_fig_manager()                                         
    mng.window.showMaximized()


    im_ms, cloud_mask = geospatialTools.get_ps_no_mask(fn)
    im_RGB = data_annotation_tool.rescale_image_intensity(im_ms[:,:,[2,1,0]], cloud_mask, 99.9)

    # Plot classes over RGB
    ax.clear()
    ax.imshow(im_RGB)

    ax.set_title('Click two points to define each transect (first point is the ' +
                      'origin of the transect and is landwards, second point seawards).\n'+
                      'When all transects have been defined, click on <ENTER>', fontsize=10)

    # initialise transects dict
    coords = []
    transects = dict([])
    counter = 0
    # loop until user breaks it by click <enter>
    while 1:
        # let user click two points
        pts = ginput(n=2, timeout=1000000)
        if len(pts) > 0:
            origin = pts[0]
        # if user presses <enter>, no points are selected
        else:
            fig.gca().set_title('Transect locations', fontsize=16)
            plt.title('Transect coordinates saved as ' + sitename + '_transects.geojson')
            plt.draw()
            # wait 2 seconds for user to visualise the transects that are saved
            ginput(n=1, timeout=2, show_clicks=True)
            plt.close(fig)
            # break the loop
            break
        
        # add selectect points to the transect dict
        counter = counter + 1
        transect = np.array([pts[0], pts[1]])
                    
        # plot the transects on the figure
        ax.plot(transect[:,0], transect[:,1], 'b-', lw=2.5)
        ax.plot(transect[0,0], transect[0,1], 'rx', markersize=10)
        ax.text(transect[-1,0], transect[-1,1], str(counter), size=16,
                    bbox=dict(boxstyle="square", ec='k',fc='w'))
        plt.draw()
        
        # Convert pix coord to real world coords WGS84
        transect_world = geospatialTools.convert_pix2world(transect[:,[1,0]], georef)
        transects[str(counter)] = transect_world
        coords.append(transect_world)
    
    auxDir = os.path.join('user_inputs', region, sitename)
    supportingFunctions.create_dir(auxDir)
    f = open(os.path.join(auxDir, f'{sitename}_transects.geojson'), "w")

    f.write("{\n")
    f.write('"type": "FeatureCollection",\n')
    f.write('"crs": { "type": "name", "properties": { "name": "' + f'urn:ogc:def:crs:EPSG::{geospatialTools.get_epsg(fn)}' +'" } },\n')
    f.write('"features": [\n')

    for i in range(0,len(coords), 1):
        # transectId = features[i]["properties"]["id"] # this will not work when using special characters
        coord = coords[i]
        #print(coord)
        if not i == (len(coords)-1):
            f.write('{ "type": "Feature", "properties": { "name": "' + str(i) + '" }, "geometry": { "type": "LineString", "coordinates": [[' + str(coord[0][0]) + ', ' + str(coord[0][1]) + '], [' + str(coord[1][0]) + ', ' + str(coord[1][1]) + ']]}},\n')
        else:
            f.write('{ "type": "Feature", "properties": { "name": "' + str(i) + '" }, "geometry": { "type": "LineString", "coordinates": [[' + str(coord[0][0]) + ', ' + str(coord[0][1]) + '], [' + str(coord[1][0]) + ', ' + str(coord[1][1]) + ']]}}\n')

    f.write("]\n")
    f.write("}")
    f.close()
    return(transects)

