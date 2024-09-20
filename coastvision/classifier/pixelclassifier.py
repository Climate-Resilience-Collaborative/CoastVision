"""
This script contains function used in the pixel classification proccess from traning modeles to inference

Author: Joel Nicolow, Climate Resiliance Collaborative, School of Ocean and Earth Science and Technology (April, 01 2022)
"""


import os
import glob
import pickle as pkl

import pandas as pd
import numpy as np

#### Feature engineering ####
def ndi(b1, b2, twoD=False):
    """Calculate the Normalized Difference Index (NDI) for two bands."""
    epsilon = 1e-6
    denominator = b1 + b2
    denominator[np.isnan(denominator) | np.isinf(denominator) | denominator == 0] = epsilon
    if twoD:
        result = np.zeros_like(b1)
        nonzero_indices = np.nonzero(denominator)
        result[nonzero_indices] = (b1 - b2)[nonzero_indices] / denominator[nonzero_indices]
        return result
    return np.divide(b1 - b2, denominator)


def calculate_features(im_ms, booleanMask=None, stdRadius=2):#, imglbp=None):
    """ if booleanMask=None you doing inference"""
    R = im_ms[:,:,2]
    G = im_ms[:,:,1]
    B = im_ms[:,:,0]
    NIR = im_ms[:,:,3]

    Rstd = coastvision.image_std(R, stdRadius)
    Gstd = coastvision.image_std(G, stdRadius)
    Bstd = coastvision.image_std(B, stdRadius)
    NIRstd = coastvision.image_std(NIR, stdRadius)
    imgStd = np.stack((Rstd, Gstd, Bstd, NIRstd), axis=-1)
    

    # need this done here because to calculate stds we need to work with the whole 2D image
    NIR_B_NDI = ndi(NIR, B)
    NIR_G_NDI = ndi(NIR, G)
    B_R_NDI = ndi(B, R)
    NDWI = ndi(G, NIR)
    NDVI = ndi(NIR, R)
    # if imglbp is None:
    #     radius = 2
    #     n_points = 8 * radius
    #     Rlbp = local_binary_pattern(im_ms[:,:,2], n_points, radius, method='uniform')
    #     Glbp = local_binary_pattern(im_ms[:,:,1], n_points, radius, method='uniform')
    #     Blbp = local_binary_pattern(im_ms[:,:,0], n_points, radius, method='uniform')
    #     NIRlbp = local_binary_pattern(im_ms[:,:,3], n_points, radius, method='uniform')
    #     imglbp = np.stack((Rlbp, Glbp, Blbp, NIRlbp), axis=-1)
    if not booleanMask is None:
        # Create a boolean mask to select pixels where the mask is True
        mask_true = np.repeat(booleanMask[:, :, np.newaxis], im_ms.shape[-1], axis=2)
        pixels_true = im_ms[mask_true].reshape((-1, im_ms.shape[-1])) # boolean indexing for the mask

        # get pixel for the current class
        # R = pixels_true[:, 2]
        # G = pixels_true[:, 1]
        # B = pixels_true[:, 0]
        # NIR = pixels_true[:, 3]

        nonzero_indices = np.argwhere(booleanMask) 
        x_coords, y_coords = nonzero_indices[:, 1], nonzero_indices[:, 0]
        # NIR_B_NDI = ndi(NIR, B)
        # NIR_G_NDI = ndi(NIR, G)
        # B_R_NDI = ndi(B, R)
        # NDWI = ndi(G, NIR)
        # NDVI = ndi(NIR, R)
        df = pd.DataFrame({
                        'X': x_coords,
                        'Y': y_coords,
                        'R': R[booleanMask], 
                        'G': G[booleanMask], 
                        'B': B[booleanMask], 
                        'NIR': NIR[booleanMask],
                        'NIR-B-NDI': NIR_B_NDI[booleanMask],
                        'NIR-G-NDI': NIR_G_NDI[booleanMask],
                        'B-R-NDI': B_R_NDI[booleanMask],
                        'NDWI': NDWI[booleanMask],
                        'NDVI': NDVI[booleanMask],
                        'stdR':imgStd[:,:,2][booleanMask],
                        'stdG':imgStd[:,:,1][booleanMask],
                        'stdB':imgStd[:,:,0][booleanMask],
                        'stdNIR':imgStd[:,:,3][booleanMask],
                        'stdNIR-B-NDI': coastvision.image_std(NIR_B_NDI, stdRadius)[booleanMask],
                        'stdNIR-G-NDI': coastvision.image_std(NIR_G_NDI, stdRadius)[booleanMask],
                        'stdB-R-NDI': coastvision.image_std(B_R_NDI, stdRadius)[booleanMask],
                        'stdNDWI': coastvision.image_std(NDWI, stdRadius)[booleanMask],
                        'stdNDVI':coastvision.image_std(NDVI, stdRadius)[booleanMask]#,
                        # 'lbpR':imglbp[:,:,2][booleanMask],
                        # 'lbpG':imglbp[:,:,1][booleanMask],
                        # 'lbpB':imglbp[:,:,0][booleanMask],
                        # 'lbpNIR':imglbp[:,:,3][booleanMask]
                        })
        return df # there is a return here so no need else statement below
    else:
        feature_array = np.empty((im_ms.shape[0], im_ms.shape[1], 18))
        # R = im_ms[:, :, 2]
        # G = im_ms[:, :, 1]
        # B = im_ms[:, :, 0]
        # NIR = im_ms[:, :, 3]

        # NIR_B_NDI = ndi(NIR, B, twoD=True)
        # NIR_G_NDI = ndi(NIR, G, twoD=True)
        # B_R_NDI = ndi(B, R, twoD=True)
        # NDWI = ndi(G, NIR, twoD=True)
        # NDVI = ndi(NIR, R, twoD=True)

        feature_array[:, :, 0] = R # R channel
        feature_array[:, :, 1] = G # G channel
        feature_array[:, :, 2] = B # B channel
        feature_array[:, :, 3] = NIR # NIR channel
        feature_array[:, :, 4] = NIR_B_NDI # NIR-B-NDI
        feature_array[:, :, 5] = NIR_G_NDI # NIR-G-NDI
        feature_array[:, :, 6] = B_R_NDI # B-R-NDI
        feature_array[:, :, 7] = NDWI # NDWI
        feature_array[:, :, 8] = NDVI # NDVI
        feature_array[:, :, 9] = imgStd[:,:,2] # stdR
        feature_array[:, :, 10] = imgStd[:,:,1] # stdG
        feature_array[:, :, 11] = imgStd[:,:,0] # stdB
        feature_array[:, :, 12] = imgStd[:,:,3] # stdNIR
        feature_array[:, :, 13] = coastvision.image_std(NIR_B_NDI, stdRadius)
        feature_array[:, :, 14] = coastvision.image_std(NIR_G_NDI, stdRadius)
        feature_array[:, :, 15] = coastvision.image_std(B_R_NDI, stdRadius)
        feature_array[:, :, 16] = coastvision.image_std(NDWI, stdRadius)
        feature_array[:, :, 17] = coastvision.image_std(NDVI, stdRadius)
        # feature_array[:, :, 13] = imglbp[:,:,2] # lbpR
        # feature_array[:, :, 14] = imglbp[:,:,1] # lbpR
        # feature_array[:, :, 15] = imglbp[:,:,0] # lbpR
        # feature_array[:, :, 16] = imglbp[:,:,3] # lbpR
        return feature_array



from coastvision import geospatialTools
# from coastvision import coastvision_oldmodel as coastvision
from coastvision import coastvision

from skimage.feature import local_binary_pattern
def get_training_data_pixel_features(modelName, traningDataDir, maskPicklePath, sitename=None):
    """
    This function creates a df with all of the features generated by claculate_features() for an image
    """
    # im_ms, cloud_mask = geospatialTools.get_ps_no_mask(imgFilePath)
    with open(maskPicklePath, 'rb') as f:
        maskDict = pkl.load(f)

    baseFn = os.path.basename(maskPicklePath)
    imgFn = baseFn.split('_annotated')[0]
    im_ms, _ = geospatialTools.get_ps_no_mask(os.path.join(traningDataDir, "raw", modelName, imgFn))

    Rstd = coastvision.image_std(im_ms[:,:,2], 2)
    Gstd = coastvision.image_std(im_ms[:,:,1], 2)
    Bstd = coastvision.image_std(im_ms[:,:,0], 2)
    NIRstd = coastvision.image_std(im_ms[:,:,3], 2)
    imgStd = np.stack((Rstd, Gstd, Bstd, NIRstd), axis=-1)
    radius = 2
    n_points = 8 * radius
    Rlbp = local_binary_pattern(im_ms[:,:,2], n_points, radius, method='uniform')
    Glbp = local_binary_pattern(im_ms[:,:,1], n_points, radius, method='uniform')
    Blbp = local_binary_pattern(im_ms[:,:,0], n_points, radius, method='uniform')
    NIRlbp = local_binary_pattern(im_ms[:,:,3], n_points, radius, method='uniform')
    imglbp = np.stack((Rlbp, Glbp, Blbp, NIRlbp), axis=-1)

    dfList = []
    for category, mask in maskDict.items():
        df = calculate_features(im_ms, booleanMask=mask, imgStd=imgStd)#, imglbp)
        df.insert(loc=0, column='class', value=category)
        df.insert(loc=0, column='filename', value=imgFn) # add file name for concating with more data later
        if not sitename is None:
            df.insert(loc=0, column='sitename', value=sitename) # if sitename is provided we will add this as a column
        dfList.append(df)
    df = pd.concat(dfList)
    
    return df


from sklearn.preprocessing import LabelEncoder, StandardScaler
def load_features(trainingDataDir, modelName, featureList=None, scale=False, sitename=None, sitenameRestriction=None, filename= None, filenameRestriction=None):
    """ 
    This function loads data from a csv of saved features and puts them into the X and y format needed for ML

    :param trainingDataDir: os.path just a path to the training_data folder which should contain raw and labeled data folders
    :param modelName: String the name of the model (matches the file name in training data)
    :param featureList: list of feature names (None if you want to use all features)
    :param scale: boolean true if features shouldbe scaled
    :param sitename: String constrict feature loading to just that site
    :param sitenameRestriction: string load features for all sites except the sitenameRestriction
    :param filename: String constrict feature loading to just that image file
    :param filenameRestriction: string load features for all images except the filenameRestriction

    """
    labeledDataDir = os.path.join(trainingDataDir, 'labeled', modelName)
    try:
        featureDf = pd.read_csv(os.path.join(labeledDataDir, f'{modelName}_training_data.csv'))
    except FileNotFoundError:
        # if there is no csv of the saved features we can compute them
        maskFiles = glob.glob(os.path.join(labeledDataDir, '*.pkl'))
        dfList = list(map(lambda maskFile: get_training_data_pixel_features(modelName, trainingDataDir, maskFile), maskFiles))
        featureDf = pd.concat(dfList)
        featureDf.to_csv(os.path.join(labeledDataDir, f'{modelName}_training_data.csv'))
    featureDf.dropna(inplace=True)
    if featureList is None:
        # featureList = ['R', 'G', 'B', 'NIR', 'NIR-B-NDI',
        # 'NIR-G-NDI', 'B-R-NDI', 'NDWI', 'NDVI', 'stdR', 'stdG', 'stdB', 'stdNIR',
        # 'lbpR', 'lbpG', 'lbpB', 'lbpNIR']
        featureList = ['R', 'G', 'B', 'NIR', 'NIR-B-NDI',
        'NIR-G-NDI', 'B-R-NDI', 'NDWI', 'NDVI',
        'stdR', 'stdG', 'stdB', 'stdNIR',]
    if not sitenameRestriction is None:
        featureDf = featureDf[featureDf['sitename'] != sitenameRestriction]
    if not sitename is None:
        featureDf = featureDf[featureDf['sitename'] == sitename]
    if not filenameRestriction is None:
        featureDf = featureDf[featureDf['filename'] != filenameRestriction]
    if not filename is None:
        featureDf = featureDf[featureDf['filename'] == filenameRestriction]
    X = featureDf[featureList].values
    y_str = featureDf['class']

    # Convert string labels to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    if scale and X.shape[0] > 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return(X, y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#### performance analysis ####
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def plot_confusion_matrix(y_true,y_pred,classes,normalize=False,cmap=plt.cm.Blues):
    """
    Function copied from the scikit-learn examples (https://scikit-learn.org/stable/)
    This function plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    """
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), ylim=[len(classes)-0.5,-0.5],
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    fig.tight_layout()
    return ax


def dice_coefficient(truth, prediction, mask=None):
    """
    Calculates the dice coeficient between a ground truth and a prediction (masking avalible)

    :param truth: 2d np array ground truth
    :param prediction: 2d np array predicted image (same shape as truth)
    :param mask: 2d boolean array same shape as other inputs

    :return: float dice coeficient for the two images
    """


    if mask is not None:
        prediction = prediction[mask]
        truth = truth[mask]
    
    intersection = np.logical_and(prediction, truth).sum()
    dice = (2.0 * intersection) / (prediction.sum() + truth.sum())
    
    return dice


def pixel_classif_full_image_stats(truth, prediction, mask=None):
    """
    This function calculates the accuracy, recall, persision, and subsiquently f1 score for a predicted image

    :param truth: 2d np array ground truth
    :param prediction: 2d np array predicted image (same shape as truth)
    :param mask: 2d boolean array same shape as other inputs

    :return: dict with statistics
    """
    if mask is not None:
        prediction = prediction[mask]
        truth = truth[mask]
    tpr = np.logical_and(truth, prediction).sum()
    fpr = np.logical_and(~truth, prediction).sum()
    tnr = np.logical_and(~truth, ~prediction).sum()
    fnr = np.logical_and(truth, ~prediction).sum()
    precision =  tpr / (tpr + fpr)
    recall = tpr / (tpr + fnr)
    f1 = 2 * ( (precision * recall) / (precision + recall) )
    accuracy = (tpr + tnr) / (tpr + fpr + tnr + fnr)

    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1': f1}



import geopandas as gpd
from coastvision import supportingFunctions
from coastvision import coastvisionPlots
import importlib
importlib.reload(coastvision)
def full_image_performance(tiffImgPath, region, sitename, clf, annotatedImg=None, contourTruth=None):
    """
    This function compares a hand-classified image against one classified by the clf
    Comparision in the from of dice loss (on a buffer around the reference shoreline) and transect intersection differences

    :param tiffImgPath: file path to tiff image (can be in the classifier/traning_data/raw folder or in the data folder or anywhere)
    :param region: string region name
    :param sitename: string sitename
    :param clf: model (needs .predict() function example sklearn SVC)
    :param annotatedImg: file path to pkl file that contains dict with class masks
    :param contourTruth: digitized shoreline in world coordinates

    :return: dict 
    """
    # get prerecs
    im_ms = geospatialTools.get_im_ms(tiffImgPath)
    udm2_file_name = os.path.basename(tiffImgPath).replace("AnalyticMS_toar_clip.tif", "udm2_clip.tif") # get cooresponding IDM fn
    print(os.path.join('data', region, sitename, udm2_file_name))
    udm_mask = geospatialTools.get_cloud_mask_from_udm2_band8(os.path.join('data', region, sitename, udm2_file_name)) # possibly its not in the same locati9on as the tiff image being referenced so look in data
    georef = geospatialTools.get_georef(tiffImgPath)
    slBuffer=coastvision.create_shoreline_buffer(region, sitename, im_ms.shape[0:2], georef, 3, 70) # get pixel size (3) and buffer size (40) from info json
    im_mask = np.logical_or(udm_mask, slBuffer) # combine slbuffer and udm mask (Note both are False for the ROI)
    infoJson = supportingFunctions.get_info_from_info_json(os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + '_info.json')))
    if infoJson is None:
        infoJson = {'max_dist_from_sl_ref':30, 'pixel_size':3}

    # classify img
    pred = coastvision.classify_image(im_ms, clf, ~im_mask) # notice as mentioned earlier mask is an inverse
    pred = coastvision.process_img_classif(pred) # morphology stuffs (binary dilation etc.)
    # plt.imshow(pred, cmap='gray')
    # plt.colorbar()

    if not annotatedImg is None:
        with open(annotatedImg, 'rb') as f:
            annotated = pkl.load(f)
        # dice coef
        counter = 0
        diceCoefDict = {}
        for key, trueClass in annotated.items():
            inferenceClass = pred == counter
            diceCoefDict[key] = dice_coefficient(truth=trueClass, prediction=inferenceClass, mask=~im_mask)
            counter += 1
    
        contourTruth = coastvision.shoreline_contour_extract_single(trueClass, region, sitename, 
                                                            georef=georef, shorelinebuff=slBuffer, min_sl_len=100, pixel_size=int(infoJson['pixel_size']), 
                                                            max_dist=int(infoJson['max_dist_from_sl_ref']), imMaskPath=None, 
                                                            saveContour=False, timestamp='truth',
                                                            smoothShorelineSigma=None, smoothWindow=None,
                                                            year=None)
        
        # get accuracy of pixel classfication
        classifStatsDict = pixel_classif_full_image_stats(truth=trueClass, prediction=inferenceClass, mask=~im_mask)
    else:
        diceCoefDict = None
        classifStatsDict = None
        if contourTruth is None:
            print('you must provide a contour (contourTruth) or an annotated image file path (annotatedImg pkl file)')
            print('otherwise there is nothing to compare the model\'s perdiction to')
            return None, None


    # difference in transect intersections
    contourPred = coastvision.shoreline_contour_extract_single(pred, region, sitename, 
                                                            georef=georef, shorelinebuff=slBuffer, min_sl_len=100, pixel_size=int(infoJson['pixel_size']), 
                                                            max_dist=int(infoJson['max_dist_from_sl_ref']), imMaskPath=None, 
                                                            saveContour=False, timestamp='pred',
                                                            smoothShorelineSigma=0.1, smoothWindow=None,
                                                            year=None)

    test = geospatialTools.convert_world2pix(contourPred, georef)
    coastvisionPlots.rgb_contour_plot_single(tiffImgPath, test, savePath=None, loudShoreline=True)
    test = geospatialTools.convert_world2pix(contourTruth, georef)
    coastvisionPlots.rgb_contour_plot_single(tiffImgPath, test, savePath=None, loudShoreline=True)

    transectPath = os.path.join(os.getcwd(), 'user_inputs', region, sitename, (sitename + "_transects.geojson"))
    transects = coastvision.transects_from_geojson(transectPath)

    intersectTruth = coastvision.transect_intersect_single(25, contourTruth, transects)
    intersectPred =  coastvision.transect_intersect_single(25, contourPred, transects)

    intersectDict = {}
    for key, item in intersectTruth.items():
        intersectDict[key] = item[0] - intersectPred[key][0] # this means a positive would be the model predicting landward

    return diceCoefDict, intersectDict, classifStatsDict
    


    





