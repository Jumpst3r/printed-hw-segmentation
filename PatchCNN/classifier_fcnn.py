import pickle
import sys
import warnings

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import skimage.io as io
import tensorflow as tf
from keras.models import load_model
from pydensecrf.utils import (create_pairwise_bilateral,
                              create_pairwise_gaussian, unary_from_labels)
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.color import gray2rgb, grey2rgb, rgb2gray
from skimage.filters import threshold_mean, threshold_sauvola
from skimage.io import imread, imsave
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.util import invert
from sklearn.svm import SVC
from tqdm import tqdm

import cv2

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def getbinim(image):
    img_denoised = ndimage.filters.median_filter(image, 3)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    return img_as_float(image < thresh_sauvola)


BOXWDITH = 512
STRIDE = 512
THRESH = 200


def castF(x):
    return K.cast(x, K.floatx())


def castB(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def IoU(true, pred):  # any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    # total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    # to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    # separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(testTrue, testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    # mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    # to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


def max_rgb_filter(image):
    image = image[:, :, ::-1]
    image = img_as_ubyte(image)
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)
    # find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    image = cv2.merge([B, G, R])
    image = img_as_float(image)
    image = image[:, :, ::-1]
    return np.ceil(image)


def weighted_categorical_crossentropy(weights):
    """ weighted_categorical_crossentropy

        Args:
            * weights<ktensor|nparray|list>: crossentropy weights
        Returns:
            * weighted categorical crossentropy function
    """
    if isinstance(weights, list) or isinstance(np.ndarray):
        weights = K.variable(weights)

    def loss(target, output, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            _epsilon = tf.convert_to_tensor(
                K.epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            weighted_losses = target * tf.log(output) * weights
            return - tf.reduce_sum(weighted_losses, len(output.get_shape()) - 1)
        else:
            raise ValueError(
                'WeightedCategoricalCrossentropy: not valid with logits')
    return loss

"""
Function which returns the labelled image after applying CRF

"""

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = The final output image after applying CRF
#Use_2d = boolean variable 
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied

def crf(original_image, annotated_image, use_2d = True):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)
        
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    print("No of labels in the Image are ")
    print(n_labels)
    
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    return MAP.reshape(original_image.shape)


def classify(image):
    model = load_model('fcnn.h5', custom_objects={
                       'loss': weighted_categorical_crossentropy([1, 0.8, 0.01]), 'IoU': IoU})
    image = gray2rgb(image)
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.ones((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            print("breaking")
            print(y + BOXWDITH)
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                np.array([(input - input.mean())/input.std()]))[0]
            x = x + STRIDE
    pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
    io.imsave('hist/im.png', pred)
    #output1 = crf(image, pred, "crf1_fcn16.png")
    plt.set_cmap('jet')
    fig, axes = plt.subplots(1, 2)
    
    # plt.imshow(rgb2gray(pred))
    axes[0].imshow(rgb2gray(pred), cmap='jet')
    crf_res = crf(image,convert(pred),"crf1_fcn16.png")
    axes[1].imshow(rgb2gray(crf_res), cmap='jet')
    plt.show()
    io.imsave('hist/red.png', mask2[0:image.shape[0], 0:image.shape[1], 0])
    io.imsave('hist/green.png', mask2[0:image.shape[0], 0:image.shape[1], 1])

def convert(image):
    image = image[:, :, ::-1]
    image = img_as_ubyte(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def postpocess(image):
    image = image[:, :, ::-1]
    image = img_as_ubyte(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_val = np.bincount(gray.ravel()).argmax()

    for x in range(3, gray.shape[1]-3):
        for y in range(3, gray.shape[0]-3):
            if gray[y, x] == bg_val:
                continue
            vals = []
            vals.append(gray[y, x])
            vals.append(gray[y, x+1])
            vals.append(gray[y, x-1])
            vals.append(gray[y+1, x])
            vals.append(gray[y-1, x])
            vals = np.array(vals)
            gray[y, x] = np.bincount(vals.ravel()).argmax()
    return gray


if __name__ == "__main__":
    classify(io.imread('classify.jpg'))
