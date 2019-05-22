"""
This file contains various helper function for image processing
"""
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.color import rgb2gray

def getbinim(image):
    if len(image.shape) >= 3:
        image = rgb2gray(image)
    thresh_sauvola = threshold_sauvola(image)
    return img_as_float(image < thresh_sauvola)

# adapted from https://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
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


# Util functions to manipulate masks


def rgb2mask(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((mask[:, :] == [255, 0, 0]).all(axis=2))] = [0, 0, 1]
    result[:, :][np.where((mask[:, :] == [0, 255, 0]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((mask[:, :] == [0, 0, 255]).all(axis=2))] = [0, 0, 4]
    result[:, :][np.where((mask[:, :] == [1, 0, 0]).all(axis=2))] = [0, 0, 1]
    result[:, :][np.where((mask[:, :] == [0, 1, 0]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [0, 0, 4]
    return result

def mask2rgb(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [1, 0, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [0, 1, 0]
    result[:, :][np.where((mask[:, :] == [0, 0, 4]).all(axis=2))] = [0, 0, 1]
    return result

def getclass(n, mask):
    result = np.zeros((mask.shape))
    if n == 1: result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [0, 0, 1]
    if n == 2: result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((result[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 4]
    return result

def getBinclassImg(n, mask):
    result = np.zeros((mask.shape))
    if n == 1: result[:, :][np.where((mask[:, :] == [0, 0, 1]).all(axis=2))] = [1, 1, 1]
    if n == 2: result[:, :][np.where((mask[:, :] == [0, 0, 2]).all(axis=2))] = [1, 1, 1]
    if n == 3: result[:, :][np.where((mask[:, :] == [0, 0, 4]).all(axis=2))] = [1, 1, 1]
    return result[:,:,0]


def get_IoU(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    return float(np.sum(intersection)) / float(np.sum(union))
