__author__      = "Nicolas Dutly"
__email__       = "nicolas.dutly(at)unifr.ch"

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.filters import threshold_sauvola

import cv2


def segment_words(image):
    '''perform word segmentation by:
    1) Denoising the src image
    2) Binarizing the image from step 1
    3) Find contours in image from step 2
    4) Find minimal bounding boxes for detected contours

    Arguments:
        image {Image} -- Input image in float format

    Returns:
        {List} -- returns list of bounding boxes and processed image
        (each element of the list is of the form [x,y,w,h])
    
    Usage example: segment_words(io.imread('image.png', as_gray=True)) -> returns list of bounding boxes
    '''
    # Denoise image
    img_denoised = ndimage.filters.median_filter(image, 3)
    # Binarize image
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    binary_sauvola = image < thresh_sauvola
    # Convert image to opencv compatible format
    cv_image = img_as_ubyte(binary_sauvola)
    # Denoise binarization
    cv_image = cv2.medianBlur(cv_image, 3)
    # Perform dilation
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    dilation = cv2.dilate(cv_image, rect_kernel, iterations=4)
    # Find contours
    _, contours, _ = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectangles = []
    # Find bounding rectangles
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x, y, w, h])

    grouped_rectangles, _ = cv2.groupRectangles(rectangles, 0)

    # Remove small bounding rectangles
    w_tresh = 0.3*grouped_rectangles.T[2].mean()
    h_tresh = 0.3*grouped_rectangles.T[3].mean()

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] > w_tresh, :]
    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[3] > h_tresh, :]

    return grouped_rectangles
