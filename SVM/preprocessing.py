import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_sauvola
from feature_extraction import *
import cv2


def preprocess(image):
    '''Preprocesses an image, performing
        1) Noise reduction
        2) Foregroud extraction
        3) Word segmentation
    
    Arguments:
        image {Image} -- Input image in float format
    
    Returns:
        (List, Image) -- returns list of bounding boxes and processed image
    '''

    img_denoised = ndimage.filters.median_filter(image,3)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    binary_sauvola = image < thresh_sauvola
    cv_image = img_as_ubyte(binary_sauvola)
    cv_image = cv2.medianBlur(cv_image,3)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    dilation = cv2.dilate(cv_image, rect_kernel, iterations = 3)
    contours, _, = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rectangles = []
    mask = np.zeros(cv_image.shape, np.uint8)
    cv2.drawContours(mask, contours,-1, 255)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x,y,w,h])

    grouped_rectangles, _ = cv2.groupRectangles(rectangles, 0)

    w_tresh = 0.3*grouped_rectangles.T[2].mean()
    h_tresh = 0.3*grouped_rectangles.T[3].mean()

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] > w_tresh,:]
    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[3] > h_tresh,:]


    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] < grouped_rectangles.T[2].max()-20,:]



    #for i,(x,y,w,h) in enumerate(grouped_rectangles):
    #    cv2.rectangle(cv_image, (x, y), (x + w, y + h), 255, 1)
    #    cv2.putText(cv_image, str(i) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255) 

    return grouped_rectangles, mask

def process_word(im):
    img_denoised = ndimage.median_filter(im, 5)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    binary_sauvola = im < thresh_sauvola
    cv_image = img_as_ubyte(binary_sauvola)
    contours, _ = cv2.findContours(cv_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(cv_image.shape, np.uint8)

    cv2.drawContours(mask, contours,-1, 255)

    return img_as_float(mask)