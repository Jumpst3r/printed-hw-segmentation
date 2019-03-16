import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.restoration import denoise_tv_chambolle

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

    img_denoised = ndimage.filters.median_filter(image, 3)
    io.imsave("denoised.png", img_denoised)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    binary_sauvola = image < thresh_sauvola
    cv_image = img_as_ubyte(binary_sauvola)
    cv_image = cv2.medianBlur(cv_image, 3)
    io.imsave("binary.png", cv_image)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    dilation = cv2.dilate(cv_image, rect_kernel, iterations=4)
    io.imsave("dilation.png", dilation)
    _, contours, _ = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectangles = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x, y, w, h])

    grouped_rectangles, _ = cv2.groupRectangles(rectangles, 0)

    w_tresh = 0.3*grouped_rectangles.T[2].mean()
    h_tresh = 0.3*grouped_rectangles.T[3].mean()

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] > w_tresh, :]
    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[3] > h_tresh, :]

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2]
                                            < grouped_rectangles.T[2].max()-20, :]

    return grouped_rectangles, cv_image