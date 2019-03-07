import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.restoration import denoise_bilateral
from skimage.filters import threshold_sauvola
from feature_extraction import *
import cv2

imagedb = io.imread_collection("data/handwritten_train/*.png")


for i,im in enumerate(imagedb):

    #img_denoised = ndimage.median_filter(im, 5)
    #thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    #binary_sauvola = im < thresh_sauvola
    cv_image = img_as_ubyte(im)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
    dilation = cv2.dilate(cv_image, rect_kernel, iterations = 3)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(cv_image.shape, np.uint8)

    cv2.drawContours(mask, contours,-1, 255)

    io.imsave("data/handwritten_train_new/"+str(i)+".png", mask)