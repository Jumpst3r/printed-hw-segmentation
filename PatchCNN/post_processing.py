# From https://github.com/lucasb-eyer/pydensecrf


import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_labels)
from skimage import img_as_float, img_as_ubyte
from skimage.color import gray2rgb
import cv2
from img_utils import *

"""
Function which returns the labelled image after applying CRF

"""


# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied
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
    Q = d.inference(2)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    return MAP.reshape(original_image.shape)


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

def remove_lines(image):
        gray = img_as_ubyte(image)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        cv2.imwrite('edges-50-150.jpg', edges)
        minLineLength = 100

        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=minLineLength, maxLineGap=10)
        print("done")
        a, b, c = lines.shape
        for i in range(a):
            x = lines[i][0][0] - lines[i][0][2]
            y = lines[i][0][1] - lines[i][0][3]
            if x != 0:
                if abs(y / x) < 1:
                    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255),
                             1, cv2.LINE_AA)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
        return getbinim(img_as_float(gray))