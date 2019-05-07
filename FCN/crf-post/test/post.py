import sys
from multiprocessing.pool import Pool
from skimage.color import *
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import skimage.io as io
from pydensecrf.utils import (unary_from_labels)
from skimage import img_as_float, img_as_ubyte
from skimage.color import gray2rgb, rgb2gray, rgba2rgb
from matplotlib import cm

from img_utils import getbinim

NB_ITERATIONS = 2

"""
Function which returns the labelled image after applying CRF
Author: https://github.com/lucasb-eyer/pydensecrf
"""


# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied
def crf(original_image, annotated_image, use_2d = True):
    original_image =  img_as_ubyte(np.array(original_image, dtype=np.int))
    annotated_image = img_as_ubyte(np.array(annotated_image, dtype = np.int))

    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,2]

    colors, labels = np.unique(annotated_label, return_inverse=True)
    #Creating a mapping back to 32 bit colors
    print('detected labels:' + str(colors))
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=3, compat=1)
        d.addPairwiseBilateral(sxy=100, srgb=1, rgbim=original_image, compat=5)
 
    Q = d.inference(NB_ITERATIONS)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))
    print(MAP.shape)
    # Convert back to correct colors

    result = np.zeros((MAP.shape[0], MAP.shape[1],3))
    result[:,:,2] = MAP

    result[:,:,2][result[:,:,2] == 2] = 4
    result[:,:,2][result[:,:,2] == 1] = 2
    result[:,:,2][result[:,:,2] == 0] = 1

    return result


def get_IoU(prediction, target):
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            return float(np.sum(intersection)) / float(np.sum(union))


def flat_labels2rgb(mask):
    mask_ = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask_[:,:,2] = mask
    return mask2rgb(mask_)

def rgb2mask(mask):
    result = np.zeros((mask.shape))
    result[:, :][np.where((result[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 4]
    result[:, :][np.where((mask[:, :] == [255, 0, 0]).all(axis=2))] = [0, 0, 1]
    result[:, :][np.where((mask[:, :] == [0, 255, 0]).all(axis=2))] = [0, 0, 2]
    result[:, :][np.where((mask[:, :] == [0, 0, 255]).all(axis=2))] = [0, 0, 4]
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

def swapRB(image):
    return image[:,:,::-1]

if __name__ == "__main__":



    inp = rgba2rgb(gray2rgb(img_as_float(io.imread(sys.argv[1]))))




    out = rgb2mask(gray2rgb(io.imread(sys.argv[2])))

    viz = sys.argv[3]


    crf_res = crf(inp, out)


    if viz:
        fig, axes = plt.subplots(1,3)
        fig.suptitle('Printed - handwritten segmentation', fontsize=20)
        axes[0].imshow(inp)
        axes[0].axis('off')
        axes[0].set_title('Input')
        axes[1].imshow(mask2rgb(out))
        axes[1].axis('off')
        #axes[1].set_title('FCN raw output [mean IoU = {:.5f}]'.format(IoU_mean_old))
        axes[2].imshow(mask2rgb(crf_res))
        axes[2].axis('off')
        #axes[2].set_title('CRF post-processing output [mean IoU = {:.5f}]'.format(IoU_mean_new))
        plt.show()