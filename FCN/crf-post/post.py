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
NB_ITERATIONS = 4

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
        U = unary_from_labels(labels, n_labels, gt_prob=0.9, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=3, compat=2)
        d.addPairwiseBilateral(sxy=80, srgb=1, rgbim=original_image, compat=4)
 
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

    plt.imshow(mask2rgb(io.imread(sys.argv[1])))
    plt.show()
    exit()


    inp = gray2rgb(io.imread(sys.argv[1]))
    out = (io.imread(sys.argv[2]))
    gt = (io.imread(sys.argv[3]))
    viz = sys.argv[4]

    crf_res = crf(inp, out)


    IoU_printed_old = get_IoU(getBinclassImg(1, out), getBinclassImg(1, gt))
    IoU_printed_new = get_IoU(getBinclassImg(1,crf_res), getBinclassImg(1,gt))




    IoU_hw_old = get_IoU(getBinclassImg(2, out), getBinclassImg(2, gt))
    IoU_hw_new = get_IoU(getBinclassImg(2,crf_res), getBinclassImg(2,gt))


    IoU_bg_old = get_IoU(getBinclassImg(3, out), getBinclassImg(3, gt))
    IoU_bg_new = get_IoU(getBinclassImg(3,crf_res), getBinclassImg(3,gt))

    IoU_mean_old = np.array([IoU_printed_old, IoU_hw_old, IoU_bg_old]).mean()
    IoU_mean_new = np.array([IoU_printed_new, IoU_hw_new, IoU_bg_new]).mean()


    print("Format:   <Class Name>  | [old IoU]-->[new IoU]")
    print("           printed      | [{:.5f}]-->[{:.5f}]".format(IoU_printed_old, IoU_printed_new))
    print("           handwritten  | [{:.5f}]-->[{:.5f}]".format(IoU_hw_old, IoU_hw_new))
    print("           background   | [{:.5f}]-->[{:.5f}]".format(IoU_bg_old, IoU_bg_new))
    print("-------------------------------------------------")
    print("           mean         | [{:.5f}]-->[{:.5f}]".format(IoU_mean_old, IoU_mean_new))
    if viz:
        fig, axes = plt.subplots(1,3)
        fig.suptitle('Printed - handwritten segmentation', fontsize=20)
        axes[0].imshow(inp)
        axes[0].axis('off')
        axes[0].set_title('Input')
        axes[1].imshow(mask2rgb(out))
        axes[1].axis('off')
        axes[1].set_title('FCN raw output [mean IoU = {:.5f}]'.format(IoU_mean_old))
        axes[2].imshow(mask2rgb(crf_res))
        axes[2].axis('off')
        axes[2].set_title('CRF post-processing output [mean IoU = {:.5f}]'.format(IoU_mean_new))
        plt.show()