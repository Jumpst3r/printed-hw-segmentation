import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.color import gray2rgb, grey2rgb
from skimage.filters import threshold_mean, threshold_sauvola
from skimage.restoration import denoise_bilateral
from skimage.util import invert
import keras.backend as K
from sklearn.svm import SVC
from tqdm import tqdm
import tensorflow as tf
from keras.models import load_model
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

def classify(image):
    model = load_model('fcnn.h5', custom_objects={'loss': weighted_categorical_crossentropy([1, 0.8, 0.01]), 'IoU': IoU})
    image = gray2rgb(ndimage.filters.median_filter(image, 3))
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.zeros((maskh, maskw, 3))
    mask2 = np.zeros((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            print("breaking")
            print(y + BOXWDITH)
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(np.array([(input - input.mean())/input.std()]))[0]
            x = x + STRIDE
    io.imsave('hist/im.png', mask2[0:image.shape[0],0:image.shape[1]])       
    io.imsave('hist/red.png', mask2[0:image.shape[0],0:image.shape[1],0])       
    io.imsave('hist/green.png', mask2[0:image.shape[0],0:image.shape[1],1])       

if __name__ == "__main__":
    classify(io.imread('classify.png'))
