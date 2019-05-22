'''
This file includes a number of helper functions for the keras models.
'''


import keras.backend as K
import numpy as np
import tensorflow as tf

"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
def weighted_categorical_crossentropy(weights):
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
######################################## IoU metric ############################################
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044
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

########################################################################################################################
