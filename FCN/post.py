import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax)
from skimage.color import gray2rgb, rgba2rgb
from skimage import img_as_ubyte
import numpy.random as rd
NB_ITERATIONS = 10

"""
Function which returns the labelled image after applying CRF
adapted from https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf
"""


def crf(original_image, annotated_image):
    rd.seed(123)
    if len(original_image.shape) < 3:
        original_image = gray2rgb(original_image)
    if len(original_image.shape) == 3 and original_image.shape[2]==4:
        original_image = rgba2rgb(original_image)
    original_image = img_as_ubyte(original_image)
    annotated_image = np.moveaxis(annotated_image, -1, 0)
    annotated_image = annotated_image.copy(order='C')

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 3)

    U = unary_from_softmax(annotated_image)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(NB_ITERATIONS)

    MAP = np.argmax(Q, axis=0).reshape(original_image.shape[0], original_image.shape[1])

    result = np.zeros((MAP.shape[0], MAP.shape[1], 3))
    result[:, :, 2] = MAP

    result[:, :, 2][result[:, :, 2] == 2] = 4
    result[:, :, 2][result[:, :, 2] == 1] = 2
    result[:, :, 2][result[:, :, 2] == 0] = 1

    return result
