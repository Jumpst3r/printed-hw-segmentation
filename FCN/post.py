import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (unary_from_softmax)
from skimage.color import gray2rgb
from skimage import img_as_ubyte
import numpy.random as rd
NB_ITERATIONS = 10

"""
Function which returns the labelled image after applying CRF

"""


# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied

def crf(original_image, annotated_image, use_2d=True):
    rd.seed(123)
    original_image = img_as_ubyte(original_image)
    if len(original_image.shape) < 3:
        original_image = gray2rgb(original_image)


    annotated_image = np.moveaxis(annotated_image, -1, 0)
    annotated_image = annotated_image.copy(order='C')
    prob = annotated_image
   # prob = prob.transpose((3,0,1))
    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], 3)

        # get unary potentials (neg log probability)
        U = unary_from_softmax(annotated_image)
        print(U.shape)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(NB_ITERATIONS)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape(original_image.shape[0], original_image.shape[1])

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.

    result = np.zeros((MAP.shape[0], MAP.shape[1], 3))
    result[:, :, 2] = MAP

    result[:, :, 2][result[:, :, 2] == 2] = 4
    result[:, :, 2][result[:, :, 2] == 1] = 2
    result[:, :, 2][result[:, :, 2] == 0] = 1

    return result