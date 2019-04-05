import sys
import warnings

import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage.color import gray2rgb

from img_utils import getbinim

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data_root = "../data/"


BOXWDITH = 512
STRIDE = 300
NB_TRAINING_SAMPLES = 10000
NB_VALID_SAMPLES = 3000


def gen_patches(img_collection: io.ImageCollection, y_lo=645, y_up=2215, x_lim=20):
    numt = 0
    numv = 0
    for im_index, image in enumerate(img_collection):
        image = ndimage.filters.median_filter(image, 3)
        image = image[:, x_lim:]
        bin_im = getbinim(image)
        bin_im = gray2rgb(bin_im)
        bin_im_original = np.copy(bin_im)
        bin_im[0:y_lo, :][np.where((bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [
            1, 0, 0]
        bin_im[y_lo:y_up, :][np.where(
            (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]
        bin_im[:, :][np.where((bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [
            0, 0, 1]

        for y in range(0, y_up, STRIDE):
            x = x_lim
            if (y + BOXWDITH > y_up):
                break
            while (x + BOXWDITH) < image.shape[1]:
                if numt <= NB_TRAINING_SAMPLES:
                    io.imsave(data_root + "fcn_masks_train/fcn_outputs/" + str(x) + "(" + str(y) + ")" +
                              str(im_index) + ".png", bin_im[y:y + BOXWDITH, x:x + BOXWDITH])
                    io.imsave(data_root + "fcn_im_in_train/fcn_inputs/" + str(x) + "(" + str(y) + ")" +
                                str(im_index) + ".png", bin_im_original[y:y + BOXWDITH, x:x + BOXWDITH])
                    x = x + STRIDE
                    numt = numt + 1
                if numt > NB_TRAINING_SAMPLES:
                    io.imsave(data_root + "fcn_masks_valid/fcn_outputs/" + str(x) + "(" + str(y) + ")" +
                              str(im_index) + ".png", bin_im[y:y + BOXWDITH, x:x + BOXWDITH])
                    io.imsave(data_root + "fcn_im_in_valid/fcn_inputs/" + str(x) + "(" + str(y) + ")" +
                                str(im_index) + ".png", bin_im_original[y:y + BOXWDITH, x:x + BOXWDITH])
                    x = x + STRIDE
                    numv = numv + 1
                    if numv > NB_VALID_SAMPLES:
                        print("Done")
                        exit()


if __name__ == "__main__":
    gen_patches(io.imread_collection(data_root + "forms/*.png"))
