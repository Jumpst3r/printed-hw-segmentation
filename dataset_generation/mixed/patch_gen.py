import sys
import warnings

import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage.color import gray2rgb
from tqdm import tqdm


from img_utils import getbinim

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data_root = "data/"


BOXWDITH = 512
STRIDE = 300
NB_TRAINING_SAMPLES = 2000
NB_VALID_SAMPLES = 100


def gen_patches(img_collection_in: io.ImageCollection, img_collection_out: io.ImageCollection):
    numt = 0
    numv = 0
    for im_index, (image_in, image_out) in enumerate(tqdm(zip(img_collection_in, img_collection_out))):
        bin_im = getbinim(image_in)

        for y in range(0, image_in.shape[0], STRIDE):
            x = 0
            if (y + BOXWDITH > image_in.shape[0]):
                break
            while (x + BOXWDITH) < image_in.shape[1]:
                if numt <= NB_TRAINING_SAMPLES:
                    io.imsave(data_root + "fcn_im_in_train/fcn_inputs/" + str(x) + "(" + str(y) + ")" +
                              str(im_index) + ".png", bin_im[y:y + BOXWDITH, x:x + BOXWDITH])
                    io.imsave(data_root + "fcn_masks_train/fcn_outputs/" + str(x) + "(" + str(y) + ")" +
                                str(im_index) + ".png", image_out[y:y + BOXWDITH, x:x + BOXWDITH])
                    x = x + STRIDE
                    numt = numt + 1
                if numt > NB_TRAINING_SAMPLES:
                    io.imsave(data_root + "fcn_im_in_valid/fcn_inputs/" + str(x) + "(" + str(y) + ")" +
                              str(im_index) + ".png", bin_im[y:y + BOXWDITH, x:x + BOXWDITH])
                    io.imsave(data_root + "fcn_masks_valid/fcn_outputs/" + str(x) + "(" + str(y) + ")" +
                                str(im_index) + ".png", image_out[y:y + BOXWDITH, x:x + BOXWDITH])
                    x = x + STRIDE
                    numv = numv + 1
                    if numv > NB_VALID_SAMPLES:
                        print("Done")
                        exit()


if __name__ == "__main__":
    gen_patches(io.imread_collection("raw_in/*"), io.imread_collection("raw_out/*"))
