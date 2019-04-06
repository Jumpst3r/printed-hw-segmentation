import sys
import warnings

import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import img_as_float
from skimage.color import gray2rgb
from tqdm import tqdm

from img_utils import getbinim

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data_root = "data/"

iam_in = "iam2segmentation/in/*"
iam_out = "iam2segmentation/out/*"

printed_in = "printedsci2segmentation/in/*"
printed_out = "printedsci2segmentation/out/*"

rimes_in = "rimes2segmentation/data/input/*"
rimes_out = "rimes2segmentation/data/output/*"


BOXWDITH = 150
STRIDE = 100
THRESH = 0.1112

def gen_patches(img_collection_in_list, img_collection_out_list):
       for col_index, (col_in, col_out) in enumerate(tqdm(zip(img_collection_in_list, img_collection_out_list), unit='collection')):
        for im_index, (image_in, image_out) in enumerate(tqdm(zip(col_in, col_out), unit='img_pair')):
            bin_im = img_as_float(getbinim(image_in))
            for y in range(0, image_in.shape[0], STRIDE):
                x = 0
                if (y + BOXWDITH > image_in.shape[0]):
                    break
                while (x + BOXWDITH) < image_in.shape[1]:
                        if img_as_float(bin_im[y:y + BOXWDITH, x:x + BOXWDITH]).mean() > THRESH:
                            io.imsave(data_root + "fcn_im_in_train/fcn_inputs/" + str(x) + "-" + str(y) + "-col_" +
                                        str(col_index) + "-" + "im_" + str(im_index) +".png", bin_im[y:y + BOXWDITH, x:x + BOXWDITH], as_gray=True)
                            io.imsave(data_root + "fcn_masks_train/fcn_outputs/" + str(x) + "-" + str(y) + "-col_" +
                                        str(col_index) + "-" + "im_" + str(im_index) +".png", image_out[y:y + BOXWDITH, x:x + BOXWDITH])
                        x = x + STRIDE

if __name__ == "__main__":
    col_list_in = [io.imread_collection(iam_in), io.imread_collection(rimes_in), io.imread_collection(printed_in)]
    col_list_out = [io.imread_collection(iam_out), io.imread_collection(rimes_out), io.imread_collection(printed_out)]
    gen_patches(col_list_in, col_list_out)