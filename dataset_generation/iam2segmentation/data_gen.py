import sys
import warnings

import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage.color import gray2rgb

from img_utils import getbinim
from tqdm import tqdm
if not sys.warnoptions:
    warnings.simplefilter("ignore")



BOXWDITH = 512
STRIDE = 300
NB_TRAINING_SAMPLES = 10000
NB_VALID_SAMPLES = 3000


def gen_patches(img_collection: io.ImageCollection, y_lo=645, y_up=2215, x_lim=20):
    numt = 0
    numv = 0
    for im_index, image in enumerate(tqdm(img_collection)):
        image = ndimage.filters.median_filter(image, 3)
        image = image[:y_up, x_lim:]
        bin_im = getbinim(image)
        bin_im = gray2rgb(bin_im)
        bin_im[0:y_lo, :][np.where((bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [
            1, 0, 0]
        bin_im[y_lo:y_up, :][np.where(
            (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]
        bin_im[:, :][np.where((bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [
            0, 0, 1]
        io.imsave('in/iam'+str(im_index)+'.png', image)
        io.imsave('out/iam'+str(im_index)+'.png', bin_im)



if __name__ == "__main__":
    gen_patches(io.imread_collection("data/*.png"))
