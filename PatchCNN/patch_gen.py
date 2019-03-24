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
from sklearn.svm import SVC
from tqdm import tqdm

import cv2

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def getbinim(image):
    img_denoised = ndimage.filters.median_filter(image, 3)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    return img_as_float(image < thresh_sauvola)


BOXWDITH = 512
STRIDE = 300
THRESH = 200


def gen_patches(img_collection: io.ImageCollection, y_lo=645, y_up=2215, x_lim=20):
    nb_im_printed = 0
    nb_im_hw = 0
    nb_im_noise = 0
    num = 0
    for im_index, image in enumerate(tqdm(img_collection)):
        image = image[:, x_lim:]
        bin_im = getbinim(image)
        bin_im = gray2rgb(bin_im)
        bin_im[0:y_lo, :][np.where((bin_im[0:y_lo, :] == [1, 1, 1]).all(axis=2))] = [
            1, 0, 0]
        bin_im[y_lo:y_up, :][np.where(
            (bin_im[y_lo:y_up, :] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]
        bin_im[:, :][np.where((bin_im[:, :] == [0, 0, 0]).all(axis=2))] = [
            0, 0, 1]

        for y in tqdm(range(0, y_up, STRIDE), unit="y_pixel"):
            x = x_lim
            if (y + BOXWDITH > y_up):
                break
            while (x + BOXWDITH) < image.shape[1]:
                io.imsave("fcn_masks/fcn_outputs/"+str(x)+"("+str(y)+")" +
                          str(im_index)+".png", bin_im[y:y+BOXWDITH, x:x+BOXWDITH])
                cv2.imwrite("fcn_im_in/fcn_inputs/"+str(x)+"("+str(y)+")" +
                            str(im_index)+".png", gray2rgb(image[y:y+BOXWDITH, x:x+BOXWDITH]))
                x = x + STRIDE
                num = num + 1
                if num > 3000:
                    print("Done")
                    exit()


if __name__ == "__main__":
    gen_patches(io.imread_collection("data/forms/*.png"))
