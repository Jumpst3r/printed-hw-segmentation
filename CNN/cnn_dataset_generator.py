import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from skimage import img_as_float, img_as_uint, io
from skimage.filters import threshold_mean, threshold_otsu

import cv2

STRIDE = 50
path = "data/printed_train/*.png"
outdir = "cnn_printed_50px"


def generate_rnd_lines(outdir):
    for i in range(20000):
        im = normal(100, 1, (50, 50))
        im /= im.max()
        k = np.random.randint(0, 50)
        im[k] = np.zeros((1, 50))
        io.imsave("data/noise/hline"+str(i)+".png", im)


def generate_noise(outdir):
    for i in range(20000):
        im = normal(100, 1, (50, 50))
        im /= im.max()
        io.imsave("data/noise/"+str(i)+".png", im)


def generate_samples(path, outdir):
    imgdb = np.array(io.imread_collection('data/cnn_printed_50px/*.png'))

    for i, im1 in enumerate(imgdb):
        im2 = normal(100, 1, (50, 50))
        im2 /= im2.max()
        im = cv2.addWeighted(img_as_float(im1), 0.7,
                             img_as_float(im2), 0.3, 0.0)
        io.imsave("data/cnn_printed_50px_new/im(" +
                  str(i)+".png", img_as_float(im))


if __name__ == "__main__":
    generate_samples(None, None)
