import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from img_utils import *
from post_processing import *
from scipy import ndimage
from tqdm import tqdm
imgdb = io.imread_collection('in/*')

for i, im in enumerate(tqdm(imgdb, unit='image')):
    im[:, :,0][im[:, :, 0] > 50] = 255
    im[:, :,1][im[:, :, 1] < 250] = 0
    im[:, :,2][im[:, :, 2] < 250] = 0

    im2 = getbinim(im)

    im2 = ndimage.median_filter(im2, size=3)


    im2 = gray2rgb(im2)
    mask = np.zeros(im.shape)

    mask[:, :][np.where((im2[:, :] == [1, 1, 1]).all(axis=2))] = [
        1, 0, 0]
    mask[:, :][np.where((im[:, :] == [255, 0, 0]).all(axis=2))] = [
        0, 1, 0]
    mask[:, :][np.where((im2[:, :] == [0, 0, 0]).all(axis=2))] = [
        0, 0, 1]

    io.imsave('out/'+ str(i) + '.png', mask)

