import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from img_utils import *
from post_processing import *

imgdb = io.imread_collection('in/*')

for im in imgdb:
    a = im[464:570, 724:790, 0]
    print(a)

    im[:, :,0][im[:, :, 0] > 100] = 255
    im[:, :,1][im[:, :, 1] < 250] = 0
    im[:, :,2][im[:, :, 2] < 250] = 0



    im2 = getbinim(im)

    plt.imshow(im)
    plt.show()
    mask = im2
    mask[:, :][np.where(
        ((im[:, :] != [255, 255, 255]) & (im[:, :] != [0, 0, 0])).all(axis=2))] = [1, 0, 0]
    plt.imshow(im)
    plt.show()

