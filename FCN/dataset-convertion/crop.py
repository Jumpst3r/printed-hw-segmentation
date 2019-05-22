'''
This file is designed to convert the dataset printed-hw-seg (link in the thesis). The version which can be downloaded
is designed to work out of the box with with DeepDiva but needs to be converted to be able to train the
models in this repository. The models assume inputs of size 256x256. The dataset contains high-res images.

As such this script can be used to extract random crops from the images. This produces a dataset which can then
be used with the models provided in this repo.
'''

import warnings

import numpy.random as random
import skimage.io as io
from skimage.color import gray2rgb

warnings.filterwarnings("ignore")

DATA_ROOT = '../printed-hw-seg/'

in_folder = DATA_ROOT + 'train/data/folder/*.png'
mask_folder = DATA_ROOT + 'train/gt/folder/*.png'

in_folder_out = DATA_ROOT + 'train/data/'
mask_folder_out = DATA_ROOT + 'train/gt/'

HEIGHT = 256
WIDTH = 256
CROP_PER_IM = 200

indb = io.imread_collection(in_folder)
maskdb = io.imread_collection(mask_folder)

print(len(indb))


def crop(img, mask):
    assert img.shape == mask.shape, print(str(img.shape) + str(mask.shape))
    x = random.randint(0, img.shape[1] - WIDTH)
    y = random.randint(0, img.shape[0] - HEIGHT)

    img_in = img[y:y + HEIGHT, x:x + WIDTH]
    img_out = mask[y:y + HEIGHT, x:x + WIDTH]

    assert img_in.shape == img_out.shape

    assert img.shape == mask.shape

    return (img_in, img_out)


for k, (im, mask) in enumerate(zip(indb, maskdb)):
    for i in range(CROP_PER_IM):
        if len(im.shape) < 3:
            im = gray2rgb(im)
        crop_in, crop_out = crop(im, mask)
        io.imsave(mask_folder_out + str(i) + str(k) + '.png', crop_out)
        io.imsave(in_folder_out + str(i) + str(k) + '.png', crop_in)
