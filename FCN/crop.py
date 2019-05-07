import cv2
import skimage.io as io
import numpy.random as random
from skimage.color import gray2rgb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

in_folder = 'printed-hw-seg/val/data/folder/*.png'
mask_folder = 'printed-hw-seg/val/gt/folder/*.png'

in_folder_out = 'printed-hw-seg/val/data/'
mask_folder_out = 'printed-hw-seg/val/gt/'


HEIGHT = 256
WIDTH = 256
CROP_PER_IM = 50

indb = io.imread_collection(in_folder)
maskdb = io.imread_collection(mask_folder)

def crop(img, mask):
    assert img.shape == mask.shape, print(str(img.shape) + str(mask.shape))
    x = random.randint(0, img.shape[1] - WIDTH)
    y = random.randint(0, img.shape[0] - HEIGHT)

    img_in = img[y:y + HEIGHT, x:x + WIDTH]
    img_out = mask[y:y + HEIGHT, x:x + WIDTH]



    assert img_in.shape == img_out.shape

    assert img.shape == mask.shape

    return (img_in, img_out)


for k,(im, mask) in enumerate(zip(indb, maskdb)):
    for i in range(CROP_PER_IM):
        if len(im.shape) < 3:
            im = gray2rgb(im)
        crop_in, crop_out = crop(im, mask)
        io.imsave(mask_folder_out + str(i) + str(k) + '.png', crop_out)
        io.imsave(in_folder_out + str(i) + str(k) + '.png', crop_in)