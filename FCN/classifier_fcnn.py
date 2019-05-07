import sys
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from keras.engine.saving import load_model
from skimage import img_as_ubyte
from skimage.color import rgb2gray, gray2rgb
from tqdm import tqdm

from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import convert, getbinim
from post_processing import max_rgb_filter, crf

if not sys.warnoptions:
    warnings.simplefilter("ignore")


BOXWDITH = 512
STRIDE = BOXWDITH

def classify(imgdb):
    model = load_model('models/fcnn_bin.h5', custom_objects={
                        'loss': weighted_categorical_crossentropy([1, 1, 0.1]), 'IoU': IoU})
    for i,image in tqdm(enumerate(imgdb), unit='image'): 
        orgim = np.copy(image)
        image = img_as_ubyte(getbinim(image))
        image = gray2rgb(image)
        maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
        maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
        mask = np.zeros((maskh, maskw, 3))
        mask2 = np.zeros((maskh, maskw, 3))
        mask[0:image.shape[0], 0:image.shape[1]] = image
        for y in range(0, mask.shape[0], STRIDE):
            x = 0
            if (y + BOXWDITH > mask.shape[0]):
                break
            while (x + BOXWDITH) < mask.shape[1]:
                input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
                std = input.std() if input.std() != 0 else 1
                mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                    np.array([(input - input.mean())/std]))[0]
                x = x + STRIDE
        pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
        io.imsave('out.png', pred)



if __name__ == "__main__":
    classify(io.imread_collection("input/*"))
