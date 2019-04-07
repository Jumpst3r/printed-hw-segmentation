import sys
import warnings
import os
import matplotlib.pyplot as plt
import skimage.io as io
from keras.engine.saving import load_model

from fcn_helper_function import *
from img_utils import *
from post_processing import *
from tqdm import tqdm
from skimage import img_as_uint

if not sys.warnoptions:
    warnings.simplefilter("ignore")


BOXWDITH = 512
STRIDE = BOXWDITH

def classify(image, model, use_postprocessing=False):
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
            std = input.std if input.std != 0 else 1
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                np.array([(input - input.mean())/input.std()]))[0]
            x = x + STRIDE
    pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
    if(use_postprocessing):
        crf_printed = crf(image,convert(pred[:,:,0]))
        crf_hw = crf(image,convert(pred[:,:,1]))
        crf_combined = np.zeros((image.shape[0],image.shape[1],3))
        crf_combined[:,:,0] = rgb2gray(crf_printed)
        crf_combined[:,:,1] = rgb2gray(crf_hw)
        pred = crf_combined
    return img_as_uint(pred)
