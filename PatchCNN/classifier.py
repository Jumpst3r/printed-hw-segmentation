import pickle

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

BOXWDITH = 50
STRIDE = 1
THRESH = 200

def gen_patches(img_collection: io.ImageCollection, y_lo=650, y_up=2600):
    model = pickle.load(open("models/cnn.modelsav", "rb"))
    for im_index, image in enumerate(tqdm(img_collection)):
        image = img_as_float(image)[700:900,400:800]
        mask =cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_GRAY2RGB)
        for y in tqdm(range(0, image.shape[0], STRIDE)):
            x = 0
            if (y + BOXWDITH > image.shape[0]):
                break
            while (x + BOXWDITH) < image.shape[1]:
                box = np.array(
                        image[y:y+BOXWDITH, x:x+BOXWDITH]).reshape(50, 50, 1)
                box = box[np.newaxis, :]
                prediction = model.predict(box)
                label = prediction.argmax()
                if label == 0:
                    mask[y:y+BOXWDITH, x:x+BOXWDITH][BOXWDITH//2,BOXWDITH//2] = [255,0,0]
                elif label == 1:
                    mask[y:y+BOXWDITH, x:x+BOXWDITH][BOXWDITH//2,BOXWDITH//2] = [0,255,0]
                elif label == 2:
                    mask[y:y+BOXWDITH, x:x+BOXWDITH][BOXWDITH//2,BOXWDITH//2] = [0,0,255]
                x = x + STRIDE
                io.imsave("res.png", mask)
        exit()
if __name__ == "__main__":
    gen_patches(io.imread_collection("data/forms/*.png"))
