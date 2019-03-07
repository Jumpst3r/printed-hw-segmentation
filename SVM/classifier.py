import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.restoration import denoise_bilateral
from skimage.color import gray2rgb
import cv2
import pickle
from sklearn.svm import SVC
from feature_extraction import get_feature_vector
from skimage.color import grey2rgb
from skimage.util import invert
from skimage.filters import threshold_sauvola
from preprocessing import *



def classify(image):
    
    grouped_rectangles, image = preprocess(image)
    svm_model = pickle.load(open("models/svm.sav", "rb"))

    cvim = cv2.imread("data/test5.PNG")

    for i,(x,y,w,h) in enumerate(grouped_rectangles):
        inputim = image[y:y+h, x:x+w]
        f_vec = get_feature_vector(inputim).reshape(1,-1)
        pred = svm_model.predict(f_vec)
        cv2.rectangle(cvim, (x, y), (x + w, y + h),  (255,0,0) if (pred == -1) else (0,255,0),2,4)
    return img_as_float(cvim)