import pickle

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from keras import Model
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPool2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from numpy.random import normal
from skimage import filters, img_as_float, img_as_ubyte, img_as_uint
from skimage.filters import threshold_mean, threshold_otsu
from sklearn.model_selection import train_test_split

import cv2

model = pickle.load(open("models/cnn_simple.sav", "rb"))

layer_name = 'flatten_1'
extractor = Model(inputs=model.input,
                  outputs=model.get_layer(layer_name).output)
pickle.dump(extractor, open("models/feature_extractor.sav", "wb"))

X = []
Y = []

print(extractor.summary())

print("Reading images...")
hw_inputs = np.array(io.imread_collection("data/cnn_hw_50px/*.png"))
hw_inputs = hw_inputs[np.random.choice(
    hw_inputs.shape[0], 20000, replace=False)]

#hw_inputs = io.imread_collection("data/cnn_hw_50px/*.png")[:30000]
printed_inputs = io.imread_collection("data/cnn_printed_50px_new/*.png")
#printed_inputs_valid = io.imread_collection("data/cnn_printed_50px/*.png")
noise_inputs = io.imread_collection("data/cnn_noise_50px/*.png")[:20000]
#hw_valid = io.imread_collection("val_hw/*.png")
#printed_valid = io.imread_collection("val_printed/*.png")

for im in hw_inputs:
    im = img_as_float(im)
    X_in = np.array([im]).reshape(50, 50, 1)
    X_in = X_in[np.newaxis, :]
    X.append(extractor.predict(X_in).flatten())
    Y.append(0)

for im in printed_inputs:
    im = img_as_float(im)
    X_in = np.array([im]).reshape(50, 50, 1)
    X_in = X_in[np.newaxis, :]
    X.append(extractor.predict(X_in).flatten())
    Y.append(1)

for im in noise_inputs:
    im = img_as_float(im)
    X_in = np.array([im]).reshape(50, 50, 1)
    X_in = X_in[np.newaxis, :]
    X.append(extractor.predict(X_in).flatten())
    Y.append(2)


X = np.array(X)
Y = np.array(Y)
print("Done!")

print("Dumping files")
pickle.dump(X, open("models/featureMatrix_extracted.sav", "wb"))
pickle.dump(Y, open("models/Yarray_extracted.sav", "wb"))
print("Done")
