import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from numpy.random import normal
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
from skimage import filters, img_as_float, img_as_ubyte, img_as_uint
from skimage.filters import threshold_mean, threshold_otsu
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
import keras

model = pickle.load(open("models/cnn_binary.sav", "rb"))

layer_name = 'flatten_1'
extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)



X = []
X2 = []
Y = []
Y2 = []


im = img_as_float(io.imread('data/cnn_hw_50px/test.png'))
print(extractor.summary())
X_in = np.array([im]).reshape(50,50,1)
X_in = X_in[np.newaxis,:]
print(extractor.predict(X_in).shape)
exit()

print("Reading images...")
hw_inputs = np.array(io.imread_collection("data/cnn_hw_50px/*.png"))
hw_inputs =  hw_inputs[np.random.choice(hw_inputs.shape[0], 20000, replace=False)]

#hw_inputs = io.imread_collection("data/cnn_hw_50px/*.png")[:30000]
printed_inputs = io.imread_collection("data/cnn_printed_50px_new/*.png")
#printed_inputs_valid = io.imread_collection("data/cnn_printed_50px/*.png")
noise_inputs = io.imread_collection("data/cnn_noise_50px/*.png")[:20000]
#hw_valid = io.imread_collection("val_hw/*.png")
#printed_valid = io.imread_collection("val_printed/*.png")

for im in hw_inputs:
    X.append(img_as_float(im))
    Y.append(0)

for im in printed_inputs:
    X.append(img_as_float(im))
    Y.append(1)

for im in noise_inputs:
    X.append(img_as_float(im))
    Y.append(2)


X = np.array(X)
Y = np.array(Y)
print("Done!")

pickle.dump(X, open("models/featureMatrix.sav", "wb"))
pickle.dump(Y, open("models/Yarray.sav", "wb"))
