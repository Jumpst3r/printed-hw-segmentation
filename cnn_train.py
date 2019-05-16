'''
This file contains the code to train the CNN for the word level classifcation task

Author: Nicolas Dutly
'''

import pickle
import sys
import warnings

import cv2
import keras
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv2D, Dense,
                          Dropout, Flatten, MaxPool2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from skimage import img_as_float, io
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

np.random.seed(123)

X = []
Y = []

print("Reading images...")

hw_inputs = np.array(io.imread_collection("words-hw-bin/*.png"))
printed_inputs = io.imread_collection("words-printed-bin/*.png")


for im in hw_inputs:
    n_im = cv2.resize(im,(50,50))
    X.append(img_as_float(n_im))
    Y.append(0)

for im in printed_inputs:
    n_im = cv2.resize(im,(50,50))
    X.append(img_as_float(n_im))
    Y.append(1)

X = np.array(X)
Y = np.array(Y)
print("Done!")

Y = to_categorical(Y)
X = X.reshape(X.shape[0], 50, 50, 1).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=True)

print('Number of training samples:' + str(len(X_train)))
print('Number of testing samples:' + str(len(y_test)))

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), activation='relu', padding='same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(2))
classifier.add(Activation('softmax'))

classifier.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy'])



print(classifier.summary())

train_gen = ImageDataGenerator(rotation_range=8,
                               width_shift_range=0.08,
                               shear_range=0.3,
                               height_shift_range=0.08,
                               zoom_range=0.08)
test_gen = ImageDataGenerator()
training_set = train_gen.flow(X_train, y_train, batch_size=32)
test_set = train_gen.flow(X_test, y_test, batch_size=32)
history = classifier.fit_generator(training_set,
                         steps_per_epoch=X_train.shape[0]//64,
                         validation_data=test_set,
                         validation_steps=X_test.shape[0]//64,
                         epochs=10)


y_pred_test = classifier.predict_classes(X_test)
#acc_train = accuracy_score(Y_train, y_pred_train)
y_test = np.array([e.argmax() for e in y_test]).flatten()
classification_report = classification_report(y_test, y_pred_test, target_names=['handwritten','printed'], digits=4)
print(classification_report)


plt.plot(history.history['acc'], 'r--')
plt.plot(history.history['val_acc'], 'b-')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'], 'r--')
plt.plot(history.history['val_loss'], 'b-')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

labels=['handwritten','printed']
data = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 20})
ax.matshow(data, cmap='Blues')
ax.set(xticks=np.arange(data.shape[1]),
           yticks=np.arange(data.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=None,
           ylabel='True label',
           xlabel='Predicted label')
for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:d}'.format(z), ha='center', va='center')

plt.show()
pickle.dump(classifier, open("models/cnn.modelsav", "wb"))
