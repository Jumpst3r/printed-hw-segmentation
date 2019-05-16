'''
This file contains the code used to train the SVM classifier with CNN extracted features.
author: Nicolas Dutly
'''

import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from keras import Model
from skimage import img_as_float
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

np.random.seed(123)

model = pickle.load(open("models/cnn.modelsav", "rb"))

layer_name = 'flatten_1'
extractor = Model(inputs=model.input,
                  outputs=model.get_layer(layer_name).output)
print("Loading image collections...")

img_collection_hw = np.array(io.imread_collection('words-hw-bin/*.png'))
img_collection_printed = np.array(io.imread_collection('words-printed-bin/*.png'))

X = []
Y = []

print("Done !")
for image in img_collection_hw:
    image = cv2.resize(image,(50,50))
    image = img_as_float(image)
    X_in = np.array([image]).reshape(50, 50, 1)
    X_in = X_in[np.newaxis, :]
    X.append(extractor.predict(X_in).flatten())
    Y.append(1)

for image in img_collection_printed:
    image = cv2.resize(image,(50,50))
    image = img_as_float(image)
    X_in = np.array([image]).reshape(50, 50, 1)
    X_in = X_in[np.newaxis, :]
    X.append(extractor.predict(X_in).flatten())
    Y.append(-1)

print("splitting data...")
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, shuffle=True)

print('Number of training samples:' + str(len(X_train)))
print('Number of testing samples:' + str(len(Y_test)))

clf = SVC(kernel='rbf', C=1000, gamma=0.001)
'''Cs = [0.0001,0.001, 0.01, 0.1, 1, 10,100,1000,10000]
gammas = [0.0001,0.001, 0.01, 0.1, 1,10,100,1000,10000]
kernels = ['linear','rbf']
param_grid = {'C': Cs, 'gamma' : gammas, 'kernel':kernels}
clf = GridSearchCV(SVC(), param_grid, cv=5, verbose=3, n_jobs=-1)
'''
clf.fit(X_train, Y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

classification_report = classification_report(Y_test, y_pred_test, target_names=['handwritten','printed'], digits=4)
labels=['handwritten','printed']

data = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 20})
ax.matshow(data, cmap='Blues')
ax.set(xticks=np.arange(data.shape[1]),
           yticks=np.arange(data.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=None,
           ylabel='True label',
           xlabel='Predicted label')

for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{:d}'.format(z), ha='center', va='center')

plt.show()
print(classification_report)


pickle.dump(clf, open("models/svm_cnn.modelsav", "wb"))
