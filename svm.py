'''
This file contains the code used to train the SVM classifier
author: Nicolas Dutly
'''


from __future__ import print_function

import pickle

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import img_as_float
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from feature_extraction import *

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

np.random.seed(123)

# Fill feature matrix and label vector

X = []
Y = []

print("Loading image collections...")

img_collection_hw = np.array(io.imread_collection('words-hw-bin/*.png'))
img_collection_printed = np.array(io.imread_collection('words-printed-bin/*.png'))


print("Done !")
for image in img_collection_hw:
    X.append(get_feature_vector(img_as_float(image)))
    Y.append(1)

for image in img_collection_printed:
    X.append(get_feature_vector(img_as_float(image)))
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
#acc_train = accuracy_score(Y_train, y_pred_train)
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


pickle.dump(clf, open("models/svm.modelsav", "wb"))
