from __future__ import print_function

import pickle

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.util import invert
from sklearn.feature_selection import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = pickle.load(open("models/featureMatrix_extracted.sav", "rb"))
Y = pickle.load(open("models/Yarray_extracted.sav", "rb"))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, shuffle=True)

clf = SVC()

#Cs = [0.001, 0.01, 0.1, 1, 10,100,1000]
#gammas = [0.001, 0.01, 0.1, 1,10,100]
#param_grid = {'C': Cs, 'gamma' : gammas}
#clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)

clf.fit(X_train, Y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
acc_train = accuracy_score(Y_train, y_pred_train)
acc_test = accuracy_score(Y_test, y_pred_test)

 
print("[Training acc: "+str(acc_train)+"] | [" + "Testing acc: " + str(acc_test) + "]")

if acc_train > 0.8 and acc_test > 0.8:
    print("[Training acc: "+str(acc_train)+"] | [" + "Testing acc: " + str(acc_test) + "]")
    print("saving model...")
    pickle.dump(clf, open("models/svm_cnn_features.sav", "wb"))
