from __future__ import print_function

import numpy as np
from feature_extraction import *
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skimage.util import invert
import matplotlib.pyplot as plt
import pickle

# Fill feature matrix and label vector

X = []
Y = []

print("Loading image collections...")

img_collection_hw = np.array(io.imread_collection('data/handwritten_train_new/*.png'))
img_collection_printed = np.array(io.imread_collection('data/printed_train_new/*.png'))

img_collection_hw =  img_collection_hw[np.random.choice(img_collection_hw.shape[0], 6000, replace=False)]
img_collection_printed =  img_collection_printed[np.random.choice(img_collection_printed.shape[0], 6000, replace=False)]



print("Done !")
for image in img_collection_hw:
    X.append(get_feature_vector(image))
    Y.append(1)

for image in img_collection_printed:
    X.append(get_feature_vector(image))
    Y.append(-1)

#X = pickle.load(open("models/featureMatrix.sav", "rb"))

#Y = pickle.load(open("models/Yarray.sav", "rb"))


#pickle.dump(X, open("models/featureMatrix.sav", "wb"))
#pickle.dump(Y, open("models/Yarray.sav", "wb"))

#X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)

print("scaling features...")

print("splitting data...")
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
    pickle.dump(clf, open("models/svm.sav", "wb"))