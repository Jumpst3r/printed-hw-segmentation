from __future__ import print_function

import numpy as np
from feature_extraction import *
from preprocessing import *
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from classifier import classify

img = io.imread("data/test5.PNG")
result = classify(img)

plt.imshow(result)
plt.show()
io.imsave("preproccessing_results/classify.png", result)
