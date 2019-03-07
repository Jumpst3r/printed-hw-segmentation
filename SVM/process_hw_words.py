import numpy as np
from feature_extraction import *
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from grayscale_preprocessing import preprocess
import matplotlib.pyplot as plt
import pickle


img = io.imread_collection("data/printed_docs/*.png")



print "Loading image collections..."

img_collection_hw = io.imread_collection('data/printed_docs/*.png')

print "Done !"

for i,image in enumerate(img_collection_hw):
    bblist, res = preprocess(image)
    for k,(x,y,w,h) in enumerate(bblist):
        io.imsave("data/printed_train/"+str(i)+"-"+str(k)+".png", image[y:y+h,x:x+w])

