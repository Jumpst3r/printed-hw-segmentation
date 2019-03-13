import pickle

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.color import gray2rgb, grey2rgb
from skimage.filters import threshold_mean, threshold_sauvola
from skimage.restoration import denoise_bilateral
from skimage.util import invert
from sklearn.svm import SVC

import cv2
from preprocessing import preprocess

STRIDE = 50
THRESH = 200


def classify(img_collection: io.ImageCollection, classifier: str = 'svm'):
    model = None
    feature_extractor = pickle.load(open("models/feature_extractor.sav", "rb"))
    feature_extractor.extract = feature_extractor.predict
    if classifier == 'svm':
        model = pickle.load(open("models/svm_cnn_features.sav", "rb"))
    elif classifier == 'hmm':
        raise Exception("HMM classifier not yet implemented.")
    else:
        raise Exception("Invalid model name.")

    for im_index, image in enumerate(img_collection):
        image = img_as_float(image)
        grouped_rectangles = preprocess(image)
        cvim = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_GRAY2RGB)
        step_v = 0
        for (x, y, w, h) in grouped_rectangles:
            im = image[y:y+h, x:x+w]
            votes_printed = 0
            votes_hw = 0
            if im.shape[0] < STRIDE:
                im = cv2.copyMakeBorder(
                    im, (STRIDE - im.shape[0]), 0, 0, 0, cv2.BORDER_WRAP)
            for step_v in range(0, im.shape[0], 30):
                step_h = 0
                if (step_v + STRIDE > im.shape[0]):
                    break
                while (step_h + STRIDE) < im.shape[1]:
                    box = np.array(
                        im[step_v:step_v+STRIDE, step_h:step_h+STRIDE]).reshape(50, 50, 1)
                    box = box[np.newaxis, :]
                    prediction = model.predict(feature_extractor.extract(box))
                    label = prediction[0]
                    if label == 0:
                        votes_hw = votes_hw + 1
                    if label == 1:
                        votes_printed = votes_printed + 1
                    step_h = step_h + 30
                if votes_hw > votes_printed:
                    cv2.rectangle(cvim, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2, 4)
                elif votes_hw < votes_printed:
                    cv2.rectangle(cvim, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2, 4)

        io.imsave("res/res"+str(im_index)+"_"+classifier+"_"+".png", img_as_float(cvim))
        print("saved image"+str(im_index)+"/"+str(len(img_collection)))
if __name__ == "__main__":
    classify(io.imread_collection("data/forms_test/*.png"))
