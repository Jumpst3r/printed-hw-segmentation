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
    model = pickle.load(open("models/try2.sav", "rb"))
    print(model.summary())
    for im_index, image in enumerate(img_collection):
        image = img_as_float(image)
        grouped_rectangles, bin_img = preprocess(image)
        mask =cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
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
                    prediction = model.predict(box)
                    label = prediction.argmax()
                    if label == 0:
                        votes_hw = votes_hw + 1
                    if label == 1:
                        votes_printed = votes_printed + 1
                    step_h = step_h + 30
                if votes_hw > votes_printed:
                    mask[y:y+h,x:x+w][np.where((mask[y:y+h,x:x+w] == [255,255,255]).all(axis = 2))] = [255,0,0]
                    cv2.rectangle(cvim, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2, 4)
                    io.imsave('pred/pred'+str(y+y+h+x+x+w)+'.png', im[step_v:step_v+STRIDE, step_h:step_h+STRIDE])
                elif votes_hw < votes_printed:
                    mask[y:y+h,x:x+w][np.where((mask[y:y+h,x:x+w] == [255,255,255]).all(axis = 2))] = [0,255,0]
                    cv2.rectangle(cvim, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2, 4)
        mask[np.where((mask == [0,0,0]).all(axis = 2))] = [0,0,255]
        mask[np.where((mask == [255,255,255]).all(axis = 2))] = [0,0,255]
        io.imsave("res/res_"+str(im_index)+"_"+classifier+"_"+".png", img_as_float(cvim))
        io.imsave("training/mask_"+str(im_index)+"_"+classifier+"_"+".png", img_as_float(mask))
        print("saved image"+str(im_index)+"/"+str(len(img_collection)))
if __name__ == "__main__":
    classify(io.imread_collection("data/forms_test/*.png"))
