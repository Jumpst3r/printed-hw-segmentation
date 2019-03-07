import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.restoration import denoise_bilateral
from skimage.color import gray2rgb
import cv2
import pickle
from sklearn.svm import SVC
from skimage.color import grey2rgb
from skimage.util import invert
from skimage.filters import threshold_sauvola, threshold_mean
from preprocessing import preprocess

STRIDE = 50
THRESH = 200



def classify(x):

    #imdb = io.imread_collection("data/forms_test/*.png")
    imdb = io.imread_collection("*.png")
    cnn_model = pickle.load(open("models/cnn_binary.sav", "rb"))

    for im_index,image in enumerate(imdb):
        image = io.imread("test.png")
        image = img_as_float(image)
        grouped_rectangles =  preprocess(image)
        cvim = cv2.cvtColor(img_as_ubyte(image),cv2.COLOR_GRAY2RGB)
        step_v = 0
        for i,(x,y,w,h) in enumerate(grouped_rectangles):
            im = image[y:y+h, x:x+w]
            votes_printed = 0
            votes_hw = 0
            if im.shape[0] < STRIDE:
                im = cv2.copyMakeBorder(im, (STRIDE - im.shape[0]),0,0,0, cv2.BORDER_WRAP)
            for step_v in range(0, im.shape[0], 30):
                step_h = 0
                if (step_v + STRIDE > im.shape[0]):
                    break
                while (step_h + STRIDE) < im.shape[1]:
                    box = np.array(im[step_v:step_v+STRIDE, step_h:step_h+STRIDE]).reshape(50,50,1)
                    box = box[np.newaxis,:]
                    label = cnn_model.predict_classes(box)
                    if label == 0:
                        votes_hw = votes_hw + 1
                        #cv2.rectangle(cvim, (x+step_h, y+step_v), (x+step_h + 50, y+step_v + 50), (255,0,0,0.4),1,4)
                        #io.imsave('pred_hw/'+str(i)+str(step_v)+str(step_h)+'.png', im[step_v:step_v+STRIDE, step_h:step_h+STRIDE])
                    if label == 1:
                        votes_printed = votes_printed + 1
                        #cv2.rectangle(cvim, (x+step_h, y+step_v), (x+step_h + 50, y+step_v + 50), (0,200,0,0.4),1,4)
                    step_h = step_h + 30
                if votes_hw > votes_printed:
                    cv2.rectangle(cvim, (x, y), (x + w, y + h), (255,0,0),2,4)
                elif votes_hw < votes_printed:
                    cv2.rectangle(cvim, (x, y), (x + w, y + h), (0,255,0),2,4)
                    
        io.imsave("res/form"+str(im_index)+".png", img_as_float(cvim))
        exit()
if __name__ == "__main__":
    classify(None)
