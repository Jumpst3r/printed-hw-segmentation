import sys
import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt
import skimage.io as io
from keras.engine.saving import load_model

from fcn_helper_function import *
from img_utils import *
from post_processing import *
from tqdm import tqdm
if not sys.warnoptions:
    warnings.simplefilter("ignore")

BOXWDITH = 512
STRIDE = 512
THRESH = 200


def classify(imdb):
    for i,image in tqdm(enumerate(imdb), unit='images'):

        image = getbinim(image)

        model = load_model('models/fcnn_bin.h5', custom_objects={
            'loss': weighted_categorical_crossentropy([1, 0.8, 0.05]), 'IoU': IoU})
        image = gray2rgb(image)

        im = model.predict(np.array([image]))[0]

        plt.imshow(im)
        plt.show()

if __name__ == "__main__":
    classify(io.imread_collection('tmp/*'))
