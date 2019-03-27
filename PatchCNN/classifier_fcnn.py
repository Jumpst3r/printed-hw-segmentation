import skimage.io as io
import sys
import warnings
import matplotlib.pyplot as plt


import cv2
from keras.engine.saving import load_model
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola

from fcn_helper_function import *
from post_processing import *
from img_utils import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def getbinim(image):
    img_denoised = ndimage.filters.median_filter(image, 3)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    return img_as_float(image < thresh_sauvola)


BOXWDITH = 512
STRIDE = 512
THRESH = 200


def classify(image):
    image = getbinim(image)
    model = load_model('models/fcnn_bin.h5', custom_objects={
        'loss': weighted_categorical_crossentropy([1, 0.8, 0.01]), 'IoU': IoU})
    image = gray2rgb(image)
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.ones((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y + BOXWDITH, x:x + BOXWDITH]
            io.imsave('input.png', input)
            mask2[y:y + BOXWDITH, x:x + BOXWDITH] = model.predict(
                np.array([(input - input.mean()) / input.std()]))[0]
            x = x + STRIDE

    # apply ceil() to FCN predicted labels
    pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
    pred = np.array(pred, dtype=int)
    raw_pred = np.copy(pred)
    io.imsave('pred.png', mask2[0:image.shape[0], 0:image.shape[1]])
    # apply CRF to handwritten / printed channels independently
    crf_printed = crf(image, pred[:, :, 0])
    crf_hw = crf(image, pred[:, :, 1])
    # combine both CRF results
    pred[:, :, 0] = crf_printed[:, :, 0]
    pred[:, :, 1] = crf_hw[:, :, 1]
    pred = max_rgb_filter(pred)
    # Plot results
    plt.tight_layout()
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(rgb2gray(image), cmap='gray')
    axes[1].imshow(rgb2gray(raw_pred), cmap='jet')
    axes[2].imshow(rgb2gray(pred), cmap='jet')
    plt.savefig('output/compare.png', dpi=900)

    io.imsave('output/red.png', img_as_float(pred[0:image.shape[0], 0:image.shape[1], 0]))
    io.imsave('output/green.png', img_as_float(pred[0:image.shape[0], 0:image.shape[1], 1]))




if __name__ == "__main__":
    classify(io.imread('classify.jpg'))
