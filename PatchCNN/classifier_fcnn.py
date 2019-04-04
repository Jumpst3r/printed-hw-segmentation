import sys
import warnings
import os
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


def classify(imgdb):
    model = load_model('models/fcnn_bin.h5', custom_objects={
                        'loss': weighted_categorical_crossentropy([1, 0.8, 0.01]), 'IoU': IoU})
    for i,image in tqdm(enumerate(imgdb), unit='image'): 
        orgim = np.copy(image)
        image = img_as_ubyte(getbinim(image))
        image = gray2rgb(image)
        maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
        maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
        mask = np.zeros((maskh, maskw, 3))
        mask2 = np.zeros((maskh, maskw, 3))
        mask[0:image.shape[0], 0:image.shape[1]] = image
        for y in range(0, mask.shape[0], STRIDE):
            x = 0
            if (y + BOXWDITH > mask.shape[0]):
                break
            while (x + BOXWDITH) < mask.shape[1]:
                input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
                std = input.std if input.std != 0 else 1
                mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                    np.array([(input - input.mean())/input.std()]))[0]
                x = x + STRIDE
        pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
        crf_printed = crf(image,convert(pred[:,:,0]))
        crf_hw = crf(image,convert(pred[:,:,1]))
        crf_combined = np.zeros((image.shape[0],image.shape[1],3))
        crf_combined[:,:,0] = rgb2gray(crf_printed)
        crf_combined[:,:,1] = rgb2gray(crf_hw)

        plt.tight_layout()
        plt.axis('off')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(rgb2gray(image), cmap='gray')
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title('Original image')
        axes[1].imshow(rgb2gray(crf_combined), cmap='Paired')
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[1].set_title('FCN+CRF combined result')
        plt.show()
        plt.savefig('output/plot'+str(i)+'.png', dpi=900)




if __name__ == "__main__":
    classify(io.imread_collection("input/*"))