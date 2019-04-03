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
        image = img_as_float(image)
        image = rgb2gray(image)
        orgim = np.copy(image)

        image = getbinim(image)

        model = load_model('models/fcnn_bin.h5', custom_objects={
            'loss': weighted_categorical_crossentropy([1, 1, 0.03]), 'IoU': IoU})
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
                input = mask[y:y + BOXWDITH, x:x + BOXWDITH]
                io.imsave('tmp/input'+str(x)+str(y)+'.png', input)
                stdv = input.std() if input.std() != 0 else 1
                mask2[y:y + BOXWDITH, x:x + BOXWDITH] = model.predict(
                    np.array([(input - input.mean()) / stdv]))[0]
                x = x + STRIDE
                mask2[y:y + BOXWDITH, x:x + BOXWDITH] = crf(img_as_ubyte(orgim[y:y + BOXWDITH, x:x + BOXWDITH]), max_rgb_filter(mask2[y:y + BOXWDITH, x:x + BOXWDITH]))

        # apply ceil() to FCN predicted labels
        plt.imshow(mask2)
        plt.show()
        exit()
        raw_pred = np.copy(mask2)
        pred = max_rgb_filter(mask2)

        io.imsave('output/pred.png', pred)
        # apply CRF to handwritten / printed channels independently
        crf_printed = crf(gray2rgb(img_as_ubyte(orgim)), pred[:orgim.shape[0], :orgim.shape[1]])
        # combine both CRF results
        #io.imsave('output/combine.png', combine)
        # Plot results
        plt.tight_layout()
        plt.axis('off')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(crf_printed, cmap='gray')
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title('Input image (IAM)')
        axes[1].imshow(pred)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[1].set_title('FCN raw segmentation')

        plt.show()
        plt.savefig('output/plot'+str(i)+'.png', dpi=900)

if __name__ == "__main__":
    classify(io.imread_collection('input/*'))
