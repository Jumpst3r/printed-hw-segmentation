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
        image = rgb2gray(image)
        print(image)
        print(image.shape)
        orgim = np.copy(image)

        image = getbinim(image)

        model = load_model('models/fcnn_bin.h5', custom_objects={
            'loss': weighted_categorical_crossentropy([1, 0.8, 0.01]), 'IoU': IoU})
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

        # apply ceil() to FCN predicted labels
        raw_pred = np.copy(mask2[0:image.shape[0], 0:image.shape[1]])
        pred = max_rgb_filter(mask2[0:image.shape[0], 0:image.shape[1]])
        io.imsave('output/pred.png', pred)
        # apply CRF to handwritten / printed channels independently
        #crf_printed = crf(gray2rgb(orgim), convert(pred[:, :, 0]))
        #crf_hw = crf(gray2rgb(orgim), convert(pred[:, :, 1]))
        # combine both CRF results
        #combine = np.zeros(pred.shape)
        #combine[:,:,0] = rgb2gray(crf_printed)
        #combine[:,:,1] = rgb2gray(crf_hw)
        #io.imsave('output/combine.png', combine)
        # Plot results
        plt.tight_layout()
        plt.axis('off')
        print(raw_pred.shape)
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(raw_pred[:,:,0], cmap='gray')
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title('Input image (IAM)')
        axes[1].imshow(raw_pred[:,:,1], cmap='gray')
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        axes[1].set_title('FCN raw segmentation')
        axes[2].imshow(raw_pred[:,:,2], cmap='gray')
        axes[2].get_xaxis().set_visible(False)
        axes[2].get_yaxis().set_visible(False)
        axes[2].set_title('FCN + CRF post-processing')
        plt.show()
        plt.savefig('output/plot'+str(i)+'.png', dpi=900)

if __name__ == "__main__":
    classify(io.imread_collection('input/*'))
