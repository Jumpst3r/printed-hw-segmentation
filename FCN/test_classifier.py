"""
This file can be used to evaluate models. You will only need the test/ folder of the
dataset printed-hw-seg which can be downloaded in the thesis.
Run python test_classifier.py for help
"""
import argparse
import sys
import warnings

import numpy as np
import skimage.io as io
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import max_rgb_filter, get_IoU, getBinclassImg, mask2rgb, rgb2mask
from keras.engine.saving import load_model
from post import crf
from skimage import img_as_float
from skimage.color import gray2rgb
from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")

BOXWDITH = 256
STRIDE = BOXWDITH - 10


def classify(imgdb):
    result_imgs = []
    print("classifying " + str(len(imgdb)) + ' images')
    for image in imgdb:
        model = load_model('models/fcnn_bin_simple.h5', custom_objects={
            'loss': weighted_categorical_crossentropy([0.4, 0.5, 0.1]), 'IoU': IoU})
        orgim = np.copy(image)
        # assume image in binary
        image = img_as_float(gray2rgb(image))
        maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
        maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
        mask = np.ones((maskh, maskw, 3))
        mask2 = np.zeros((maskh, maskw, 3))
        mask[0:image.shape[0], 0:image.shape[1]] = image
        for y in tqdm(range(0, mask.shape[0], STRIDE), unit='batch'):
            x = 0
            if (y + BOXWDITH > mask.shape[0]):
                break
            while (x + BOXWDITH) < mask.shape[1]:
                input = mask[y:y + BOXWDITH, x:x + BOXWDITH]
                std = input.std() if input.std() != 0 else 1
                mean = input.mean()
                mask2[y:y + BOXWDITH, x:x + BOXWDITH] = model.predict(
                    np.array([(input - mean) / std]))[0]
                x = x + STRIDE
        result_imgs.append(mask2[0:image.shape[0], 0:image.shape[1]])
    return result_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_folder", help="ground truth folder")
    parser.add_argument("test_folder", help="test folder")
    args = parser.parse_args()

    image_col = io.imread_collection(args.test_folder + '*')
    mask_col = io.imread_collection(args.ground_truth_folder + '*')
    crf_orgim = io.imread_collection(mask_col + 'crf-orgim/*')
    out_raw = classify(image_col)
    out_crf = [crf(inim, outim) for inim, outim in zip(crf_orgim, out_raw)]

    results_folder = 'test_out/'

    IoUs_hw_old = []
    IoUs_hw_new = []
    IoUs_printed_old = []
    IoUs_printed_new = []
    IoUs_bg_old = []
    IoUs_bg_new = []
    IoUs_mean_old = []
    IoUs_mean_new = []

    for i, (out, crf_res, gt) in enumerate(zip(out_raw, out_crf, mask_col)):
        IoU_printed_old = get_IoU(getBinclassImg(1, rgb2mask(max_rgb_filter(out))), getBinclassImg(1, gt))
        IoU_hw_old = get_IoU(getBinclassImg(2, rgb2mask(max_rgb_filter(out))), getBinclassImg(2, gt))
        IoU_bg_old = get_IoU(getBinclassImg(3, rgb2mask(max_rgb_filter(out))), getBinclassImg(3, gt))
        IoU_mean_old = np.array([IoU_printed_old, IoU_hw_old, IoU_bg_old]).mean()

        IoU_printed_new = get_IoU(getBinclassImg(1, crf_res), getBinclassImg(1, gt))
        IoU_hw_new = get_IoU(getBinclassImg(2, crf_res), getBinclassImg(2, gt))
        IoU_bg_new = get_IoU(getBinclassImg(3, crf_res), getBinclassImg(3, gt))
        IoU_mean_new = np.array([IoU_printed_new, IoU_hw_new, IoU_bg_new]).mean()

        IoUs_hw_old.append(IoU_hw_old)
        IoUs_hw_new.append(IoU_hw_new)
        IoUs_printed_new.append(IoU_printed_new)
        IoUs_printed_old.append(IoU_printed_old)
        IoUs_bg_old.append(IoU_bg_old)
        IoUs_bg_new.append(IoU_bg_new)
        IoUs_mean_old.append(IoU_mean_old)
        IoUs_mean_new.append(IoU_mean_new)

        print("--------------- IoU test results for image " + str(i) + " ---------------")
        print("Format:   <Class Name>  | [old IoU]-->[new IoU]")
        print("           printed      | [{:.2f}]-->[{:.2f}]".format(IoU_printed_old, IoU_printed_new))
        print("           handwritten  | [{:.2f}]-->[{:.2f}]".format(IoU_hw_old, IoU_hw_new))
        print("           background   | [{:.2f}]-->[{:.2f}]".format(IoU_bg_old, IoU_bg_new))
        print("-------------------------------------------------")
        print("           mean         | [{:.2f}]-->[{:.2f}]".format(IoU_mean_old, IoU_mean_new))
        print("\n")

        io.imsave(results_folder + 'fcn_out_' + str(i) + '.png', max_rgb_filter(out))
        print('saved fcn_out_' + str(i) + '.png')
        io.imsave(results_folder + 'fcn_out_crf_' + str(i) + '.png', mask2rgb(crf_res))
        print('saved fcn_out_crf_' + str(i) + '.png')

        print("\n")

    IoUs_hw_old = np.array(IoUs_hw_old)
    IoUs_hw_new = np.array(IoUs_hw_new)
    IoUs_printed_old = np.array(IoUs_printed_old)
    IoUs_printed_new = np.array(IoUs_printed_new)
    IoUs_bg_old = np.array(IoUs_bg_old)
    IoUs_bg_new = np.array(IoUs_bg_new)
    IoUs_mean_old = np.array(IoUs_mean_old)
    IoUs_mean_new = np.array(IoUs_mean_new)

    print('\n')
    print("--------------- IoU mean test results ---------------")
    print("Format:   <Class Name>  | [old IoU mean]-->[new IoU mean]")
    print("           printed      | [{:.2f}]-->[{:.2f}]".format(IoUs_printed_old.mean(), IoUs_printed_new.mean()))
    print("           handwritten  | [{:.2f}]-->[{:.2f}]".format(IoUs_hw_old.mean(), IoUs_hw_new.mean()))
    print("           background   | [{:.2f}]-->[{:.2f}]".format(IoUs_bg_old.mean(), IoUs_bg_new.mean()))
    print("-------------------------------------------------")
    print("           mean         | [{:.2f}]-->[{:.2f}]".format(IoUs_mean_old.mean(), IoUs_mean_new.mean()))
    print("\n")
