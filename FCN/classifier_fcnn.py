import sys
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from keras.engine.saving import load_model
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2gray, gray2rgb

import argparse
from fcn_helper_function import weighted_categorical_crossentropy, IoU
from img_utils import getbinim, max_rgb_filter, get_IoU, getBinclassImg, mask2rgb, rgb2mask
from post import crf
if not sys.warnoptions:
    warnings.simplefilter("ignore")


BOXWDITH = 256
STRIDE = BOXWDITH - 10

def classify(image):
    model = load_model('/input/models/fcnn_bin.h5', custom_objects={
                        'loss': weighted_categorical_crossentropy([0.4,0.5,0.1]), 'IoU': IoU})
    orgim = np.copy(image)
    image = img_as_float(gray2rgb(getbinim(image)))
    maskw = int((np.ceil(image.shape[1] / BOXWDITH) * BOXWDITH)) + 1
    maskh = int((np.ceil(image.shape[0] / BOXWDITH) * BOXWDITH))
    mask = np.ones((maskh, maskw, 3))
    mask2 = np.zeros((maskh, maskw, 3))
    mask[0:image.shape[0], 0:image.shape[1]] = image
    print("classifying image...")
    for y in range(0, mask.shape[0], STRIDE):
        x = 0
        if (y + BOXWDITH > mask.shape[0]):
            break
        while (x + BOXWDITH) < mask.shape[1]:
            input = mask[y:y+BOXWDITH, x:x+BOXWDITH]
            std = input.std() if input.std() != 0 else 1
            mean = input.mean()
            mask2[y:y+BOXWDITH, x:x+BOXWDITH] = model.predict(
                    np.array([(input-mean)/std]))[0]
            x = x + STRIDE
    return mask2[0:image.shape[0], 0:image.shape[1]]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enableviz", help="Plot results of segmentation",
                        action="store_true")
    parser.add_argument("--enableCRF", help="Use crf for postprocessing",
                        action="store_true")
    parser.add_argument("--ground_truth", help="ground truth file name. Classes must be encoded in the blue channel as 1:printed, 2:hw, 4:bg")
    parser.add_argument("input_image", help="input image file name")
    parser.add_argument("output_folder", help="output folder")
    args = parser.parse_args()
    inputim = io.imread(args.input_image)
    output_folder = io.imread(args.output_folder)


    out = classify(inputim)

    if args.enableCRF:
        crf_res = crf(inputim, out)
    else:
        crf_res = None

    if args.ground_truth is not None:
        gt = io.imread(args.ground_truth)

        mask_out = rgb2mask(max_rgb_filter(out))

        IoU_printed_old = get_IoU(getBinclassImg(1, rgb2mask(max_rgb_filter(out))), getBinclassImg(1, gt))
        IoU_hw_old = get_IoU(getBinclassImg(2, rgb2mask(max_rgb_filter(out))), getBinclassImg(2, gt))
        IoU_bg_old = get_IoU(getBinclassImg(3, rgb2mask(max_rgb_filter(out))), getBinclassImg(3, gt))
        IoU_mean_old = np.array([IoU_printed_old, IoU_hw_old, IoU_bg_old]).mean()

        if crf_res is not None:
            IoU_printed_new = get_IoU(getBinclassImg(1, crf_res), getBinclassImg(1, gt))
            IoU_hw_new = get_IoU(getBinclassImg(2, crf_res), getBinclassImg(2, gt))
            IoU_bg_new = get_IoU(getBinclassImg(3, crf_res), getBinclassImg(3, gt))
            IoU_mean_new = np.array([IoU_printed_new, IoU_hw_new, IoU_bg_new]).mean()

            print("Format:   <Class Name>  | [old IoU]-->[new IoU]")
            print("           printed      | [{:.5f}]-->[{:.5f}]".format(IoU_printed_old, IoU_printed_new))
            print("           handwritten  | [{:.5f}]-->[{:.5f}]".format(IoU_hw_old, IoU_hw_new))
            print("           background   | [{:.5f}]-->[{:.5f}]".format(IoU_bg_old, IoU_bg_new))
            print("-------------------------------------------------")
            print("           mean         | [{:.5f}]-->[{:.5f}]".format(IoU_mean_old, IoU_mean_new))

            if args.enableviz:
                fig, axes = plt.subplots(1, 3)
                fig.suptitle('Printed - handwritten segmentation', fontsize=20)
                axes[0].imshow(inputim)
                axes[0].axis('off')
                axes[0].set_title('Input')
                axes[1].imshow(max_rgb_filter(out))
                axes[1].axis('off')
                axes[1].set_title('FCN raw output [mean IoU = {:.5f}]'.format(IoU_mean_old))
                axes[2].imshow(mask2rgb(crf_res))
                axes[2].axis('off')
                axes[2].set_title('CRF post-processing output [mean IoU = {:.5f}]'.format(IoU_mean_new))
                plt.show()
            else:
                io.imsave(output_folder + 'fcn_out.png', max_rgb_filter(out))
                print('saved fcn_out.png')
                io.imsave(output_folder + 'fcn_out_crf.png', mask2rgb(crf_res))
                print('saved fcn_out_post.png')

        else:
            print("Format:   <Class Name>  | [IoU]")
            print("           printed      | [{:.5f}]".format(IoU_printed_old))
            print("           handwritten  | [{:.5f}]".format(IoU_hw_old))
            print("           background   | [{:.5f}]".format(IoU_bg_old))
            print("-------------------------------------------------")
            print("           mean         | [{:.5f}]".format(IoU_mean_old))

            if args.enableviz:
                fig, axes = plt.subplots(1, 2)
                fig.suptitle('Printed - handwritten segmentation', fontsize=20)
                axes[0].imshow(inputim)
                axes[0].axis('off')
                axes[0].set_title('Input')
                axes[1].imshow(max_rgb_filter(out))
                axes[1].axis('off')
                axes[1].set_title('FCN raw output [mean IoU = {:.5f}]'.format(IoU_mean_old))
                plt.show()
            else:
                io.imsave(output_folder + 'fcn_out.png', max_rgb_filter(out))
                print('saved fcn_out.png')
    else:
        if crf_res is not None:
            if args.enableviz:
                fig, axes = plt.subplots(1, 3)
                fig.suptitle('Printed - handwritten segmentation', fontsize=20)
                axes[0].imshow(inputim)
                axes[0].axis('off')
                axes[0].set_title('Input')
                axes[1].imshow(max_rgb_filter(out))
                axes[1].axis('off')
                axes[1].set_title('FCN raw output')
                axes[2].imshow(mask2rgb(crf_res))
                axes[2].axis('off')
                axes[2].set_title('CRF post-processing output')
                plt.show()
            else:
                io.imsave(output_folder + 'fcn_out.png', max_rgb_filter(out))
                print('saved fcn_out.png')
                io.imsave(output_folder + 'fcn_out_crf.png', mask2rgb(crf_res))
                print('saved fcn_out_post.png')

        else:
            if args.enableviz:
                fig, axes = plt.subplots(1, 2)
                fig.suptitle('Printed - handwritten segmentation', fontsize=20)
                axes[0].imshow(inputim)
                axes[0].axis('off')
                axes[0].set_title('Input')
                axes[1].imshow(max_rgb_filter(out))
                axes[1].axis('off')
                axes[1].set_title('FCN raw output')
                plt.show()
            else:
                io.imsave(output_folder + 'fcn_out.png', max_rgb_filter(out))
                print('saved fcn_out.png')
