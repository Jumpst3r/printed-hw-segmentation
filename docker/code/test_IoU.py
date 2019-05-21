import subprocess

import numpy as np
import pytest
import skimage.io as io


class Test:

    def test_runnable(self):
        return_code = subprocess.call("./printed-hw-segmentation input.png ./", shell=True)
        assert return_code == 0, 'Launcher script or python runnable failed'
        print("Successfully ran python runnable")




    def test_IoU(self):
        THRESH = 0.95
        def get_IoU(prediction, target):
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            return float(np.sum(intersection)) / float(np.sum(union))

        im_output = io.imread('fcn_out_post.png')
        im_mask = io.imread('mask.png')
        # Compute label-wise IoU scores
        IoUs = []
        for channel in range(3):
            IoUs.append(get_IoU(im_output[:, :, channel], im_mask[:, :, channel]))
        IoUs = np.array(IoUs)
        assert IoUs[0] > 0.7, 'IoU for label [printed] on trained data is less than 0.7 (IoU=' + str(IoUs[0]) + ')'
        assert IoUs[1] > 0.7, 'IoU for label [handwritten] on trained data is less than 0.7 (IoU=' + str(
            IoUs[1]) + ')'
        assert IoUs[2] > 0.7, 'IoU for label [background] on trained data is less than 0.7 (IoU=' + str(IoUs[2]) + ')'
        print("IoU test passed with threshold [" + str(THRESH) + "]")
        print("IoU for label [printed]: " + str(IoUs[0]))
        print("IoU for label [handwritten]: " + str(IoUs[1]))
        print("IoU for label [background]: " + str(IoUs[2]))
        print("Mean IoU: " + str(IoUs.mean()))
