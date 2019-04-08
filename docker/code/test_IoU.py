import subprocess
import skimage.io as io
import numpy as np
import pytest


class Test:

    def test_runnable(self):
        return_code = subprocess.call("conda activate base && python printed-hw-seg.py input.png output/", shell=True)
        assert return_code == 0, 'Conda environment activation or python script failed'
        print("Successfully activated conda environment and ran target")




    def test_IoU(self):
        THRESH = 0.95
        def get_IoU(prediction, target):
            intersection = np.logical_and(target, prediction)
            union = np.logical_or(target, prediction)
            return np.sum(intersection) / np.sum(union)

        im_output = io.imread('output/output.png')
        im_mask = io.imread('mask.png')
        # Compute label-wise IoU scores
        IoUs = []
        for channel in range(3):
            IoUs.append(get_IoU(im_output[:, :, channel], im_mask[:, :, channel]))
        IoUs = np.array(IoUs)
        assert IoUs[0] > 0.95, 'IoU for label [printed] on trained data is less than 0.95 (IoU=' + str(IoUs[0]) + ')'
        assert IoUs[1] > 0.95, 'IoU for label [handwritten] on trained data is less than 0.95 (IoU=' + str(
            IoUs[1]) + ')'
        assert IoUs[2] > 0.95, 'IoU for label [background] on trained data is less than 0.95 (IoU=' + str(IoUs[2]) + ')'
        print("IoU test passed with threshold [" + str(THRESH) + "]")
        print("IoU for label [printed]: " + str(IoUs[0]))
        print("IoU for label [handwritten]: " + str(IoUs[1]))
        print("IoU for label [background]: " + str(IoUs[2]))
        print("Mean IoU: " + str(IoUs.mean()))
