import skimage.io as io
import cv2
import os
import xmltodict
import matplotlib.pyplot as plt
import numpy as np
from pyexpat import ExpatError

from img_utils import *

XML_DATA_PATH = 'xml/'
IM_DATA_PATH = 'images/'

IM_OUT_PATH = 'data/output/'


def xml2segmentation(image, ground_truth, name):
    """
    Takes an image file and groundtruth file from the RIMES dataset
    and outputs an RGB image with pixel-wise labels:
    R : printed
    G : handwritten
    B : Background / noise
    :param name: output name (without extension)
    :param img: input image file name
    :param xml: input xml file name
    :return: pixel-wise annotated image
    """
    orgim = np.copy(image)
    image = gray2rgb(getbinim(image))

    mask = image
    try:
        doc = xmltodict.parse(ground_truth.read())
    except ExpatError:
        print('XML file malformated: ' + name + '.xml' + ' skipping..')
        return

    bboxes = doc['annotation']['box']

    for boxes in bboxes:
        x1, y1 = int(boxes['@top_left_x']), int(boxes['@top_left_y'])
        x2, y2 = int(boxes['@bottom_right_x']), int(boxes['@bottom_right_y'])
        type = boxes['type']

        if 'DactylographiÃ©' in type:
            mask[y1:y2, x1:x2][np.where(
                (image[y1:y2, x1:x2] == [1, 1, 1]).all(axis=2))] = [1, 0, 0]
        if 'Manuscrit' in type:
            mask[y1:y2, x1:x2][np.where(
                (image[y1:y2, x1:x2] == [1, 1, 1]).all(axis=2))] = [0, 1, 0]

    mask[:, :][np.where(
        (image[:, :] == [0, 0, 0]).all(axis=2))] = [0, 0, 1]
    mask[:, :][np.where(
        (image[:, :] == [1, 1, 1]).all(axis=2))] = [0, 0, 1]

    io.imsave(IM_OUT_PATH + name + '.png', mask)
    io.imsave('data/input' + name + '.png', orgim)


if __name__ == '__main__':
    input_files = io.imread_collection(IM_DATA_PATH + '/*')
    xml_files = os.listdir(XML_DATA_PATH)
    for xml, img in zip(xml_files, input_files):
        name = os.path.splitext(xml)[0]
        with open(XML_DATA_PATH + xml, 'r') as xml_doc:
            xml2segmentation(img, xml_doc, name)
