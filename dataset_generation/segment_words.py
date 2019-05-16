'''
This file contains code for extracting words and saving them as binary image from IAM forms by
using the provided ground truth XML files
author: Nicolas Dutly
'''


import os

import numpy as np
import skimage.io as io
import xmltodict
from pyexpat import ExpatError
from tqdm import tqdm

from img_utils import *

XML_DATA_PATH = 'xml/'
IM_DATA_PATH = 'forms/'

IM_OUT_PATH = 'words-printed-bin/'

import string
import random
def id_generator(size=10, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))


def preprocess(image):
    cv_image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    dilation = cv2.dilate(cv_image, rect_kernel, iterations=3)
    _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectangles = []
    mask = np.zeros(cv_image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rectangles.append([x, y, w, h])

    grouped_rectangles, _ = cv2.groupRectangles(rectangles, 0)

    w_tresh = 0.3 * grouped_rectangles.T[2].mean()
    h_tresh = 0.3 * grouped_rectangles.T[3].mean()

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] > w_tresh, :]
    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[3] > h_tresh, :]

    grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] < grouped_rectangles.T[2].max() - 20, :]

    for i,(x,y,w,h) in enumerate(grouped_rectangles):
        cv2.imwrite(IM_OUT_PATH + id_generator() + '.png',cv_image[y:y+h,x:x+w])


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

    x_lo = 100
    x_up = image.shape[1] - 100

    y_lo = 328
    y_up = int(doc['form']['handwritten-part']['line'][0]['@asy'])

    printed_part = image[y_lo:y_up,x_lo:x_up]

    preprocess(printed_part)


if __name__ == '__main__':
    input_files = io.imread_collection(IM_DATA_PATH + '/*')
    xml_files = os.listdir(XML_DATA_PATH)
    random.seed(123)

    for _ in tqdm(range(75), unit='form'):
        rnd_index = random.randint(0, len(input_files)-1)
        xml = xml_files[rnd_index]
        img = input_files[rnd_index]
        name = os.path.splitext(xml)[0]
        with open(XML_DATA_PATH + xml, 'r') as xml_doc:
            xml2segmentation(img, xml_doc, name)