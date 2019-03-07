import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert
from scipy import ndimage
from preprocessing import process_word
import cv2

def get_feature_vector(image):
    #image = process_word(image)
    return np.array([get_variance(image), get_mean(image), get_bb_density(image)])
    #return np.array([get_mean(image), get_variance(image)])

def get_variance(image):
    return image.var()

def get_top_pixel_count(image):
    return image[0].mean()

def get_bot_pixel_count(image):
    return image[-1].mean()

def get_horizontal_variance(image):
    '''Calculates and returns the horizontal projection variance
    
    Arguments:
        image {numpy.ndarray} -- Binary array containing the image to process
    
    Returns:
        int -- [horizontal projection variance]
    '''

    return image.mean(axis=1).var()


def get_bb_density(image):
    '''Calculates and returns pixel density
    
    Arguments:
        image {numpy.ndarray} -- Binary array containing the image to process
    
    Returns:
        float -- White pixel density
    '''

    return image.mean()

def get_vertical_variance(image):
    '''Calculates and returns the vertical projection variance
    
    Arguments:
        image {numpy.ndarray} -- Binary array containing the image to process
    
    Returns:
        int -- [vertical projection variance]
    '''

    return image.sum(axis=0).var()

def get_vertical_mean(image):
    '''Calculates and returns the vertical projection mean
    
    Arguments:
        image {numpy.ndarray} -- Binary array containing the image to process
    
    Returns:
        int -- [vertical projection mean]
    '''

    return image.mean(axis=0).mean()

def get_mean(image):
    return image.mean()
