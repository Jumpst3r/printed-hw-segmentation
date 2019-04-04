import cv2

from scipy import ndimage
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_sauvola

def getbinim(image):
    img_denoised = ndimage.filters.median_filter(image, 3)
    thresh_sauvola = threshold_sauvola(img_denoised, window_size=25)
    return img_as_float(image < thresh_sauvola)

def convert(image):
    if len(image.shape) < 3:
        image = gray2rgb(image)
    image = image[:, :, ::-1]
    image = img_as_ubyte(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray