import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy import ndimage
from skimage import filters, img_as_float, img_as_ubyte
from skimage.restoration import denoise_bilateral
from skimage.filters import threshold_sauvola
from feature_extraction import *
import cv2

image = io.imread("data/test5.PNG", as_gray=True)

img_denoised = ndimage.median_filter(image, 5)
window_size = 25
thresh_sauvola = threshold_sauvola(img_denoised, window_size=window_size)
binary_sauvola = image < thresh_sauvola
cv_image = img_as_ubyte(binary_sauvola)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
dilation = cv2.dilate(cv_image, rect_kernel, iterations = 2)

contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
rectangles = []

mask = np.zeros(binary_sauvola.shape, np.uint8)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    rectangles.append([x,y,w,h])

cv2.drawContours(mask, contours,-1, 255)

grouped_rectangles, _ = cv2.groupRectangles(rectangles, 0, 10e5)

w_tresh = 1.1*grouped_rectangles.T[2].mean()
h_tresh = 1.1*grouped_rectangles.T[3].mean()


grouped_rectangles = grouped_rectangles[grouped_rectangles.T[2] > w_tresh,:]
grouped_rectangles = grouped_rectangles[grouped_rectangles.T[3] > h_tresh,:] 

x1,y1,w1,h1 = grouped_rectangles[107]
x2,y2,w2,h2 = grouped_rectangles[25]

cv2.imshow("sss1",mask[y1:y1+h1,x1:x1+w1])
cv2.waitKey()
cv2.imshow("sss2",mask[y2:y2+h2,x2:x2+w2])
cv2.waitKey()

im1 = mask[y1:y1+h1,x1:x1+w1].mean(axis=0)
im2 = mask[y2:y2+h2,x2:x2+w2].mean(axis=0)

plt.plot(im1)
plt.plot(im2)
plt.show()
