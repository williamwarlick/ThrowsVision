import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('images/stereoLeft/imageL8.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('images/stereoRight/imageR8.png', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)


disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()