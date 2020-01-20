# example: $python3 detect_barcode.py homography.jpg

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageEnhance
import argparse
import imutils
import cv2
import sys
from PIL import Image



infile = sys.argv[1]
plt.rcParams["figure.figsize"] = (15,12)

# load the image and convert it to grayscale

image_src = cv2.imread(infile)
image = cv2.imread(infile)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#===============================================
#		Find the border
#===============================================

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient0 = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient0)

# blur and threshold the image
blurred = cv2.blur(gradient, (8, 8))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed0 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed1 = cv2.erode(closed0, None, iterations = 4)
closed = cv2.dilate(closed1, None, iterations = 4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

#===============================================
#		image Crop
#===============================================
image_bound = image
box_start = (0,0)
box_end = (0,0)
box_x = rect[0][0]
box_y = rect[0][1]
box_w = rect[1][0]
box_h = rect[1][1]
box_start = (int(box_x - box_w *12/10/2),int(box_y-box_h *14/10/2))
box_end = (int(box_x + box_w *12/10/2),int(box_y+box_h *14/10/2))

#cv2.drawContours(image_bound, [box], -1, (0, 255, 0), 1)
cv2.rectangle(image_bound,box_start,box_end,(255,0,0),1)
image_crop = image_src[box_start[1]:box_end[1], box_start[0]:box_end[0]]
cv2.imwrite("image_corp.jpg", image_crop)

#===============================================
#		image scale up
#===============================================
basewidth = 1000
img_scale = PIL.Image.open("image_corp.jpg")
wpercent = (basewidth / float(img_scale.size[0]))
hsize = int((float(img_scale.size[1]) * float(wpercent)))
img_scale = img_scale.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

img_scale = PIL.ImageEnhance.Sharpness(img_scale).enhance(1)
img_scale.save("img_scale.jpg", "JPEG")



#===============================================
#		image insert
#===============================================
img_combine = np.zeros((800,1000,3), np.uint8)
img_combine[:,0:1000] = (255,255,255) 
img_blank = cv2.imread("blank.jpg")
img_combine[0:img_scale.size[1], 0:img_scale.size[0]] = img_scale

#img_combine.save("img_combine.jpg", "JPEG")
cv2.imwrite("img_combine.jpg", img_combine)


#===============================================
#		threshold
#===============================================

img_scale1 = cv2.imread('img_scale.jpg',0)
ret, thresh1 = cv2.threshold(img_scale1,81,255,cv2.THRESH_TOZERO)
ret, thresh2 = cv2.threshold(thresh1,185,255,cv2.THRESH_TRUNC)
cv2.imwrite("img_threshold.jpg", thresh2)

#===============================================
#		Show image in multi window
#===============================================


plt.subplot(2,4,1), plt.imshow(image_src, 'gray')
plt.subplot(2,4,2), plt.imshow(gradient0, 'gray')
plt.subplot(2,4,3), plt.imshow(gradient, 'gray')
plt.subplot(2,4,4), plt.imshow(blurred, 'gray')
plt.subplot(2,4,5), plt.imshow(closed0, 'gray')
plt.subplot(2,4,6), plt.imshow(closed, 'gray')

plt.subplot(2,4,7), plt.imshow(image_bound, 'gray')
#plt.subplot(2,4,4), plt.imshow(image_bound, 'gray')
#plt.subplot(2,4,5), plt.imshow(img_scale, 'gray')
#plt.subplot(2,4,6), plt.imshow(thresh1, 'gray')
#plt.subplot(2,4,7), plt.imshow(thresh2, 'gray')


plt.show()
cv2.waitKey(0)


