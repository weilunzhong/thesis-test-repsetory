import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
import urllib2

#load images
img_1 = cv2.imread('/home/weilun/Documents/vionlabs/proj3/data/test/bedroom/image_0003.jpg')
gray_1= cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

img_2 = cv2.imread('/home/weilun/Documents/vionlabs/proj3/data/test/bedroom/image_0004.jpg')
gray_2= cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

# Compute both keypoints and sift features
kp_1, des_1 = sift.detectAndCompute(gray_1,None)
kp_2, des_2 = sift.detectAndCompute(gray_2,None)

print (des_1[0])