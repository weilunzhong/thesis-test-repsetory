import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
import urllib2

#load images
img_1 = cv2.imread('E:/Weilun_thesis/testing code/images.jpg')
gray_1= cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

img_2 = cv2.imread('E:/Weilun_thesis/testing code/box_in_scene.png')
gray_2= cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

# Compute both keypoints and sift features
kp_1, des_1 = sift.detectAndCompute(gray_1,None)
kp_2, des_2 = sift.detectAndCompute(gray_2,None)


# Knn methods for feature matching
raw_matches = matcher.knnMatch(des_1, trainDescriptors = des_2, k = 2)
print(raw_matches[0])

"""
matches has a format of DMatch.distance; Dmatch.trainIdx; DMatch.queryIdx; DMatch.imgIdx
"""

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    """
    Keep only matches that have distance ratio to 
    second closest point less than 'ratio'.
    """
    mkp1, mkp2 = [], []
    for m in matches:
        if m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, mkp1, mkp2


# Finding the matching pair os sift features
p1, p2, kp_pairs, mkp1, mkp2 = filter_matches(kp_1, kp_2, raw_matches)
img_withfeature=cv2.drawKeypoints(gray_1,mkp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('image',img_withfeature)
cv2.waitKey(0)



# Calculating the 
H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
#print (H)



h1, w1 = gray_1.shape[:2]
h2, w2 = gray_2.shape[:2]
vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
vis[:h1, :w1] = gray_1
vis[:h2, w1:w1+w2] = gray_2
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)


# plot the matches
color = (0, 255, 0)
for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
    if inlier:
        cv2.circle(vis, (x1, y1), 2, color, -1)
        cv2.circle(vis, (x2, y2), 2, color, -1)
        cv2.line(vis, (x1, y1), (x2, y2), color)

#cv2.imshow('image',vis)
#cv2.waitKey(0)






#print (good)
#print (kp_1[0])
#print (np.shape(des_1))
# print (np.shape(kp_1))
# print (np.shape(kp_2))



#plt.imshow(img3,),plt.show()



# img_withfeature=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# print (np.shape(img_withfeature))



#cv2.imshow(img_withfeature)
#cv2.imwrite('sift_keypoints.jpg',img_withfeature)

# print (np.shape(kp))
# print (np.shape(des))
# print (kp[0])
 
#img=cv2.drawKeypoints(gray,kp)


#cv2.imshow('image',img_withfeature)
#cv2.waitKey(0)
