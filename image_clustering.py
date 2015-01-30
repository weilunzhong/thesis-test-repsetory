import numpy as np
import cv2

#load images
#img_1 = cv2.imread('/home/weilun/Documents/vionlabs/proj3/data/test/bedroom/image_0003.jpg')
img_1 = cv2.imread('/home/vion-labs-server/Documents/Weilun thesis/proj3/data/test/bedroom/image_0003.jpg')
gray_1= cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

#img_2 = cv2.imread('/home/weilun/Documents/vionlabs/proj3/data/test/bedroom/image_0004.jpg')
img_2 = cv2.imread('/home/vion-labs-server/Documents/Weilun thesis/proj3/data/test/bedroom/image_0004.jpg')
gray_2= cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

# Compute both keypoints and sift features
kp_1, des_1 = sift.detectAndCompute(gray_1,None)
kp_2, des_2 = sift.detectAndCompute(gray_2,None)



Z = img_1.reshape((-1,3))
#print (Z.shape)


# convert to np.float32
Z = np.float32(Z)
#print (np.shape(Z))


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


K = 8
#ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
ret,label,center = cv2.kmeans(des_1, K=10, bestLabels=None,
criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 10), attempts=1, 
flags=cv2.KMEANS_RANDOM_CENTERS) 

print (ret)
print (label)
print (center.shape)

img_withfeature=cv2.drawKeypoints(gray_1,center,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('image',img_withfeature)
#cv2.waitKey(0)



"""
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_1.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""