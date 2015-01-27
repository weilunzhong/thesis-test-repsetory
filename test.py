import cv2
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

#plt.show()

knn = cv2.KNearest()
newcomers = np.random.randint(0,100,(10,2)).astype(np.float32)

for x in xrange (0,10):
	print (newcomers[x,:])
	print (x)
	plt.scatter(newcomers[x,0], newcomers[x,1],80,'g','o')
# 	knn.train(trainData,responses)
# 	ret, results, neighbours ,dist = knn.find_nearest(newcomers[1,:], 3)

# 	print "result: ", results,"\n"
# 	print "neighbours: ", neighbours,"\n"
# 	print "distance: ", dist, "\n"



# newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
# print (newcomer)
# plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

# knn = cv2.KNearest()
# knn.train(trainData,responses)
# ret, results, neighbours ,dist = knn.find_nearest(newcomer, 3)

# print "result: ", results,"\n"
# print "neighbours: ", neighbours,"\n"
# print "distance: ", dist,"\n"


# newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
# print (newcomer)
# plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv2.KNearest()
knn.train(trainData,responses)
ret, results, neighbours ,dist = knn.find_nearest(newcomers, 3)

print "result: ", results,"\n"
print "neighbours: ", neighbours,"\n"
print "distance: ", dist,"\n"


plt.show()
