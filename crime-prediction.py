import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_train = pd.read_table("crimedata-train.txt")
df_test = pd.read_table("crimedata-test.txt")

# this is the function to calculate soft
def soft(a, sigma):
	return np.sign(a) * max(abs(a) - sigma, 0)

# this is the function to calculate lasso using murphy algorithm
def lasso(X, y, lambdaa):
	transposeX = np.transpose(X)
	firstDot = np.dot(transposeX,X)
	identity = np.identity(len(X[0]))
	secondDot = np.dot(lambdaa,identity)
	inverse = np.linalg.inv(firstDot +secondDot)
	w = np.dot(np.dot(inverse, transposeX), y)
	stop = False
	while not stop:
		wold = w[:]
		for j in range(len(X[0])):
		 	aj = 0
		 	cj = 0
		 	for i in range(len(X)):
		 		xij = X[i][j]
		 		wj = w[j]
		 		xi = X[i]
		 		yi = y[i]
		 		aj += xij**2
		 		firstDot = np.dot(np.transpose(w),xi)
		 		secondDot = np.dot(wj,xij)
		 		cj += xij * (yi - firstDot + secondDot)
		 	aj *= 2
		 	cj *= 2
		 	w[j] = soft(cj / aj, lambdaa/aj)
		stop = np.abs(np.linalg.norm(wold) - np.linalg.norm(w)) < 10**(-6)
	return w


# take the data
y = df_train.values[:, :1]
X = df_train.values[:, 1:]

# each title coresponded to an index
indexMapColumn = {}
for i in range(len(df_train.columns)):
	indexMapColumn[df_train.columns[i]] = i

###############################################################
# I calculate the regularization paths for 5 these coefficients
indexList = []
indexList.append(indexMapColumn["agePct12t29"])
indexList.append(indexMapColumn["pctWSocSec"])
indexList.append(indexMapColumn["PctKids2Par"])
indexList.append(indexMapColumn["PctIlleg"])
indexList.append(indexMapColumn["HousVacant"])


def calculateRegularizationPaths(X, y, indexList):
	lambdaa = 600
	count = 0
	while count < 10:
		w = lasso(X, y, np.log(lambdaa))
		for i in indexList:
			plt.plot(np.log(lambdaa), w[i], 'ro')
		lambdaa /= 2
		count += 1

	plt.show()

#calculateRegularizationPaths(X, y, indexList)
##############################################################

# Calculate the squared error in the training Data
def SqErTraining(X, y):
	lambdaa = 600
	count = 0
	copyY = y[:]
	while count < 10:
		copyY = y[:]
		w = lasso(X, y, np.log(lambdaa))
		dot = np.dot(np.transpose(w), np.transpose(X))
		transposeY = np.transpose(copyY)
		copyY = transposeY - dot
		transposeY = np.transpose(copyY)
		copyY = np.dot(copyY, transposeY)
		plt.plot(np.log(lambdaa), copyY, 'ro')
		lambdaa /= 2
		count += 1
	plt.show()
	
#SqErTraining(X,y)

###########################################

# Calculate the squared error in test data
def SqErTesting(X, y, Xtest, ytest):
	lambdaa = 600
	count = 0
	copyY = y[:]
	while count < 10:
		copyY = ytest[:]
		w = lasso(X, y, np.log(lambdaa))
		dot = np.dot(np.transpose(w), np.transpose(Xtest))
		transposeY = np.transpose(copyY)
		copyY = transposeY - dot
		transposeY = np.transpose(copyY)
		copyY = np.dot(copyY, transposeY)
		plt.plot(np.log(lambdaa), copyY, 'ro')
		lambdaa /= 2
		count += 1
	plt.show()

ytest = df_test.values[:, :1]
Xtest = df_test.values[:, 1:]
#SqErTesting(X,y, Xtest, ytest)

#############################################
# plot the graph based on the number of non zero coefficients
def estimateZeroBasedOnLambda(X, y):
	lambdaa = 600
	count = 0
	while count < 10:
		w = lasso(X, y, lambdaa)
		res = np.count_nonzero(w)
		plt.plot(lambdaa, res, 'ro')
		lambdaa /=2
		count += 1
	plt.show()


# estimateZeroBasedOnLambda(Xtest,ytest)
#############################################

def bestLambda(X, y, Xtest, ytest):
	lambdaa = 600
	count = 0
	minVal = float("inf")
	minLam = None
	while count < 10:
		copyY = ytest[:]
		w = lasso(X, y, lambdaa)
		dot = np.dot(np.transpose(w), np.transpose(Xtest))
		transposeY = np.transpose(copyY)
		copyY = transposeY - dot
		transposeY = np.transpose(copyY)
		copyY = np.dot(copyY, transposeY)
		if copyY < minVal:
			minVal = copyY
			minLam = lambdaa
		plt.plot(lambdaa, copyY, 'ro')
		lambdaa /= 2
		count += 1
	print minLam

	newW = lasso(Xtest, ytest, minLam)
	ma = np.argmax(newW)
	mi = np.argmin(newW)
	print "max: ", df_test.columns[ma]
	print "min: ", df_test.columns[mi]
	plt.show()


bestLambda(X,y, Xtest, ytest)

# The lambda that gave the best test set performance is 18. 
# The variable that had the largest Lasso coefficient is NumStreet. 
# And the most negative one is PctFam2Par. This is makes sense where the 
# place has the most homeless people counted in the street will likely has more 
# crimes. The family with kids that are headed by two parents seem educated, 
# this will be likely to have less crimes.

################################################

