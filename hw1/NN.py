import numpy as np
import pandas as pd
import sys
from itertools import combinations

import math
import csv
import random



def kernel(x):

	xnormal = x - np.mean(x, axis=0) / np.std(x, axis=0)
	
	xs = x**2 
	# x = xnormal
	xs = xs - np.mean(xs, axis=0) / np.std(xs, axis=0)
	
	timefactor = np.array([float(i) for i in range(1,10) for j in range(18)]*2)
	xconcat = np.concatenate((x, xs), axis=1)
	xconcat = timefactor * xconcat

	x = xconcat
	# x = (x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
	return x




class linearRegression(object):


	def __init__(self, featureSize, W=[], b=None):
		self.featureSize = featureSize
		self.W1_N = 100
		self.W2_N = 100
		self.W3_N = 1
		self.W1 = np.random.uniform(-0.001, 0.001, (self.W1_N, featureSize))
		self.b1 = np.random.uniform(-0.001, 0.001, (self.W1_N, 1))

		self.W2 = np.random.uniform(-0.001, 0.001, (self.W2_N, self.W1_N))
		self.b2 = np.random.uniform(-0.001, 0.001, (self.W2_N, 1))

		self.W3 = np.random.uniform(-0.001, 0.001, (self.W3_N, self.W2_N))
		self.b3 = np.random.uniform(-0.001, 0.001, (self.W3_N, 1))


		self.learningRate = 1e-9#5e-8
		self.adaW1 = np.ones((self.W1_N, featureSize))
		self.adaW2 = np.ones((self.W2_N, self.W1_N))
		self.adaW3 = np.ones((self.W3_N, self.W2_N))
		self.adab1 = np.ones((self.W1_N, 1))
		self.adab2 = np.ones((self.W2_N, 1))
		self.adab3 = np.ones((self.W3_N, 1))
		self.C = 0.0# 10.0 # regularization hyper
		self.mgW1 = np.zeros((self.W1_N, featureSize))
		self.mgW2 = np.zeros((self.W2_N, self.W1_N))
		self.mgW3 = np.zeros((self.W3_N, self.W2_N))
		self.mgb1 = np.zeros((self.W1_N, 1))
		self.mgb2 = np.zeros((self.W2_N, 1))
		self.mgb3 = np.zeros((self.W3_N, 1))
		self.M = 0.0
		# self.inW = self.W
		# self.inb = self.b

	# def actv(self, x):
	# 	#sigmoid
	# 	return np.log(1 + np.exp(x))

	# def de_actv(self, x):
	# 	return 1.0 / (1 + np.exp(-x))

	def actv(self, x):
		return x

	def de_actv(self, x):
		return 1.0

	def cost(self, x, y):
		# square error
		
		# c_partial = self.predict(x) - y
		# gw1 = 2 * np.dot(np.dot(self.W2.T, c_partial), x) + 2 * self.C * self.W1


		# gb1 = 2 * np.dot(c_partial, np.dot(np.ones((x.shape[0], self.W2_N)), self.W2)).T

	
		# # print c_partial.shape, self.W2.shape, gb1.shape
		# gw2 = 2 * (np.dot(np.dot(self.W1, x.T), c_partial.T) + self.b1).T + 2 * self.C * self.W2

		# gb2 = 2 * np.sum(c_partial)

		z1 = np.dot(self.W1, x.T) + self.b1
		a1 = self.actv(z1)

		z2 = np.dot(self.W2, a1) + self.b2
		a2 = self.actv(z2)

		z3 = np.dot(self.W3, a2) + self.b3
		a3 = self.actv(z3)

		# cost = (a2-y) ** 2 + C * || w ||^2

		# back prop

		de_cost = 2 * (a3 - y)

		theta3 = self.de_actv(z3) * de_cost
		theta2 = self.de_actv(z2) * np.dot(self.W3.T, theta3)
		theta1 = self.de_actv(z1) * np.dot(self.W2.T, theta2)

		# print self.W1.shape
		
		gw1 = np.dot(theta1, x) + self.C * self.W1
		gw2 = np.dot(theta2, a1.T) + self.C * self.W2
		gw3 = np.dot(theta3, a2.T) + self.C * self.W3
		# print gw2.shape, theta2.shape, a1.shape


		gb1 = np.dot(theta1, np.ones((theta1.shape[1], 1)))
		gb2 = np.dot(theta2, np.ones((theta2.shape[1], 1)))
		gb3 = np.dot(theta3, np.ones((theta3.shape[1], 1)))




	




		return gw1, gb1, gw2, gb2, gw3, gb3




	def fit(self, x, y):

		gw1, gb1, gw2, gb2, gw3, gb3 = self.cost(x, y)
		
		#update parameters


		self.mgW1 = self.M * self.mgW1 + self.learningRate * gw1 / np.sqrt(self.adaW1)
		self.mgW2 = self.M * self.mgW2 + self.learningRate * gw2 / np.sqrt(self.adaW2)
		self.mgW3 = self.M * self.mgW3 + self.learningRate * gw3 / np.sqrt(self.adaW3)
		self.mgb1 = self.M * self.mgb1 + self.learningRate * gb1 / np.sqrt(self.adab1)
		self.mgb2 = self.M * self.mgb2 + self.learningRate * gb2 / np.sqrt(self.adab2)
		self.mgb3 = self.M * self.mgb3 + self.learningRate * gb3 / np.sqrt(self.adab3)




		self.W1 = self.W1 - self.mgW1
		self.b1 = self.b1 - self.mgb1
		self.W2 = self.W2 - self.mgW2
		self.b2 = self.b2 - self.mgb2
		self.W3 = self.W3 - self.mgW3
		self.b3 = self.b3 - self.mgb3




		# self.mgw = self.M * self.mgw + self.learningRate / np.sqrt(self.adaW) * gw
		# self.mgb = self.M * self.mgb + self.learningRate / np.sqrt(self.adaB) * gb
		# self.W = self.W - self.mgw
		# self.b = self.b - self.mgb

		self.adaW1 += (self.learningRate * gw1) ** 2
		self.adab1 += (self.learningRate * gb1) ** 2
		self.adaW2 += (self.learningRate * gw2) ** 2
		self.adab2 += (self.learningRate * gb2) ** 2
		self.adaW3 += (self.learningRate * gw3) ** 2
		self.adab3 += (self.learningRate * gb3) ** 2
		

		# return rmse, c

		# cost = 0.
		# rmse = 0.
		# for i, dat in enumerate(x):
		# 	diff = np.dot(self.W.T, dat) + self.b - y[i]
		# 	cost +=  diff * diff + self.C * np.sum(self.W**2)
		# 	# rmse += diff * diff

		# 	self.W -= self.learningRate * (diff * dat + self.C * self.W) / np.sqrt(self.adaW)
		# 	self.b -= self.learningRate * diff / np.sqrt(self.adaB)

		# 	self.adaW += self.learningRate * (diff * dat + self.C * self.W) * self.learningRate  * (diff * dat + self.C * self.W)
		# 	self.adaB += self.learningRate  * diff * self.learningRate  * diff


		# return 0, cost




	
	def eval(self, x, y):

		c_partial = self.predict(x) - y

		# print "aaaa", self.W
		# print "bbbb", x

		rmse = np.sqrt(np.mean(c_partial ** 2))
		
		# w_ab = np.absolute(self.W)
		# w_sq = self.W ** 2

		# c = c_partial ** 2
		c = np.sqrt((np.sum(c_partial ** 2) + self.C * (np.sum(self.W1**2)+np.sum(self.W2**2)))/len(y)) #+ self.C * (np.sum(w_sq**2)))/len(y))
		return rmse, c



	def predict(self, x):

		z1 = np.dot(self.W1, x.T) + self.b1
		a1 = self.actv(z1)

		z2 = np.dot(self.W2, a1) + self.b2
		a2 = self.actv(z2)

		z3 = np.dot(self.W3, a2) + self.b3
		a3 = self.actv(z3)


		return a3



if __name__ == "__main__":

	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	outputFile = sys.argv[3]
	validRate = int(sys.argv[4])

	batchSize = 100

	trainData = np.load(trainFile).astype('float64')
	testData = np.load(testFile)

	validSize = trainData.shape[0] // validRate

	np.random.shuffle(trainData) 
	yValid, xValid = trainData[0:validSize, 0], trainData[0:validSize, 1:] 
	yTrain, xTrain = trainData[validSize:, 0], trainData[validSize:, 1:]


	xTrainKernel = kernel(xTrain)
	xValidKernel = kernel(xValid)
	print xTrainKernel.shape
	print xValidKernel.shape

	model = linearRegression(xTrainKernel.shape[1])

	iterations = 10000
	stop = iterations
	min_valid = float('Inf')

	prev_valid = float('Inf')
	count = 0
	
	for t in range(iterations):

		for xi, yi in [(xTrainKernel[i:i+batchSize], yTrain[i:i+batchSize]) for i in range(0,len(yTrain),batchSize)]:
		
			model.fit(xi, yi)
		
		validrmse, validScore = model.eval(xValidKernel, yValid)

		if t % 1 == 0:
			rmse, score = model.eval(xTrainKernel, yTrain)
			print >> sys.stderr, "iter: ", t, rmse, score, validrmse, validScore


		# if validrmse < prev_valid:
		# 	prev_valid = validrmse
		# 	count = 0
		# else:
		# 	if t > 0:
		# 		count += 1
		# 		prev_valid = validrmse
		# 		if count > 10 :
		# 			stop = t
		# 			break

	# generate test prediciton

	yTrain, xTrain = trainData[:, 0], trainData[:, 1:]
	xTest = testData

	xTrainKernel = kernel(xTrain)
	xTestKernel = kernel(xTest)

	model = linearRegression(xTrainKernel.shape[1])

	for t in range(stop):


		for xi, yi in [(xTrainKernel[i:i+batchSize], yTrain[i:i+batchSize]) for i in range(0,len(yTrain),batchSize)]:
		
			model.fit(xi, yi)	
		
		if t % 1 == 0 or t > 1000:
			rmse, score = model.eval(xTrainKernel, yTrain)
			print >> sys.stderr, "iter: ", t, rmse, score

	rmse, score = model.eval(xTrainKernel, yTrain)
	yPredict = model.predict(xTestKernel).T[:, 0]

	yPredict = np.round(yPredict)
	print yPredict.shape

	df = pd.DataFrame({'id':['id_'+str(i) for i in range(len(yPredict))], 'value':yPredict})
	outputFile = ""
	l = [("lr",model.learningRate), ("c", model.C), ("dim", model.featureSize), ("iter", stop), ("ein", rmse)]
	outputFile = "[NN]" + "".join(['['+t[0]+'='+str(t[1])+']' for t in l])+".csv"

	df.to_csv(outputFile, index=False)

