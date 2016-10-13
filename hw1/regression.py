import numpy as np
import pandas as pd
import sys
from itertools import combinations

import math
import csv
import random



def kernel(x):
	#print x.max(axis=0) == x.min(axis=0) 

	
	# xk = (x[:, 162-18 * 1 + 9] / 6 - x[:, 162-18*5+9] / 8)[:, None]
	# xk1 = np.sum(x[:, range(162-18*4+9,162, 18)], axis=1)[:, None]/4
	
	# d = []
	# d += [i for i in range(10, 162, 18)]
	# d += [i for i in range(2, 162, 18)]
	# # # d += [i for i in range(13, 162, 18)]
	# # # d += [i for i in range(1, 162, 18)]
	# # # d += [i for i in range(4, 162, 18)]
	# # # d += [i for i in range(12, 162, 18)]
	# # # d += [i for i in range(3, 162, 18)]
	# # # d += [i for i in range(7, 162, 18)]
	


	# x = np.delete(x, d, axis=1)

	xnormal = x / np.sum(x, axis=0)
	xs = xnormal**2 
	# xq = np.array([ [xi[p]*xi[q] for p, q in combinations(range(x.shape[1]), 2)] for xi in xnormal])
	# xq = np.array([[xi[p]*xi[q] for p in range(0, 162, 18) for q in range(0, 162) if p != q ]for xi in xnormal])
	# print xq.shape
	# xlog = np.log(x-x.min(axis=0)+0.001)
	# c = 10
	timefactor = np.array([float(i) for i in range(1,10) for j in range(18)]*2)
	xconcat = np.concatenate((x, xs), axis=1)
	xconcat = timefactor * xconcat
	# # # xconcat = np.concatenate((x, xk, xk1, xs, xq, xlog), axis=1)
	# xconcat = np.concatenate((x, xs), axis=1)
	x = xconcat
	# # x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
	return x




class linearRegression(object):


	def __init__(self, featureSize, W=[], b=None):
		self.featureSize = featureSize
		self.W = W if not W == [] else np.random.uniform(-0.01, 0.01, featureSize)# np.random.normal(0, 0.01, featureSize)
		self.b = b if b else (np.random.random() - 0.5 ) / 5000.0
		self.learningRate = 3e-9
		self.adaW = np.array([1.0] * featureSize)
		self.adaB = 1.0
		self.C = 0.1# 10.0 # regularization hyper
		self.mgw = 0.0
		self.mgb = 0.0
		self.M = 0.3
		self.inW = self.W
		self.inb = self.b

	def cost(self, x, y):
		# square error
		

		c_partial = np.dot(x, self.W) + self.b - y
		

		# rmse = np.sum(c_partial ** 2)
		
		# w_ab = np.absolute(self.W)
		w_sq = self.W ** 2

		# c = rmse + self.C * (np.sum(w_sq))
		
		
		gw = np.dot(c_partial, x) + self.C * (self.W )
		# gw /= len(y)
		gb = np.sum(c_partial) #+ self.B * (y * (c_partial + y) < 0) * (-y)
		# gb /= len(y)



		return gw, gb




	def fit(self, x, y):

		gw, gb = self.cost(x, y)
		

		#update parameters
		self.mgw = self.M * self.mgw + self.learningRate / np.sqrt(self.adaW) * gw
		self.mgb = self.M * self.mgb + self.learningRate / np.sqrt(self.adaB) * gb
		self.W = self.W - self.mgw
		self.b = self.b - self.mgb

		self.adaW += (self.learningRate * gw) ** 2
		self.adaB += (self.learningRate * gb) ** 2
		

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

		c_partial = np.dot(x, self.W) + self.b - y

		# print "aaaa", self.W
		# print "bbbb", x

		rmse = np.sqrt(np.mean(c_partial ** 2))
		
		# w_ab = np.absolute(self.W)
		w_sq = self.W ** 2

		# c = c_partial ** 2
		c = np.sqrt((np.sum(c_partial ** 2) + self.C * (np.sum(w_sq**2)))/len(y))




		return rmse, c



	def predict(self, x):

		return np.dot(x, self.W) + self.b



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

	# validSize = 12 * 20 // validRate

	# validMonths = np.random.choice(range(0, 12 * 20), validSize, replace=False)

	# dValid = []
	# dTrain = []

	# # a months = 471 instances (WSIZE=10)

	# for i in range(0, 12 * 20):
	# 	if i in validMonths:
	# 		dValid += [j for j in range(15*i, 15*(i+1))]
	# 	else:
	# 		dTrain += [j for j in range(15*i, 15*(i+1))]

	

	# yValid, xValid = trainData[dValid, 0], trainData[dValid, 1:]
	# yTrain, xTrain = trainData[dTrain, 0], trainData[dTrain, 1:]

	# yValid, xValid = trainData[-validSize:, 0], trainData[-validSize:, 1:]
	# yTrain, xTrain = trainData[:-validSize:, 0], trainData[:-validSize, 1:]

	


	xTrainKernel = kernel(xTrain)
	xValidKernel = kernel(xValid)
	print xTrainKernel.shape
	print xValidKernel.shape



	model = linearRegression(xTrainKernel.shape[1])

	iterations = 10000
	stop = iterations
	min_valid = float('Inf')


	# w, b = train(xTrainKernel, yTrain, xValidKernel, yValid)


	prev_valid = float('Inf')
	count = 0
	
	# for t in range(iterations):
	# 	# score = model.fit(xTrainKernel, yTrain)
	# 	# if t % 100 == 0:
	# 	# 	T = np.concatenate((yTrain[:,None], xTrainKernel), axis=1)
	# 	# 	np.random.shuffle(T)
	# 	# 	yTrain, xTrainKernel = T[:, 0], T[:, 1:]
		
	# 	for xi, yi in [(xTrainKernel[i:i+batchSize], yTrain[i:i+batchSize]) for i in range(0,len(yTrain),batchSize)]:
		
	# 		model.fit(xi, yi)

	# 		# rmse += r
	# 		# score += s	





		
	# 	validrmse, validScore = model.eval(xValidKernel, yValid)

	# 	if t % 100 == 0:
	# 		rmse, score = model.eval(xTrainKernel, yTrain)
	# 		print >> sys.stderr, "iter: ", t, rmse, score, validrmse, validScore


		# if validrmse < prev_valid:
		# 	prev_valid = validrmse
		# 	count = 0
		# else:
		# 	if t > 1000:
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

	model = linearRegression(xTrainKernel.shape[1], model.inW, model.inb)



	for t in range(stop):
		# score = model.fit(xTrainKernel, yTrain)
		# if t % 100 == 0:
		# 	T = np.concatenate((yTrain[:,None], xTrainKernel), axis=1)
		# 	np.random.shuffle(T)
		# 	yTrain, xTrainKernel = T[:, 0], T[:, 1:]


		for xi, yi in [(xTrainKernel[i:i+batchSize], yTrain[i:i+batchSize]) for i in range(0,len(yTrain),batchSize)]:
		
			model.fit(xi, yi)	
		

		
		if t % 100 == 0 or t > 9000:
			rmse, score = model.eval(xTrainKernel, yTrain)
			print >> sys.stderr, "iter: ", t, rmse, score

	rmse, score = model.eval(xTrainKernel, yTrain)
	yPredict = model.predict(xTestKernel)

	yPredict = np.round(yPredict)
	#print yPredict

	df = pd.DataFrame({'id':['id_'+str(i) for i in range(len(yPredict))], 'value':yPredict})
	# outputFile = ""
	# l = [("lr",model.learningRate), ("c", model.C), ("dim", model.featureSize), ("iter", stop), ("ein", rmse)]
	# outputFile = "".join(['['+t[0]+'='+str(t[1])+']' for t in l])+".csv"

	df.to_csv(outputFile, index=False)









