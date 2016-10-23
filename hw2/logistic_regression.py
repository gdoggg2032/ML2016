import numpy as np
import random
import time
import cPickle as pickle
import pandas as pd
import argparse
import sys


class Data(object):

	def __init__(self, args):

		if args.mode % 2 == 0: # 0 or 2

			self.args = args
			self.data = np.genfromtxt(args.train, delimiter=",")

			self.names = self.data[:, 0].astype("int")
			self.labels = self.data[:, -1]
			self.features = self.kernel(self.data[:, 1:-1])
			# np.random.shuffle(self.data)

			self.n_samples = self.data.shape[0]
			self.n_dims = self.features.shape[1]

			self.index = 0

		if args.mode > 0: # 1 or 2
			self.test = np.genfromtxt(args.test, delimiter=",")

			self.names_test = self.test[:, 0].astype("int")
			self.features_test = self.kernel(self.test[:, 1:])

	def kernel(self, x):
		# xnormal = x / np.sum(x, axis=0)
		# xnormal = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
		xnormal = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
		xconcat = np.concatenate((xnormal, xnormal**2), axis=1)
		# xconcat = xnormal
		return xconcat

		

	def shuffle(self):

		p = np.random.permutation(self.n_samples)
		self.names = self.names[p]
		self.labels = self.labels[p]
		self.features = self.features[p]

		
	def next_batch(self, batch_size):

		if self.index == 0:
			self.shuffle()

		if self.index + batch_size > self.n_samples:

			batch_x = self.features[self.index :]
			batch_y = self.labels[self.index :]

			self.index = 0
			
		else:

			batch_x = self.features[self.index : self.index + batch_size]
			batch_y = self.labels[self.index : self.index + batch_size]
			self.index += batch_size
		
		return batch_x, batch_y


def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='./spam_data/spam_train.csv', type=str)
	parser.add_argument('--test', default='./spam_data/spam_test.csv', type=str)
	parser.add_argument('--lr', default=0.05, type=float)
	parser.add_argument('--C', default=0.0, type=float)
	parser.add_argument('--epochs', default=3000, type=int)
	parser.add_argument('--batch', default=10, type=int)
	parser.add_argument('--mode', default=2, type=int)

	# mode 0: only train, mode 1: only test, model 2: both train and test
	parser.add_argument('--model', default="./model_tmp", type=str)
	parser.add_argument('--predict', default='./spam_predict', type=str)
	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args

class logistic_regressor(object):

	def __init__(self, args, feature_size):

		self.feature_size = feature_size
		self.learning_rate = args.lr

		self.W = np.random.uniform(-0.01, 0.01, self.feature_size)
		self.b = (np.random.random() - 0.5 ) / 5000.0

		self.adaW = np.array([1.0] * self.feature_size)
		self.adab = 1.0

		self.C = args.C

	def activ(self, x):

		sigmoid = 1.0 / (1.0 + np.exp(-x))

		# return np.clip(sigmoid, 1e-10, 1.0 - 1e-10)
		return sigmoid

	def de_activ(self, x):

		return self.activ(x) * (1 - self.activ(x))


	def cost(self, x, y):

		c_partial = self.activ(np.dot(x, self.W) + self.b) - y

		gw = np.dot(c_partial, x) + self.C * self.W

		gb = np.nansum(c_partial)

		return gw, gb

	def fit(self, x, y):

		gw, gb = self.cost(x, y)

		self.W = self.W - self.learning_rate * gw / np.sqrt(self.adaW)
		self.b = self.b - self.learning_rate * gb / np.sqrt(self.adab)


		self.adaW += gw ** 2
		self.adab += gb ** 2




	def eval(self, x, y):

		c_partial = self.activ(np.dot(x, self.W) + self.b)



		cross_entropy = np.nansum(-(y * np.log(c_partial) + (1 - y) * np.log(1 - c_partial)))
		c = cross_entropy + self.C * (np.sum(self.W ** 2))
		return cross_entropy, c

	def predict(self, x):

		# >= 0.5 -> 1, < 0.5 -> 0
		c_partial = self.activ(np.dot(x, self.W) + self.b)
		return (c_partial >= 0.5).astype("int")

	def classify_error(self, x, y):

		predict_y = self.predict(x)
		return np.sum((y != predict_y).astype("int")) / float(len(y))









def logistic_regression(train, args, display_step=1000):

	batch_size = args.batch
	
	training_epochs = args.epochs
	

	#sigmoid(WTx+b)
	#mini batch

	model = logistic_regressor(args, train.n_dims)

	start_time = time.time()

	for epoch in range(training_epochs):

		

		avg_cost = 0.0

		total_batch = int((train.n_samples+batch_size) / float(batch_size))

		for i in range(total_batch):
			# if (i+1) % display_step == 0:
			# 	print >> sys.stderr, i, "/", total_batch


			batch_x, batch_y = train.next_batch(batch_size)

			model.fit(batch_x, batch_y)

		avg_cost, r_cost = model.eval(train.features, train.labels)
		ein = model.classify_error(train.features, train.labels)
		if (epoch+1) % display_step == 0:
			print >> sys.stderr, "Epoch:", epoch+1, "cost=", \
			avg_cost, "r_cost=", r_cost, "ein=", ein, "time: ", time.time() - start_time

	return model, model.classify_error(train.features, train.labels)






def dump_model(model_file, model):
	pickle.dump(model, open(model_file, "w"))

def load_model(model_file):
	return pickle.load(open(model_file, "r"))

def dump_predict(predict_file, predict, names):
	
	df = pd.DataFrame({'id':names, 'label':predict})
	df.to_csv(predict_file, index=False)
	


	





def main():

	args = arg_parse()

	if args.mode % 2 == 0:

		print >> sys.stderr, "load train data"
		train = Data(args)

		print >> sys.stderr, "training"
		model, ein = logistic_regression(train, args)
		print >> sys.stderr, "ein:", ein

		print >> sys.stderr, "dumping model"
		dump_model("./model"+"[logreg][ein="+str(ein)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"][C="+str(args.C)+"]", model)
		dump_model(args.model, model)

	if args.mode > 0:

		print >> sys.stderr, "load test data"
		if args.mode == 1:
			test = Data(args)
		elif args.mode == 2:
			test = train

		print >> sys.stderr, "load model"
		model = load_model(args.model)

		print >> sys.stderr, "predicting"
		predict_y = model.predict(test.features_test)

		print >> sys.stderr, "dumping prediction"
		if args.mode == 1:
			dump_predict(args.predict, predict_y, test.names_test)
		elif args.mode == 2:
			dump_predict("./model"+"[logreg][ein="+str(ein)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"][C="+str(args.C)+"].predict", predict_y, test.names_test)



if __name__ == "__main__":
	main()













