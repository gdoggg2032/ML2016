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
		# xconcat = np.concatenate((xnormal, xnormal**2), axis=1)
		xconcat = xnormal
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


class Data2(object):

	def __init__(self, x, y):

	
		self.labels = y
		self.features = x
		# np.random.shuffle(self.data)

		self.n_samples = x.shape[0]
		self.n_dims = x.shape

		self.index = 0

	
		

	def shuffle(self):

		p = np.random.permutation(self.n_samples)
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
	parser.add_argument('--lr', default=0.07, type=float)
	parser.add_argument('--C', default=0.0, type=float)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--batch', default=100, type=int)
	parser.add_argument('--n_estimators', default=50, type=int)
	parser.add_argument('--bag_dim', default=0.9, type=float)
	parser.add_argument('--mode', default=2, type=int)

	# mode 0: only train, mode 1: only test, model 2: both train and test
	parser.add_argument('--model', default="./model_tmp", type=str)
	parser.add_argument('--predict', default='./spam_predict', type=str)
	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args

class neuron_network(object):

	def __init__(self, args, feature_size):

		self.feature_size = feature_size
		self.learning_rate = args.lr
		self.layers = [feature_size, 50, 50, 10,1]

		self.W = {}
		self.b = {}
		self.adaW = {}
		self.adab = {}
		for i in range(len(self.layers)-1):
			self.W[i] = np.random.uniform(-0.01, 0.01, (self.layers[i], self.layers[i+1]))
			self.b[i] = np.random.uniform(-0.001, 0.001, self.layers[i+1])
			self.adaW[i] = np.ones((self.layers[i], self.layers[i+1]))
			self.adab[i] = np.ones(self.layers[i+1])


		

		self.C = args.C

	def activ(self, x):
		sigmoid = 1.0 / (1.0 + np.exp(-x))
		# return np.clip(sigmoid, 1e-5, 1 - 1e-5)
		return sigmoid

	def de_activ(self, x):
		return self.activ(x) * (1 - self.activ(x))


	def cost(self, x, y):

		Z = {}
		A = {}

		inp = x
		for i in range(len(self.layers)-1):
			Z[i] = np.dot(inp, self.W[i]) + self.b[i]
			A[i] = self.activ(Z[i])
			inp = A[i]

		de_cost = -y[:, None] / inp + (1 - y[:, None]) / (1 - inp)
		
		theta = {}

		inp = de_cost

		for i in range(len(self.layers)-2, -1, -1):
			theta[i] = self.de_activ(Z[i]) * inp
			inp = np.dot(theta[i], self.W[i].T)

		gw = {}
		gb = {}

		inp = x.T
		for i in range(len(self.layers)-1):
			gw[i] = np.dot(inp, theta[i]) + self.C * self.W[i]
			inp = A[i].T
			gb[i] = np.sum(theta[i], axis=0)

		return gw, gb

	def fit(self, x, y):

		gw, gb = self.cost(x, y)

		for i in range(len(self.layers)-1):
			self.W[i] = self.W[i] - self.learning_rate * gw[i] / np.sqrt(self.adaW[i])
			self.b[i] = self.b[i] - self.learning_rate * gb[i] / np.sqrt(self.adab[i])

			self.adaW[i] += (self.learning_rate * gw[i]) ** 2
			self.adab[i] += (self.learning_rate * gb[i]) ** 2

	def eval(self, x, y):

		Z = {}
		A = {}

		inp = x
		for i in range(len(self.layers)-1):
			Z[i] = np.dot(inp, self.W[i]) + self.b[i]
			A[i] = self.activ(Z[i])
			inp = A[i]

		c_partial = inp

		cross_entropy = np.nansum(-(y[:, None] * np.log(c_partial) + (1 - y[:, None]) * np.log(1 - c_partial)))

		# c = cross_entropy + self.C * (np.sum([np.sum(self.W[i] ** 2 ) for i in range(len(self.layers)-1)])
		c = 0
		return cross_entropy, c

	def predict(self, x):

		# >= 0.5 -> 1, < 0.5 -> 0
		# c_partial = self.activ(np.dot(x, self.W) + self.b)

		Z = {}
		A = {}

		inp = x
		for i in range(len(self.layers)-1):
			Z[i] = np.dot(inp, self.W[i]) + self.b[i]
			A[i] = self.activ(Z[i])
			inp = A[i]


		c_partial = inp

		return (c_partial >= 0.5).astype("int")[:, 0]

	def classify_error(self, x, y):

		predict_y = self.predict(x)
		return np.sum((y != predict_y).astype("int")) / float(len(y))











class ensemble(object):
	def __init__(self, train, args):
		self.n_estimators = args.n_estimators
		self.bag_dim_rate = args.bag_dim

		self.features = train.features
		self.labels = train.labels
		self.args = args
		self.train = train
		

		self.feature_record = {}
		self.data_record = {}
		self.models = {}
		self.trains = {}



	def bootstrap(self, x, y):

		self.n_samples = x.shape[0]
		self.n_dims = x.shape[1]
		self.n_bagging_dims = int(self.bag_dim_rate * self.n_dims)
		self.n_bagging_samples = self.n_samples

		for i in range(self.n_estimators):
			p = np.random.permutation(self.n_dims)[:self.n_bagging_dims]
			self.feature_record[i] = p
			p = np.random.permutation(self.n_samples)[:self.n_bagging_samples]
			self.data_record[i] = p

			self.trains[i] = Data2(x[self.data_record[i],:][:, self.feature_record[i]], y[self.data_record[i]])



	def fit(self, x, y, display_step=10):

		self.bootstrap(x, y)

		for i in range(self.n_estimators):
			print >> sys.stderr, "tree: ", i+1, "/", self.n_estimators
			self.models[i] = neuron_network(self.args, self.n_bagging_dims)

			#fix bugs
			self.models[i].n_dims = self.n_bagging_dims
			self.models[i].n_samples = self.n_bagging_samples


			training_epochs = self.args.epochs

			start_time = time.time()

			batch_size = self.args.batch

			for epoch in range(training_epochs):

				avg_cost = 0.0
				avg_cost = 0.0

				total_batch = int((self.n_samples+batch_size) / float(batch_size))

				for j in range(total_batch):
					# if (i+1) % display_step == 0:
					# 	print >> sys.stderr, i, "/", total_batch


					batch_x, batch_y = self.trains[i].next_batch(batch_size)

					self.models[i].fit(batch_x, batch_y)

				# if (epoch+1) % display_step == 0:
				# 	print >> sys.stderr, "Epoch:", epoch+1, "time: ", time.time() - start_time

				if (epoch+1) % display_step == 0:
					avg_cost, r_cost = self.models[i].eval(self.features[:, self.feature_record[i]], self.labels)
					ein = self.models[i].classify_error(self.features[:, self.feature_record[i]], self.labels)
				
					print >> sys.stderr, "Epoch:", epoch+1, "cost=", \
					avg_cost, "ein=", ein, "time: ", time.time() - start_time

			self.trains[i] = -1


	def predict(self, x):

		r = np.zeros((x.shape[0]))

		for i in range(self.n_estimators):
			predict_y = self.models[i].predict(x[:, self.feature_record[i]])
			r += predict_y
		r /= self.n_estimators
		return (r >= 0.5).astype("int")

	def classify_error(self, x, y):

		predict_y = self.predict(x)
		return np.sum((y != predict_y).astype("int")) / float(len(y))

def ensemble_logistic_regression(train, args):

	en = ensemble(train, args)
	x = train.features
	y = train.labels

	en.fit(x, y)

	ein = en.classify_error(x, y)

	print >> sys.stderr, "ein=", ein

	return en, ein







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
		model, ein = ensemble_logistic_regression(train, args)
		print >> sys.stderr, "ein:", ein

		print >> sys.stderr, "dumping model"
		dump_model("./model"+"[enNN][ein="+str(ein)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"][bag_dim="+str(args.bag_dim)+"][n_esti="+str(args.n_estimators)+"]", model)
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
			dump_predict("./model"+"[enNN][ein="+str(ein)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"][bag_dim="+str(args.bag_dim)+"][n_esti="+str(args.n_estimators)+"].predict", predict_y, test.names_test)



if __name__ == "__main__":
	main()













