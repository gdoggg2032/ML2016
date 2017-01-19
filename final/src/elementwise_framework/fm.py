import tensorflow as tf
import numpy as np
import sys
import csv
import argparse
import time
import progressbar as pb 



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='./clicks_split_train.fmf', type=str)
	parser.add_argument('--valid', default='./clicks_valid.fmf', type=str)
	parser.add_argument('--test', default='./clicks_test.fmf', type=str)
	parser.add_argument('--lr', default=0.2, type=float)
	parser.add_argument('--regular', default=0.00002, type=float)
	parser.add_argument('--epochs', default=1, type=int)
	parser.add_argument('--batch', default=500, type=int)
	parser.add_argument('--dim', default=4, type=int)
	parser.add_argument('--num_features', default=20, type=int)
	parser.add_argument('--max_features', default=1675952, type=int)
	# hash size = 33129486, filtered to 2071310
	# too long, have to cut some features with low frequencies
	# hash size = 1437077, not need to filter (feature_extract4.py)
	# hash size = 1437096, not need to filter (feature_extract5.py, half of day and hour)
	# hash size = 1437114, not need to filter (feature_extract5.py)
	# hash size = 1437208, not need to filter (feature_extract6.py)
	# hash size = 1437298, not need to filter (feature_extract6.py, separate ad,display doc cate)
	# hash size = 1437898, not need to filter (feature_extract6.py, separate ad,display doc cate topic)
	# hash size = 1450464, not need to filter (feature_extract6.py, separate ad,display doc cate meta)
	# hash size = 1451064, not need to filter (feature_extract6.py, separate ad,display doc cate topic meta)
	# hash size = 1675352, not need to filter (feature_extract6.py, separate ad,display doc cate topic meta entity)
	parser.add_argument('--model', default="./fm_model", type=str)
	parser.add_argument('--predict', default="./predict", type=str)
	parser.add_argument('--mode', default=0, type=int)

	args = parser.parse_args()

	return args

class mf(object):

	def __init__(self, args):
		self.args = args

		
		self.index = 0
		self.index_test = 0

	def load_data(self, train=True):	
		if train:
			print >> sys.stderr, "load train_data"
			# self.train_data = np.loadtxt(self.args.train, dtype=int)
			self.train_data = np.fromstring(open(self.args.train, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, self.args.num_features+1)
			print self.train_data.shape
			# with open(self.args.train, "r") as f:
			# 	self.train_data = [[int(x) for x in row] for row in csv.reader(f, delimiter=' ')]
			print >> sys.stderr, "load val_data"
			# with open(self.args.valid, "r") as f:
			# 	self.val_data = [[int(x) for x in row] for row in csv.reader(f, delimiter=' ')]
			# self.val_data = np.loadtxt(self.args.valid, dtype=int)
			self.val_data = np.fromstring(open(self.args.valid, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, self.args.num_features+1)
			print self.val_data.shape

			# self.train_data = self.shuffle(self.train_data)
			np.random.shuffle(self.train_data)

		else:
			print >> sys.stderr, "load test_data"
			self.test_data = np.fromstring(open(self.args.test, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, self.args.num_features+1)
			print self.test_data.shape

	def shuffle(self, data):
		p = np.random.permutation(len(data))
		data = data[p]
		return data


	def next_batch(self, batch_size, train=True):
		if train:
			data = self.train_data[self.index : batch_size + self.index]
			# data = np.array(data)
			if self.index + batch_size >= len(self.train_data):
				self.index = 0
				# self.train_data = self.shuffle(self.train_data)
				np.random.shuffle(self.train_data)
			else:
				self.index = self.index + batch_size
			return data
		else:
			data = self.test_data[self.index_test : batch_size + self.index_test]
			self.index_test = self.index_test + batch_size
			return data

	def add_model(self):
		with tf.device('/cpu:0'):
			x = tf.placeholder(tf.int32, [None, self.args.num_features])
			y = tf.placeholder(tf.float32, [None])

			b = tf.Variable(tf.random_uniform(
									 	[1], -.1, .1))

			embedding_w = tf.concat(0, [tf.constant([[0.] * 1], dtype=tf.float32), 
									 tf.Variable(tf.random_uniform(
									 	[self.args.max_features, 1], -.1, .1))
									])
		
			embedding_v = tf.concat(0, [tf.constant([[0.] * self.args.dim], dtype=tf.float32), 
									 tf.Variable(tf.random_uniform(
									 	[self.args.max_features, self.args.dim], -.1, .1))
									])


		with tf.device('/gpu:0'):
			embed_w = tf.nn.embedding_lookup(embedding_w, x)
			embed_v = tf.nn.embedding_lookup(embedding_v, x)

			w_x = tf.reduce_sum(embed_w, [1, 2])
			# print w_x.get_shape()

			m = tf.batch_matmul(embed_v, tf.transpose(embed_v, perm=[0, 2, 1]))
			# mask = np.array([[1 if j > i else 0 for j in range(self.args.num_features) ] for i in range(self.args.num_features)]).astype(bool)
			


			m_l = []
			for i in range(self.args.num_features):
				for j in range(i+1, self.args.num_features):
					m_l.append(tf.expand_dims(m[:,i,j],1))

			mm = tf.concat(1, m_l)
			
			w_mm_dim = (self.args.num_features ** 2 - self.args.num_features) / 2 # C(n, 2)
			
			# w1_mm = tf.Variable(tf.random_uniform([w_mm_dim, 1], -.1, .1))
			# b1_mm = tf.Variable(tf.random_uniform([1], -.1, .1))
			# w1_mm = tf.Variable(tf.random_uniform([w_mm_dim, 1], 1.0, 1.0))
			# b1_mm = tf.Variable(tf.random_uniform([1], 0.0, 0.0))
			w1_mm = tf.random_uniform([w_mm_dim, 1], 1.0, 1.0)
			b1_mm = tf.random_uniform([1], 0.0, 0.0)
			
			a1_mm = tf.matmul(mm, w1_mm) + b1_mm

			z1_mm = a1_mm # linear


			mmm = z1_mm

			vv_x = tf.squeeze(mmm, [1])

			# w_mm = tf.Variable(tf.random_uniform([w_mm_dim], -.1, .1))
			# w_mm = tf.Variable(tf.constant([2.0]*mm.get_shape()[1]))
			# mmm = mm * w_mm
			# vv_x =  tf.reduce_sum(mmm, [1])
			# mm =  tf.matrix_band_part(m, 0, -1) - tf.matrix_band_part(m, 0, 0)
			# w_mm = tf.Variable(tf.random_uniform([self.args.num_features, self.args.num_features], -.1, .1))

			# mmm = mm * w_mm
			# vv_x =  tf.reduce_sum(mmm, [1, 2])



			# vv_x = (tf.reduce_sum(tf.batch_matmul(embed_v, tf.transpose(embed_v, perm=[0, 2, 1])), [1, 2]) \
			# 	- tf.reduce_sum(embed_v ** 2, [1, 2]) ) #/ 2 #+ tf.reduce_sum(embed_v ** 2, [1, 2])
			# print vv_x.get_shape()
			all_x = b + w_x + vv_x
			# print all_x.get_shape()
			# this can only be used in tensorflow 0.12, due to tf.trace() 
			# m = tf.batch_matmul(embed, tf.transpose(embed, perm=[0, 2, 1]))
			# wTx = ( tf.reduce_sum(m, [1, 2]) - tf.trace(m) ) / 2
			
			clip_all_x = tf.clip_by_value(all_x, -35., 35.)
			p = 1.0 / (1.0 + tf.exp(-clip_all_x))
			clip_p = tf.clip_by_value(p, 10e-8, 1.0 - 10e-8)

			# cost: logloss
			cost = -tf.reduce_sum(y * tf.log(clip_p) + (1.0-y) * tf.log(1.0-clip_p)) \
				+ self.args.regular * (tf.nn.l2_loss(embed_w) \
					+ tf.nn.l2_loss(embed_v) + tf.nn.l2_loss(w1_mm) )

			opt = tf.train.AdagradOptimizer(self.args.lr).minimize(cost)


			return {'x': x,
					'y': y,
					'p': clip_p,
					'cost': cost,
					'opt': opt}

	def train(self):
		self.load_data(train=True)
		with tf.Graph().as_default():
			print >> sys.stderr, "add model"
			var = self.add_model()

			saver = tf.train.Saver()

			# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True
			# config.gpu_options.per_process_gpu_memory_fraction = 0.6

			sess = tf.Session(config = config)

			sess.run(tf.initialize_all_variables())

			total_batch = int(np.ceil(len(self.train_data) / float(self.args.batch)))

			for epoch in xrange(self.args.epochs):

				total_loss = 0.0
				total_count = 0
				pbar = pb.ProgressBar(widgets=["[TRAIN] ", pb.DynamicMessage('loss'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()

				for i in xrange(total_batch):
					batch = self.next_batch(self.args.batch)
					_, loss = sess.run([var['opt'], var['cost']], feed_dict={var['y']:batch[:, 0], var['x']:batch[:,1:]})
					total_loss += loss
					total_count += len(batch)
					pbar.update(i, loss=total_loss/total_count)
				pbar.finish()

				v_loss = sess.run(var['cost'], feed_dict={var['y']:self.val_data[:, 0], var['x']:self.val_data[:,1:]})

				print >> sys.stderr, \
					"Epoch {}: tr_logloss: {}, v_logloss: {}".format(epoch, total_loss / len(self.train_data), v_loss / len(self.val_data))

			print >> sys.stderr, "save model"
			save_path = saver.save(sess, self.args.model)
			print >> sys.stderr, "save model in path", save_path

	def predict(self):
		self.load_data(train=False)
		with tf.Graph().as_default():

			var = self.add_model()

			saver = tf.train.Saver()

			config = tf.ConfigProto(allow_soft_placement=True)
			config.gpu_options.allow_growth = True

			sess = tf.Session(config = config)
			saver.restore(sess, self.args.model)
			print >> sys.stderr, "restore model from ", self.args.model

			total_batch = int(np.ceil(len(self.test_data) / float(self.args.batch)))
			p = open(self.args.predict, "w")
			pbar = pb.ProgressBar(widgets=["[TEST] ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
			prob_l = []
			for i in xrange(total_batch):
				batch = self.next_batch(self.args.batch, train=False)
				prob = sess.run(var['p'], feed_dict={var['x']:batch[:,1:]})
				# print prob.shape
				out_l = [str(x) for x in prob.tolist()]
				out = "\n".join(out_l)+'\n'
				p.write(out)
				pbar.update(i)
			pbar.finish()






def main():

	args = arg_parse()

	model = mf(args)

	if args.mode % 2 == 0:

		start_time = time.time()

		model.train()

		print >> sys.stderr, "training time: {}".format(time.time()-start_time)
	
	if args.mode > 0:

		start_time = time.time()

		model.predict()

		print >> sys.stderr, "testing time: {}".format(time.time()-start_time)



if __name__ == "__main__":
	main()
	