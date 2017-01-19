import tensorflow as tf
import numpy as np
import sys
import csv
import argparse
import time
import progressbar as pb 



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='./clicks_split_train.fm', type=str)
	parser.add_argument('--valid', default='./clicks_valid.fm', type=str)
	parser.add_argument('--test', default='./clicks_test.fm', type=str)
	parser.add_argument('--lr', default=0.2, type=float)
	parser.add_argument('--regular', default=0.00002, type=float)
	parser.add_argument('--epochs', default=1, type=int)
	parser.add_argument('--batch', default=500, type=int)
	parser.add_argument('--dim', default=4, type=int)
	parser.add_argument('--num_features', default=11, type=int)
	parser.add_argument('--num_features_category', default=10, type=int)
	parser.add_argument('--num_features_numeric', default=1, type=int)
	parser.add_argument('--max_list_num', default=12, type=int)
	parser.add_argument('--max_features', default=829117, type=int)
	# hash size = 1251142, not need to filter (feature_extract7.py)
	# hash size = 829117, filter by low 5 (feature_extract7.py)
	# hash size = 1251152 UNK (featurex_extract8.py)
	parser.add_argument('--model', default="./fm_model", type=str)
	parser.add_argument('--predict', default="./predict", type=str)
	parser.add_argument('--mode', default=0, type=int)

	args = parser.parse_args()

	return args

class mf(object):

	def __init__(self, args):
		self.args = args

		
		self.index = 0
		self.index_valid = 0
		self.index_test = 0


		self.args.num_features 

	def load_data(self, train=True):	
		if train:

			print >> sys.stderr, "load train_data"
			# self.train_data = np.loadtxt(self.args.train, dtype=int)
			self.train_data = np.fromstring(open(self.args.train, "r").read(), \
			dtype=np.float32, sep=' ').reshape(-1, self.args.max_list_num, self.args.num_features+1)
			# self.train_data[:,:,8] = self.train_data[:,:,8] % 7
			print self.train_data.shape

			# normalize numeric part
			print >> sys.stderr, "normalize train_data"
			# self.train_data[:,:,(1+self.args.num_features_category):] /= 100.0 
			# mean = np.mean(self.train_data[:,:,(1+self.args.num_features_category):], axis=(0,1))
			# std = np.std(self.train_data[:,:,(1+self.args.num_features_category):], axis=(0,1))
			# self.train_data[:,:,(1+self.args.num_features_category):] = (self.train_data[:,:,(1+self.args.num_features_category):] - mean) / std


			# with open(self.args.train, "r") as f:
			# 	self.train_data = [[int(x) for x in row] for row in csv.reader(f, delimiter=' ')]
			print >> sys.stderr, "load val_data"
			# with open(self.args.valid, "r") as f:
			# 	self.val_data = [[int(x) for x in row] for row in csv.reader(f, delimiter=' ')]
			# self.val_data = np.loadtxt(self.args.valid, dtype=int)
			self.val_data = np.fromstring(open(self.args.valid, "r").read(), \
			dtype=np.float32, sep=' ').reshape(-1, self.args.max_list_num, self.args.num_features+1)
			print self.val_data.shape

			# normalize numeric part
			# self.val_data[:,:,(1+self.args.num_features_category):] /= 100.0 

			# self.val_data[:,:,8] = self.val_data[:,:,8] % 7

			s_time = time.time()
			self.shuffle()
			print >> sys.stderr, "shuffle time: ", time.time() - s_time
			# self.train_data = self.shuffle(self.train_data)
			# np.random.shuffle(self.train_data)

		else:
			print >> sys.stderr, "load test_data"
			self.test_data = np.fromstring(open(self.args.test, "r").read(), \
			dtype=np.float32, sep=' ').reshape(-1, self.args.max_list_num, self.args.num_features+1)
			print self.test_data.shape

			# normalize numeric part
			# self.test_data[:,:,(1+self.args.num_features_category):] /= 100.0 


			# self.test_data[:,:,8] = self.test_data[:,:,8] % 7



	

	def shuffle(self):
		np.random.shuffle(self.train_data)
		p = np.random.permutation(self.args.max_list_num)
		self.train_data = self.train_data[:, p, :]
		# p = np.random.permutation(len(data))
		# data = data[p]
		# return data


	def next_batch(self, batch_size, dtype="train"):
		if dtype == "train":
			data = self.train_data[self.index : batch_size + self.index]
			# data = np.array(data)
			if self.index + batch_size >= len(self.train_data):
				self.index = 0
				self.shuffle()
				# self.train_data = self.shuffle(self.train_data)
				# np.random.shuffle(self.train_data)
			else:
				self.index = self.index + batch_size
			return data

		elif dtype == "valid":
			data = self.val_data[self.index_valid : batch_size + self.index_valid]
			# data = np.array(data)
			if self.index_valid + batch_size >= len(self.val_data):
				self.index_valid = 0
			else:
				self.index_valid = self.index_valid + batch_size
			return data

		elif dtype == "test":
			data = self.test_data[self.index_test : batch_size + self.index_test]
			self.index_test = self.index_test + batch_size

			return data

	def add_model(self):
		with tf.device('/cpu:0'):



			x = tf.placeholder(tf.float32, [None, self.args.max_list_num, self.args.num_features])
			x_category = tf.cast(x[:,:,:self.args.num_features_category], tf.int32)
			x_numeric = x[:,:,self.args.num_features_category:]

			y = tf.placeholder(tf.float32, [None, self.args.max_list_num])
			
			b = tf.Variable(tf.random_uniform(
									 	[1, 1], -.1, .1))

			embedding_w = tf.concat(0, [tf.constant([[0.] * 1], dtype=tf.float32), 
									 tf.Variable(tf.random_uniform(
									 	[self.args.max_features, 1], -.1, .1))
									])
		
			embedding_v = tf.concat(0, [tf.constant([[0.] * self.args.dim], dtype=tf.float32), 
									 tf.Variable(tf.random_uniform(
									 	[self.args.max_features, self.args.dim], -.1, .1))
									])

			# b = tf.Variable(tf.random_uniform(
			# 						 	[1, 1], 0.0, .1))

			# w = tf.random_uniform(self.args.max_features, 1], 0.0, .1)
			# w_sum = tf.reduce_sum(w)
			# w_c = tf.constant(1.0 / 1.0, shape=[self.args.max_features, 1])

			# embedding_w = tf.concat(0, [tf.constant([[0.] * 1], dtype=tf.float32), 
			# 						 tf.Variable(w/w_sum*w_c)
			# 						])

			# v = tf.random_uniform(self.args.max_features, self.args.dim], 0.0, .1)
			# v_sum = tf.reduce_sum(v)
			# v_c = tf.constant(1.0 / self.args.dim, shape=[self.args.max_features, self.args.dim])
		
			# embedding_v = tf.concat(0, [tf.constant([[0.] * self.args.dim], dtype=tf.float32), 
			# 						 tf.Variable(v/v_sum*v_c)
			# 						])


		with tf.device('/gpu:0'):
			embed_w = tf.nn.embedding_lookup(embedding_w, x_category)
			embed_v = tf.nn.embedding_lookup(embedding_v, x_category)

			# w_x = tf.reduce_sum(embed_w, [1, 2])
			w_x = tf.reduce_sum(embed_w, [2, 3])
			# print w_x.get_shape()

			# m = tf.batch_matmul(embed_v, tf.transpose(embed_v, perm=[0, 2, 1]))
			m = tf.batch_matmul(embed_v, tf.transpose(embed_v, perm=[0, 1, 3, 2]))
			# mask = np.array([[1 if j > i else 0 for j in range(self.args.num_features) ] for i in range(self.args.num_features)]).astype(bool)
			


			m_l = []
			for i in range(self.args.num_features_category):
				for j in range(i+1, self.args.num_features_category):
					m_l.append(tf.expand_dims(m[:,:,i,j], 2))

			mm = tf.concat(2, m_l)

			mm_numeric = tf.concat(2, [mm, x_numeric])
			
			w_mm_dim = (self.args.num_features_category ** 2 - self.args.num_features_category) / 2 + self.args.num_features_numeric # C(n, 2)
			
			
			w_mm = tf.Variable(tf.random_uniform([w_mm_dim, 1], -.1, .1))
			b_mm = tf.Variable(tf.random_uniform([1], -.1, .1))

			# w_mm = tf.ones([w_mm_dim, 1])
			# b_mm = tf.zeros([1])
			

			mm_reshape = tf.reshape(tf.expand_dims(mm_numeric, 2), [-1, w_mm_dim])
			a_mm = tf.matmul(mm_reshape, w_mm) + b_mm
			mmm = tf.reshape(a_mm, [-1, self.args.max_list_num])

			vv_x = mmm

			# a1_mm = tf.matmul(mm, w1_mm) + b1_mm

			# z1_mm = a1_mm # linear


			# mmm = z1_mm

			# vv_x = tf.squeeze(mmm, [1])

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
			
			# # clip_all_x = tf.clip_by_value(all_x, -35., 35.)
			# # p = 1.0 / (1.0 + tf.exp(-clip_all_x))
			# # clip_p = tf.clip_by_value(p, 10e-8, 1.0 - 10e-8)

			# # cost: logloss
			# cost = -tf.reduce_sum(y * tf.log(clip_p) + (1.0-y) * tf.log(1.0-clip_p)) \
			# 	+ self.args.regular * (tf.nn.l2_loss(embed_w) \
			# 		+ tf.nn.l2_loss(embed_v) + tf.nn.l2_loss(w1_mm) )
			
			cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(all_x, y))
			# pred = tf.argmax(all_x, 1)

			v_sorted, i_sorted = tf.nn.top_k(all_x, self.args.max_list_num)
			pred = i_sorted
			y_label = tf.expand_dims(tf.cast(tf.argmax(y, 1), tf.int32), 1)


			# rank = i_sorted[y_label]
			rank = tf.where(tf.equal(i_sorted, y_label))[:, 1]
			rr = 1.0 / (1.0 + tf.cast(rank, tf.float32))
			rr_sum = tf.reduce_sum(rr)

			opt = tf.train.AdagradOptimizer(self.args.lr).minimize(cost)


			return {'x': x,
					'y': y,
					'pred': pred,
					'cost': cost,
					'opt': opt,
					'rr_sum': rr_sum}

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
				total_rr = 0.0
				total_count = 0
				pbar = pb.ProgressBar(widgets=["[TRAIN] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('mrr'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()

				for i in xrange(total_batch):
					batch = self.next_batch(self.args.batch)
					_, loss, rr = sess.run([var['opt'], var['cost'], var['rr_sum']], feed_dict={var['y']:batch[:, :, 0], var['x']:batch[:, :, 1:]})
					total_loss += loss
					total_rr += rr
					total_count += len(batch)
					pbar.update(i, loss=total_loss/total_count, mrr=total_rr/total_count)
				pbar.finish()

				# v_loss, v_rr = sess.run([var['cost'], var['rr_sum']], feed_dict={var['y']:self.val_data[:, :, 0], var['x']:self.val_data[:, :, 1:]})
				v_loss, v_rr, v1_rr, v2_rr = self.eval(sess, var)

				print >> sys.stderr, \
					"Epoch {}:\n  tr_cross_entropy: {}, tr_mrr: {}\n  v_cross_entropy: {}, v_mrr: {}\n  v1_rr: {}, v2_rr: {}".format(\
						epoch, total_loss / len(self.train_data), total_rr / len(self.train_data), \
						v_loss / len(self.val_data), v_rr / len(self.val_data), \
						v1_rr / (len(self.val_data)/2), v2_rr / (len(self.val_data)/2) )

			print >> sys.stderr, "save model"
			save_path = saver.save(sess, self.args.model)
			print >> sys.stderr, "save model in path", save_path

	def eval(self, sess, var):
		batch_size = 10000
		
		total_batch = int(np.ceil(len(self.val_data) / float(batch_size)))


		total_loss = 0.0
		total_rr = 0.0
		total_count = 0
		total_v1_rr = 0.0
		total_v2_rr = 0.0
		pbar = pb.ProgressBar(widgets=["[VALID] ", pb.DynamicMessage('loss'), " ", pb.DynamicMessage('mrr'), " ", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()

		for i in xrange(total_batch):
			batch = self.next_batch(batch_size, dtype="valid")
			loss, rr = sess.run([var['cost'], var['rr_sum']], feed_dict={var['y']:batch[:, :, 0], var['x']:batch[:, :, 1:]})
			total_loss += loss
			total_rr += rr
			total_count += len(batch)
			if i < total_batch / 2:
				total_v1_rr += rr
			else:
				total_v2_rr += rr
			pbar.update(i, loss=total_loss/total_count, mrr=total_rr/total_count)
		pbar.finish()

		return total_loss, total_rr, total_v1_rr, total_v2_rr


	def predict(self):
		# not yet here
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
				batch = self.next_batch(self.args.batch, dtype='test')
				pred = sess.run(var['pred'], feed_dict={var['x']:batch[:,:,1:]})
				# print prob.shape
				out_l = [str(xx) for x in pred.tolist() for xx in x]
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
	