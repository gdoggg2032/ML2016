import argparse
import cPickle as pickle
import sys
import time
import progressbar as pb
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import random




def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument("--name", default="cnn", type=str)
	parser.add_argument("--label", default="./data/all_label.p", type=str)
	parser.add_argument("--unlabel", default="./data/all_unlabel.p", type=str)
	parser.add_argument("--test", default="./data/test.p", type=str)
	parser.add_argument("--batch", default=100, type=int)
	parser.add_argument("--epochs", default=2000, type=int)
	parser.add_argument("--lr", default=5e-3, type=float)
	parser.add_argument("--model", default="./model", type=str)
	parser.add_argument("--conti", default=0, type=int)
	parser.add_argument("--conti_model", default="./model", type=str)
	parser.add_argument("--predict", default="./predict.csv", type=str)
	parser.add_argument("--mode", default=2, type=int)

	parser.add_argument("--val_rate", default=0.1, type=int)
	parser.add_argument("--early_stop", default=200000, type=int)


	args = parser.parse_args()
	print >> sys.stderr, args
	args.prints = "[{}][lr={}][b={}]".format(args.name, args.lr, args.batch)
	if args.conti == 1:
		args.prints += "[conti_m={}]".format(args.conti_model.split("/")[-1])
	

	return args

class cnn(object):

	def __init__(self, args):
		self.args = args
		self.load_data()
		self.index = 0

	def load_data(self):

		if self.args.mode % 2 == 0:
			train_label_data = np.array(pickle.load(open(self.args.label, "r")))
			train_unlabel_data = np.array(pickle.load(open(self.args.unlabel, "r")))
			


			num_per_class = 500
			num_per_val = int(self.args.val_rate * num_per_class)
			p_val = []
			p_train = []
			for i in range(10):
				per = np.random.permutation(500) + i * num_per_class
				p_val.append(per[0 : num_per_val])
				p_train.append(per[num_per_val:])
			p_val = np.concatenate(p_val)
			p_train = np.concatenate(p_train)




			# only use labeled data
			self.train_label_data = train_label_data.reshape(train_label_data.shape[0] * train_label_data.shape[1], train_label_data.shape[2])
			self.train_label_labels = np.array([[i]*train_label_data.shape[1] for i in range(train_label_data.shape[0])]).flatten()
			
			self.val_label_data, self.val_label_labels = self.train_label_data[p_val], self.train_label_labels[p_val]
			self.train_label_data, self.train_label_labels = self.train_label_data[p_train], self.train_label_labels[p_train]

			self.train_label_weights = np.ones(self.train_label_labels.shape)
			self.val_label_weights = np.ones(self.val_label_labels.shape)

			self.train_label_data, self.train_label_labels, self.train_label_weights = self.shuffle(self.train_label_data, self.train_label_labels, self.train_label_weights)
			
			
			# self.val_label_data = self.train_label_data[0 : self.args.val_size]
			# self.val_label_labels = self.train_label_labels[0 : self.args.val_size]
			
			# self.train_label_data = self.train_label_data[self.args.val_size :]
			# self.train_label_labels = self.train_label_labels[self.args.val_size :]
			
			self.train_data = self.train_label_data
			self.train_labels = self.train_label_labels
			self.train_weights = np.ones(self.train_labels.shape)

			self.train_unlabel_data = train_unlabel_data

		if self.args.mode > 0:
			test_data = pickle.load(open(self.args.test, "r"))
			self.test_ids = np.array(test_data['ID'])
			self.test_data = np.array(test_data['data'])




	
		

	def shuffle(self, data, labels, weights):
		p = np.random.permutation(len(data))
		data = data[p]
		labels = labels[p]
		weights = weights[p]
		return data, labels, weights

	def next_batch(self, batch_size):

		data = self.train_data[self.index : batch_size + self.index]

		# data = self.sess.run(self.distorted_image, feed_dict={self.image:data})

		labels = self.train_labels[self.index : batch_size + self.index]

		weights = self.train_weights[self.index : batch_size + self.index]
		
		if self.index + batch_size >= len(self.train_data):
			self.index = 0
			self.train_data, self.train_labels, self.train_weights = self.shuffle(self.train_data, self.train_labels, self.train_weights)
		else:
			self.index = self.index + batch_size
		return data, labels, weights


	def add_image_vars(self):
		image = tf.placeholder(tf.float32, shape=[None, 1024 * 3])
		x_image = tf.transpose(tf.reshape(image, [-1, 3, 32, 32]), [0, 2, 3, 1])

		distorted_image = x_image
		# r = random.randint(0, 10) # 0, 1, 2, 3, 4, 4 for doing nothing
		# Randomly crop a [height, width] section of the image.
		# distorted_image = tf.random_crop(x_image, [32, 32, 3])


		# num = tf.placeholder(tf.int32)
		num = tf.random_shuffle(tf.range(-10, 4))[0]
		# order = tf.placeholder(tf.int32, [4])
		order = tf.random_shuffle(tf.range(5))

		r = x_image

		z = tf.constant(0)
		def b(z, r):

			def f0():
				return r
			def f1():
				return tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), r)
			def f2():
				return tf.map_fn(lambda img: tf.image.random_flip_left_right(img), r)
			def f3():
				return tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=63), r)
			def f4():
				return tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), r)

			cond1 = tf.equal(tf.constant(1), order[z])
			cond2 = tf.equal(tf.constant(2), order[z])
			cond3 = tf.equal(tf.constant(3), order[z])
			cond4 = tf.equal(tf.constant(4), order[z])

			case = tf.case({cond1:f1, cond2:f2, cond3:f3, cond4:f4}, default=f0, exclusive=True)
			case.set_shape([None, 32, 32, 3])
			return z+1, case

		def c(z, r):
			return tf.less(z, num)

		# _, distorted_image = tf.while_loop(c, b, [z, r], shape_invariants=[z.get_shape(), r.get_shape()])
		# print distorted_image.get_shape()






		# numberOfOperations = random.randint(-3, ) # prob 0.5 for do nothing

		# operationsOrder = random.shuffle([0, 1, 2, 3])


		# for counter in range(numberOfOperations):
		# 	if operationsOrder[counter] == 0:
		# 		distorted_image = tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), distorted_image)
		# 	elif operatiosnOrder[counter] == 1:
		# 		distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), distorted_image)
		# 	elif operationsOrder[counter] == 2:
		# 		distorted_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=63), distorted_image)
		# 	elif operationsOrder[counter] == 3:
		# 		distorted_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), distorted_image)

		# if r == 0:
		# 	distorted_image = tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), distorted_image)

		# Randomly flip the image horizontally.
		# distorted_image = tf.image.random_flip_left_right(distorted_image)
		# if r == 1:
		# 	distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), distorted_image)
		# Because these operations are not commutative, consider randomizing
		# the order their operation.
		# distorted_image = tf.image.random_brightness(distorted_image,
		#
		# if r == 2:									
		# 	distorted_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=63), distorted_image)
		
		# distorted_image = tf.image.random_contrast(distorted_image,
		#	
		# if r == 3:								  
		# 	distorted_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8), distorted_image)
		# Subtract off the mean and divide by the variance of the pixels.
		# float_image = tf.image.per_image_whitening(distorted_image)
		# distorted_image = tf.map_fn(lambda img: tf.image.per_image_whitening(img), distorted_image)
		
		float_image = tf.reshape(tf.transpose(distorted_image, [0, 3, 1, 2]), [-1, 1024 * 3])
		return float_image, image


	def add_model_vars(self):
		def weight_variable(shape, dev):
			# initial = tf.truncated_normal(shape, stddev=0.1)
			initial = tf.random_normal(shape, stddev=dev)
			return tf.Variable(initial)

		def bias_variable(shape):
			# initial = tf.constant(0.0, shape=shape)
			initial = tf.random_normal(shape)
			return tf.Variable(initial)

		def conv2d(x, W):
			return tf.nn.conv2d(x, W, 
				strides=[1,1,1,1], padding='SAME')

		def max_pool_2x2(x):
			return tf.nn.max_pool(x, ksize=[1,2,2,1],
				strides=[1,2,2,1], padding='SAME')

		# x = tf.placeholder(tf.float32, shape=[None, 1024 * 3])
		x = tf.placeholder(tf.float32, shape=[None, 1024 * 3])
		y = tf.placeholder(tf.int64, shape=[None])

		w = tf.placeholder(tf.float32, shape=[None])

		keep_prob = tf.placeholder(tf.float32)


		# x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])
		x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])
		

		W_conv1 = weight_variable([3, 3, 3, 24], 0.01)
		b_conv1 = bias_variable([24])
		
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)
		

		W_conv2 = weight_variable([3, 3, 24, 48], 0.01)
		b_conv2 = bias_variable([48])
		
		h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)



		W_conv3 = weight_variable([3, 3, 48, 48], 0.01)
		b_conv3 = bias_variable([48])

		h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
		h_pool3 = max_pool_2x2(h_conv3)
		h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)


		W_fc1 = weight_variable([4 * 4 * 48, 256], 0.01)
		b_fc1 = bias_variable([256])
		h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 4 * 4 * 48])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = weight_variable([256, 10], 1.0)
		b_fc2 = bias_variable([10])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y) * w)

		# train_step = tf.train.AdamOptimizer(self.args.lr).minimize(cross_entropy)
		train_step = tf.train.AdagradOptimizer(self.args.lr).minimize(cross_entropy)
		predicts = tf.argmax(y_conv, 1)

		probs = tf.nn.softmax(y_conv)
		self.probs = tf.reduce_max(probs, 1)

		correct_prediction = tf.equal(predicts, y)
		acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		return x, y, keep_prob, train_step, acc, cross_entropy, predicts, w

	def add_unlabel_data(self, top_k):
		s_time = time.time()

		batch_size = 1000
		total_batch = int(np.ceil(self.train_unlabel_data.shape[0] / float(batch_size)))

		probs_unlabel_l = []
		labels_unlabel_l = []
		for i in range(total_batch):
			batch = self.train_unlabel_data[i*batch_size : (i+1)*batch_size]
			probs_unlabel, labels_unlabel = self.sess.run([self.probs, self.predicts], feed_dict={self.x:batch, self.keep_prob:1.0})
			probs_unlabel_l.append(probs_unlabel)
			labels_unlabel_l.append(labels_unlabel)
		probs_unlabel = np.concatenate(probs_unlabel_l, axis=0)
		labels_unlabel = np.concatenate(labels_unlabel_l, axis=0)
		print >> sys.stderr, "probs", time.time() - s_time

		self.train_data = self.train_label_data
		self.train_labels = self.train_label_labels

		total = 0
		# top_k = 450#int(2000 * 0.85)
		prob_tf = 0.0
		unlabel_weight = 1.5
		del_x = []
		for i in range(10):
			
			can = [(index,o) for index, (o, c) in enumerate(zip(probs_unlabel, labels_unlabel)) if c == i if o>prob_tf]
			xcan = [c[0] for c in  sorted(can, key=lambda x: x[1], reverse=True)][:top_k]
			del_x += xcan
			total += len(xcan)
			self.train_data = np.concatenate([self.train_data, self.train_unlabel_data[xcan]], axis=0)
			self.train_labels = np.concatenate([self.train_labels, i * np.ones(len(xcan))], axis=0)
			self.train_weights = np.concatenate([self.train_weights, unlabel_weight * np.ones(len(xcan))], axis=0)
		# self.train_unlabel_data = np.delete(self.train_unlabel_data, del_x, axis=0)
		print >> sys.stderr, "add unlabel data: {}, time cost: {}".format(total, time.time() - s_time)
		print >> sys.stderr, "total train_data number: {}".format(len(self.train_data))

		self.train_data, self.train_labels, self.train_weights = self.shuffle(self.train_data, self.train_labels, self.train_weights)


	def train(self):
		with tf.Graph().as_default():
			x, y, keep_prob, train_step, acc, cross_entropy, predicts, w = self.add_model_vars()
			
			self.x = x
			self.y = y
			self.w = w
			self.keep_prob = keep_prob
			self.predicts = predicts

			self.acc = acc
			self.cross_entropy = cross_entropy

			self.distorted_image, self.image = self.add_image_vars()

			self.saver = tf.train.Saver()

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			# self.sess = tf.Session()

			if self.args.conti == 1:
				self.saver.restore(self.sess, self.args.conti_model)
				print >> sys.stderr, "restore model from ", self.args.conti_model
			else:
				self.sess.run(tf.initialize_all_variables())

			stopped = -1
			best_val_loss = float('inf')
			best_val_acc = 0.0
			best_val_epoch = 0

			val_warm_up = -5

			val_th = 0.55

			top_k = 450


			for epoch in range(self.args.epochs):

				total_batch = int(np.ceil(len(self.train_data) / float(self.args.batch)))

				pbar = pb.ProgressBar(widgets=["train:", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
				for j in xrange(total_batch):
					# print j
					pbar.update(j)
					batchx, batchy, batchw = self.next_batch(self.args.batch)
					self.sess.run(train_step, feed_dict={x:batchx, y:batchy, keep_prob:0.7, w:batchw})
				pbar.finish()

				# val_acc, val_loss  = self.sess.run([acc, cross_entropy], feed_dict={x:self.val_label_data, y:self.val_label_labels, keep_prob:1.0})
				val_acc, val_loss  = self.eval(self.val_label_data, self.val_label_labels, self.val_label_weights)
				
				# if val_loss < best_val_loss:
				# 	best_val_loss = val_loss
				# 	best_val_epoch = epoch

				val_warm_up += 1
				

				# if epoch - best_val_epoch > self.args.early_stop:
				# 	stopped = epoch
				if val_acc > best_val_acc:
					best_val_acc = val_acc
					best_val_epoch = epoch



				if epoch - best_val_epoch > self.args.early_stop:
					stopped = epoch


				if epoch % 1 == 0 or epoch +1 == self.args.epochs or epoch - best_val_epoch > self.args.early_stop:
					print >> sys.stderr, "evaluating"
					# label_acc, label_loss = self.sess.run([acc, cross_entropy], feed_dict={x:self.train_label_data, y:self.train_label_labels, keep_prob:1.0})
					# train_acc, train_loss = self.sess.run([acc, cross_entropy], feed_dict={x:self.train_data, y:self.train_labels, keep_prob:1.0})
					label_acc, label_loss = self.eval(self.train_label_data, self.train_label_labels, self.train_label_weights)
					train_acc, train_loss = self.eval(self.train_data, self.train_labels, self.train_weights)
					
					print >> sys.stderr, "step {}: val_th:{}, label_size:{}, total_size:{}".format(epoch, val_th, len(self.train_label_data), len(self.train_data))
					print >> sys.stderr, "  [train_acc: {:6.6}, train_loss: {:6.6}]".format(train_acc, train_loss)
					print >> sys.stderr, "  [label_acc: {:6.6}, label_loss: {:6.6}]".format(label_acc, label_loss)
					print >> sys.stderr, "  [valid_acc: {:6.6}, valid_loss: {:6.6}]".format(val_acc, val_loss)
					if epoch - best_val_epoch > self.args.early_stop:
						stopped = epoch
						break
				# if val_acc > val_th and val_warm_up > 0:
				# 	print >> sys.stderr, "assign unlabel data"
				# 	self.add_unlabel_data()
				# 	val_warm_up = -5
				# 	val_th += (val_acc - val_th) * 0.5
				# 	print >> sys.stderr, "update val_th to {}".format(val_th)
				# 	self.sess.run(tf.initialize_all_variables())
				# 	print >> sys.stderr, "reset variables"

				if val_acc > val_th and val_warm_up > 0:
					print >> sys.stderr, "assign unlabel data"
					self.add_unlabel_data(top_k)
					val_warm_up = -0
					val_th += 0.5 *(val_acc-val_th)#+= 0.0 #0.005#(label_acc - val_th) * 0.5
					top_k += 50
					print >> sys.stderr, "update val_th to {}".format(val_th)
					# self.sess.run(tf.initialize_all_variables())
					# print >> sys.stderr, "reset variables"

					tmp_model = "[l_acc={:.4}][v_acc={:.4}][epochs={}]{}".format(label_acc, val_acc, epoch, self.args.prints)
					save_path = self.saver.save(self.sess, "model_{}".format(tmp_model),  write_meta_graph=False)
					print >> sys.stderr, "save tmp model in path", save_path

			if stopped == -1:
				stopped = self.args.epochs
			print >> sys.stderr, "stop at Epoch {}".format(stopped)
			stopped += 1
			print >> sys.stderr, "save model"
			save_path = self.saver.save(self.sess, self.args.model, write_meta_graph=False)
			print >> sys.stderr, "save model in path", save_path
			
			self.args.prints = "[l_acc={:.4}][v_acc={:.4}][epochs={}]{}".format(label_acc, val_acc, self.args.epochs, self.args.prints)
			save_path = self.saver.save(self.sess, "model_{}".format(self.args.prints),  write_meta_graph=False)
			print >> sys.stderr, "save model in path", save_path
			# print "test_acc: {}".format(acc.eval(feed_dict={x:self.test_data, y:mnist.test.labels, keep_prob:1.0}))

	def eval(self, data, labels, weights):
		batch_size = 1000
		total_batch = int(np.ceil(data.shape[0] / float(batch_size)))

		total_loss = 0.0
		total_acc = 0.0
		for i in range(total_batch):
			data_batch = data[i*batch_size : (i+1)*batch_size]
			labels_batch = labels[i*batch_size : (i+1)*batch_size]
			weights_batch = weights[i*batch_size : (i+1)*batch_size]
			acc, loss = self.sess.run([self.acc, self.cross_entropy], feed_dict={self.x:data_batch, self.y:labels_batch, self.keep_prob:1.0, self.w:weights_batch})
			total_acc += acc * len(data_batch)
			total_loss += loss * len(data_batch)
		total_acc /= data.shape[0]
		total_loss /= data.shape[0]
		return total_acc, total_loss


	def test(self):
		with tf.Graph().as_default():
			x, y, keep_prob, train_step, acc, cross_entropy, predicts, w = self.add_model_vars()

			self.saver = tf.train.Saver()
			self.sess = tf.Session()

			self.saver.restore(self.sess, self.args.model)
			print >> sys.stderr, "restore model from ", self.args.model
			
			batch_size = 100
			total_batch = int(np.ceil(self.test_data.shape[0] / float(batch_size)))
			preds_l = []
			for i in range(total_batch):
				data_batch = self.test_data[i*batch_size : (i+1)*batch_size]
				preds_batch = self.sess.run(predicts, feed_dict={x:data_batch, keep_prob:1.0})
				preds_l.append(preds_batch)


			# test_preds = self.sess.run(predicts, feed_dict={x:self.test_data, keep_prob:1.0})
			# print test_preds.shape
			test_preds = np.concatenate(preds_l, axis=0)





			# test_preds = self.sess.run(predicts, feed_dict={x:self.test_data, keep_prob:1.0})
			# print test_preds.shape


			df = pd.DataFrame({'ID':[i for i in self.test_ids], 'class':test_preds})
			outputFile = "predict_{}.csv".format(self.args.prints)

			df.to_csv(outputFile, index=False)
			df.to_csv(self.args.predict, index=False)







def main():

	args = arg_parse()

	model = cnn(args)

	

	if args.mode % 2 == 0:
		s_time = time.time()

		model.train()

		print >> sys.stderr, "training time: {}".format(time.time()-s_time)

	if args.mode / 2 == 0:
		start_time = time.time()

		model.test()

		print >> sys.stderr, "testing time: {}".format(time.time()-start_time)




if __name__ == "__main__":
	main()