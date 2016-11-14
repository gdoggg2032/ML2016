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
import math




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

	parser.add_argument("--auto", default=0, type=int)
	parser.add_argument("--auto_model", default="./autoencoder", type=str)
	parser.add_argument("--auto_epochs", default=100, type=int)


	args = parser.parse_args()
	print >> sys.stderr, args
	args.prints = "[{}][lr={}][b={}]".format(args.name, args.lr, args.batch)
	if args.conti == 1:
		args.prints += "[conti_m={}]".format(args.conti_model.split("/")[-1])
	

	return args
class Convolution2D(object):
	'''
	  constructor's args:
		  input	 : input image (2D matrix)
		  input_siz ; input image size
		  in_ch	 : number of incoming image channel
		  out_ch	: number of outgoing image channel
		  patch_siz : filter(patch) size
		  weights   : (if input) (weights, bias)
	'''
	def __init__(self, input, input_siz, in_ch, out_ch, patch_siz, activation='relu'):
		self.input = input	  
		self.rows = input_siz[0]
		self.cols = input_siz[1]
		self.in_ch = in_ch
		self.activation = activation
		
		wshape = [patch_siz[0], patch_siz[1], in_ch, out_ch]
		
		w_cv = tf.Variable(tf.random_normal(wshape, stddev=0.1), 
							trainable=True)
		b_cv = tf.Variable(tf.random_normal([out_ch], stddev=0.1), 
							trainable=True)
		
		self.w = w_cv
		self.b = b_cv
		self.params = [self.w, self.b]
		
	def output(self):
		shape4D = [-1, self.rows, self.cols, self.in_ch]
		
		x_image = tf.reshape(self.input, shape4D)  # reshape to 4D tensor
		linout = tf.nn.conv2d(x_image, self.w, 
				  strides=[1, 1, 1, 1], padding='SAME') + self.b
		if self.activation == 'relu':
			self.output = tf.nn.relu(linout)
		elif self.activation == 'sigmoid':
			self.output = tf.sigmoid(linout)
		else:
			self.output = linout
		
		return self.output

# Max Pooling Layer   
class MaxPooling2D(object):
	'''
	  constructor's args:
		  input  : input image (2D matrix)
		  ksize  : pooling patch size
	'''
	def __init__(self, input, ksize=None):
		self.input = input
		if ksize == None:
			ksize = [1, 2, 2, 1]
			self.ksize = ksize
	
	def output(self):
		self.output = tf.nn.max_pool(self.input, ksize=self.ksize,
					strides=[1, 2, 2, 1], padding='SAME')
  
		return self.output

class Conv2Dtranspose(object):
	'''
	  constructor's args:
		  input	  : input image (2D matrix)
		  output_siz : output image size
		  in_ch	  : number of incoming image channel
		  out_ch	 : number of outgoing image channel
		  patch_siz  : filter(patch) size
	'''
	def __init__(self, input, output_siz, in_ch, out_ch, patch_siz, activation='relu'):
		self.input = input	  
		self.rows = output_siz[0]
		self.cols = output_siz[1]
		self.out_ch = out_ch
		self.activation = activation
		
		wshape = [patch_siz[0], patch_siz[1], out_ch, in_ch]	# note the arguments order
		
		w_cvt = tf.Variable(tf.random_normal(wshape, stddev=0.1), 
							trainable=True)
		b_cvt = tf.Variable(tf.random_normal([out_ch], stddev=0.1), 
							trainable=True)
		self.batsiz = tf.shape(input)[0]
		self.w = w_cvt
		self.b = b_cvt
		self.params = [self.w, self.b]
		
	def output(self):
		shape4D = [self.batsiz, self.rows, self.cols, self.out_ch]	  
		linout = tf.nn.conv2d_transpose(self.input, self.w, output_shape=shape4D,
							strides=[1, 2, 2, 1], padding='SAME') + self.b
		if self.activation == 'relu':
			self.output = tf.nn.relu(linout)
		elif self.activation == 'sigmoid':
			self.output = tf.sigmoid(linout)
		else:
			self.output = linout
		
		return self.output


def autoencoder(input_shape=[None, 3*32*32],
				n_filters=[3, 72, 48, 48],
				filter_sizes=[3, 3, 3, 3],
				corruption=False):
	"""Build a deep denoising autoencoder w/ tied weights.
	Parameters
	----------
	input_shape : list, optional
		Description
	n_filters : list, optional
		Description
	filter_sizes : list, optional
		Description
	Returns
	-------
	x : Tensor
		Input placeholder to the network
	z : Tensor
		Inner-most latent representation
	y : Tensor
		Output reconstruction of the input
	cost : Tensor
		Overall cost to use for training
	Raises
	------
	ValueError
		Description
	"""
	# %%
	# input to the network
	x = tf.placeholder(
		tf.float32, input_shape, name='x')
	x_tensor = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])
	keep_prob = tf.placeholder(tf.float32)


	# # %%
	# # ensure 2-d is converted to square tensor.
	# if len(x.get_shape()) == 2:
	#	 x_dim = np.sqrt(x.get_shape().as_list()[1])
	#	 if x_dim != int(x_dim):
	#		 raise ValueError('Unsupported input dimensions')
	#	 x_dim = int(x_dim)
	#	 x_tensor = tf.reshape(
	#		 x, [-1, x_dim, x_dim, n_filters[0]])
	# elif len(x.get_shape()) == 4:
	#	 x_tensor = x
	# else:
	#	 raise ValueError('Unsupported input dimensions')
	current_input = x_tensor

	# %%
	# Optionally apply denoising autoencoder
	if corruption:
		current_input = corrupt(current_input)

	# %%
	# Build the encoder
	encoder = []
	shapes = []
	for layer_i, n_output in enumerate(n_filters[1:]):
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		W = tf.Variable(
			tf.random_uniform([
				filter_sizes[layer_i],
				filter_sizes[layer_i],
				n_input, n_output],
				-1.0 / math.sqrt(n_input),
				1.0 / math.sqrt(n_input)))
		print >> sys.stderr, W.get_shape()
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		# output = tf.nn.relu(
		# 	tf.add(tf.nn.conv2d(
		# 		current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
		output = tf.nn.relu(
			tf.add(tf.nn.conv2d(
				current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b))
		# output = tf.add(tf.nn.conv2d(
		# 		current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
		output = MaxPooling2D(output).output()
		print >> sys.stderr, output.get_shape()
		output = tf.nn.dropout(output, keep_prob)
		current_input = output

	# # fully connected layer
	current_input = tf.reshape(current_input, [-1, 4*4*48])

	W = tf.Variable(tf.random_uniform([4*4*48, 2*2*48], -0.01, 0.01))
	b = tf.Variable(tf.zeros([2*2*48]))
	
	current_input = tf.matmul(current_input, W) + b
	current_input = tf.nn.relu(current_input)

	# W2 = tf.Variable(tf.random_uniform([4*4*48, 4*4*48], -0.01, 0.01))
	# b2 = tf.Variable(tf.zeros([4*4*48]))

	# current_input = tf.matmul(current_input, W2) + b2
	# current_input = tf.nn.relu(current_input)

	# store the latent representation
	# z = tf.reshape(current_input, [-1, 4*4*48])
	z = current_input

	encoder.reverse()
	shapes.reverse()

	# W2_transpose = tf.transpose(W2)
	# b2_transpose = tf.Variable(tf.zeros([4*4*48]))

	# current_input = tf.matmul(current_input, W2_transpose) + b2_transpose
	# current_input = tf.nn.relu(current_input)

	W_transpose = tf.transpose(W)
	b_transpose = tf.Variable(tf.zeros([4*4*48]))

	current_input = tf.matmul(current_input, W_transpose) + b_transpose
	current_input = tf.nn.relu(current_input)

	current_input = tf.reshape(current_input, [-1, 4, 4, 48])

	# %%
	# Build the decoder using the same weights
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		output = tf.nn.relu(tf.add(
			tf.nn.conv2d_transpose(
				current_input, W,
				tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
				strides=[1, 2, 2, 1], padding='SAME'), b))
		# output = tf.add(
		# 	tf.nn.conv2d_transpose(
		# 		current_input, W,
		# 		tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
		# 		strides=[1, 2, 2, 1], padding='SAME'), b)
		output = tf.nn.dropout(output, keep_prob)
		current_input = output

	# %%
	# now have the reconstruction through the network
	y = current_input
	# cost function measures pixel-wise difference
	cost = tf.reduce_mean(tf.square(y - x_tensor))

	learning_rate = 5e-5
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	# %%
	return {'x': x, 'z': z, 'y': y, 'cost': cost, 'keep_prob':keep_prob, 'optimizer':optimizer}


# class auto_encoder(object):

# 	def __init__(self, args):
# 		self.args = args
# 		self.index = 0

# 	def shuffle(self, data):
# 		p = np.random.permutation(len(data))
# 		data = data[p]
# 		return data

# 	def next_batch(self, batch_size):

# 		data = self.data[self.index : batch_size + self.index]
		
# 		if self.index + batch_size >= len(self.data):
# 			self.index = 0
# 			self.data = self.shuffle(self.data)
# 		else:
# 			self.index = self.index + batch_size
# 		return data

# 	def add_model_vars(self):


# 		x = tf.placeholder(tf.float32, shape=[self.args.batch, 1024 * 3])
# 		x_image = tf.transpose(tf.reshape(x, [self.args.batch, 3, 32, 32]), [0, 2, 3, 1])

# 		# x_gray = tf.image.rgb_to_grayscale(x_image)
# 		# print x_gray.get_shape()


# 		conv1 = Convolution2D(x_image, (32, 32), 3, 48, (3, 3), activation='relu')

# 		conv1_out = conv1.output()
# 		# print conv1_out.get_shape()

# 		pool1 = MaxPooling2D(conv1_out)
# 		pool1_out = pool1.output()
# 		# print pool1_out.get_shape()

# 		conv2 = Convolution2D(pool1_out, (16, 16), 48, 16, (3, 3), activation='relu')

# 		conv2_out = conv2.output()
# 		# print conv2_out.get_shape()
	
# 		pool2 = MaxPooling2D(conv2_out)
# 		pool2_out = pool2.output()
# 		# print pool2_out.get_shape()

# 		conv3 = Convolution2D(pool2_out, (8, 8), 16, 8, (3, 3), activation='relu')
# 		conv3_out = conv3.output()
# 		# print conv3_out.get_shape()

# 		pool3 = MaxPooling2D(conv3_out)
# 		pool3_out = pool3.output()
# 		# print pool3_out.get_shape()

# 		# 4 * 4 * 8
# 		encoded = tf.reshape(pool3_out, [-1, 4*4*8])
# 		# print encoded.get_shape()

# 		conv_t1 = Conv2Dtranspose(pool3_out, (8, 8), 8, 16, (3, 3), activation='relu')
		
# 		conv_t1_out = conv_t1.output()
# 		# print conv_t1_out.get_shape()
# 		conv_t2 = Conv2Dtranspose(conv_t1_out, (16, 16), 16, 48, (3, 3), activation='relu')
# 		conv_t2_out = conv_t2.output()

# 		conv_t3 = Conv2Dtranspose(conv_t2_out, (32, 32), 48, 3, (3, 3), activation='relu')
# 		decoded = conv_t3.output()
# 		print decoded.get_shape()

# 		# conv_last = Convolution2D(conv_t3_out, (32, 32), 16, 1, (3, 3), activation='relu')
# 		# decoded = conv_last.output()
		
# 		# print decoded.get_shape()

# 		# decoded = tf.reshape(decoded, [-1, 1*32*32])

# 		# cross_entropy = -1. * x * tf.log(tf.clip_by_value(decode,1e-10,1.0)) - (1. - x) * tf.log(tf.clip_by_value(1.-decode,1e-10,1.0))
# 		# loss = tf.reduce_mean(cross_entropy)
# 		loss = tf.reduce_mean((x_image - decoded) ** 2)

# 		op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 		self.x = x 
# 		self.loss = loss 
# 		self.op = op
# 		# self.decoded = decoded
# 		self.encoded = encoded

		
# 		# def conv2d(inp, name):
# 		# 	w = weights[name]
# 		# 	b = biases[name]
# 		# 	var = tf.nn.conv2d(inp, w, [1, 1, 1, 1], padding='SAME')
# 		# 	var = tf.nn.bias_add(var, b)
# 		# 	var = tf.nn.relu(var)
# 		# 	return var

# 		# def conv2d_transpose(inp, name, dropout_prob):
# 		# 	w = weights[name]
# 		# 	b = biases[name]

# 		# 	dims = inp.get_shape().dims[:3]
# 		# 	dims.append(w.get_shape()[-2]) # adpot channels from weights (weight definition for deconv has switched input and output channel!)
# 		# 	print dims
# 		# 	out_shape = tf.TensorShape(dims)
# 		# 	print out_shape

# 		# 	var = tf.nn.conv2d_transpose(inp, w, out_shape, strides=[1, 1, 1, 1], padding="SAME")
# 		# 	var = tf.nn.bias_add(var, b)
# 		# 	if not dropout_prob is None:
# 		# 		var = tf.nn.relu(var)
# 		# 		var = tf.nn.dropout(var, dropout_prob)
# 		# 	return var

# 		# def fc2d(inp, name):
# 		# 	w = weights[name]
# 		# 	b = biases[name]
# 		# 	var = tf.matmul(inp, w) + b
# 		# 	var = tf.nn.relu(var)
# 		# 	return var


# 		# weights = {
# 		# 	"conv1":	tf.Variable(tf.random_normal([3, 3,  1, 32], stddev=0.1)),
# 		# 	"conv2":	tf.Variable(tf.random_normal([3, 3, 32, 16], stddev=0.1)),
# 		# 	"conv3":	tf.Variable(tf.random_normal([3, 3, 16, 1], stddev=0.1)),
# 		# 	"fc1":		tf.Variable(tf.random_normal([32 * 32 * 1, 256], stddev=0.1)),
# 		# 	"fc2":		tf.Variable(tf.random_normal([256, 32 * 32 * 1], stddev=0.1)),
# 		# 	"deconv2":  tf.Variable(tf.random_normal([3, 3, 16, 1], stddev=0.1)),
# 		# 	"deconv1":  tf.Variable(tf.random_normal([3, 3,  1, 16], stddev=0.1)) }

# 		# biases = {
# 		# 	"conv1":	tf.Variable(tf.random_normal([32], stddev=0.1)),
# 		# 	"conv2":	tf.Variable(tf.random_normal([16], stddev=0.1)),
# 		# 	"conv3":	tf.Variable(tf.random_normal([1], stddev=0.1)),
# 		# 	"fc1":		tf.Variable(tf.random_normal([256], stddev=0.1)),
# 		# 	"fc2":		tf.Variable(tf.random_normal([32*32*1], stddev=0.1)),
# 		# 	"deconv2":  tf.Variable(tf.random_normal([16], stddev=0.1)),
# 		# 	"deconv1":  tf.Variable(tf.random_normal([ 3], stddev=0.1)) }


# 		# ## Build Miniature CEDN
# 		# x = tf.placeholder(tf.float32, shape=[self.args.batch, 1024 * 3])
# 		# x_image = tf.transpose(tf.reshape(x, [self.args.batch, 3, 32, 32]), [0, 2, 3, 1])
# 		# y = tf.placeholder(tf.float32, shape=[self.args.batch, 1024 * 3])
# 		# y_image = tf.transpose(tf.reshape(y, [self.args.batch, 3, 32, 32]), [0, 2, 3, 1])
# 		# p = tf.placeholder(tf.float32)

# 		# x_gray = (x_image[:,:,:,0]*0.2989 + x_image[:,:,:,1] * 0.5870 + x_image[:,:,:,2]*0.1140)[:,:,:, None]
# 		# y_gray = (y_image[:,:,:,0]*0.2989 + y_image[:,:,:,1] * 0.5870 + y_image[:,:,:,2]*0.1140)[:,:,:, None]
# 		# conv1								   = conv2d(x_gray, "conv1")

# 		# conv2								   = conv2d(conv1, "conv2")

# 		# conv3								   = conv2d(conv2, "conv3")

# 		# conv3_flatten = tf.reshape(conv3, [self.args.batch, 32 * 32 * 1])

# 		# fc1 								  = fc2d(conv3_flatten, "fc1")
# 		# fc2  								  = fc2d(fc1, "fc2")

# 		# fc2_unflat = tf.reshape(fc2, [self.args.batch, 32, 32, 1])

		

# 		# # feature_dim = tf.reshape(conv3, [self.args.batch, 32 * 32 * 1])

# 		# deconv2								 = conv2d_transpose(fc2_unflat, "deconv2", p)

# 		# deconv1								 = conv2d_transpose(deconv2, "deconv1", None)

# 		# # loss		= tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(deconv1, y))
# 		# loss		= tf.reduce_sum((deconv1 - y_gray) ** 2)
# 		# op   = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# 		# self.x = x
# 		# self.y = y
# 		# self.p = p
# 		# self.loss = loss
# 		# self.op = op
# 		# self.feature_dim = fc1


# 	def fit(self, data):
# 		self.data = data
# 		self.add_model_vars()

# 		self.sess = tf.Session()
# 		self.sess.run(tf.initialize_all_variables())

# 		for epoch in range(70):
# 			total_batch = int(np.ceil(len(self.data) / float(self.args.batch)))

# 			pbar = pb.ProgressBar(widgets=["auto:", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
# 			total_loss = 0.0
# 			for j in xrange(total_batch):
# 				# print j
# 				pbar.update(j)
# 				batch = self.next_batch(self.args.batch)
# 				_, loss = self.sess.run([self.op, self.loss], feed_dict={self.x:batch})
# 				total_loss += loss
# 			pbar.finish()
# 			total_loss /= total_batch * 32 * 32 

# 			print >> sys.stderr, "epoch {}  [loss: {:6.6}]".format(epoch, total_loss)

# 	def encode(self, data):
# 		batch_size = 100
# 		total_batch = int(np.ceil(data.shape[0] / float(batch_size)))

# 		s_time = time.time()

# 		encodes_l = []
# 		for i in range(total_batch):
# 			batch = data[i*batch_size : (i+1)*batch_size]
# 			encode = self.sess.run(self.encoded, feed_dict={self.x:batch})
# 			encodes_l.append(encode)

# 		encodes = np.concatenate(encodes_l, axis=0)
		

# 		return encodes




class auto_encoder(object):

	def __init__(self, args):
		self.args = args
		self.index = 0

	def shuffle(self, data):
		p = np.random.permutation(len(data))
		data = data[p]
		return data

	def next_batch(self, batch_size):

		data = self.data[self.index : batch_size + self.index]
		
		if self.index + batch_size >= len(self.data):
			self.index = 0
			self.data = self.shuffle(self.data)
		else:
			self.index = self.index + batch_size
		return data


	def fit(self, data):

		# self.mean_img = np.mean(data, axis=1)[:, None]
		# self.std_img = np.std(data, axis=1)[:, None]
		self.mean_img = np.mean(data, axis=0)
		self.std_img = np.std(data, axis=0)
		# normalize
		self.data = (data - self.mean_img) / self.std_img  
		self.data = np.exp(self.data)
		# self.data = self.data / np.sum(self.data, axis=1)[:,None]
		
		

		

		

		# %%
		# We create a session to use the graph
		if self.args.auto == 0:
			self.ae = autoencoder()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.sess.run(tf.initialize_all_variables())

		# %%
		# Fit all training data
		batch_size = 10
		n_epochs = self.args.auto_epochs
		
		total_batch = int(np.ceil(len(self.data) / float(self.args.batch)))
		for epoch_i in range(n_epochs):
			
			pbar = pb.ProgressBar(widgets=["auto:", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
			for batch_i in range(total_batch):
				pbar.update(batch_i)
				batch_xs = self.next_batch(self.args.batch)
				# train = np.array([img - mean_img for img in batch_xs])
				loss = self.sess.run(self.ae['optimizer'], feed_dict={self.ae['x']: batch_xs, self.ae['keep_prob']:1.0})
				
			pbar.finish()
			print >> sys.stderr, "Epoch {}: loss: {}".format(epoch_i, self.eval(self.data))


		# self.add_model_vars()

		# self.sess = tf.Session()
		# self.sess.run(tf.initialize_all_variables())

		# for epoch in range(70):
		# 	total_batch = int(np.ceil(len(self.data) / float(self.args.batch)))

		# 	pbar = pb.ProgressBar(widgets=["auto:", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
		# 	total_loss = 0.0
		# 	for j in xrange(total_batch):
		# 		# print j
		# 		pbar.update(j)
		# 		batch = self.next_batch(self.args.batch)
		# 		_, loss = self.sess.run([self.op, self.loss], feed_dict={self.x:batch})
		# 		total_loss += loss
		# 	pbar.finish()
		# 	total_loss /= total_batch * 32 * 32 

		# 	print >> sys.stderr, "epoch {}  [loss: {:6.6}]".format(epoch, total_loss)
	def eval(self, data):
		batch_size = 50
		total_batch = int(np.ceil(data.shape[0] / float(batch_size)))

		total_loss = 0.0
		for i in range(total_batch):
			data_batch = data[i*batch_size : (i+1)*batch_size]
			loss = self.sess.run(self.ae['cost'], feed_dict={self.ae['x']: data_batch, self.ae['keep_prob']:1.0})
			
			total_loss += loss * len(data_batch)
		total_loss /= data.shape[0]
		return total_loss

	def dump_model(self, path):

		meta = {'mean':self.mean_img, 'std':self.std_img}
		pickle.dump(meta, open(path,"w"))
		print >> sys.stderr, "save autoencoder meta in path", path

		
		save_path = self.saver.save(self.sess, path+".weight", write_meta_graph=False)

		print >> sys.stderr, "save autoencoder weights in path", save_path

					

	def load_model(self, path):
		meta = pickle.load(open(path, "r"))
		self.mean_img = meta['mean']
		self.std_img = meta['std']
		self.ae = autoencoder()
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		self.saver.restore(self.sess, path+".weight")
		print >> sys.stderr, "restore autoencoder from", path



	def encode(self, data):

		# self.mean_img = np.mean(data, axis=1)[:, None]
		# self.std_img = np.std(data, axis=1)[:, None]
		data = (data - self.mean_img) / self.std_img  # normalize
		data = np.exp(data)
		# data = data / np.sum(data, axis=1)[:,None]

		batch_size = 50
		total_batch = int(np.ceil(data.shape[0] / float(batch_size)))

		s_time = time.time()

		encodes_l = []
		for i in range(total_batch):
			batch = data[i*batch_size : (i+1)*batch_size]
			encode = self.sess.run(self.ae['z'], feed_dict={self.ae['x']:batch, self.ae['keep_prob']:1.0})
			encodes_l.append(encode)

		encodes = np.concatenate(encodes_l, axis=0)
		
		return encodes






class cnn(object):

	def __init__(self, args):
		self.args = args
		self.auto_encoder = auto_encoder(self.args)
		self.load_data()
		self.index = 0
		
		
		

	def load_data(self):

		if self.args.mode % 2 == 0:

			train_label_data = np.array(pickle.load(open(self.args.label, "r")))
			train_unlabel_data = np.array(pickle.load(open(self.args.unlabel, "r")))
			test_data = pickle.load(open(self.args.test, "r"))


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

			
			self.train_label_data, self.train_label_labels = self.shuffle(self.train_label_data, self.train_label_labels)
			
		
			self.test_ids = np.array(test_data['ID'])
			self.test_data = np.array(test_data['data'])

		
		

			self.train_unlabel_data = train_unlabel_data

			if self.args.auto == 0:
				print >> sys.stderr, "fit auto_encoder"
				data = np.concatenate([self.train_label_data, self.train_unlabel_data, self.test_data], axis=0)
				self.auto_encoder.fit(data)
				self.auto_encoder.dump_model(self.args.auto_model)
			else:
				self.auto_encoder.load_model(self.args.auto_model)
				data = np.concatenate([self.train_label_data, self.train_unlabel_data, self.test_data], axis=0)
				self.auto_encoder.fit(data)
				self.auto_encoder.dump_model(self.args.auto_model)

			self.train_label_data = self.auto_encoder.encode(self.train_label_data)
			self.train_unlabel_data = self.auto_encoder.encode(self.train_unlabel_data)

			# with open("vectors", "w") as f:
			# 	for i, d in enumerate(self.train_label_data):
			# 		print >> f, " ".join([str(self.train_label_labels[i])]+[str(dd) for dd in d])
			# 	for d in self.train_unlabel_data:
			# 		print >> f, " ".join(["u"]+[str(dd) for dd in d])
		


			self.val_label_data = self.auto_encoder.encode(self.val_label_data)
			self.test_data = self.auto_encoder.encode(self.test_data)

			self.train_data = self.train_label_data
			self.train_labels = self.train_label_labels
			# self.train_data = np.concatenate([self.train_label_data]+unlabel_data_l, axis=0)
			# self.train_labels = np.concatenate([self.train_label_labels]+unlabel_labels_l, axis=0)

		elif self.args.mode > 0:

			self.auto_encoder.load_model(self.args.auto_model)
			test_data = pickle.load(open(self.args.test, "r"))
			self.test_ids = np.array(test_data['ID'])
			self.test_data = np.array(test_data['data'])
			self.test_data = self.auto_encoder.encode(self.test_data)



	def shuffle(self, data, labels):
		p = np.random.permutation(len(data))
		data = data[p]
		labels = labels[p]
		return data, labels

	def next_batch(self, batch_size):

		data = self.train_data[self.index : batch_size + self.index]

		# data = self.sess.run(self.distorted_image, feed_dict={self.image:data})

		labels = self.train_labels[self.index : batch_size + self.index]
		if self.index + batch_size >= len(self.train_data):
			self.index = 0
			self.train_data, self.train_labels = self.shuffle(self.train_data, self.train_labels)
		else:
			self.index = self.index + batch_size
		return data, labels



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

		x = tf.placeholder(tf.float32, shape=[None, 2*2*48])
		y = tf.placeholder(tf.int64, shape=[None])
		keep_prob = tf.placeholder(tf.float32)


		W_fc1 = weight_variable([2*2*48, 256], 0.01)
		b_fc1 = bias_variable([256])
		h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = weight_variable([256, 64], 0.01)
		b_fc2 = bias_variable([64])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
		h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

		W_fc3 = weight_variable([64, 10], 0.01)
		b_fc3 = bias_variable([10])

		y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
	
		# x = tf.placeholder(tf.float32, shape=[None, 1024 * 3])
		# y = tf.placeholder(tf.int64, shape=[None])

		# keep_prob = tf.placeholder(tf.float32)


		# x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])
		

		# W_conv1 = weight_variable([3, 3, 3, 24], 0.01)
		# b_conv1 = bias_variable([24])
		
		# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		# h_pool1 = max_pool_2x2(h_conv1)
		# h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob)
		

		# W_conv2 = weight_variable([3, 3, 24, 48], 0.01)
		# b_conv2 = bias_variable([48])
		
		# h_conv2 = tf.nn.relu(conv2d(h_pool1_drop, W_conv2) + b_conv2)
		# h_pool2 = max_pool_2x2(h_conv2)
		# h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)



		# W_conv3 = weight_variable([3, 3, 48, 48], 0.01)
		# b_conv3 = bias_variable([48])

		# h_conv3 = tf.nn.relu(conv2d(h_pool2_drop, W_conv3) + b_conv3)
		# h_pool3 = max_pool_2x2(h_conv3)
		# h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)


		# W_fc1 = weight_variable([4 * 4 * 48, 256], 0.01)
		# b_fc1 = bias_variable([256])
		# h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 4 * 4 * 48])
		# h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# W_fc2 = weight_variable([256, 10], 1.0)
		# b_fc2 = bias_variable([10])

		# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y))

		# train_step = tf.train.AdamOptimizer(self.args.lr).minimize(cross_entropy)
		train_step = tf.train.AdagradOptimizer(self.args.lr).minimize(cross_entropy)
		predicts = tf.argmax(y_conv, 1)

		probs = tf.nn.softmax(y_conv)
		self.probs = tf.reduce_max(probs, 1)

		correct_prediction = tf.equal(predicts, y)
		acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		self.x = x 
		self.y = y 
		self.keep_prob = keep_prob
		self.train_step = train_step
		self.acc = acc 
		self.cross_entropy = cross_entropy
		self.predicts = predicts



	def train(self):
		with tf.Graph().as_default():



			self.add_model_vars()
			


			# self.distorted_image, self.image = self.add_image_vars()

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

			val_th = 0.60


			for epoch in range(self.args.epochs):

				total_batch = int(np.ceil(len(self.train_data) / float(self.args.batch)))

				pbar = pb.ProgressBar(widgets=["train:", pb.FileTransferSpeed(unit="batchs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=total_batch).start()
				for j in xrange(total_batch):
					# print j
					pbar.update(j)
					batchx, batchy = self.next_batch(self.args.batch)
					self.sess.run(self.train_step, feed_dict={self.x:batchx, self.y:batchy, self.keep_prob:0.7})
				pbar.finish()

				# val_acc, val_loss  = self.sess.run([acc, cross_entropy], feed_dict={x:self.val_label_data, y:self.val_label_labels, keep_prob:1.0})
				val_acc, val_loss  = self.eval(self.val_label_data, self.val_label_labels)
				
				# if val_loss < best_val_loss:
				# 	best_val_loss = val_loss
				# 	best_val_epoch = epoch

		
				

				# if epoch - best_val_epoch > self.args.early_stop:
				# 	stopped = epoch
				if val_acc > best_val_acc and val_acc > val_th:
					best_val_acc = val_acc
					best_val_epoch = epoch
					tmp_model = "[l_acc={:.4}][v_acc={:.4}][epochs={}]{}".format(label_acc, val_acc, epoch, self.args.prints)
					save_path = self.saver.save(self.sess, "model_{}".format(tmp_model),  write_meta_graph=False)
					print >> sys.stderr, "save tmp model in path", save_path



				if epoch - best_val_epoch > self.args.early_stop:
					stopped = epoch


				if epoch % 1 == 0 or epoch +1 == self.args.epochs or epoch - best_val_epoch > self.args.early_stop:
					print >> sys.stderr, "evaluating"
					# label_acc, label_loss = self.sess.run([acc, cross_entropy], feed_dict={x:self.train_label_data, y:self.train_label_labels, keep_prob:1.0})
					# train_acc, train_loss = self.sess.run([acc, cross_entropy], feed_dict={x:self.train_data, y:self.train_labels, keep_prob:1.0})
					label_acc, label_loss = self.eval(self.train_label_data, self.train_label_labels)
					train_acc, train_loss = self.eval(self.train_data, self.train_labels)
					
					print >> sys.stderr, "step {}: val_th:{}, label_size:{}, total_size:{}".format(epoch, val_th, len(self.train_label_data), len(self.train_data))
					print >> sys.stderr, "  [train_acc: {:6.6}, train_loss: {:6.6}]".format(train_acc, train_loss)
					print >> sys.stderr, "  [label_acc: {:6.6}, label_loss: {:6.6}]".format(label_acc, label_loss)
					print >> sys.stderr, "  [valid_acc: {:6.6}, valid_loss: {:6.6}]".format(val_acc, val_loss)
					if epoch - best_val_epoch > self.args.early_stop:
						stopped = epoch
						break



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

	def eval(self, data, labels):
		batch_size = 5000
		total_batch = int(np.ceil(data.shape[0] / float(batch_size)))

		total_loss = 0.0
		total_acc = 0.0
		for i in range(total_batch):
			data_batch = data[i*batch_size : (i+1)*batch_size]
			labels_batch = labels[i*batch_size : (i+1)*batch_size]
			acc, loss = self.sess.run([self.acc, self.cross_entropy], feed_dict={self.x:data_batch, self.y:labels_batch, self.keep_prob:1.0})
			total_acc += acc * len(data_batch)
			total_loss += loss * len(data_batch)
		total_acc /= data.shape[0]
		total_loss /= data.shape[0]
		return total_acc, total_loss

	def test(self):
		with tf.Graph().as_default():
			self.add_model_vars()

			self.saver = tf.train.Saver()
			self.sess = tf.Session()

			self.saver.restore(self.sess, self.args.model)
			print >> sys.stderr, "restore model from ", self.args.model


			batch_size = 100
			total_batch = int(np.ceil(self.test_data.shape[0] / float(batch_size)))
			preds_l = []
			for i in range(total_batch):
				data_batch = self.test_data[i*batch_size : (i+1)*batch_size]
				preds_batch = self.sess.run(self.predicts, feed_dict={self.x:data_batch, self.keep_prob:1.0})
				preds_l.append(preds_batch)


			# test_preds = self.sess.run(predicts, feed_dict={x:self.test_data, keep_prob:1.0})
			# print test_preds.shape
			test_preds = np.concatenate(preds_l, axis=0)


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

	if args.mode > 0:
		start_time = time.time()

		model.test()

		print >> sys.stderr, "testing time: {}".format(time.time()-start_time)




if __name__ == "__main__":
	main()