import sys

import argparse
import time
import numpy as np

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='./clicks_split_train.fm', type=str)
	parser.add_argument('--valid', default='./clicks_valid.fm', type=str)
	parser.add_argument('--test', default='./clicks_test.fm', type=str)
	parser.add_argument('--train_f', default='./clicks_split_train.fmf', type=str)
	parser.add_argument('--valid_f', default='./clicks_valid.fmf', type=str)
	parser.add_argument('--test_f', default='./clicks_test.fmf', type=str)
	parser.add_argument('--low', default=5, type=int)
	parser.add_argument('--num_features', default=10, type=int)
	


	args = parser.parse_args()

	return args

def filtering(args):
	
	train_text = np.fromstring(open(args.train, "r").read(), \
			dtype=np.int32, sep=' ')

	count = np.bincount(train_text)

	c = np.where(count >= args.low)[0]

	x = np.concatenate([np.array([i for i in range(args.num_features+1)]), c])
	c_rev_map = np.unique(x)
	c_rev_map.sort()

	c_map = np.zeros(count.shape)
	for i, v in enumerate(c_rev_map):
		c_map[v] = i

	print >> sys.stderr, "origin size: ", len(count)
	print >> sys.stderr, "filtered size: ", len(c_rev_map)

	


	# train
	print >> sys.stderr, "filtering train"
	data = train_text
	data = c_map[data]
	# data = np.vectorize(lambda v:c_map[v])(train_text)
	del train_text
	print >> sys.stderr, "1"
	data = data.reshape(-1, args.num_features+1)

	print >> sys.stderr, "2"
	for i in range(1, args.num_features+1):
		data[data[:,i] == 0, i] = i

	print >> sys.stderr, "3"
	np.savetxt(args.train_f, data, fmt='%d')
	


	print >> sys.stderr, "filtering valid"
	data = np.fromstring(open(args.valid, "r").read(), \
			dtype=np.int32, sep=' ')
	# data = np.vectorize(lambda v:c_map[v])(data)
	data = c_map[data]
	data = data.reshape(-1, args.num_features+1)
	for i in range(1, args.num_features+1):
		data[data[:,i] == 0, i] = i
	np.savetxt(args.valid_f, data, fmt='%d')


	print >> sys.stderr, "filtering test"
	data = np.fromstring(open(args.test, "r").read(), \
			dtype=np.int32, sep=' ')
	# data = np.vectorize(lambda v:c_map[v])(data)
	data = c_map[data]
	data = data.reshape(-1, args.num_features+1)
	for i in range(1, args.num_features+1):
		data[data[:,i] == 0, i] = i
	np.savetxt(args.test_f, data, fmt='%d')








def main():

	s = time.time()
	args = arg_parse()
	filtering(args)


	print >> sys.stderr, "time cost: {}".format(time.time() - s)


if __name__ == "__main__":
	main()