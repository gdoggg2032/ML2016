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
	parser.add_argument('--max_list_num', default=12, type=int)


	args = parser.parse_args()

	return args

def filtering(args):
	
	# build mapping
	# 0 map to 0, 1 map to 1
	int_text = []
	f = open(args.train, "r")
	for line in f:
		int_l = map(int, line.strip().split())
		int_text.extend(int_l)

	max_i = max(int_text)

	c = [0] * (max_i + 1)

	for i in int_text:
		c[i] += 1

	del int_text

	
	c_rev_map = [i for i in range(args.num_features+1)]
# 	{0: 0, 'ad_id_UNK': 1, 'ad_info_document_id_UNK': 2, 'ad_info_campaign_id_UNK': 3, 'ad_info_advertiser_id_UNK': 4, 'event_document_id_UNK': 5, 'e
# vent_platform_UNK': 6, 'event_loc_UNK': 7, 'event_day_UNK': 8, 'event_hour_UNK': 9, 'leak_uuids_UNK': 10, 'ad_doc_cate_UNK': 11, 'display_doc_cat
# e_UNK': 12, 'ad_doc_topic_UNK': 13, 'display_doc_topic_UNK': 14}

	for i, d in enumerate(c[(args.num_features+1):]):
		if d >= args.low:
			c_rev_map.append(i+args.num_features+1)

	c_map = [0] * (max_i + 1)
	for i, d in enumerate(c_rev_map):
		c_map[d] = i

	print >> sys.stderr, "origin size: ", len(c)
	print >> sys.stderr, "filtered size: ", len(c_rev_map)

	# train
	print >> sys.stderr, "filtering train"
	f = open(args.train, "r")
	p = open(args.train_f, "w")
	for line in f:

		l_f = map(lambda x:str(c_map[int(x)]), line.strip().split())
		out = " ".join(l_f)+'\n'
		p.write(out)

	# valid
	print >> sys.stderr, "filtering valid"
	f = open(args.valid, "r")
	p = open(args.valid_f, "w")
	for line in f:
		print line
		l_f = map(lambda x:str(c_map[int(x)]), line.strip().split())
		out = " ".join(l_f)+'\n'
		p.write(out)

	# test
	print >> sys.stderr, "filtering test"
	f = open(args.test, "r")
	p = open(args.test_f, "w")
	for line in f:
		l_f = map(lambda x:str(c_map[int(x)]), line.strip().split())
		out = " ".join(l_f)+'\n'
		p.write(out)

	

	print >> sys.stderr, "assign UNK train"
	data = np.fromstring(open(args.train_f, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, args.num_features+1)
	for i in range(1, args.num_features+1):
		data[data[:,i] == 0, i] = i
	np.savetxt(args.train_f, data, fmt='%d')

	print >> sys.stderr, "assign UNK valid"
	data = np.fromstring(open(args.valid_f, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, args.num_features+1)
	for i in range(1, args.num_features+1):
		data[data[:,i] == 0, i] = i
	np.savetxt(args.valid_f, data, fmt='%d')

	print >> sys.stderr, "assign UNK test"
	data = np.fromstring(open(args.test_f, "r").read(), \
			dtype=np.int32, sep=' ').reshape(-1, args.num_features+1)
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