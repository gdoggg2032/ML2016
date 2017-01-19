import argparse
import time
import sys
import csv 
import random
import pandas as pd
import numpy as np



def arg_parse():

	parser = argparse.ArgumentParser()

	
	parser.add_argument('--clicks_tr', default='../input_data/clicks_train.csv', type=str)
	parser.add_argument('--tr_distrib', default='./clicks_train.distrib', type=str)
	parser.add_argument('--te_distrib', default='./clicks_test.distrb', type=str)
	parser.add_argument('--train', default='./clicks_split_train.csv', type=str)
	parser.add_argument('--valid', default='./clicks_valid.csv', type=str)
	# parser.add_argument('--valid_rate', default=0.01, type=float)
	args = parser.parse_args()

	return args

def split(args):

	# split the last 2 day (11, 12) and equal number of the others

	df = pd.read_csv(args.tr_distrib)
	df_valid1 = df[df.day > 10]
	df_train1 = df[df.day <= 10]
	del df 
	num = len(df_valid1.display_id.unique()) # num of display_ids

	selected = np.random.choice(df_train1.display_id.unique(), num, replace = False)
	df_valid2 = df_train1[df_train1.display_id.isin(selected)]
	df_train2 = df_train1[~df_train1.display_id.isin(selected)]
	df_valid = df_valid2.append(df_valid1)
	df_valid = df_valid[['display_id', 'ad_id', 'clicked']]
	df_train = df_train2[['display_id', 'ad_id', 'clicked']]
	df_train.to_csv(args.train, index=False)
	df_valid.to_csv(args.valid, index=False)
	# valid = [] # 11, 12
	# train = []
	# with open(args.tr_distrib, "r") as f:
	# 	reader = csv.DictReader(f)
	# 	for t, row in enumerate(reader):
	# 		day = int(row['day'])
	# 		hour = int(row['hour'])
	# 		if day in [11, 12]:
	# 			valid.append(t)
	# 		else:
	# 			train.append(t)

	# random.shuffle(train)
	# # use the same number as last 2 days
	# num = len(valid)
	# valid, train = train[:num], train[num:]




	# with open(args.clicks_tr, "r") as f:
	# 	with open(args.train, "w") as p_train:
	# 		with open(args.valid, "w") as p_valid:
		
	# 			reader = csv.DictReader(f)
	# 			fieldnames = ['display_id', 'ad_id', 'clicked']
	# 			writer_train = csv.DictWriter(p_train, fieldnames=fieldnames)
	# 			writer_valid = csv.DictWriter(p_valid, fieldnames=fieldnames)
	# 			writer_train.writeheader()
	# 			writer_valid.writeheader()
	# 			for t, row in enumerate(reader):
	# 				display_id = int(row['display_id'])
	# 				ad_id = int(row['ad_id'])
	# 				clicked = int(row['clicked'])
	# 				out = {'display_id':display_id, 'ad_id':ad_id, 'clicked':clicked}
	# 				if t in valid:
	# 					writer_valid.writerow(out)
	# 				if t in train:
	# 					writer_train.writerow(out)



def main():

	args = arg_parse()

	start_time = time.time()

	split(args)

	print >> sys.stderr, "splitting time: {}".format(time.time()-start_time)


if __name__ == "__main__":
	main()
	