import pandas as pd 
import sys
import numpy as np
import argparse
import time
from csv2ffm import csv2ffm

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--click_tr', default='./clicks_train.csv', type=str)
	parser.add_argument('--click_te', default='./clicks_test.csv', type=str)
	parser.add_argument('--tr', default='./clicks_train', type=str)
	parser.add_argument('--te', default='./clicks_test', type=str)
	parser.add_argument('--ad_info', default='./promoted_content.csv', type=str)
	parser.add_argument('--event', default='./events.csv', type=str)
	parser.add_argument('--doc_topic', default='./documents_topics.csv', type=str)
	parser.add_argument('--doc_cate', default='./documents_categories.csv', type=str)
	parser.add_argument('--doc_meta', default='./documents_meta.csv', type=str)
	args = parser.parse_args()

	return args

def print_info(task, part):
	print >> sys.stderr, "{}: {}".format(task, part)

def merge_data(df, args, train=True):

	task = "TRAIN" if train else "TEST"

	df1 = pd.read_csv(args.ad_info)
	print_info(task, "df1")

	result = pd.merge(df, df1, how='left', on='ad_id')
	print_info(task, 'result')
	del df, df1

	df2 = pd.read_csv(args.doc_topic)
	df2_2 = df2[df2.groupby(['document_id'])['confidence_level'].transform(max) == df2['confidence_level']]
	print_info(task, 'df2')
	del df2

	result2 = pd.merge(result, df2_2, how='left', on='document_id')
	print_info(task, 'result2')
	del result, df2_2

	df3 = pd.read_csv(args.doc_cate)
	df3_2 = df3[df3.groupby(['document_id'])['confidence_level'].transform(max) == df3['confidence_level']]
	print_info(task, 'df3')
	del df3

	result3 = pd.merge(result2, df3_2, how='left', on='document_id')
	print_info(task, 'result3')
	del result2, df3_2

	if not train:
		result3['clicked'] = 0

	df_f1 = result3[['clicked','display_id', 'ad_id', 'document_id', 'topic_id', 'category_id']]
	print_info(task, 'df_f1')
	del result3

	return df_f1





def extract_dump(args):

	# train
	df = pd.read_csv(args.click_tr) 

	df_merge = merge_data(df, args)

	csv2ffm(df_merge, './train.libffm')

	# del df_merge

	# test

	df = pd.read_csv(args.click_te) 

	df_merge = merge_data(df, args, train=False)

	csv2ffm(df_merge, './test.libffm')

	# del df_merge

# def extract(args):

# 	# train
# 	print >> sys.stderr, "read training data"
# 	df = pd.read_csv(args.click_tr)  

# 	df1 = pd.read_csv(args.ad_info)
# 	print >> sys.stderr, "train df1"

# 	result = pd.merge(df, df1, how='left', on='ad_id')
# 	print >> sys.stderr, "train result"
# 	del df

# 	df2 = pd.read_csv(args.doc_topic)

# 	df2_2 = df2[df2.groupby(['document_id'])['confidence_level'].transform(max) == df2['confidence_level']]

# 	# df3 = pd.concat([df2_2, pd.get_dummies(df2_2['topic_id'], sparse=True).rename(columns=lambda x: 'topic_'+str(x))], axis=1)
# 	result2 = pd.merge(result, df2_2, how='left', on='document_id')
# 	print >> sys.stderr, "train result2"
# 	del result
# 	del df2 

# 	df3 = pd.read_csv(args.doc_cate)

# 	df3_2 = df3[df3.groupby(['document_id'])['confidence_level'].transform(max) == df3['confidence_level']]

# 	del df3

# 	result3 = pd.merge(result2, df3_2, how='left', on='document_id')
# 	print >> sys.stderr, "train result3"
# 	del result2

# 	df_f1 = result3[['clicked', 'topic_id', 'category_id']]
# 	print >> sys.stderr, "train df_f1"
# 	del result3

# 	# df_f2 = pd.concat([df_f1, pd.get_dummies(df_f1['topic_id']).rename(columns=lambda x: 'topic_'+str(x))], axis=1)
# 	# print >> sys.stderr, "train df_f2"
# 	# del df_f1

# 	# df_f3 = pd.concat([df_f2, pd.get_dummies(df_f2['category_id']).rename(columns=lambda x: 'category_'+str(x))], axis=1)
# 	# print >> sys.stderr, "train df_f3"
# 	# del df_f2

# 	# df_f4 = df_f3.drop(['topic_id', 'category_id'], axis=1)
# 	# print >> sys.stderr, "train df_f4"
# 	# del df_f3

# 	print >> sys.stderr, "dump training data"
# 	df_f1.to_csv(args.tr, index=False, sep=" ", header=False)

# 	del df_f1

	

# 	# test

# 	print >> sys.stderr, "read testing data"

# 	df = pd.read_csv(args.click_te)  
# 	print >> sys.stderr, "test df"


# 	result = pd.merge(df, df1, how='left', on='ad_id')
# 	print >> sys.stderr, "test result"

# 	del df

# 	# df3 = pd.concat([df2_2, pd.get_dummies(df2_2['topic_id'], sparse=True).rename(columns=lambda x: 'topic_'+str(x))], axis=1)
# 	result2 = pd.merge(result, df2_2, how='left', on='document_id')
# 	print >> sys.stderr, "test result2"
# 	del result


# 	# df3_2 = df3[df3.groupby(['document_id'])['confidence_level'].transform(max) == df3['confidence_level']]

# 	result3 = pd.merge(result2, df3_2, how='left', on='document_id')
# 	print >> sys.stderr, "test result3"
# 	del result2

# 	result3['clicked'] = 0

# 	df_f1 = result3[['clicked', 'document_id', 'topic_id', 'category_id']]
# 	print >> sys.stderr, "test df_f1"
# 	del result3

# 	# df_f2 = pd.concat([df_f1, pd.get_dummies(df_f1['topic_id']).rename(columns=lambda x: 'topic_'+str(x))], axis=1)
# 	# print >> sys.stderr, "test df_f2"
# 	# del df_f1

# 	# df_f3 = pd.concat([df_f2, pd.get_dummies(df_f2['category_id']).rename(columns=lambda x: 'category_'+str(x))], axis=1)
# 	# print >> sys.stderr, "test df_f3"
# 	# del df_f2

# 	# df_f4 = df_f3.drop(['topic_id', 'category_id'], axis=1)
# 	# print >> sys.stderr, "test df_f4"
# 	# del df_f3

# 	print >> sys.stderr, "dump testing data"
# 	df_f1.to_csv(args.te, index=False, sep=" ", header=False)



# 	return 0



def main():

	s = time.time()
	args = arg_parse()
	extract_dump(args)
	# data = extract(args)
	# dump(data)

	print >> sys.stderr, "time cost: {}".format(time.time() - s)


if __name__ == "__main__":
	main()