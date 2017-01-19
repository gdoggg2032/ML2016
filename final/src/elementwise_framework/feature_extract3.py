import csv

from csv import DictReader


import sys

import argparse
import time

from array import array

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--click_tr', default='./clicks_train.csv', type=str)
	parser.add_argument('--click_te', default='./clicks_test.csv', type=str)
	parser.add_argument('--tr_fm', default='./clicks_train.fm', type=str)
	parser.add_argument('--te_fm', default='./clicks_test.fm', type=str)
	parser.add_argument('--ad_info', default='./promoted_content.csv', type=str)
	parser.add_argument('--event', default='./events.csv', type=str)
	parser.add_argument('--doc_topic', default='./documents_topics.csv', type=str)
	parser.add_argument('--doc_cate', default='./documents_categories.csv', type=str)
	parser.add_argument('--doc_meta', default='./documents_meta.csv', type=str)
	args = parser.parse_args()

	return args

def print_info(*info_list):
	print >> sys.stderr, " ".join([str(info) for info in info_list])

num_missing = 0
num_missing_from_display_id_or_ad_id = 0
hash_dict = {0:0} # 0 for missing value
def d_hash(item, train=True):
	if train:
		if item not in hash_dict:
			hash_dict[item] = len(hash_dict)
		return hash_dict[item]
	else:
		return hash_dict.get(item, None)
		# if item not in hash_dict:
		# 	return None
		# else:
		# 	return hash_dict[item]

def extract_dump(args):

	global num_missing
	global num_missing_from_display_id_or_ad_id

	ad_info_dict = {}

	print_info("Content")
	with open(args.ad_info) as f:
		ad_info = csv.reader(f)
		#prcont_header = (prcont.next())[1:]
		ad_info_header = next(ad_info)[1:]
		ad_info_header = ['ad_info_'+a for a in ad_info_header]
	
		# ['document_id', 'campaign_id', 'advertiser_id']
		for i, row in enumerate(ad_info):
			ad_info_dict[int(row[0])] = row[1:]
			if i % 100000 == 0:
				print_info("Content: ", i)
			
		print_info("Content Size: ", len(ad_info_dict))
	del ad_info

	event_dict = {}
	# uuid, doc_id, platform, geo_location, country, state, dma

	print_info("Events")
	with open(args.event) as f:
		events = csv.reader(f)
		#events.next()
		next(events)
		event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma']
		event_header = ['event_'+a for a in event_header]
		
		for i, row in enumerate(events):
			tlist = row[1:3] + row[4:6]
			loc = row[5].split('>')
			if len(loc) == 3:
				tlist.extend(loc[:])
			elif len(loc) == 2:
				tlist.extend( loc[:]+[''])
			elif len(loc) == 1:
				tlist.extend( loc[:]+['',''])
			else:
				tlist.append(['','',''])	
			event_dict[int(row[0])] = tlist[:] 
			if i % 1000000 == 0:
				print_info("Events : ", i)
			
		print_info("Events Size: ", len(event_dict))
	del events

	feature_dict = {'ad_info': ad_info_dict,		
					'event': event_dict	
					}
	header_dict = {'ad_info_header': ad_info_header,
				   'event_header': event_header
				}

	dump(args.click_tr, args.tr_fm, feature_dict, header_dict, train=True)
	dump(args.click_te, args.te_fm, feature_dict, header_dict, train=False)
	print >> sys.stderr, "hash_size: ", len(hash_dict), "num_missing", num_missing
	print >> sys.stderr, "num_missing_from_display_id_or_ad_id", num_missing_from_display_id_or_ad_id


def data(data_path, feature_dict, header_dict, train=True):

	D = 2 ** 20
	global num_missing
	global num_missing_from_display_id_or_ad_id

	# feild_id_map = {
	# 		'display_id':0,
	# 		'ad_id':1,
	# 		'uuid',2
	# 		'document_id':3,
	# 		'campaign_id':4, 
	# 		'advertiser_id':5,
	# 		'event_document_id':6, 
	# 		'platform':7,
	# 		'geo_location':8,
	# 		'loc_country':9,
	# 		'loc_state':10,
	# 		'loc_dma':11
	# }

	feild_list = ['display_id', 'ad_id'] + header_dict['ad_info_header'] + header_dict['event_header']
	# feild_list = header_dict['ad_info_header'] + header_dict['event_header']
	print_info("feild_list", feild_list)

	feild_id_map = {v:i for i, v in enumerate(feild_list)}
	print_info("feild_id_map", feild_id_map)



	for t, row in enumerate(DictReader(open(data_path, "r"))):
		
		# process id
		display_id = int(row['display_id'])
		ad_id = int(row['ad_id'])

		# process clicks
		y = 0
		if 'clicked' in row:
			if row['clicked'] == '1':
				y = 1
			del row['clicked']

		# x for display_id and ad_id
		x = []
		for key in row:
			# x.append(abs(hash(key+'_'+row[key])) % D + 1) # 0 for missing value
			h = d_hash(key+'_'+row[key], train)
			if h == None:
				x.append(0)
				num_missing += 1
				num_missing_from_display_id_or_ad_id += 1
			else:
				x.append(h)
			# x[feild_id_map[key]] = row[key]
			# x[feild_id_map[key]] = abs(hash(key+'_'+row[key])) % D

		# x for ad_info
		ad_info_dict = feature_dict['ad_info']
		ad_info_header = header_dict['ad_info_header']

		row = ad_info_dict.get(ad_id, [0]*len(ad_info_header))
		ad_doc_id = -1
		for i, val in enumerate(row):
			if i == 0:
				ad_doc_id = int(val)
			if val == "":
				# missing value
				x.append(0)
				num_missing += 1
			else:
				# x.append(abs(hash(ad_info_header[i]+'_'+val)) % D + 1) # 0 for missing value
				h = d_hash(ad_info_header[i]+'_'+val, train)
				if h == None:
					x.append(0)
					num_missing += 1
				else:
					x.append(h)
				# x[feild_id_map[ad_info_header[i]]] = val
			# x[feild_id_map[ad_info_header[i]]] = abs(hash(ad_info_header[i]+'_'+val)) % D

		# x for event
		event_dict = feature_dict['event']
		event_header = header_dict['event_header']

		row = event_dict.get(display_id, [0]*len(event_header))
		display_doc_id = -1
		for i, val in enumerate(row):
			if i == 0:
				uuid_val = val
			if i == 1:
				display_doc_id = int(val)

			if val == "":
				# missing value
				x.append(0)
				num_missing += 1
			else:
				# x.append(abs(hash(event_header[i]+'_'+val)) % D + 1)
				h = d_hash(event_header[i]+'_'+val, train)
				if h == None:
					x.append(0)
					num_missing += 1
				else:
					x.append(h)
			# if val.isdigit():
			# 	x[feild_id_map[event_header[i]]] = val
			# else:
			# 	x[feild_id_map[event_header[i]]] = hash(val) % D
			# x[feild_id_map[event_header[i]]] = abs(hash(event_header[i]+'_'+val)) % D

		# check if x not len == 12
		if len(x) != 12:
			print "len x error ", len(x)
		yield t, display_id, ad_id, x, y





def dump(data_path, output_data_path, feature_dict, header_dict, train=True):

	
	with open(output_data_path, "w") as p:
		for t, display_id, ad_id, x, y in data(data_path, feature_dict, header_dict, train):
			# t is instance counter
			# display_id, ad_id... you know
			# x: feature dictionary
			# y: label (click)


			# convert to fm.py format "label f1 f2 f3 ...."

			out_list = [y] + x 
			out = " ".join([str(o) for o in out_list]) + '\n'
			p.write(out)


			# convert to libffm format

			# out_list = [str(y)]
			# for key in sorted(x.keys()):
			# 	feild_id = key 
			# 	item_id = x[key]
			# 	feature_str = "{}:{}:1".format(feild_id, item_id)
			# 	out_list.append(feature_str)
			# out = " ".join(out_list) + '\n'
			# p.write(out)

			if t % 1000000 == 0:
				print_info("Processed : ", t, "hash_size: ", len(hash_dict), "num_missing: ", num_missing, "num_display_or_ad", num_missing_from_display_id_or_ad_id)


def main():

	s = time.time()
	args = arg_parse()
	extract_dump(args)
	# data = extract(args)
	# dump(data)

	print >> sys.stderr, "time cost: {}".format(time.time() - s)


if __name__ == "__main__":
	main()