import csv

from csv import DictReader


import sys

import argparse
import time
import datetime

from array import array

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--click_tr', default='./clicks_split_train.csv', type=str)
	parser.add_argument('--click_va', default='./clicks_valid.csv', type=str)
	parser.add_argument('--click_te', default='../input_data/clicks_test.csv', type=str)
	parser.add_argument('--tr_fm', default='./clicks_split_train.fm', type=str)
	parser.add_argument('--va_fm', default='./clicks_valid.fm', type=str)
	parser.add_argument('--te_fm', default='./clicks_test.fm', type=str)
	parser.add_argument('--ad_info', default='../input_data/promoted_content.csv', type=str)
	parser.add_argument('--event', default='../input_data/events.csv', type=str)
	parser.add_argument('--leak', default='../input_data/leak.csv', type=str)
	parser.add_argument('--leak_cate', default='../input_data/leak_cate.csv', type=str)
	parser.add_argument('--leak_topic', default='../input_data/leak_topic.csv', type=str)
	parser.add_argument('--doc_topic', default='../input_data/documents_topics.csv', type=str)
	parser.add_argument('--doc_cate', default='../input_data/documents_categories.csv', type=str)
	parser.add_argument('--doc_meta', default='../input_data/documents_meta.csv', type=str)
	parser.add_argument('--num_features', default=12, type=int)
	parser.add_argument('--mode', default=0, type=int)
	# mode 0 = use valid, mode 1 = not use valid
	args = parser.parse_args()

	return args

def print_info(*info_list):
	print >> sys.stderr, " ".join([str(info) for info in info_list])

num_missing = {i:0 for i in range(10)}
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

day_train = set()
day_valid = set()

leak_count = [0]*10

def extract_dump(args):

	global num_missing

	# or leak may too big
	csv.field_size_limit(sys.maxsize)

	ad_info_dict = {}

	print_info("Content")
	with open(args.ad_info, "r") as f:
		ad_info = csv.reader(f)
		#prcont_header = (prcont.next())[1:]
		ad_info_header = next(ad_info)[1:]
		ad_info_header = ['ad_info_'+a for a in ad_info_header]
	
		# ad_id: ['document_id', 'campaign_id', 'advertiser_id']
		for i, row in enumerate(ad_info):
			ad_info_dict[int(row[0])] = row[1:]
			if i % 100000 == 0:
				print_info("Content: ", i)
			
		print_info("Content Size: ", len(ad_info_dict))
	del ad_info

	event_dict = {}
	# display_id, uuid, document_id, timestamp, platform, geo_location
	# display_id: [uuid, doc_id, platform, loc, day, hour]

	print_info("Events")
	with open(args.event, "r") as f:
		events = csv.reader(f)
		#events.next()
		next(events)
		# event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma']
		# event_header = ['uuid', 'document_id', 'platform', 'loc_country', 'loc_state', 'loc_dma']
		# event_header = ['uuid', 'document_id', 'platform', 'loc']
		event_header = ['uuid', 'document_id', 'platform', 'loc', 'day', 'hour']
		# event_header = ['platform', 'loc']
		# loc only "US" get state level, others only country
		event_header = ['event_'+a for a in event_header]

		for i, row in enumerate(events):

			tlist = row[1:(1+2)] + row[4:5] # uuid, doc_id, platform
			# tlist = row[4:5]

			# loc
			loc = row[5].split('>')
			loc_list = []
			if len(loc) == 3:
				# full info
				loc_list.extend(loc[:])
			elif len(loc) == 2:
				# may country > loc_dma or country > loc_state
				if loc[1].isdigit():
					# country > loc_dma
					loc_list.extend([loc[0], "", loc[1]])
				else:
					# country > loc_state
					loc_list.extend( loc[:]+[""])
			elif len(loc) == 1:
				# assume is country

				loc_list.extend( loc[:]+["", ""])
			else:
				loc_list.append(["", "", ""])

			# only US to state-level
			if loc_list[0] == 'US':
				loc_label = "{}>{}".format(loc_list[0], loc_list[1])
			else:
				loc_label = loc_list[0]
			if loc_label == "": # no loc -> "UNK"
				loc_label = "UNK"
			tlist.append(loc_label)

			# hour, day
			# timestamp = int(row[3])
			# hour = (timestamp // (3600 * 1000)) % 24 
			# day = timestamp // (3600 * 24 * 1000) 
			timestamp = int(row[3]) + 1465876799998
			date = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)
			day = date.day - 14
			hour = date.hour
			# to UTC-4
			if hour >= 4:
				hour = hour - 4
			else:
				hour = hour - 4
				day = day - 1

			
			day = day % 7

			tlist.extend([str(day), str(hour)])


			event_dict[int(row[0])] = tlist[:] 
			if i % 1000000 == 0:
				print_info("Events : ", i)

			# tlist = row[1:3] + row[4:6]
			# loc = row[5].split('>')
			# if len(loc) == 3:
			# 	tlist.extend(loc[:])
			# elif len(loc) == 2:
			# 	tlist.extend( loc[:]+[''])
			# elif len(loc) == 1:
			# 	tlist.extend( loc[:]+['',''])
			# else:
			# 	tlist.append(['','',''])	
			# event_dict[int(row[0])] = tlist[:] 
			# if i % 1000000 == 0:
			# 	print_info("Events : ", i)
			
		print_info("Events Size: ", len(event_dict))
	del events

	print_info("Leak")
	# document_id, uuid
	# uuids
	leak_uuid_dict = {}
	with open(args.leak, "r") as f:
		leaks = csv.reader(f)
		next(leaks)
		leak_uuid_header = ['uuids']
		leak_uuid_header = ['leak_'+a for a in leak_uuid_header]
		for i, row in enumerate(leaks):
			doc_id = int(row[0])
			leak_uuid_dict[doc_id] = set(row[1].split(' '))
			if i % 1000 == 0:
				print_info("Leaks : ", i)

		print_info("Leaks Size: ", len(leak_uuid_dict))
	del leaks

	print_info("Leak_cate")
	# cate_id, uuid
	# uuids
	leak_cate_uuid_dict = {}
	with open(args.leak_cate, "r") as f:
		leak_cate = csv.reader(f)
		next(leak_cate)
		for i, row in enumerate(leak_cate):
			cate_id = int(row[0])
			leak_cate_uuid_dict[cate_id] = set(row[1].split(' '))
			if i % 20 == 0:
				print_info("Leak_cate : ", i)
		print_info("Leak_cate Size: ", len(leak_cate_uuid_dict))
	del leak_cate

	print_info("Leak_topic")
	# topic_id, uuid
	# uuids
	leak_topic_uuid_dict = {}
	with open(args.leak_topic, "r") as f:
		leak_topic = csv.reader(f)
		next(leak_topic)
		for i, row in enumerate(leak_topic):
			topic_id = int(row[0])
			leak_topic_uuid_dict[topic_id] = set(row[1].split(' '))
			if i % 30 == 0:
				print_info("Leak_topic : ", i)
		print_info("Leak_topic Size:", len(leak_topic_uuid_dict))
	del leak_topic



	print_info("Doc_categories")
	# document_id,category_id,confidence_level
	# doc_id: cate_id
	doc_cate_dict = {}
	with open(args.doc_cate, "r") as f:
		cates = csv.reader(f)
		next(cates) # header line
		doc_cate_header = ['doc_cate']
		for i, row in enumerate(cates):
			doc_id = int(row[0])
			cate_id = int(row[1])
			confidence_level = float(row[2])
			if doc_id not in doc_cate_dict:
				doc_cate_dict[doc_id] = set()
			if confidence_level > 0.3:
				doc_cate_dict[doc_id].add(cate_id)
			if i % 1000000 == 0:
				print_info("Doc_categories : ", i)
		print_info("Doc_categories Size: ", len(doc_cate_dict))
	del cates

	print_info("Doc_topics")
	# document_id,topic_id,confidence_level
	# doc_id: topic_id

	doc_topic_dict = {}
	with open(args.doc_topic, "r") as f:
		topics = csv.reader(f)
		next(topics) # header line
		doc_topic_header = ['doc_topic']
		for i, row in enumerate(topics):
			doc_id = int(row[0])
			topic_id = int(row[1])
			confidence_level = float(row[2])
			if doc_id not in doc_topic_dict:
				doc_topic_dict[doc_id] = set()
			if confidence_level > 0.3:
				doc_topic_dict[doc_id].add(topic_id)
			if i % 1000000 == 0:
				print_info("Doc_topics : ", i)
		print_info("Doc_topics Size: ", len(doc_topic_dict))
	del topics






	feature_dict = {'ad_info': ad_info_dict,		
					'event': event_dict,
					'leak': leak_uuid_dict,
					'doc_cate': doc_cate_dict,
					'doc_topic': doc_topic_dict,
					'leak_cate': leak_cate_uuid_dict,
					'leak_topic': leak_topic_uuid_dict
					}
	header_dict = {'ad_info_header': ad_info_header,
				   'event_header': event_header,
				   'leak_header': leak_uuid_header,
				   'doc_cate_header': doc_cate_header,
				   'doc_topic_header': doc_topic_header
				}

	global day_train
	global day_valid
	global leak_count

	dump(args, args.click_tr, args.tr_fm, feature_dict, header_dict, train=True)
	print day_train
	print leak_count
	if args.mode == 0:
		dump(args, args.click_va, args.va_fm, feature_dict, header_dict, train=False)
		print day_valid
		print leak_count
	else:
		dump(args, args.click_te, args.te_fm, feature_dict, header_dict, train=False)
		print leak_count
	print >> sys.stderr, "hash_size: ", len(hash_dict)
	print >> sys.stderr, "num_missing: ", num_missing
	
		
	print day_train, day_valid

def initial_hash_dict(feild_list):
	label = "UNK"
	for feild in feild_list:
		d_hash(feild+'_'+label, train=True)



def data(args, data_path, feature_dict, header_dict, train=True):

	D = 2 ** 20
	global num_missing
	global day_train
	global day_valid
	global leak_count


	feild_list = ['ad_id'] + header_dict['ad_info_header'] \
	+ header_dict['event_header'][1:] + header_dict['leak_header'] \
	+ ['ad_doc_cate', 'display_doc_cate'] \
	+ ['ad_doc_topic', 'display_doc_topic']
	# feild_list = header_dict['ad_info_header'] + header_dict['event_header']
	print_info("feild_list", feild_list)

	initial_hash_dict(feild_list)
	if train == True:
		print hash_dict

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
		for i, key in enumerate(row):
			if i == 1: # ad_id
				# x.append(abs(hash(key+'_'+row[key])) % D + 1) # 0 for missing value
				h = d_hash(key+'_'+row[key], train)
				if h == None:
					# x.append(0)
					# num_missing[i+0] += 1
					h = d_hash(key+'_'+"UNK", train)
					if h == None:
						print "error!"
					x.append(h)
				else:
					x.append(h)
				# x[feild_id_map[key]] = row[key]
				# x[feild_id_map[key]] = abs(hash(key+'_'+row[key])) % D

		# x for ad_info
		ad_info_dict = feature_dict['ad_info']
		ad_info_header = header_dict['ad_info_header']

		row = ad_info_dict.get(ad_id, [0]*len(ad_info_header))
		# print row
		ad_doc_id = -1
		for i, val in enumerate(row):
			if i == 0:
				ad_doc_id = int(val)
			if val == "":
				# missing value
				h = d_hash(ad_info_header[i]+'_'+"UNK", train)
				if h == None:
					print "error!"
				x.append(h)
			else:
				# x.append(abs(hash(ad_info_header[i]+'_'+val)) % D + 1) # 0 for missing value
				h = d_hash(ad_info_header[i]+'_'+val, train)
				if h == None:
					h = d_hash(ad_info_header[i]+'_'+"UNK", train)
					if h == None:
						print "error!"
					x.append(h)
				else:
					x.append(h)
				# x[feild_id_map[ad_info_header[i]]] = val
			# x[feild_id_map[ad_info_header[i]]] = abs(hash(ad_info_header[i]+'_'+val)) % D

		# x for event
		event_dict = feature_dict['event']
		event_header = header_dict['event_header']

		row = event_dict.get(display_id, [0]*len(event_header))
		# print row
		display_doc_id = -1
		for i, val in enumerate(row):
			if i == 0:
				uuid_val = val
			if i == 1:
				display_doc_id = int(val)
			if i not in [0]: # uuid too many missing value in test data: 0.83
				if val == "":
					h = d_hash(event_header[i]+'_'+"UNK", train)
					if h == None:
						print "error!"
					x.append(h)
				else:
					# x.append(abs(hash(event_header[i]+'_'+val)) % D + 1)
					h = d_hash(event_header[i]+'_'+val, train)
					if h == None:
						h = d_hash(event_header[i]+'_'+"UNK", train)
						if h == None:
							print "error!"
						x.append(h)
					else:
						x.append(h)
			if i == 4:
				if train:
					day_train.add(val)
				else:
					day_valid.add(val)
			# if val.isdigit():
			# 	x[feild_id_map[event_header[i]]] = val
			# else:
			# 	x[feild_id_map[event_header[i]]] = hash(val) % D
			# x[feild_id_map[event_header[i]]] = abs(hash(event_header[i]+'_'+val)) % D

		leak_dict = feature_dict['leak']
		leak_header = header_dict['leak_header']

		if (ad_doc_id in leak_dict) and (uuid_val in leak_dict[ad_doc_id]):
			x.append(d_hash("leak_found", train))
			leak_count[0] += 1
		else:
			x.append(d_hash("leak_not_found", train))
			leak_count[1] += 1


		leak_cate_dict = feature_dict['leak_cate']
		doc_cate_dict = feature_dict['doc_cate']

		leak_cate_flag = 0
		if ad_doc_id in doc_cate_dict:
			for cate in doc_cate_dict[ad_doc_id]:
				if cate in leak_cate_dict and uuid_val in leak_cate_dict[cate]:
					leak_cate_flag = 1

		if leak_cate_flag == 1:
			x.append(d_hash("leak_cate_found", train))
			leak_count[2] += 1
		else:
			x.append(d_hash("leak_cate_not_found", train))
			leak_count[3] += 1


		if display_doc_id in doc_cate_dict \
		and ad_doc_id in doc_cate_dict \
		and doc_cate_dict[display_doc_id] == doc_cate_dict[ad_doc_id]:
			x.append(d_hash("cate_are_the_same", train))
			leak_count[6] += 1
		else:
			x.append(d_hash("cate_are_not_the_same", train))
			leak_count[7] += 1

		leak_topic_dict = feature_dict['leak_topic']
		doc_topic_dict = feature_dict['doc_topic']

		leak_topic_flag = 0
		if ad_doc_id in doc_topic_dict:
			for topic_id in doc_topic_dict[ad_doc_id]:
				if topic_id in leak_topic_dict and uuid_val in leak_topic_dict[topic_id]:
					leak_topic_flag = 1
					
		if leak_topic_flag == 1:
			x.append(d_hash("leak_topic_found", train))
			leak_count[4] += 1
		else:
			x.append(d_hash("leak_topic_not_found", train))
			leak_count[5] += 1

		if display_doc_id in doc_topic_dict \
		and ad_doc_id in doc_topic_dict \
		and doc_topic_dict[display_doc_id] == doc_topic_dict[ad_doc_id]:
			x.append(d_hash("topic_are_the_same", train))
			leak_count[8] += 1
		else:
			x.append(d_hash("topic_are_not_the_same", train))
			leak_count[9] += 1



		# doc_cate_dict = feature_dict['doc_cate']
		# doc_topic_dict = feature_dict['doc_topic']
		# # display_doc_id
		# h = d_hash(doc_cate_dict[display_doc_id], train)
		# x.append(h)
		# h = d_hash(doc_topic_dict[display_doc_id], train)
		# x.append(h)



		# x_total = 
		 	# # display_id
		 	# ad_id 
		 	# ad_doc_id
		 	# ad_camp_id
		 	# ad_ader_id
		 	# # event_u_id 
		 	# event_doc_id
		 	# event_platform
		 	# event_loc
		 	# event_day
		 	# event_hour
		 	# # event_loc_country
		 	# # event_loc_state
		 	# # event_loc_dma
		 	# leak_or_not
		 	# leak_cate_or_not
		 	# leak_topic_or_not



		# check if x not len == 12
		if len(x) != args.num_features:
			print "len x error ", len(x)
		yield t, display_id, ad_id, x, y





def dump(args, data_path, output_data_path, feature_dict, header_dict, train=True):

	global num_missing
	prev_display_id = -1
	prev_count = 0
	with open(output_data_path, "w") as p:
		for t, display_id, ad_id, x, y in data(args, data_path, feature_dict, header_dict, train):
			# t is instance counter
			# display_id, ad_id... you know
			# x: feature dictionary
			# y: label (click)


			# convert to fm.py format "label f1 f2 f3 ...."

			if prev_display_id != -1 and display_id != prev_display_id:
				for i in range(prev_count, 12):
					out_list = ['0'] + [0] * args.num_features
					out = " ".join([str(o) for o in out_list]) + '\n'
					p.write(out)
				prev_count = 0
			prev_display_id = display_id



			out_list = [y] + x 
			out = " ".join([str(o) for o in out_list]) + '\n'
			p.write(out)
			prev_count += 1


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
				print_info("Processed : ", t, "hash_size: ", len(hash_dict))
				print_info("num_missing: ", num_missing)
		
		for i in range(prev_count, 12):
			out_list = ['0'] + [0] * args.num_features
			out = " ".join([str(o) for o in out_list]) + '\n'
			p.write(out)

def main():

	s = time.time()
	args = arg_parse()
	extract_dump(args)
	# data = extract(args)
	# dump(data)

	print >> sys.stderr, "time cost: {}".format(time.time() - s)


if __name__ == "__main__":
	main()