
import argparse
import time
import sys
import csv 
import datetime



def arg_parse():

	parser = argparse.ArgumentParser()

	
	parser.add_argument('--tr', default='../input_data/clicks_train.csv', type=str)
	parser.add_argument('--te', default='../input_data/clicks_test.csv', type=str)
	parser.add_argument('--tr_distrib', default='./clicks_train.distrib', type=str)
	parser.add_argument('--te_distrib', default='./clicks_test.distrib', type=str)
	parser.add_argument('--event', default='../input_data/events.csv', type=str)

	args = parser.parse_args()

	return args

def print_info(*info_list):
	print >> sys.stderr, " ".join([str(info) for info in info_list])


def get_distrib(args):

	event_dict = {}
	# display_id, uuid, document_id, timestamp, platform, geo_location

	print_info("Events")
	with open(args.event, "r") as f:
		events = csv.reader(f)
		event_header = ['day', 'hour']
		next(events)

		for i, row in enumerate(events):

			tlist = []

			# hour, day
			timestamp = int(row[3]) + 1465876799998
			date = datetime.datetime.utcfromtimestamp(timestamp / 1000.0)
			day = date.day - 14
			hour = date.hour
			if hour >= 4:
				hour = hour - 4
			else:
				hour = (hour - 4) + 24
				day = day - 1
			# hour = (timestamp // (3600 * 1000)) % 24 
			# day = timestamp // (3600 * 24 * 1000) 

			tlist.extend([str(day), str(hour)])


			event_dict[int(row[0])] = tlist[:] 
			if i % 1000000 == 0:
				print_info("Events : ", i)

		print_info("Events Size: ", len(event_dict))
	del events

	
	with open(args.tr, "r") as f:
		with open(args.tr_distrib, "w") as p:
			reader = csv.DictReader(f)
			fieldnames = ['display_id', 'ad_id', 'day', 'hour', 'clicked']
			writer = csv.DictWriter(p, fieldnames=fieldnames)
			writer.writeheader()
			for t, row in enumerate(reader):
				display_id = int(row['display_id'])
				ad_id = int(row['ad_id'])
				clicked = int(row['clicked'])
				day = event_dict[display_id][0]
				hour = event_dict[display_id][1]
				out = {'display_id': display_id,
					   'ad_id': ad_id,
					   'day': day,
					   'hour': hour,
					   'clicked': clicked}
				writer.writerow(out)

	# not need to get test distribution
	# with open(args.te, "r") as f:
	# 	with open(args.te_distrib, "w") as p:
	# 		reader = csv.DictReader(f)
	# 		fieldnames = ['day', 'hour']
	# 		writer = csv.DictWriter(p, fieldnames=fieldnames)
	# 		writer.writeheader()
	# 		for t, row in enumerate(reader):
	# 			display_id = int(row['display_id'])

	# 			writer.writerow({'day': event_dict[display_id][0], 'hour': event_dict[display_id][1]})



def main():

	args = arg_parse()

	start_time = time.time()

	get_distrib(args)

	print >> sys.stderr, "get_distrib time: {}".format(time.time()-start_time)


if __name__ == "__main__":
	main()
	