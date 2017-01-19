import sys
import argparse
import csv 
from itertools import izip

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--test', default='./clicks_test.csv', type=str)
	parser.add_argument('--predict', default='./predict', type=str)
	parser.add_argument('--output', default='./output.csv', type=str)
	args = parser.parse_args()

	return args
args = arg_parse()

test_file = args.test 
predict_file = args.predict 
output_file = args.output



def print_info(*info_list):
	print >> sys.stderr, " ".join([str(info) for info in info_list])

test_reader = csv.DictReader(open(test_file, "r"))
test_header = ["display_id", "ad_id"]#next(test_reader)
print_info("test_header", test_header)

pred_f = open(predict_file, "r")

# writer = csv.writer(open(output_file, "w"))
# writer.writerow(test_header)
writer = open(output_file, "w")
writer.write(",".join(test_header)+'\n')

print_info("Converting")
prev_display_id = None
buckets = []
for i, (row, pred_line) in enumerate(izip(test_reader, pred_f)):
	# print row, pred_line
	display_id = row['display_id']
	ad_id = row['ad_id']
	prob = float(pred_line)
	if prev_display_id and prev_display_id != display_id:
		# dump 
		sorted_list = sorted(buckets, key=lambda x:x[1], reverse=True)
		# print sorted_list
		# writer.writerow([prev_display_id, " ".join([ad_idx for (ad_idx, _) in sorted_list])])
		writer.write(",".join([prev_display_id, " ".join([ad_idx for (ad_idx, _) in sorted_list])])+'\n')
		# for (ad_idx, _) in sorted_list:
		buckets = []

	prev_display_id = display_id
	buckets.append((ad_id, prob))

	if i % 1000000 == 0:
		print_info("Processed : ", i)

sorted_list = sorted(buckets, key=lambda x:x[1], reverse=True)
# writer.writerow([prev_display_id, " ".join([ad_idx for (ad_idx, _) in sorted_list])])
writer.write(",".join([prev_display_id, " ".join([ad_idx for (ad_idx, _) in sorted_list])])+'\n')


# predicts = [float(line) for line in open(predict_file, "r")]
# print >> sys.stderr, "predicts"

# df = pd.read_csv(test_file, dtype=str)
# print >> sys.stderr, "df"

# df['prob'] = predicts

# f = lambda x: x.sort_values(by='prob', ascending=False)

# df2 = df.groupby('display_id').apply(f).drop(['prob'], axis=1)
# print >> sys.stderr, "df2"

# # df3 = df2
# df4 = df2.groupby('display_id').agg(lambda x: ' '.join(x))
# print >> sys.stderr, "df4"


# df4.to_csv(output_file)
