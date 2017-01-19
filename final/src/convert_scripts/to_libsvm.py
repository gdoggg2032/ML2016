import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
max_topic = 301
max_category = 32406

p = open(output_file, "w") 

with open(input_file, "r") as f:
	for line in f:
		ll = line.strip().split()
		# print ll
		if len(ll) == 1:
			label = int(ll[0])
			doc_topic = 300
			doc_category = 32405
		elif len(ll) == 2:
			label = int(ll[0])
			doc_topic = 300
			doc_category = int(float(ll[1]))
		else:

			label = int(ll[0])
			doc_topic = int(float(ll[1]))
			doc_category = int(float(ll[2]))
		out = "{} {}:1 {}:1".format(label, doc_topic, doc_category+max_topic)
		print >> p, out