#result3[['clicked','display_id', 'ad_id', 'document_id', 'topic_id', 'category_id']]
import sys

f = open(sys.argv[1], "r")
feature_path = "./features/"

p = {
	'label':open(feature_path+"label.libffm", "w"),
	'display_id':open(feature_path+"display_id.libffm", "w"),
	'ad_id':open(feature_path+"ad_id.libffm", "w"),
	'doc_id':open(feature_path+"doc_id.libffm", "w"),
	'topic_id':open(feature_path+"topic_id.libffm", "w"),
	'category_id':open(feature_path+"category_id.libffm", "w")
	}

for line in f:
	ll = line.strip().split(" ")
	label = ll[0]
	features = ll[1:]
	p['label'].write(label+'\n')
	fea_d = {v: None for v in ['display_id','ad_id','doc_id','topic_id','category_id']}
	for fea in features:

		fea_type = fea.split(":")[0]
		if fea_type == '0': # display_id
			fea_d['display_id'] = fea
		elif fea_type == '1': # ad_id
			fea_d['ad_id'] = fea 
		elif fea_type == '2': # doc_id 
			fea_d['doc_id'] = fea 
		elif fea_type == '3': # topic_id 
			fea_d['topic_id'] = fea 
		elif fea_type == '4':
			fea_d['category_id'] = fea

	for v in ['display_id','ad_id','doc_id','topic_id','category_id']:
		fea_str = fea_d[v]
		if fea_str == None:
			p[v].write("\n")
		else:
			p[v].write(fea_str+'\n')


