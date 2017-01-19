# full leak can be extracted with 10 GB memory
# but you can extract a subset of leak if you have less memory
# pypy leak.py takes 30 mins

import csv
import os

memory = 20 # stands for 10GB, write your memory here
limit = 114434838 / 10 * memory 

topic_set = set()
doc_topic_dict = {}
# document_id,topic_id,confidence_level

for c, row in enumerate(csv.DictReader(open("../input_data/documents_topics.csv"))):
    doc_id = int(row['document_id'])
    if doc_id not in doc_topic_dict:
        doc_topic_dict[doc_id] = set()
    doc_topic_dict[doc_id].add(int(row['topic_id']))
    topic_set.add(row['topic_id'])
    

leak = {}
c = 0
for c,row in enumerate(csv.DictReader(open('../input_data/promoted_content.csv'))):
    if row['document_id'] != '':
        doc_id = int(row['document_id'])
        if doc_id in doc_topic_dict:
            for topic in doc_topic_dict[doc_id]:
                leak[topic] = 1
        else:
            c += 1
print(len(leak))
print("missing:", c, len(doc_topic_dict))
print("topic:", len(topic_set))

count = 0
filename = '../input_data/page_views.csv'
# uuid,document_id,timestamp,platform,geo_location,traffic_source
# 1fd5f051fba643,120,31905835,1,RS,2
# 8557aa9004be3b,120,32053104,1,VN>44,2
# filename = '../input/page_views_sample.csv' # comment this out locally
for c,row in enumerate(csv.DictReader(open(filename))):
    # if count>limit:
	   #  break
    if c%1000000 == 0:
        print (c,count)
    doc_id = int(row['document_id'])
    if doc_id not in doc_topic_dict:
	    continue
    for topic_id in doc_topic_dict[doc_id]:
        if topic_id not in leak:
            continue
        if leak[topic_id] == 1:
            leak[topic_id] = set()
        lu = len(leak[topic_id])
        leak[topic_id].add(row['uuid'])
        if lu != len(leak[topic_id]):
            count += 1

    # cate_id = doc_cate_dict[doc_id]
    # if leak[cate_id] == 1:
    #     leak[cate_id] = set()
    # lu = len(leak[cate_id])
    # leak[cate_id].add(row['uuid'])
    # if lu != len(leak[cate_id]):
    #     count += 1
    # if leak[row['document_id']]==1:
	   #  leak[row['document_id']] = set()
    # lu = len(leak[row['document_id']])
    # leak[row['document_id']].add(row['uuid'])
    # if lu!=len(leak[row['document_id']]):
	   #  count+=1
fo = open('../input_data/leak_topic.csv','w')
fo.write('topic_id,uuid\n')
for i in leak:
    if leak[i]!=1:
	    tmp = list(leak[i])
	    fo.write('%s,%s\n'%(i,' '.join(tmp)))
	    del tmp
fo.close()	