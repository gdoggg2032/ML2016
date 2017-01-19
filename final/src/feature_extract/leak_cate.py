# full leak can be extracted with 10 GB memory
# but you can extract a subset of leak if you have less memory
# pypy leak.py takes 30 mins

import csv
import os

memory = 20 # stands for 10GB, write your memory here
limit = 114434838 / 10 * memory 

cate_set = set()
real_cate_set = set()
doc_cate_dict = {}
# document_id,category_id,confidence_level
# only append first cate
for c, row in enumerate(csv.DictReader(open("../input_data/documents_categories.csv"))):
    doc_id = int(row['document_id'])
    if doc_id not in doc_cate_dict:
        doc_cate_dict[doc_id] = int(row['category_id'])
        real_cate_set.add(row['category_id'])
    cate_set.add(row['category_id'])

leak = {}
c = 0
for c,row in enumerate(csv.DictReader(open('../input_data/promoted_content.csv'))):
    if row['document_id'] != '':
        if int(row['document_id']) in doc_cate_dict:
            leak[doc_cate_dict[int(row['document_id'])]] = 1
        else:
            c += 1
print(len(leak))
print("missing:", c, len(doc_cate_dict))
print("cate:", len(cate_set), len(real_cate_set))

count = 0
filename = '../input_data/page_views.csv'
# uuid,document_id,timestamp,platform,geo_location,traffic_source
# 1fd5f051fba643,120,31905835,1,RS,2
# 8557aa9004be3b,120,32053104,1,VN>44,2
# filename = '../input/page_views_sample.csv' # comment this out locally
user_id_vocab = {}
user_list = []

for c,row in enumerate(csv.DictReader(open(filename))):
    # if count>limit:
	   #  break
    if c%1000000 == 0:
        print (c,count, len(user_list))
    doc_id = int(row['document_id'])
    if doc_id not in doc_cate_dict or doc_cate_dict[doc_id] not in leak:
	    continue
    cate_id = doc_cate_dict[doc_id]
    if leak[cate_id] == 1:
        leak[cate_id] = set()
    lu = len(leak[cate_id])
    if row['uuid'] not in user_id_vocab:
        user_id_vocab[row['uuid']] = len(user_id_vocab)
        user_list.append(row['uuid'])
    leak[cate_id].add(user_id_vocab[row['uuid']])
    if lu != len(leak[cate_id]):
        count += 1
    # if leak[row['document_id']]==1:
	   #  leak[row['document_id']] = set()
    # lu = len(leak[row['document_id']])
    # leak[row['document_id']].add(row['uuid'])
    # if lu!=len(leak[row['document_id']]):
	   #  count+=1

# dump user_id_vocab
fv = open("../input_data/leak_uuid.vocab", "w")
for uuid in user_list:
    fv.write("%s\n"%(uuid))

fo = open('../input_data/leak_cate.csv','w')
fo.write('category_id,uuid\n')
for i in leak:
    if leak[i]!=1:
	    tmp = list(leak[i])
	    fo.write('%s,%s\n'%(i,' '.join(tmp)))
	    del tmp
fo.close()	