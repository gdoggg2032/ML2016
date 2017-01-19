import pandas as pd 
import numpy as np 
from itertools import izip
import sys

def csv2ffm(merge_dat, output_name):

    map_col = lambda dat,col: col+"-"+dat.map(str)
    map_target = lambda dat,col: dat.map(int)

    def hashstr(item):
        return int(float(item))

    gen_hash_item = lambda field, feat: '{0}:{1}:1'.format(field,hashstr(feat))
    def gen_hash_row(feats,label):
        result = []
        for idx, item in enumerate(feats):
            val = item.split('-')[-1]
            if val != 'nan':
                result.append(gen_hash_item(idx,val))
        lbl = 1
        if label == 0:
            lbl = -1
        return str(lbl) + ' ' + ' '.join(result)+'\n'

    # merge_dat_val = merge_dat.drop(['display_id','clicked'],axis=1)
    merge_dat_target = merge_dat[['clicked']]
    merge_dat_val = merge_dat.drop(['clicked'], axis=1)
    del merge_dat

    print >> sys.stderr, "features"
    cols = merge_dat_val.columns
    features = []
    for col in merge_dat_val.columns:
        features.append(map_col(merge_dat_val[col],col))
    features_1 = np.array(features).T

    del features
    features = features_1
    del merge_dat_val

    print >> sys.stderr, "targets"
    targets = []
    col = 'clicked'
    targets.append(map_target(merge_dat_target[col], col))
    targets_1 = np.array(targets).T
    del targets
    targets = targets_1
    del merge_dat_target


    with open(output_name,'w') as f_tr:
        i = 0;
        for item, label in izip(features,targets):
            if(i%200000==0):
                print i
            row = gen_hash_row(item,label)
            f_tr.write(row)
            i += 1


def main():
    merge_dat = pd.read_csv("clicks_train.csv")
    output_name = 'clicks_train.libsvm'
    csv2ffm(merge_dat, output_name)

if __name__ == "__main__":
    main()