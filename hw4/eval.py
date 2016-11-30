from sklearn.metrics import f1_score



import sys
import pandas as pd
import numpy as np


in_file = sys.argv[1]
nega = "hard_title_answer_nega.txt"
one = "hard_title_answer_one.txt"
zero = "hard_title_answer_zero.txt"

df1 = pd.read_csv(in_file)
df_nega = pd.read_csv(nega)
df_one = pd.read_csv(one)
df_zero = pd.read_csv(zero)

ans1 = df1.Ans.as_matrix()
ans_nega = df_nega.Ans.as_matrix()
ans_one = df_one.Ans.as_matrix()
ans_zero = df_zero.Ans.as_matrix()

predict = []
ans = []
total = 0
unk = 0
for (a, p) in zip(ans_nega, ans1):
	if a != -1:
		ans.append(a)
		predict.append(p)
	else:
		unk += 1
	total += 1

print "unk: {}, total: {}, rate:{}".format(unk, total, float(unk)/total)
print "nega: {}, one_rate: {}".format(f1_score(ans, predict), np.mean(ans))
print "one: {}, one_rate: {}".format(f1_score(ans_one, ans1), np.mean(ans_one))
print "zero: {}, , one_rate: {}".format(f1_score(ans_zero, ans1), np.mean(ans_zero))


