import pandas as pd 
import sys
import numpy as np

input_file = sys.argv[1]
output_file = sys.argv[2]


df = pd.read_csv(input_file)
if len(df.columns) == 2:
	# test format
	df['clicked'] = -1
else:
	df['clicked'][df.clicked==0] = -1
	df = df.iloc[np.random.permutation(len(df))]
df.to_csv(output_file,index=False, sep=" ", header=False)