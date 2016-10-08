import pandas as pd
import numpy as np
import sys






if __name__ == "__main__":

	dataFile = sys.argv[1]

	dumpFile = sys.argv[2]

	windowSize = int(sys.argv[3])

	# if data is a training data, set = 1
	training = int(sys.argv[4])


	if training:
		df = pd.read_csv(dataFile)
	else:
		df = pd.read_csv(dataFile, header=None)

	# drop out "place" column because they are all the same
	if training:
		df = df.drop(df.columns[1], 1)

	# replace "NR" to 0.0
	df = df.replace('NR', 0.0)

	df2 = df

	# drop date and test_item_name
	df2 = df2.drop([df2.columns[0], df2.columns[1]], axis=1)

	N = df2.shape[0]
	df3 = pd.DataFrame()

	for i in range(0, N, 18):
		x = df2[i:i+18]
		x.reset_index(drop=True, inplace=True)
		df3 = pd.concat([df3, x], axis=1)

	#df3.loc[18] = df3.columns
	df3 = df3.transpose().astype('float')
	df3.reset_index(drop=True, inplace=True)

	df4 = pd.DataFrame()
	M = df3.shape[0]

	if training:


		for j in range(0, M - windowSize+1, 480):
			for i in range(0, 480 - windowSize + 1):
				k = 0
				X = df3.iloc[j+k+i:j+k+i+windowSize, :]
				XX = X.iloc[0:windowSize-1, :]
				XX = pd.DataFrame([XX.values.flatten()])
				if training:
					XX[XX.shape[1]] = X.iloc[windowSize-1, 9]
				df4 = pd.concat([df4, XX])

		cols = df4.columns
		df4 = pd.concat([df4[len(cols)-1], df4[cols[:len(cols)-1]]], axis=1)
		



		# for j in range(0, M - windowSize+1, 480):
		# 	# for i in range(0, 480 - windowSize + 1, windowSize):
		# 	for k in range(0, 480, 24):
		# 		for i in range(0, 24 - windowSize + 1):
		# 			X = df3.iloc[j+k+i:j+k+i+windowSize, :]
		# 			XX = X.iloc[0:windowSize-1, :]
		# 			XX = pd.DataFrame([XX.values.flatten()])
		# 			if training:
		# 				XX[XX.shape[1]] = X.iloc[windowSize-1, 9]
		# 			df4 = pd.concat([df4, XX])

		# cols = df4.columns
		# df4 = pd.concat([df4[len(cols)-1], df4[cols[:len(cols)-1]]], axis=1)
		



		# for i in range(0, M - windowSize+1):

		# 	X = df3.iloc[i:i+windowSize, :]
		# 	XX = X.iloc[0:windowSize-1, :]
		# 	XX = pd.DataFrame([XX.values.flatten()])
		# 	if training:
		# 		XX[XX.shape[1]] = X.iloc[windowSize-1, 9]
		# 	df4 = pd.concat([df4, XX])

		# if training:
			

		

	else :
		windowSize -= 1
		for i in range(0, M, 9):
			X = df3.iloc[i+9-windowSize:i+9]
			XX = X.iloc[0:windowSize, :]
			XX = pd.DataFrame([XX.values.flatten()])
			df4 = pd.concat([df4, XX])


	Data = df4.astype('float').as_matrix()

	Data.dump(dumpFile)
	




