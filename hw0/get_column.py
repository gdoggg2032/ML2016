import numpy as np
import sys

def getColumn(col, mat):

	m = np.genfromtxt(mat)

	if col < m.shape[1]:

		return np.sort(m.T[col])
	else:
		print >> sys.stderr, "illegal column"
		return ""

if __name__ == "__main__":

	col = int(sys.argv[1])

	mat = sys.argv[2]

	colmat = getColumn(col, mat)

	colstr = ",".join([str(x) for x in colmat])

	print colstr

	
