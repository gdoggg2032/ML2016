import sys



inputfile = sys.argv[1]
outputfile = sys.argv[2]

p = open(outputfile, "w")

with open(inputfile, "r") as f:
	for i, line in enumerate(f):
		ll = line.strip().split()
		label = ll[0]
		fea = set(ll[1:])
		output = label+" "+" ".join([ff+":1" for ff in fea])+'\n'
		p.write(output)
		if i % 1000000 == 0:
			print >> sys.stderr, "to_libsvm:", i
