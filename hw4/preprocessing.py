import sys
import re
from nltk.stem.porter import *
import time
import progressbar as pb
import cPickle as pickle

def clear_text(input_str, vocab=None, stem=True):

	stemmer = PorterStemmer()

	sl = input_str.split("\n")

	stop_list = ['i','it', 'for', 'this', 'you', 
	'have', 'has', 'that', 'but', 'my', 'an', 'not',
	'be', 'a', 'the', 'is', 'are', 'am', 'do', 'can',
	'm', 't', 'can', 'from', 'as', 'like', 'if', 'the', 
	'using', 'there', 'so', 'on', 'how', 's', 'what', 'when',
	'in', 'at', 'of', 'and', 'or', 'to', 'with', 'get', 'use'
	]
	
	# clear word
	new_sl = []
	pbar = pb.ProgressBar(widgets=["clear:", pb.FileTransferSpeed(unit="docs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(sl)).start()
	for i, s in enumerate(sl):
		pbar.update(i)
		wl = s.split()
		new_wl = []
		# df_set = set()
		for w in wl:
			ws = clear_word(w)
			for ws_w in ws:
				if ws_w != "" and ws_w not in stop_list:
					if stem == True:
						try:
							ws_w = stemmer.stem(ws_w)
						except:
							ws_w = ws_w

					new_wl.append(ws_w)

		new_sl.append(new_wl)
	pbar.finish()
	sl = new_sl
	
	if vocab is not None:
		new_sl = []

		for wl in sl:
			new_wl = []
			for w in wl:
				if w in vocab:
					iw = vocab[w]
					new_wl.append(iw)
			new_sl.append(new_wl)
		sl = new_sl

	return sl


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def clear_word(w):

	w = re.sub("[\xc2\xe2,#\?\&\+\|\$`\^0-9\"\':_\.;<>\(\)\[\]\{\}=\*/\\\\-]", " ", w)
	ws = w.split()
	wss = []
	for w in ws:

		a = camel_case_split(w)
		for aa in a:
			aa = re.sub(r"^[^a-zA-Z]*", "", aa)
			if aa != "C++" and aa != "c++":
				aa = re.sub(r"[^a-zA-Z]*$", "", aa)
			aa = re.sub(r"[^a-zA-Z\s]*", "", aa)
			aa = aa.lower()
			wss.append(aa)
		
	return wss

def main():
	input_file = sys.argv[1]
	output_file = sys.argv[2]
	
	T = open(input_file, "r").read().strip()

	patterns = pickle.load(open("title.patterns", "r"))

	for p in patterns:
		p_re = patterns[p][0]
		p_sub = patterns[p][1]
		not_ignore = patterns[p][2]
		if not_ignore == 0:
			pattern = re.compile(p_re, re.IGNORECASE)
		else:
			pattern = re.compile(p_re)
		T = pattern.sub(p_sub, T)

	TT = clear_text(T)
	print >> sys.stderr, "clear_text"

	output = "\n".join([" ".join(d) for d in TT])+'\n'

	with open(output_file, "w") as p:
		p.write(output)

if __name__ == "__main__":
	s_time = time.time()
	main()
	print >> sys.stderr, "time cost: ", time.time() - s_time
