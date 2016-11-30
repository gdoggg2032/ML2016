import sys
import argparse

# import preprocessing as pp
import pandas as pd

import numpy as np
import progressbar as pb
import time
from itertools import izip

# import word2vec
import cPickle as pickle

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--docs', default='./docs.txt', type=str)
	parser.add_argument('--title', default='./clear_title.txt', type=str)
	parser.add_argument('--index', default='./check_index.csv', type=str)
	parser.add_argument('--predict', default='./predict.csv', type=str)
	parser.add_argument('--embed', default='./title.embed', type=str)
	parser.add_argument('--vocab', default='./vocab.txt', type=str)
	args = parser.parse_args()

	return args


class clustering(object):

	def __init__(self, args):
		self.args = args
		self.load_data()

	def load_data(self):

		# docs_text = open(self.args.docs, "r").read().strip()
		title_text = open(self.args.title, "r").read().strip()

		all_text = title_text

		self.load_embed()

		texts = self.load_texts(all_text, self.vocab)

		self.re_vocab = {v:k for k, v in self.vocab.iteritems()}
		title_num = len(title_text.strip().split("\n"))
		self.titles = texts[-title_num:]
		print >> sys.stderr, "title num:", len(self.titles)
		print >> sys.stderr, "vocab size:", len(self.vocab)

		title_vocab = set([w for title in self.titles for w in title])
		print >> sys.stderr, "title vocab rate:", len(title_vocab) / float(len(self.vocab)), len(title_vocab), len(self.vocab)

		self.cluster_df = pd.read_csv(self.args.index)

	def load_texts(self, text, vocab):
		sl = []
		text = text.split("\n")
		for d in text:
			wl = []
			d = d.split(" ")
			for w in d:
				if w not in self.vocab:
					w = "UNK"
				wi = self.vocab[w]
				wl.append(wi)
			sl.append(wl)

		return sl


	def load_embed(self):

		
		self.embed = pickle.load(open(self.args.embed, "r"))
		self.embed_size = self.embed.shape[1]
		self.vocab = {}
		with open(self.args.vocab, "r") as f:
			for i, line in enumerate(f):
				l = line.strip().split(" ")
				# print l
				self.vocab[l[0]] = i

	def train(self):

		self.vectors = []
		for t in self.titles:
			v = self.d2v(t)[None, :]
			self.vectors.append(v)

		self.vectors = np.concatenate(self.vectors, axis=0)
		print >> sys.stderr, self.vectors.shape

		# normalize
		self.vectors = (self.vectors - np.mean(self.vectors, axis=0)) / np.std(self.vectors, axis=0)

		from sklearn.cluster import KMeans

		from sklearn.metrics.pairwise import cosine_similarity
		def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
			return 1.0 - cosine_similarity(X,Y)


		from sklearn.cluster import k_means_
		k_means_.euclidean_distances = new_euclidean_distances 

		n_clusters = 22
		print >> sys.stderr, "fitting kmeans, n_clusters =",n_clusters
		kmeans = KMeans(n_clusters=n_clusters, n_init=20)
		kmeans.fit(self.vectors)

		self.clusters = kmeans.predict(self.vectors)

		# dump results
		with open("result.txt", "w") as f:
			for i in range(len(self.clusters)):
				output = [str(self.clusters[i])] + [str(v) for v in self.vectors[i]]
				output = " ".join(output)
				print >> f, output

	def score(self, d1, d2):

		t1 = self.clusters[d1]
		t2 = self.clusters[d2]
		if t1 == t2:
			return 1
		else:
			return 0

	def d2v(self, t):

		v = np.array([0.]*self.embed_size)
		total = 0.0
		for iw in t:
			if iw == 0:
				continue
			v += self.embed[iw]
			total += 1

		if total == 0:
			v = np.mean(self.embed[1:], axis=0)
			total += 1
		v /= total
		return v

	def predict(self):

		df = self.cluster_df
		
		ans = []

		x_ids = df['x_ID'].as_matrix()
		y_ids = df['y_ID'].as_matrix()

		pbar = pb.ProgressBar(widgets=["predict:", pb.FileTransferSpeed(unit="pairs"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(df)).start()

		for i, (x, y) in enumerate(izip(x_ids, y_ids)):
			pbar.update(i)
			p = self.score(x, y)
			ans.append(p)

		pbar.finish()

		print >> sys.stderr, "1 rate", np.mean(ans)

		df['Ans'] = ans
		df = df[['ID', 'Ans']]

		df.to_csv(self.args.predict, index=False)

if __name__ == "__main__":

	args = arg_parse()

	start_time = time.time()

	model = clustering(args)

	model.train()

	print >> sys.stderr, "training time: {}".format(time.time()-start_time)

	model.predict()
