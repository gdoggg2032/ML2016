from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np 
import cPickle as pickle 
import time

s_time = time.time()

sentences = LineSentence("clear_title.txt")

embed_size = 300

model = Word2Vec(sentences, size=embed_size, min_count=2, sg=1,
	negative=100, window=3, workers=4, iter=50, batch_words=50, sample=0)

vocab = {w:(i+1) for i, (w,_) in enumerate(model.vocab.iteritems())}

vocab['UNK'] = 0

v = np.zeros([len(vocab), embed_size])

for w, wi in vocab.iteritems():
	if w != 'UNK':
		v[wi] = model[w]
v[0] = np.mean(v[1:],axis=0)
revocab = {v:k for k,v in vocab.iteritems()}
with open("vocab.txt", "w") as p:
	for i in range(len(revocab)):
		w = revocab[i]
		output = "{} {}\n".format(w,i)
		p.write(output)
		        
pickle.dump(v, open("title.embed", "w"))

print "total time cost: ", time.time() - s_time