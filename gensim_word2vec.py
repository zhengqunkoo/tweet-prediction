# adapted from https://rare-technologies.com/word2vec-tutorial/
import gensim, logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)

class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def get_dirname(self):
		return self.dirname

	def __iter__(self):
		'''
		returns generator that converts space-separated words into list of space-separated words
		'''
		dirname = self.get_dirname()
        for fname in os.listdir(dirname):
            for line in open(os.path.join(dirname, fname)):
                yield line.split()
		print('finished iterating over {}'.format(dirname))

if __name__ == '__main__':
	pj = ParseJSON('example_training_data.json')
	shortened_to_full = pj.parse_json()

	# directories containing only training files and testing files, in .txt
	traindir = 'train/txt'
	testdir = 'test/txt'
	sentences = Sentences(traindir)
	# :para iter: parse sentences iter times, first parse frequency count,
	#			  then iter - 1 parses trains neural model
	# :para min_count: discards words with frequency less than min_count
	# :para size: size of neural network layers
	# :para workers: number of cores working in parallel, needs Cython to work
	model = gensim.models.Word2Vec(sentences, iter=5, min_count=5, size=200, workers=4)

