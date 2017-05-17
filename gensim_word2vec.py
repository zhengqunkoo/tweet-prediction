# adapted from https://rare-technologies.com/word2vec-tutorial/
import gensim, logging
import os

class Sentences(object):
	def __init__(self, dirname, filetype='.txt', encoding='utf8'):
		self.dirname = dirname
		self.filetype = filetype
		self.encoding = encoding

	def get_dirname(self):
		return self.dirname

	def __iter__(self):
		'''
		returns generator that converts space-separated words into list of space-separated words
		'''
		dirname = self.get_dirname()
		for fname in [f for f in os.listdir(dirname) if f.split('.')[1] == self.filetype]:
			for line in open(os.path.join(dirname, fname), 'r', encoding=self.encoding):
				yield line.split()
			print('finished yielding {}'.format(fname))
		print('finished yielding {}'.format(dirname))

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# directories containing only training files and testing files, in .txt
	traindir = 'train/txt/'
	testdir = 'test/txt/'
	# the numbers of the .txt files gensim model will be training on (purely for logging purposes)
	start_number = '001'
	end_number = '060'
	# either '.txt' with duplicates, or '.unique' without duplicates
	filetype = 'unique'
	sentences = Sentences(traindir, filetype)
	# :para iter: parse sentences iter times, first parse frequency count,
	#			  then iter - 1 parses trains neural model
	# :para min_count: discards words with frequency less than min_count
	# :para size: size of neural network layers
	# :para workers: number of cores working in parallel, needs Cython to work

	# min_count = 1 since need to convert all training words into vectors
	# size = 300 since keras model requires 300 dimension vectors

	itr = 5 # default 5
	min_count = 1 # default 5
	size = 300 # default 100
	workers = 4 # default 1
	model = gensim.models.Word2Vec(sentences, iter=itr, min_count=min_count, size=size, workers=workers)
	model.save('models/model-start{}-end{}-{}-{}-{}-{}-{}'.format(start_number, end_number,
																  itr, min_count, size, workers,
																  filetype))