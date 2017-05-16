# adapted from https://radimrehurek.com/gensim/tut1.html#from-strings-to-vectors

from gensim import corpora
from collections import defaultdict
from ParseJSON import ParseJSON
from os import path

class TextToCorpus(object):
	def __init__(self, filename):
		self.filename = filename

	def get_filename(self):
		return self.filename

	def remove_words(self, tweets):
		'''
		:para tweets: list of tuples, each tuple is tuple of space-separated words
		
		outputs list of lists, each list is still space-separated words, but certain words removed
		'''
		# remove common words and tokenize
		stoplist = set('for a of the and to in'.split())
		removed_stop = [[word for word in tweet if word not in stoplist] for tweet in tweets]
		# don't remove unique words for now (because dataset is small)
		return removed_stop
		'''
		# remove words that only appear once in entire removed_stop
		frequency = defaultdict(int)
		for text in removed_stop:
			for word in text:
				frequency[word] += 1
		removed_unique = [[word for word in text if frequency[word] > 1] for text in removed_stop]
		return removed_unique
		'''

	def save_processed_texts(self, tweets):
		filename = self.get_filename()
		# avoid overwriting existing file
		if path.isfile(filename):
			print('{} already exists'.format(filename))
		else:
			with open(filename, 'a+') as f:
				for tweet in tweets:
					# separate each tweet by return char
					f.write(' '.join(tweet) + '\n')
			print('{} written'.format(filename))

		# write dictionary if dictionary does not exists
		dictname = filename.split('.')[0] + '.dict'
		if path.isfile(dictname):
			print('{} already exists'.format(dictname))
		else:
			dictionary = corpora.Dictionary(tweets)
			dictionary.save(dictname)

	def __iter__(self):
		'''
		outputs generator that converts space-separated words into Bag of Words
		'''
		filename = self.get_filename()
		# raise exception if file does not exist
		if not path.isfile(filename):
			raise Exception('{} does not exist'.format(filename))

		dictname = filename.split('.')[0] + '.dict'
		if not path.isfile(dictname):
			raise Exception('{} does not exist'.format(dictname))
		
		dictionary = corpora.Dictionary.load(dictname)
		for line in open(filename):
			yield dictionary.doc2bow(line.split())

if __name__ == '__main__':
	pj = ParseJSON('example_training_data.json')
	shortened_to_full = pj.parse_json()

	text = TextToCorpus('example.txt')
	removed_stop_and_unique = text.remove_words(shortened_to_full.values())
	text.save_processed_texts(removed_stop_and_unique)
	for i in text:
		print(i)