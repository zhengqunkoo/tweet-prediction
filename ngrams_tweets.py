from collections import defaultdict, Counter
import math
import json

from pprint import pprint
from sys import getsizeof

ENC = 'utf8'
STARTPAD = u'\u0002'
ENDPAD = u'\u0003'
UNKNOWN = u'\ufffd'

def nested_defaultdict(n, endtype):
	'''
	:para n: number of nested defaultdicts
	:para endtype: type at the innermost defaultdict
	'''
	if n == 0:
		return defaultdict(endtype)
	return defaultdict(lambda: nested_defaultdict(n - 1, endtype))

def ngrams_tweets(filename, filetype, filedir, startn, endn):
	# given csv file of processed tweets,

	# returns defaultdict, keys of first letters, values of:
	# defaultdict, keys of words, values of frequency of words

	# where length of words range from startn to endn
	n = endn - startn
	ngrams = nested_defaultdict(n, int)

	with open(filedir + filename + filetype, 'r', encoding=ENC) as f:
		for line in f:
			# remove endline
			line = line.strip('\n')
			words = list(line.lower().split(' '))
			# remove empty strings
			while '' in words:
				del words[words.index('')]
			# add special character, denote start and end of sentence
			words = [STARTPAD] + words + [ENDPAD]
			# convert to tuple
			words = tuple(words)
			start = 0
			end = len(words) - n
			while start < end:
				ptr = ngrams
				for key in words[start : start + n]:
					ptr = ptr[key]
				ptr[words[start + n]] += 1
				start += 1
	return ngrams

def save_json_counts(counters, jsonname, countname, counttype, startn, endn):
	'''
	one-time updating of counts,
	so don't have to traverse till leaf just to get bigram counts

	now, only leaves of defaultdict tree have counts
	add up counts per layer to get counts of defaultdict branches
	store count along key, in a new tuple (key, count)
	leaf word doesn't change (still a dictionary of word : count)
	'''

	def sum_level(tree, n):
		new_tree = nested_defaultdict(n, int)
		if n == 0:
			return sum(tree.values()), tree
		total_sum = 0
		for k, v in tree.items():
			new_sum, new_val = sum_level(v, n - 1)
			total_sum += new_sum
			# ensure format fits JSON format (no tuples)
			# tuple data separated by Unicode Information Separator One
			new_tree[k] = [new_sum, new_val]
		return total_sum, new_tree

	n = endn - startn
	total_sum, counters = sum_level(counters, n)
	json.dump(counters, open(countname + counttype, 'w'))
	print('{} written!'.format(countname + counttype))

def load_json_counts(countname, counttype):
	counters = json.load(open(countname + counttype, 'r'))
	return counters

def normalize_bigrams_by_unigrams(counters):
	'''
	input: dictionary of unigram keys, with value as array of [unigram_count, {...}]
	where ... represents dictionary of next word (bigram) with same reucrsive structure as unigram

	output: table of bigram probabilities, normalized by unigrams, stored as log value
	'''
	probabilities = {}
	for unigram, array in counters.items():
		unigram_count, bigram_dictionary = array
		for bigram, bi_array in bigram_dictionary.items():
			# probabilities in negative log space
			probabilities[bigram] = math.log(unigram_count, 2) - math.log(bi_array[0], 2)
	return probabilities

def save_or_load_counters(filename, filetype, filedir, startn, endn, jsonname, load=False):
	if load:
		print('loading json...')
		return json.load(open(jsonname + '.json', 'r'))
	counters = ngrams_tweets(filename, filetype, filedir, startn, endn)
	print('saving json...')
	json.dump(counters, open(jsonname + '.json', 'w'))
	return counters

if __name__ == '__main__':
	filename = 'tweets.20170508-191053'
	filetype = '.csv'
	filedir = 'C:/Users/zheng/twitter-files/'
	startn, endn = 1, 5
	
	jsonname = 'counters'
	countname = 'counters'
	counttype = '.count'
	counters = save_or_load_counters(filename, filetype, filedir, startn, endn, jsonname)
	save_json_counts(counters, jsonname, countname, counttype, startn, endn)
	counters = load_json_counts(countname, counttype)
	pprint(normalize_bigrams_by_unigrams(counters))