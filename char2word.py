"""
This file houses the main model with keras it defines a few functions
The idea is take the character vector pass it through several LSTMs and then
get an approximate word vector. Then use nearest neighbour to find the closest
word vector.
"""
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
import spacy
import numpy as np
import math
import ParseJSON
import keras.preprocessing.sequence as sq

import gensim
import os

def char2vec(char_sequence):
	"""
	para: char_sequence - a sequence of characters
	This function takes a char_sequence and encodes it as a series of onehot vectors
	>>> char2vec("a")
	[[....0,0,1,0,0,0,0,0,0,0,0,...]]
	"""
	vector = [] 
	for c in char_sequence:
		char_vec = [0]*129
		char_vec[ord(c)] = 1
		vector.append(np.array(char_vec))
	#vector.append([0]*128+[1])
	return vector


def gensim_model(filepath='models/model-start001-end060-5-1-300-4-unique'):
	"""
	A small convenience function with a wrapper around 
	"""
	if os.path.exists(filepath):
		return gensim.models.Word2Vec.load(filepath)
	raise Exception('{} does not exist. You need to run the gensim_word2vec.py \
		on either .txt or .unique files'.format(filepath))

def wordseq2vec(words, gensim):
	"""
	This takes an array of words and changes it into a word vector (numpy array)
	used list comprehension
	"""
	return [gensim.wv[word] for word in words]

def keras_model():
	"""
	Returns a keras model which we can try to fit.
	"""
	model = Sequential()
	# Get rid of sparse characters. Null, etc. unlikely to be part of our model
	model.add(LSTM(70, input_shape=(None,129), return_sequences=True))
	model.add(Bidirectional(LSTM(90, return_sequences=True)))
	model.add(Bidirectional(LSTM(100, return_sequences=True)))
	model.add(Bidirectional(LSTM(200, return_sequences=True)))
	model.add(Bidirectional(LSTM(150, return_sequences=True)))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def match_model_to_words(gensim, vectors):
	"""
	This is the nearest neighbour function that is run after the networks output. It translates
	the vector model back into words.

	for each vector, find the most similar word, returns list using list comprehension
	"""
	# 0-indexing extracts each word from its (word, similarity_score) tuple
	return [gensim.most_similar(positive=[vector], negative=[], topn=1)[0] for vector in vectors]

def test_training_proc():
	"""
	Unit test for the model
	"""
	model = keras_model()
	gensim = gensim_model()
	key_strokes = "Ilef"
	vectors = wordseq2vec(["I","love","eating","fish"], gensim) 
	model.fit(np.array([char2vec(key_strokes)]),np.array([vectors]), epochs=100)
	outcome = model.predict(np.array([char2vec(key_strokes)]))
	print(outcome)
	return match_model_to_words(gensim, outcome[0]) 

def train_against_file(model, filename):
	"""
	Actual training code
	TODO: split the files even smaller otherwise memory error raised
	"""
	my_file = ParseJSON.ParseJSON(filename)
	gensim = gensim_model()
	keys, labels = list(zip(*my_file.parse_json()))
	char_vecs = list(map(char2vec, keys))
	labels = list(map(lambda x: wordseq2vec(x, gensim), labels))
	for i in range(0,len(char_vecs),50):
		if len(char_vecs) - i < 50: 
			print("last round")
			X = sq.pad_sequences(char_vecs[i:], maxlen=100)
			Y = sq.pad_sequences(labels[i:], maxlen=100)
			model.fit(X, Y, epochs=1)
		else:
			print("data", 100*i/(len(char_vecs))//50)
			X = sq.pad_sequences(char_vecs[i:i+100], maxlen=100)
			Y = sq.pad_sequences(labels[i:i+100], maxlen=100)
			model.fit(X, Y, epochs=1)
	return match_model_to_words(gensim, model.predict(np.array([char2vec("Ilef")]))[0])
 
#print(test_matching_function())
#print(test_training_proc())
#print(test_word2vec())
#print(train_against_file(keras_model(),"example_training_data.json"))
print(train_against_file(keras_model(), "train/tmlc1-training-001.json"))
