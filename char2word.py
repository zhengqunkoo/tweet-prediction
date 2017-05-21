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
import keras.callbacks as cb
import keras.preprocessing.sequence as sq
from keras.models import load_model
def char2words(char_sequence,filename="american-english.txt"):
	"""
	A quick and dirty example of how to enumerate all 
        possibilities in the path
	"""
	with open(filename) as words:
		possibilities = [[] for c in char_sequence] 
		for word in words:
			for i in range(len(char_sequence)):
				if word[0] == char_sequence[i]:
					possibilities[i].append(word[:-1])
		return possibilities

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
	vector.append([0]*128+[1])
	return vector


def spacy_model():
	"""
	A small convenience function with a wrapper around 
	"""
	return spacy.load("en_core_web_md")

def wordseq2vec(words, spacy_model):
	"""
	This takes an array of words and changes it into
	"""
	my_vec = []
	for word in words:
		word_vec = spacy_model(word).vector
		word_vec = np.append(word_vec,[0,0])
		my_vec.append(word_vec)
	my_vec.append(np.array([0]*301+[1]))
	return my_vec

def keras_model():
	"""
	Returns a keras model which we can try to fit.
	"""
	model = Sequential()
	# Get rid of sparse characters. Null, etc. unlikely to be part of our model
	model.add(LSTM(70, input_shape=(None,129), return_sequences=True))
	model.add(Bidirectional(LSTM(90, return_sequences=True)))
	model.add(Bidirectional(LSTM(100, return_sequences=True)))
	model.add(Bidirectional(LSTM(120, return_sequences=True)))
	model.add(Bidirectional(LSTM(120, return_sequences=True)))
	model.add(Bidirectional(LSTM(151, return_sequences=True)))
	model.compile(loss='mse', optimizer='adam')
	return model

def match_model_to_words(spacy, keystrokes, vectors):
	"""
	This is the nearest neighbour function that is run after the networks output. It translates
	the vector model back into words.
	"""
	possible_words = char2words(keystrokes)
	pred = []
	for i in range(len(vectors)):
		min_num = 0
		min_word = ""
		word_preds = []
		for word in possible_words[i]:
			dv = spacy(word).vector
			try:
				dist = np.dot(dv,vectors[i])/(np.linalg.norm(dv)*np.linalg.norm(vectors[i]))
			except:
				pass
			else:
				if min_num < dist:
					min_num = dist
					min_word = word
					word_preds.append(word)
		pred.append(min_word)
	return pred	

def test_matching_function():
	"""
	Test for the matching function
	"""
	nlp = spacy_model()
	key_strokes = "Ilef"
	outcome = wordseq2vec(["I","love","eating","fish"],nlp)
	return match_model_to_words(nlp, key_strokes, outcome)

def test_training_proc():
	"""
	Unit test for the model
	"""
	model = keras_model()
	nlp = spacy_model()
	key_strokes = "Ilef"
	vectors = wordseq2vec(["I","love","eating","fish"],nlp) 
	model.fit(np.array([char2vec(key_strokes)]),np.array([vectors]), epochs=100)
	outcome = model.predict(np.array([char2vec(key_strokes)]))
	print(outcome)
	return match_model_to_words(nlp, key_strokes, outcome[0]) 

def train_against_file(model, filename, start=0):
	"""
	Actual training code
	TODO: split the files even smaller otherwise memory error raised
	"""
	keys = [['entitiesShortened', 'value'], ['entitiesFull', 'value']]
	my_file = ParseJSON.ParseJSON(filename, keys)
	nlp = spacy_model()
	keys, labels = list(zip(*my_file.parse_json()))
	char_vecs = list(map(char2vec, keys))
	labels = list(map(lambda x: wordseq2vec(x,nlp), labels))
	tcb = cb.TensorBoard(log_dir='./logs')
	for i in range(start,len(char_vecs),50):
		if len(char_vecs) - i < 50: 
			print("last round")
			X = sq.pad_sequences(char_vecs[i:], maxlen=100)
			Y = sq.pad_sequences(labels[i:], maxlen=100)
			model.fit(X, Y, epochs=100, callbacks=[tcb])
		else:
			print("data", i,len(char_vecs))
			X = sq.pad_sequences(char_vecs[i:i+100], maxlen=100)
			Y = sq.pad_sequences(labels[i:i+100], maxlen=100)
			model.fit(X, Y, epochs=20,  batch_size=50, callbacks=[tcb])
		print("backing up")
		model.save("model.h5")
	return match_model_to_words(nlp,"Ilef",model.predict(np.array([char2vec("Ilef")]))[0])

def predictor(nlp, model, keystrokes):
	return match_model_to_words(nlp,keystrokes,model.predict(np.array([char2vec(keystrokes)]))[0]) 
#print(test_matching_function())
#print(test_training_proc())
#print(test_word2vec())
#print(train_against_file(keras_model(),"example_training_data.json"))
if __name__ == "__main__":
	train_against_file(keras_model(), "/media/arjo/EXT4ISAWESOME/tmlc1-training-01/tmlc1-training-001.json", start=750)
