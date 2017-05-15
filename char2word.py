from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
def char2words(char_sequence,filename):
	"""
	A quick and dirty example of how to enumerate all 
        possibilities in the path
	"""
	with open(filename) as words:
		possibilities = [[] for c in char_sequence] 
		for word in words:
			for i in range(len(char_sequence)):
				if word[0] == char_sequence[i]:
					possibilities[i].append(word)
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
		char_vec = [0]*128
		char_vec[ord(c)] = 1
		vector.append(char_vec)
	return vector

def keras_model():
	"""
	Returns a keras model which we can try to fit.
	"""
	model = Sequential()
	# Get rid of sparse characters. Null, etc. unlikely to be part of our model
	model.add(LSTM(70, input_shape=(10,128), return_sequences=True))
	model.add(Bidirectional(LSTM(90, return_sequences=True)))
	model.add(Bidirectional(LSTM(90, return_sequences=True)))
	model.add(Bidirectional(LSTM(200, return_sequences=True)))
	model.add(Bidirectional(LSTM(300, return_sequences=True)))
	return model

	
