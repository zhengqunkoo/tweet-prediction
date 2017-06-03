import ParseJSON
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Merge, GRU
from keras.callbacks import TensorBoard, ModelCheckpoint

def get_ch2ix_ix2ch(chars):
	"""
	gets all unique characters across all files unique_file,
	and enumerates them according to their order in the ASCII table

	return tuple of three elements: (mapping of characters to index,
									 mapping of index to characters
									 )
	these mappings produce a unique one-hot vector for each character
	"""
	sorted_chars = sorted(set(chars))
	ch2ix = {ch:ix for ix,ch in enumerate(sorted_chars)}
	ix2ch = {ix:ch for ix,ch in enumerate(sorted_chars)}
	return ch2ix, ix2ch


def char2vec(char_sequence, ch2ix):
	"""
	:param char_sequence - a sequence of characters
	:param ch2ix: mapping of characters to index

	This function takes a char_sequence and encodes it as a series of onehot vectors
	all vectors are numpy arrays to save time
	>>> char2vec("a")
	[[....0,0,1,0,0,0,0,0,0,0,0,...]]
	"""
	input_length = len(ch2ix)
	vector = np.zeros((len(char_sequence), input_length))
	for i in range(len(char_sequence)):
		c = chr(char_sequence[i])
		char_vec = np.zeros((input_length))
		char_vec[ch2ix[c]] = 1
		vector[i] = char_vec
	return vector


def parse_data(f, ch2ix):
	line = f.readline()
	# read line by line, strip newline, split by '\t' into keys
	line = line.strip().split('\t'.encode('ascii', 'backslashreplace'))
	# dirty way to remove 'id' and 'user' keys
	_, _, created, media, reply, quote, entities_full, entities_shortened = line
	# join by null byte
	# entities_full also ends in null byte
	train_data = char2vec(b'\0'.join([created, media, reply, quote, entities_shortened]), ch2ix)
	test_data = char2vec(entities_full, ch2ix)

	train_data = np.vstack((train_data, np.zeros((TRAINSIZE-train_data.shape[0], train_data.shape[1]))))
	test_data = np.vstack((test_data, np.zeros((TRAINSIZE-test_data.shape[0], test_data.shape[1]))))

	return train_data, test_data


def build_batch(f, batch_size, ch2ix):
	'''
	:param f: file pointer to a .unique file
	'''
	input_length = len(ch2ix)
	batch_train = []
	batch_test = []
	# read :batch_size: lines from file
	# if reach EOF before batch_size read, just return the np.arrays with extra zeros at the end
	for i in range(batch_size):
		train_data, test_data = parse_data(f, ch2ix)
		batch_train.append(train_data)
		batch_test.append(test_data)
	return np.array(batch_train), np.array(batch_test)


def predict(model, f, ch2ix, ix2ch):
	"""
	reads all metadata from unique_file, passes all except entitiesFull into model
	model output space separated, match with actual entitiesFull

	scoring does not include Soundex / Levenshtein
	"""
	train_data, test_data = parse_data(f, ch2ix)
	predictions = model.predict(train_data)

	# numpy arrays to words
	for word_array in predictions:
		# normalize each tweet with respect to itself
		# such that char_array can be distinguished from one another
		norms = np.linalg.norm(word_array, axis=1, ord=np.inf)
		word_array = word_array / norms[:, None]
		# space separated
		indeces = set()
		words = []
		word = ''
		for char_array in word_array:
			ix = np.argmax(char_array)
			indeces.add(ix)
			ch = ix2ch[ix]
			if ch == ' ':
				words.append(word)
				word = ''
			else:
				word += ch
		# append last word (but remove terminating byte)
		words.append(word[:-1])

		print(word_array)
		print(words)
		print(indeces)
		print(entities_full)
		"""
		score = 0
		for i in range(len(entities_full)):
			score += entities_full[i] == words[i]
		print(score)
		"""


def k_best_options(mat, k):
	"""
	Returns the k best 
	"""
	best = []
	for i in range(k):
		b = np.argmax(mat)
		best.append((chr(b),mat[b]))
		mat[b] = 0
	return best


def get_key_strokes(string):
	return len(list(filter(lambda x: x == " ", string)))


def beam_search(model, keystrokes, ix2ch, thickness=2, pruning=10, context=["a",1]):
	"""
	Beam search: this takes a model and uses beam search as a method to find most probable
	   string. The aim is to allow for better predictions without being overly greedy.
	try prediction without special cases (treat NULLBYTE and spaces as normal characters)
	"""
	stack = []
	for current, c_prob in context:
		"""
		if current[-1] == " ": 
			res = model.predict(np.array([char2vec(current+keystrokes[get_key_strokes(current)])]))
			best = np.argmax(res)
			predictions = [(ix2ch[best],res[best])]
		elif current[-1] == NULLBYTE:
			continue
		else:
		"""
		res = model.predict(np.array([char2vec(current+keystrokes[get_key_strokes(current)])]))
		predictions = k_best_options(res, thickness)
		for prediction, probability in predictions:
			stack.append((current+prediction, c_prob*probability))
	context = sorted(stack,key=lambda x: x[2])
	return context


def charRNN_model(ch2ix):
	"""
    This Builds a character RNN based on kaparthy's infamous blog post
	:return: None
	"""
	input_length = len(ch2ix)
	model = Sequential()
	model.add(GRU(512, input_shape=(None, input_length), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(GRU(512, input_shape=(None, input_length), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(Dense(input_length, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model


def train_model_twitter(ch2ix, unique_path, batch_size, steps_per_epoch, epochs, loops=0, unique_number=None, model=None):
	"""
	This function trains the data on the character network
	:return: 
	"""
	if model == None:
		model = charRNN_model(ch2ix)
	# loop over files to fit
	while True:
		for unique_file in [f for f in os.listdir(unique_path) if os.path.isfile(os.path.join(unique_path, f)) and f.split('.')[1] == 'unique']:
			if not unique_number or int(unique_file.split('.')[0][-3:]) > unique_number:
				with open(os.path.join(unique_path, unique_file), 'rb') as f:
					print("training on {}...".format(unique_file))

					batch_train, batch_test = build_batch(f, batch_size, ch2ix)
					history_callback = model.fit(batch_train,
												 batch_test,
												 epochs=epochs,
												 callbacks=[ModelCheckpoint("hdf5/weights.500-350-200.{}.{}.hdf5".format(unique_file, loops))],
												 )

					# log loss history in txt file, since tensorboard graph overlaps
					loss_history = history_callback.history["loss"]
					np_loss_history = np.array(loss_history)
					with open("log/dnn_loss-batch{}-epoch{}.txt".format(batch_size, epochs), 'ab') as f:
						np.savetxt(f, np_loss_history, delimiter="\n")
					# predict on file that was just trained
					predict(model, f, ch2ix, ix2ch)
		# restart from first file
		unique_number = 0
		loops += 1


if __name__ == "__main__":
	NULLBYTE = '\0'
	NEWLINE = '\37'
	CREATEDSIZE = 14
	MEDIASIZE = 1
	REPLYSIZE = 1
	# dangerous estimate: number of entities in entities_shortened
	SHORTENSIZE = 30
	TWEETSIZE = 171
	TRAINSIZE = CREATEDSIZE + MEDIASIZE + REPLYSIZE + SHORTENSIZE + TWEETSIZE
	# assume replace_types identical to replace_types in JSON2Text
	# if replace_types different, wrong prediction
	# hashtag, userMention are not replaced
	replace_types = {'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}
	ascii_nonspec_chars = [chr(x) for x in range(32, 128)]
	ascii_spec_chars = [NULLBYTE, NEWLINE] + list(replace_types.values())
	chars = ascii_nonspec_chars + ascii_spec_chars

	# character array will enumerate according to sorted characters.
	# if the order of the ASCII special chars in replace_types changes,
	# e.g.      'number':'\33', 'url':'\34' => number < url
	# becomes   'number':'\34', 'url':'\33' => url < number
	# then enumeration index changes, and old model will not be compatible with new model
	ch2ix, ix2ch = get_ch2ix_ix2ch(chars)
	
	# parameters to continue training
	unique_path = "train/txt"
	unique_number = 2 # continue training for files strictly after this number
	unique_str = str(unique_number)
	unique_str = "0"*(2 - len(unique_str)) + unique_str
	loops = 0 # how many times trained over entire fileset
	hdf5_file = "hdf5/weights.500-350-200.tmlc1-training-0{}.unique.{}.hdf5".format(unique_str, loops)
	# train on 18000 lines per file
	batch_size = 10
	steps_per_epoch = 100
	epochs = 18
	train_model_twitter(ch2ix, unique_path, batch_size, steps_per_epoch, epochs, loops=loops, unique_number=unique_number)#, model=load_model(hdf5_file))