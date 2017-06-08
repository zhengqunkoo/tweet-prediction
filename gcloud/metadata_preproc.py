"""
Defines functions to handle metadata
"""
import ijson.backends.yajl2_cffi as ijson
import numpy as np
import json
import os
from keras.models import load_model
import re
from string import punctuation, digits
from math import log

def parse_test_case(test_case):
	"""
	Parses a JSON file and yields 3 objects:
	tweet_id
	an initial string for predicting on
	entities_shortened (letters only)
	"""
	
	for obj in json.loads(test_case):
		user = obj["user"]
		entities_shortened = obj["entitiesShortened"]
		inputs, letters = [], []
		for item in entities_shortened:
			if item["type"] == "userMention":
				inputs.append("\1@"+item["value"]+"\1")
			elif item["type"] == "hashtag":
				inputs.append("\2#"+item["value"]+"\2")
			elif item["type"] == "url":
				inputs.append("\3<link>\3")
			else:
				inputs.append(item["value"])
				letters.append(item["value"])
		yield obj["id"], "".join(inputs), "".join(letters)

def get_probabilities(model, string):
	"""
	:param model - the model
	:param string- the seed with which to feed the model
	returns all probabilities for each of the next characters
	"""
	return model.predict(np.array([char2vec(string)]))


def get_k_highest_probabilities(probabilities, k=5):
	"""
	given a numpy matrix of probabilities,
	return dictionary of k letters with the highest probabilities
	"""
	max_probs = {}
	for i in range(k):
		letter = chr(np.argmax(probabilities))
		max_probs[letter] = probabilities[0][ord(letter)]
		probabilities[0][ord(letter)] = 0
	return max_probs


def beam_search(model, seed, letters, k=3, j=10):
	"""
	:param model: the model
	:param k: number of probabilities to keep at every step, default 3.
	:param j: number of probabilities to search at every step, default 10.
	:param seed: initial input to model (metadata)
	:param letters: string of first letters from entities_shortened

	beam search through the RNNs
	at every step
	1) initialize top k probabilities
	2) get top j probabilities
	3) keep top k probabilities

	returns: list of strings with top k probabilities
	"""
	# top_k: key: seed string, value: [logarithmic probability so far, letter_ind]
	# each prediction tracks its own letter_ind
	top_k = {}
	# strip ' ' from seed so don't skip first word
	seed = seed.strip(' ')
	for _ in range(k):
		top_k[seed] = [0, 0]
	while True:
		for seed, value in top_k.items():
			c_prob, letter_ind = value
			new_top_k = {}
			if seed[-1] == " ":
				try:
					new_top_k[seed + letters[letter_ind]] = [c_prob, letter_ind+1]
				except IndexError:
					# finished predicting last word, return
					# remove extra space at end of prediction
					return [prediction.strip(' ') for prediction in list(top_k.keys())]
			else:
				max_probs = get_k_highest_probabilities(get_probabilities(model, seed), j)
				for letter, prob in max_probs.items():
					# use logarithmic probs
					new_top_k[seed + letter] = [c_prob + log(prob), letter_ind]
			# from j candidates, keep top k probabilities
			top_k = dict(sorted(new_top_k.items(), key=lambda x : x[1][0], reverse=True)[:k])
	

def parse_input(fname):
        """
        :param fname - file name
        This generator takes an input and parses it splitting it into tuples of (inputs, outputs)
        The generator sanitizes the data to prevent problems from occuring
        """
        with open(fname) as f:
                for obj in ijson.items(f,"item"):
                        user = obj["user"]
                        entities_shortened = obj["entitiesShortened"]
                        inputs = []
                        for item in entities_shortened:
                                if item["type"] == "userMention":
                                        inputs.append("\1@"+item["value"]+"\1")
                                elif item["type"] == "hashtag":
                                        inputs.append("\2#"+item["value"]+"\2")
                                elif item["type"] == "url":
                                        inputs.append("\3<link>\3")
                                else:
                                        inputs.append(item["value"])
                        entities_full = obj["entitiesFull"]
                        expected_out = []
                        for item in entities_full:
                                if item["type"] == "url":
                                        expected_out.append("%s")
                                else:
                                        expected_out.append(item["value"])

                        yield "".join(inputs)," ".join(expected_out)


def mix_generators(*args):
	"""
	Takes a bunch of generators and returns a generator which samples
	each generator
	"""
	generators = list(args)
	i = 0
	while len(generators) > 0:
		try:
			yield next(generators[i%len(generators)])
		except:
			del generators[i%len(generators)]
		finally:
			i+=1
			

def _input2training_batch(fname, max_len=300):
	"""
	sanitizes the input data... prevents things from overflowing
	"""
	for inputs, outputs in parse_input(fname):
		curr_buff = inputs+"\t"
		if len(outputs) + len(inputs) == 3:
			# skip too long
			# TODO: write to seperate file if tweet data is too long
			continue
		for c in outputs:
			yield curr_buff,c
			curr_buff = curr_buff + c


def strip_prediction(string):
	return string.split("\5")


def char2vec(char_sequence):
	"""
	:param char_sequence - a sequence of characters
	This function takes a char_sequence and encodes it as a series of onehot vectors
	>>> char2vec("a")
	[[....0,0,1,0,0,0,0,0,0,0,0,...]]
	"""
	vector = []
	for c in char_sequence:
		char_vec = [0]*128
		try:
			char_vec[ord(c)] = 1
		except IndexError:
			pass # Not an ascii character
		vector.append(np.array(char_vec))
	return vector


def pad_upto(array, length = 300):
	"""
	:param array - the array to zeropad
	:param length - the length to pad up to
	:returns array prepended with zeros.
	>>>  len(pad_upto([[0]*80]))
	300
	"""
	return [np.array([0]*128) for i in range(length-len(array))] + array

def training_batch_generator(fname, length = 300):
	"""
	:param fname - file name
	Train on this generator to get one file's data
	"""
	for inputs, expectation in _input2training_batch(fname, max_len=length):
		yield np.array([char2vec(inputs)]),np.array(char2vec(expectation))


def test_model_twitter(tweet_ids, jsonpath, modelpath, k=3, j=10, window_size=20):
	"""
	:param jsonpath: path to JSON file
	:param modelpapth: path to the model
	:param k: top k probabilities returned from beam search, default 3.
	:param j: number of probabilities to search at every step of beam search, default 10.
	:param window_size: ideally, same number used in training model, default 20.

	yields dictionary of format {<tweet_id>:[<space-separated sentence>]}
	outputs (not much variance, maybe increase j?):
	{'rens0erfsao': ['500p/P"LC2"bJC-\x03<ling', '.onithic', 'pooitioear', '.onol', 'anterestinetingsurotero', 'a', 'chelugivetes', '.ade', 'cating', 'tere', 'tho', 'peritesiogs', 'a', 'forme', 'capintietsion']}
	{'rens0erfsao': ['500p/P"LC2"bJC-\x03<ling', '.onithic', 'pooitioear', '.onol', 'anterestinetingsurotero', 'a', 'chelugivetes', '.ade', 'cating', 'tere', 'tho', 'peritesiogs', 'a', 'forme', 'capintietsiou']}
	{'rens0erfsao': ['500p/P"LC2"bJC-\x03<ling', '.onithic', 'pooitioear', '.onol', 'anterestinetingsurotero', 'a', 'chelugivetes', '.ade', 'cating', 'tere', 'tho', 'peritesiogs', 'a', 'forme', 'capintietsiom']}
	{'revfvdonwg0': ['R\x01@Eman36\x01:\x01@mikezigg', 'and', 'the', 'readersioua', 'ofrate', 'yations', 'tho', 'corsemitere', '.ereath', 'yat', 'ofedesteris', '.egentiog', 'ano', 'peater', 'teede', 'yourrcanl', 'teed', 'th']}
	{'revfvdonwg0': ['R\x01@Eman36\x01:\x01@mikezigg', 'and', 'the', 'readersioua', 'ofrate', 'yations', 'tho', 'corsemitere', '.ereath', 'yat', 'ofedesteris', '.egentiog', 'ano', 'peater', 'teede', 'yourrcanl', 'teed', 'to']}
	{'revfvdonwg0': ['R\x01@Eman36\x01:\x01@mikezigg', 'and', 'the', 'readersioua', 'ofrate', 'yations', 'tho', 'corsemitere', '.ereath', 'yat', 'ofedesteris', '.egentiog', 'ano', 'peater', 'teede', 'yourrcanl', 'teed', 'te']}
	"""
	with open(jsonpath, 'r') as f:
		model = load_model(modelpath)
		for tweet_id, seed, letters in parse_test_case(f.readline()):
			if tweet_id not in tweet_ids:
				# seed string is same length that was used in training
				top_k = beam_search(load_model(modelpath), seed, letters, k=int(k), j=int(j))
				# for the same user, yield each of the top_k predictions
				for prediction in top_k:
					prediction = prediction[len(seed)+1:]
					print(seed, letters, parse_output(prediction))
					yield {tweet_id : parse_output(prediction)}


def parse_output(s):
	s = s.replace("%s", "")
	specchars = ['\1', '\2', '\3']
	for spec in specchars:
		indeces = [i for i,x in enumerate(s) if x == spec]
		for i in range(0, len(indeces)-1, 2):
			s = s[:indeces[i]] + s[indeces[i+1]+1:]
	r = re.compile(r'[\s{}]+'.format(re.escape(punctuation + digits + '\1\2\3')))
	s = [ch.strip(' ') for ch in r.split(s) if ch.strip(' ') != '']
	return s
	# return re.sub(r'<link>', b' ', s)


if __name__ == "__main__":
	import character_rnn
	import sys
	if len(sys.argv) >= 2 and sys.argv[1] == "train":
		print("Starting {}ing...".format(sys.argv[1]))
		if len(sys.argv) >= 3:
			count = 0
			pre = ""
			for file in sys.argv[2:]:
				print("loading model weights.",end="")
				if count == 0:
					filenum = str(int(file[-8:-5])-1)
					pre = file[:-8] + '0'*(3-len(filenum)) + filenum + file[-5:]
				"""
					print("{}.hdf5".format(file))
					character_rnn.train_model_twitter(file, model=load_model("weights.{}.hdf5".format(file)), generator=training_batch_generator)
				else:
				"""
				print("{}.hdf5".format(pre))
				character_rnn.train_model_twitter(file, model=load_model("weights.{}.hdf5".format(pre)), generator=training_batch_generator)
				count += 1
				pre = file
