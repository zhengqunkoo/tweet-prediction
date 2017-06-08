"""
Defines functions to handle metadata
"""
import ijson
import numpy as np
import json
import os
from keras.models import load_model


def parse_test_case(test_case):
        """
        Parses a JSON file and yields 2 objects:
        tweet_id
        an initial string for predicting on
        """
        for obj in json.loads(test_case):
                user = obj["user"]
                entities_shortened = obj["entitiesShortened"]
                inputs = []
                first_item = None
                for item in entities_shortened:
                        if not first_item:
                                first_item = item["value"]
                        if item["type"] == "userMention":
                                inputs.append("\1@"+item["value"]+"\1")
                        elif item["type"] == "hashtag":
                                inputs.append("\2#"+item["value"]+"\2")
                        elif item["type"] == "url":
                                inputs.append("\3<link>\3")
                        else:
                                inputs.append(item["value"])
                yield obj["id"], "".join(inputs)+"\t"+first_item


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


def beam_search(model, seed, letters = [], k=3, j=10, length=140, limit_by_word_num = False):
        """
        :param model: the model
        :param seed: string provided to model on initialization
        :param k: number of probabilities to keep at every step, default 3.
        :param j: number of probabilities to search at every step, default 10.
        :param length: maximum length of predicted strings, default 140.

        beam search through the RNNs
        at every step
        1) initialize top k probabilities
        2) get top j probabilities
        3) keep top k probabilities

        returns: list of strings with top k probabilities
        """
        # top_k: key: seed string, value: probability so far
        top_k = {}
        for _ in range(k):
                top_k[seed] = 1

        # assume all strings in top_k are same length,
        # then checking stopping condition for one string, means all strings satisfy the condition as well
        letter_ind = 0
        while len(list(top_k.keys())[0]) <= length:
                items = top_k.items()
                for seed, c_prob in items:
                        new_top_k = {}
                        if seed[-1] == " ":
                                letter_ind += 1
                                try:
                                        top_k[seed + letters[letter_ind]] = top_k
                                        continue
                                except IndexError:
                                        continue
                        max_probs = get_k_highest_probabilities(get_probabilities(model, seed), j)
                        for letter, prob in max_probs.items():
                                new_top_k[seed + letter] = c_prob * prob
                        # from j candidates, keep top k probabilities
                        top_k = dict(sorted(new_top_k.items(), key=lambda x : x[1], reverse=True)[:k])
        return list(top_k.keys())
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
        return string.split("\t")[1]

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


def test_model_twitter(jsonpath, modelpath, k=3, j=10, window_size=20):
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
                model =load_model(modelpath)
                for tweet_id, string in parse_test_case(f.readline()):
                        # seed string is same length that was used in training
                        top_k = beam_search(model, string[:], k=k, j=j, length=140)
                        # for the same user, yield each of the top_k predictions
                        for prediction in top_k:
                                yield {tweet_id : prediction.split(' ')}

if __name__ == "__main__":
        import character_rnn
        import sys
        print("Starting training...")
        if len(sys.argv) > 2:
                character_rnn.train_model_twitter(sys.argv[1], model=load_model(sys.argv[2]), generator=training_batch_generator)
        else:
                print("Usage: %s [json files]"%sys.argv[0])

