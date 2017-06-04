"""
Defines functions to handle metadata
"""
import ijson
import numpy as np

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
    for inputs, expectation in _input2training_batch(fname, maxlen=length):
        yield np.arrray([char2vec(inputs)]),np.array(char2vec(expectation))

if __name__ == "__main__":
    import character_rnn
    import sys
    print("Starting training...")
    character_rnn.train_model_twitter(sys.argv[1],generator = training_batch_generator)
    
