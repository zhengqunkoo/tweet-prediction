import ijson
import numpy as np
def parse_input(fname):
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
            

def input2training_batch(fname, max_len=300):
    for inputs, outputs in parse_input(fname):
        curr_buff = inputs+"\t"
        for c in outputs:
            yield curr_buff,c
            if len(curr_buff) < max_len:
                curr_buff = curr_buff + c
            else:
                curr_buff = inputs+"\t"

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
    
