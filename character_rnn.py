import ParseJSON
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import TensorBoard,ModelCheckpoint


def training_set(string, window_size=10):
    """
    :param string - Takes an input string and returns a list of generators
    :param window_size - The size of the context window
    :return: a generator containing all the training pairs
    
    >>> for x,y  in training_set("I love eating"): print(x,y)
    I love eat i
     love eati n
    love eatin g
    """
    if len(string) < window_size:
        yield string[:-1], string[-1]
    for i in range(len(string)-window_size):
        current_buffer = string[i:i+window_size]
        output = string[i+window_size]
        yield current_buffer, output


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


def build_batch(json_file, window_size=20):
    # extract sentence from tweetdata
    for tweetdata in json_file.parse_json():
        sentence = tweetdata[0]
        print(sentence)
        inputs, labels =[],[]
        result = ""
        for item in sentence: result = result + " " + item
        sentence = result+"\0"
        try:
            for context, character in training_set(sentence, window_size=window_size):
                # TODO: Optimize this shit - can use itertools.chain... and maybe move char2vec
                # into the training set
                #inputs.append(char2vec(context))
                #labels.append(char2vec(character))
                print(context,character)
                yield (np.array([char2vec(context)]), np.array(char2vec(character)))
        except:
            pass


def predict(model, context, length=1):
    for i in range(length):
        res = model.predict(np.array([char2vec(context)]))
        mx = np.argmax(res)
        context = context+chr(mx)
    return context


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


def stopping_condition(context, key_strokes):
    cond = True
    for i in context:
        if len(context) > 140:
            return True
        if context[-1] != "\0":
            cond = False
        if get_key_strokes(i) > key_strokes:
            cond = False
    return cond


def beam_search(model, keystrokes, thickness =2, pruning=10, context=["a",1]):
    """
    Beam search: this takes a model and uses beam search as a method to find most probable
       string. The aim is to allow for better predictions without being overly greedy.
    TODO: write unit test
    """
    while not stopping_condition(context, len(keystrokes)):
        stack = []
        for current, c_prob in context:
            if current[-1] == " ": 
                res = model.predict(np.array([char2vec(current+keystrokes[get_key_strokes(current)])]))
                best = np.argmax(res)
                predictions = [(chr(best),res[best])]
            elif current[-1] == "\0":
                continue
            else:
                res = model.predict(np.array([char2vec(current+keystrokes[get_key_strokes(current)])]))
                predictions = k_best_options(res,thickness)
            for prediction, probability in predictions:
                stack.append((current+prediction, c_prob*probability))
        context = sorted(stack,key=lambda x: x[2])
    return context

def charRNN_model():
    """
    This Builds a character RNN based on kaparthy's infamous blog post
    :return: None
    """
    model = Sequential()
    model.add(LSTM(512, input_shape=(None, 128)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def train_model_twitter(file, model=charRNN_model(), generator = build_batch):
    """
    This function trains the data on the character network
    :return: 
    """
    model.fit_generator(generator(file), steps_per_epoch=100, epochs=4000,
                        callbacks=[TensorBoard("./log"), ModelCheckpoint("weights.{epoch:02d}.hdf5")])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: %s [json files]"%sys.argv[0])
    elif len(sys.argv) == 2:
        train_model_twitter(sys.argv[1])
    else:
        train_multiple_files(sys.argv[1:])
