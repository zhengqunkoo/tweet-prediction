import ParseJSON
import keras.models
import numpy as np
from keras.layers import GRU, Dropout, Dense
from keras.callbacks import TensorBoard,ModelCheckpoint
from itertools import chain
import os

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
    """
    # this code moved to build_batch()
    if len(string) < window_size:
        yield string[:-1], string[-1]
    """
    for i in range(len(string)-window_size):
        current_buffer = string[i:i+window_size]
        output = string[i+window_size]
        yield current_buffer, output


def char2vec(char_sequence):
    """
    :param char_sequence: a sequence of ASCII bytes
    This function takes a char_sequence and encodes it as a series of onehot vectors
    >>> char2vec("a")
    [[....0,0,1,0,0,0,0,0,0,0,0,...]]
    """
    vector = []
    # convert to ascii string
    for c in char_sequence:
        char_vec = [0]*128
        try:
            char_vec[ord(c)] = 1
        except IndexError:
            # Not an ascii character
            raise Exception("'{}' from file {} not ASCII".format(sentence, unique_file))
        vector.append(np.array(char_vec))
    return vector


def build_batch(unique_file, batch_size, window_size=15):
    '''
    :param unique_file: file pointer to a .unique file
    '''
    while True:
        batch_context = np.zeros((batch_size, window_size, 128))
        batch_character = np.zeros((batch_size, 128))
        # read :batch_size: lines from file
        for i in range(batch_size):
            # read line by line, strip newline
            sentence = unique_file.readline().strip()
            # skip sentences of length < window_size
            # numpy cannot broadcast input array of shape 1 into shape 2
            if len(sentence) < window_size:
                continue
            # else, convert to ascii format
            sentence = str(sentence, 'ascii')
            for context, character in training_set(sentence, window_size=window_size):
                batch_context[i] = np.array(char2vec(context))
                batch_character[i] = np.array(char2vec([character]))
        yield batch_context, batch_character


def predict(model, context, length=1):
    for i in range(length):
        res = model.predict(np.array([char2vec(context)]))
        mx = np.argmax(res)
        context = context+chr(mx)
    return context

def charRNN_model():
    """
    This Builds a character RNN based on kaparthy's infamous blog post
    :return: None
    """
    model = keras.models.Sequential()
    model.add(GRU(512, input_shape=(None, 128), return_sequences=True))
    model.add(GRU(512, return_sequences=True))
    model.add(GRU(512))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model_twitter(unique_path, batch_size, steps_per_epoch=50, epochs=10, model=charRNN_model()):
    """
    This function trains the data on the character network
    :return: 
    """
    for unique_file in [f for f in os.listdir(unique_path) if os.path.isfile(os.path.join(unique_path, f)) and f.split('.')[1] == 'unique']:
        with open(os.path.join(unique_path, unique_file), 'rb') as f:
            print("training on {}...".format(unique_file))
            # total lines trained per file = batch_size * steps_per_epoch * epochs
            model.fit_generator(build_batch(f, batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs,
                                callbacks=[TensorBoard("./log"), ModelCheckpoint("hdf5/weights.{epoch:02d}.hdf5")])

if __name__ == "__main__":
    unique_path = "train/txt"
    """
    train on 15000 lines per file
    batch_size = 30
    steps_per_epoch=50
    epochs=10
    """
    batch_size = 30
    train_model_twitter(unique_path, batch_size)