import ParseJSON
import keras.models
import numpy as np
from keras.layers import GRU, Dropout, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from itertools import chain
import os

def training_set(string, window_size):
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


def char2vec(char_sequence, window_size):
    """
    :param char_sequence: a sequence of ASCII bytes
    This function takes a char_sequence and encodes it as a series of onehot vectors
    >>> char2vec("a")
    [[....0,0,1,0,0,0,0,0,0,0,0,...]]
    """
    # if char_sequence not window_size, return numpy array with extra zero padding appended
    vector = np.zeros((window_size, 128))
    for i in range(len(char_sequence)):
        char_vec = [0]*128
        char_vec[ord(char_sequence[i])] = 1
        vector[i] = np.array(char_vec)
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
            # else, convert to ascii format
            sentence = str(sentence, 'ascii')
            for context, character in training_set(sentence, window_size):
                batch_context[i] = np.array(char2vec(context, window_size))
                # window_size 1 because length char == 1
                batch_character[i] = np.array(char2vec([character], window_size=1))
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
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model_twitter(unique_path, batch_size, train_validate_split, loops=0, unique_number=None, steps_per_epoch=80, epochs=1, model=charRNN_model()):
    """
    This function trains the data on the character network
    :return: 
    """
    # loop over files to fit
    while True:
        for unique_file in [f for f in os.listdir(unique_path) if os.path.isfile(os.path.join(unique_path, f)) and f.split('.')[1] == 'unique']:
            if not unique_number or int(unique_file.split('.')[0][-3:]) > unique_number:
                with open(os.path.join(unique_path, unique_file), 'rb') as f:
                    print("training on {}...".format(unique_file))
                    # total lines trained per file = batch_size * steps_per_epoch * epochs
                    # split batch_size into train_size and validation_size
                    train_size = int(batch_size * train_validate_split)
                    validation_size = batch_size - train_size
                    model.fit_generator(build_batch(f, train_size),
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        callbacks=[TensorBoard("./log"), ModelCheckpoint("hdf5/weights.{}.{}.hdf5".format(unique_file, loops))],
                                        validation_data=build_batch(f, validation_size),
                                        validation_steps=steps_per_epoch
                                        )
        # restart from first file
        unique_number = 0
        loops += 1

if __name__ == "__main__":
    unique_path = "train/txt"
    unique_number = 39 # continue training for files strictly after this number
    loops = 3 # how many times trained over entire fileset
    hdf5_file = "hdf5/weights.tmlc1-training-0{}.unique.{}.hdf5".format(unique_number, loops)
    """
    train on 16000 lines per file
    batch_size = 200
    steps_per_epoch = 8
    epochs = 1
    """
    batch_size = 200
    train_model_twitter(unique_path, batch_size, 0.8, loops=loops, unique_number=unique_number, model=keras.models.load_model(hdf5_file))
    print(predict(keras.models.load_model(hdf5_file), "hello baby", 100))