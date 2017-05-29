import ParseJSON
import keras.models
import numpy as np
import sys
from keras.layers import GRU, Dropout, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from itertools import chain
import os

<<<<<<< HEAD
def training_set(string, window_size):
=======

def training_set(string, window_size=10):
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51
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


<<<<<<< HEAD
def build_batch(unique_file, batch_size, window_size=25):
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
            if sentence:
                for context, character in training_set(sentence, window_size):
                    batch_context[i] = np.array(char2vec(context, window_size))
                    # window_size 1 because length char == 1
                    batch_character[i] = np.array(char2vec([character], window_size=1))
        yield batch_context, batch_character
=======
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
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51


def predict(model, context, window_size=25, length=1):
    for i in range(length):
        res = model.predict(np.array([char2vec(context, window_size)]))
        mx = np.argmax(res)
        context = context+chr(mx)
    return context

<<<<<<< HEAD
=======

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


def beam_search(model, keystrokes, thickness =2, pruning=10, context=["a",1]):
    """
    Beam search: this takes a model and uses beam search as a method to find most probable
       string. The aim is to allow for better predictions without being overly greedy.
    """
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
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51
def charRNN_model():
    """
    This Builds a character RNN based on kaparthy's infamous blog post
    :return: None
    """
    model = keras.models.Sequential()
<<<<<<< HEAD
    model.add(GRU(512, input_shape=(None, 128), return_sequences=True))
    model.add(GRU(512, return_sequences=True))
    model.add(GRU(512))
    model.add(Dropout(0.1))
=======
    model.add(GRU(512, input_shape=(None, 128)))
    model.add(Dropout(0.2))
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51
    model.add(Dense(128, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model_twitter(unique_path, train_validate_split, batch_size, steps_per_epoch, epochs, loops=0, unique_number=None, model=charRNN_model()):
    """
    This function trains the data on the character network
    :return: 
    """
<<<<<<< HEAD
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
                    history_callback = model.fit_generator(build_batch(f, train_size),
                                                           steps_per_epoch=steps_per_epoch,
                                                           epochs=epochs,
                                                           callbacks=[ModelCheckpoint("hdf5/weights.{}.{}.hdf5".format(unique_file, loops))],
                                                           validation_data=build_batch(f, validation_size),
                                                           validation_steps=steps_per_epoch
                                                           )

                    # log loss history in txt file, since tensorboard graph overlaps
                    loss_history = history_callback.history["loss"]
                    np_loss_history = np.array(loss_history)
                    with open("log/loss_history.txt", 'ab') as f:
                        np.savetxt(f, np_loss_history, delimiter="\n")
        # restart from first file
        unique_number = 0
        loops += 1
=======
    # let output of ParseJSON always be entitiesFull of values
    keys = [['entitiesFull', 'value']]
    json_file = ParseJSON.ParseJSON(file, keys)
    model.fit_generator(build_batch(json_file), steps_per_epoch=100, epochs=2000,
                        callbacks=[TensorBoard("./log"), ModelCheckpoint("weights.{epoch:02d}.hdf5")])
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51


if __name__ == "__main__":
<<<<<<< HEAD
    unique_path = "train/txt"
    unique_number = 55 # continue training for files strictly after this number
    unique_str = str(unique_number)
    unique_str = "0"*(2 - len(unique_str)) + unique_str
    loops = 102 # how many times trained over entire fileset
    hdf5_file = "hdf5/weights.tmlc1-training-0{}.unique.{}.hdf5".format(unique_str, loops)
    """
    train on 16000 lines per file
    """
    batch_size = 50
    steps_per_epoch = 80
    epochs = 4
    print(predict(keras.models.load_model(hdf5_file), "hello baby", 100))
    train_model_twitter(unique_path, 0.9, batch_size, steps_per_epoch, epochs,
        loops=loops, unique_number=unique_number, model=keras.models.load_model(hdf5_file))
=======
    if len(sys.argv) == 1:
        print("Usage: %s [json files]"%sys.argv[0])
    elif len(sys.argv) == 2:
        train_model_twitter(sys.argv[1])
    else:
        train_multiple_files(sys.argv[1:])
>>>>>>> 0c34bb1ce42bbbed06ad217a64d08530663afb51
