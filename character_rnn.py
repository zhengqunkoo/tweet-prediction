import ParseJSON
import keras.models
import numpy as np
from keras.layers import GRU, Dropout, Dense
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
    for _,sentence in json_file.parse_json():
        inputs, labels =[],[]
        result = ""
        for item in sentence: result = result + " " + item
        sentence = result
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


def train_model_twitter(file, model=charRNN_model()):
    """
    This function trains the data on the character network
    :return: 
    """
    json_file = ParseJSON.ParseJSON(file)
    model.fit_generator(build_batch(json_file), steps_per_epoch=100, epochs=2000,
                        callbacks=[TensorBoard("./log"), ModelCheckpoint("weights.{epoch:02d}.hdf5")])

if __name__ == "__main__":
    train_model_twitter("/media/arjo/EXT4ISAWESOME/tmlc1-training-01/tmlc1-training-001.json")
