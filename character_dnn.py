import ParseJSON
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Merge, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

"""
merge multiple neural networks
https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
"""

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
    all vectors are numpy arrays
    >>> char2vec("a")
    [[....0,0,1,0,0,0,0,0,0,0,0,...]]
    """
    input_length = len(ch2ix)
    vector = np.zeros((len(char_sequence), input_length))
    for i in range(len(char_sequence)):
        c = chr(char_sequence[i])
        char_vec = np.zeros((input_length))
        try:
            char_vec[ch2ix[c]] = 1
        except:
            raise Exception(str(char_sequence) + ' ' + str(c) + ' ' + str(i)+' ' + str(ch2ix))
        vector[i] = char_vec
    return vector


def build_batch(f, batch_size, ch2ix):
    '''
    :param f: file pointer to a .unique file
    '''
    input_length = len(ch2ix)
    num2ix = {str(ix):ix for ix in range(10)}
    bin2ix = {str(ix):ix for ix in range(2)}

    # assume tweets are 140 characters long
    batch_training = np.zeros((batch_size, 5, 140))
    batch_test = np.zeros((batch_size, 140))
    # read :batch_size: lines from file
    # if reach EOF before batch_size read, just return the np.arrays with extra zeros at the end
    for i in range(batch_size):
        line = f.readline()
        # read line by line, strip newline, split by '\t' into keys
        line = line.strip().split('\t'.encode('ascii', 'backslashreplace'))
        # entitiesShortened has all information
        # entitiesFull only contains values of 'type':'word'
        _, user, created, media, reply, quote, entities_full, entities_shortened = line

        # perhaps user should go into neural net, but need find max length of twitter username
        # then pad zeros to all shorter twitter usernames

        # the values below have fixed lengths for all users
        created = char2vec(created, num2ix)
        media = char2vec(media, bin2ix)
        reply = char2vec(reply, bin2ix)
        quote = char2vec(quote, bin2ix)
        entities_shortened = char2vec(entities_shortened, ch2ix)
        entities_full = char2vec(entities_full, ch2ix)
        
        batch_training[i] = np.array([created, media, reply, quote, entities_shortened])
        batch_test[i] = entities_full

    yield batch_training, batch_test

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
        elif current[-1] == NULLBYTE:
            continue
        else:
            res = model.predict(np.array([char2vec(current+keystrokes[get_key_strokes(current)])]))
            predictions = k_best_options(res,thickness)
        for prediction, probability in predictions:
            stack.append((current+prediction, c_prob*probability))
    context = sorted(stack,key=lambda x: x[2])

def charDNN_model():
    """
    This Builds a character RNN based on kaparthy's infamous blog post
    :return: None
    """
    created_branch = Sequential()
    media_branch = Sequential()
    reply_branch = Sequential()
    quote_branch = Sequential()
    entities_shortened_branch = Sequential()

    created_branch.add(Dense(16, input_shape=(16,), activation='softmax'))
    media_branch.add(Dense(2, input_shape=(2,), activation='relu'))
    reply_branch.add(Dense(2, input_shape=(2,), activation='relu'))
    quote_branch.add(Dense(140, input_shape=(140,), activation='softmax'))
    entities_shortened_branch.add(Dense(140, input_shape=(140,), activation='softmax'))

    created_branch.add(BatchNormalization())
    media_branch.add(BatchNormalization())
    reply_branch.add(BatchNormalization())
    quote_branch.add(BatchNormalization())
    entities_shortened_branch.add(BatchNormalization())

    model = Sequential()
    model.add(Merge([created_branch, media_branch, reply_branch, quote_branch, entities_shortened_branch], mode='concat'))
    model.add(Dense(280, activation='softmax'))
    model.add(Dense(280, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(Dense(140, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model_twitter(ch2ix, unique_path, train_validate_split, batch_size, steps_per_epoch, epochs, loops=0, unique_number=None, model=charDNN_model()):
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
                    history_callback = model.fit_generator(build_batch(f, train_size, ch2ix),
                                                           steps_per_epoch=steps_per_epoch,
                                                           epochs=epochs,
                                                           callbacks=[ModelCheckpoint("hdf5/weights.{}.{}.hdf5".format(unique_file, loops))],
                                                           validation_data=build_batch(f, validation_size, ch2ix),
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



if __name__ == "__main__":
    NULLBYTE = '\0'
    # assume replace_types identical to replace_types in JSON2Text
    # if replace_types different, wrong prediction

    # hashtag, userMention are not replaced

    # character array will enumerate according to sorted characters.
    # if the order of the ASCII special chars in replace_types changes,
    # e.g.      'number':'\33', 'url':'\34' => number < url
    # becomes   'number':'\34', 'url':'\33' => url < number
    # then enumeration index changes, and old model will not be compatible with new model

    replace_types = {'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}
    # ASCII non-special chars from byte 32 to byte 127 will definitely be included
    # ASCII special chars are just NULLBYTE and replace_types
    ascii_nonspec_chars = [chr(x) for x in range(32, 128)]
    ascii_spec_chars = [NULLBYTE] + list(replace_types.values())
    chars = ascii_nonspec_chars + ascii_spec_chars

    ch2ix, _ = get_ch2ix_ix2ch(chars)
    
    # parameters to continue training
    unique_path = "train/txt"
    unique_number = 0 # continue training for files strictly after this number
    unique_str = str(unique_number)
    unique_str = "0"*(2 - len(unique_str)) + unique_str
    loops = 200 # how many times trained over entire fileset
    hdf5_file = "hdf5/weights.tmlc1-training-0{}.unique.{}.hdf5".format(unique_str, loops)
    """
    train on 16000 lines per file
    """
    train_validate_split = 0.9
    batch_size = 50
    steps_per_epoch = 80
    epochs = 4
    # print(predict(keras.models.load_model(hdf5_file), "hello baby", 100))
    train_model_twitter(ch2ix,
                        unique_path,
                        train_validate_split,
                        batch_size,
                        steps_per_epoch,
                        epochs,
                        loops=loops,
                        unique_number=unique_number
                        )# , model=keras.models.load_model(hdf5_file))