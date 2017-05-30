import ParseJSON
import numpy as np
import os
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from pprint import pprint

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

    # assume tweets are TWEETSIZE characters long
    batch_train = []
    batch_test = []
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
        # quote is either 0 or full quote
        created = char2vec(created, num2ix)
        media = char2vec(media, bin2ix)
        reply = char2vec(reply, bin2ix)
        quote = char2vec(quote, ch2ix)
        entities_shortened = char2vec(entities_shortened, ch2ix)
        entities_full = char2vec(entities_full, ch2ix)
        
        data = [created, media, reply, quote, entities_shortened]
        try:
            data = [np.vstack((x, np.zeros((TWEETSIZE-x.shape[0], x.shape[1])))) for x in data]
        except Exception as e:
            print(str(e))
            print([x.shape for x in data])
        batch_train.append(data)
        batch_test.append(np.vstack((entities_full, np.zeros((TWEETSIZE-entities_full.shape[0], entities_full.shape[1])))))

    return batch_train, np.array(batch_test)

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

def charDNN_model(ch2ix):
    """
    functional API
    https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
    :return: None
    """
    input_length = len(ch2ix)

    created_branch = Input(shape=(TWEETSIZE, 10), dtype='float32', name='created_branch')
    media_branch = Input(shape=(TWEETSIZE, 2), dtype='float32', name='media_branch')
    reply_branch = Input(shape=(TWEETSIZE, 2), dtype='float32', name='reply_branch')
    quote_branch = Input(shape=(TWEETSIZE, input_length), dtype='float32', name='quote_branch')
    entities_shortened_branch = Input(shape=(TWEETSIZE, input_length), dtype='float32', name='entities_shortened_branch')

    x = concatenate([created_branch, media_branch, reply_branch, quote_branch, entities_shortened_branch], axis=2)
    x = Dense(280, dtype='float32', activation='softmax')(x)
    x = Dense(280, dtype='float32', activation='softmax')(x)
    x = Dense(280, dtype='float32', activation='softmax')(x)
    x = Dense(280, dtype='float32', activation='softmax')(x)
    x = Dropout(0.2)(x)
    
    main_output = Dense(101, dtype='int32', activation='softmax', name='main_output')(x)

    model = Model(inputs=[created_branch,
                          media_branch,
                          reply_branch,
                          quote_branch,
                          entities_shortened_branch
                          ],
                  outputs=[main_output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model_twitter(ch2ix, unique_path, batch_size, epochs, loops=0, unique_number=None, model=None):
    """
    This function trains the data on the character network
    :return: 
    """
    if model == None:
        model = charDNN_model(ch2ix)
    # loop over files to fit
    while True:
        for unique_file in [f for f in os.listdir(unique_path) if os.path.isfile(os.path.join(unique_path, f)) and f.split('.')[1] == 'unique']:
            if not unique_number or int(unique_file.split('.')[0][-3:]) > unique_number:
                with open(os.path.join(unique_path, unique_file), 'rb') as f:
                    print("training on {}...".format(unique_file))

                    input_data, entities_full_data = build_batch(f, batch_size, ch2ix)
                    created_data, media_data, reply_data, quote_data, entities_shortened_data = map(lambda x : np.asarray(x), zip(*input_data))
                    history_callback = model.fit({'created_branch':created_data,
                                                  'media_branch':media_data,
                                                  'reply_branch':reply_data,
                                                  'quote_branch':quote_data,
                                                  'entities_shortened_branch':entities_shortened_data
                                                 },
                                                 {'main_output':entities_full_data},
                                                 epochs=epochs,
                                                 batch_size=batch_size,
                                                 callbacks=[ModelCheckpoint("hdf5/weights.{}.{}.hdf5".format(unique_file, loops))]
                                                 )

                    # log loss history in txt file, since tensorboard graph overlaps
                    loss_history = history_callback.history["loss"]
                    np_loss_history = np.array(loss_history)
                    with open("log/dnn_loss-batch{}-epoch{}.txt".format(batch_size, epochs), 'ab') as f:
                        np.savetxt(f, np_loss_history, delimiter="\n")
        # restart from first file
        unique_number = 0
        loops += 1



if __name__ == "__main__":
    NULLBYTE = '\0'
    NEWLINE = '\37'
    TWEETSIZE = 160
    # assume replace_types identical to replace_types in JSON2Text
    # if replace_types different, wrong prediction

    # hashtag, userMention are not replaced

    # character array will enumerate according to sorted characters.
    # if the order of the ASCII special chars in replace_types changes,
    # e.g.      'number':'\33', 'url':'\34' => number < url
    # becomes   'number':'\34', 'url':'\33' => url < number
    # then enumeration index changes, and old model will not be compatible with new model

    replace_types = {'number':'\33', 'url':'\34', 'punctuation':'\35', 'emoji':'\36'}
    ascii_nonspec_chars = [chr(x) for x in range(32, 128)]
    ascii_spec_chars = [NULLBYTE, NEWLINE] + list(replace_types.values())
    chars = ascii_nonspec_chars + ascii_spec_chars

    ch2ix, _ = get_ch2ix_ix2ch(chars)
    
    # parameters to continue training
    unique_path = "train/txt"
    unique_number = 23 # continue training for files strictly after this number
    unique_str = str(unique_number)
    unique_str = "0"*(2 - len(unique_str)) + unique_str
    loops = 0 # how many times trained over entire fileset
    hdf5_file = "hdf5/weights.tmlc1-training-0{}.unique.{}.hdf5".format(unique_str, loops)
    """
    train on 17000 lines per file
    """
    batch_size = 100
    epochs = 170
    # print(predict(keras.models.load_model(hdf5_file), "hello baby", 100))
    train_model_twitter(ch2ix,
                        unique_path,
                        batch_size,
                        epochs,
                        loops=loops,
                        unique_number=unique_number,
                        model=load_model(hdf5_file))