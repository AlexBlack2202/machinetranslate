#!/usr/bin/python
# -*- coding: utf8 -*-
import collections

import helper
import numpy as np
#import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y



def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')



def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])




# Load no accent data 
non_accent_sentences = helper.load_data('data/data_noaccent_clean.txt')
# Load accent data
accent_sentences = helper.load_data('data/data_clean.txt')

print('Dataset Loaded')

# add space at begin and end in the sentence


non_accent_sentences = [" "+sentence +" " for sentence in non_accent_sentences]
accent_sentences = [" "+sentence +" " for sentence in accent_sentences]


split_non_accent_words =  [" "+word.strip()+" " for sentence in non_accent_sentences for word in sentence.split()]
split_accent_words =  [" "+word.strip()+" " for sentence in accent_sentences for word in sentence.split()]

words_dict = dict()


for _row in range(len(split_non_accent_words)):
    for _col in range(len(split_non_accent_words[_row])):
        word = split_non_accent_words[_row][_col]
        if(word not in words_dict):
            words_dict.append(word,{})
        words_dict[word].add(split_accent_words[_row][_col])


unique_words = {}

for key,val in words_dict:
    if len(val) == 1:
        unique_words.add(key)


# dumplicate_data= set(split_non_accent_words).intersection(split_accent_words)

# unique_words =set([x for x in split_non_accent_words if x not in dumplicate_data])

# with open("logs/dumplicate_data.txt",'w+',encoding='utf-8') as f:
#     for item in dumplicate_data:
#         f.writelines(item)

with open("logs/unique_words.txt",'w+',encoding='utf-8') as f:
    for item in unique_words:
        f.writelines(item)

# for sentence in  non_accent_sentences:
    

nonaccent_words_counter = collections.Counter(split_non_accent_words)


accent_words_counter = collections.Counter(split_accent_words)

#thống kê dữ liệu

print('{} non accent words'.format(len(split_non_accent_words)))
print('{} non accent unique words'.format(len(nonaccent_words_counter)))

with open("logs/non_accent_counter.txt",'w+') as f:
    for k,v in  nonaccent_words_counter.most_common():
        f.write( "{} {}\n".format(k,v) )

print('{} accent words'.format(len(split_accent_words)))
print('{} accent unique words'.format(len(accent_words_counter)))

with open("logs/accent_counter.txt",'w+',encoding='utf-8') as f:
    for k,v in  accent_words_counter.most_common():
        f.write( "{} {}\n".format(k.encode("utf-8").decode("utf-8"),v) )

# def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
#     """
#     Build and train a basic RNN on x and y
#     :param input_shape: Tuple of input shape
#     :param output_sequence_length: Length of output sequence
#     :param english_vocab_size: Number of unique English words in the dataset
#     :param french_vocab_size: Number of unique French words in the dataset
#     :return: Keras model built, but not trained
#     """
#     # TODO: Build the layers
#     learning_rate = 1e-3
#     input_seq = Input(input_shape[1:])
#     rnn = GRU(64, return_sequences = True)(input_seq)
#     logits = TimeDistributed(Dense(french_vocab_size))(rnn)
#     model = Model(input_seq, Activation('softmax')(logits))
#     model.compile(loss = sparse_categorical_crossentropy, 
#                  optimizer = Adam(learning_rate), 
#                  metrics = ['accuracy'])
    
#     return model

# preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =preprocess(non_accent_sentences, accent_sentences)

# max_english_sequence_length = preproc_english_sentences.shape[1]
# max_french_sequence_length = preproc_french_sentences.shape[1]
# english_vocab_size = len(english_tokenizer.word_index)
# french_vocab_size = len(french_tokenizer.word_index)

# # Reshaping the input to work with a basic RNN
# tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# # Train the neural network
# simple_rnn_model = simple_model(
#     tmp_x.shape,
#     max_french_sequence_length,
#     english_vocab_size,
#     french_vocab_size)
# simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=16, epochs=10, validation_split=0.2)


# # Print prediction(s)
# print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))


# # serialize model to YAML
# model_yaml = simple_rnn_model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# simple_rnn_model.save_weights("model.h5")
# print("Saved model to disk")