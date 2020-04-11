import json
import keras
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras import initializers
from keras import optimizers
from keras import backend as K

from keras.layers import Embedding,Reshape, Activation, RepeatVector, Permute, Lambda, GlobalMaxPool1D
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, LSTM, Bidirectional, GRU
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

# Numpy Seed
from numpy.random import seed
seed(1)

# TensorFlow Seed
import tensorflow as tf
tf.random.set_seed(2)

# Random Seed
import random
random.seed(3)

# Python Hash Seed
import os
os.environ['PYTHONHASHSEED'] = '0'

KERAS_INIT_SEED = 1

INPUT_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/data/processed/'
LEGIT_INPUT_FILE = 'legit-iwspa.txt'
PHISH_INPUT_FILE = 'phish-iwspa.txt'

MODEL_DIRECTORY = '/Users/shahriar/Documents/Research/Code/themis/models/'
MODEL_FILE = MODEL_DIRECTORY + 'themis-small.h5'

TOKEN_HW_FILE = MODEL_DIRECTORY + 'tokenhw.pkl'
TOKEN_HC_FILE = MODEL_DIRECTORY + 'tokenhc.pkl'
TOKEN_BW_FILE = MODEL_DIRECTORY + 'tokenbw.pkl'
TOKEN_BC_FILE = MODEL_DIRECTORY + 'tokenbc.pkl'

PHISHY = 1
NOT_PHISHY = 0
SMALL_DATASET_SIZE = 300
TRAIN_TEST_SPLIT_PERC = 0.2

MAX_TOKENS_HEADER_WORD = 50
MAX_TOKENS_HEADER_CHAR = 100
# Original was 150
MAX_TOKENS_BODY_WORD = 150
# Original was 300
MAX_TOKENS_BODY_CHAR = 300
EMBEDDING_DIM = 256
NUM_EPOCHS = 15

def read_input_emails(input_file_path):
    with open(input_file_path, "r") as file:
        email_list = json.load(file)

    return email_list

def build_model(sequences, vocab_sizes):
    max_hw = MAX_TOKENS_HEADER_WORD
    max_hc = MAX_TOKENS_HEADER_CHAR
    max_bw = MAX_TOKENS_BODY_WORD
    max_bc = MAX_TOKENS_BODY_CHAR

    vocab_size_hw, vocab_size_hc, vocab_size_bw, vocab_size_bc = vocab_sizes
    hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input = sequences

    hc_embedding_layer = Embedding(vocab_size_hc,
                                       EMBEDDING_DIM,
                                       input_length=max_hc, trainable=True,
                                       embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=KERAS_INIT_SEED))
    hc_sequence_input = Input(shape=(max_hc,), name="headerchar_input")
    hc_embedded_sequences = hc_embedding_layer(hc_sequence_input)

    hc_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(hc_embedded_sequences)
    hc_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(hc_embedded_sequences)
    # hc_z_concat = merge([hc_z_pos, hc_embedded_sequences, hc_z_neg], mode='concat', concat_axis=-1)
    hc_z_concat =  Concatenate(axis=-1)([hc_z_pos, hc_embedded_sequences, hc_z_neg])

    hc_z = Dense(512, activation='tanh')(hc_z_concat)
    hc_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(hc_z)
    # -----headerword------
    hw_embedding_layer = Embedding(vocab_size_hw,
                                       EMBEDDING_DIM,
                                       input_length= max_hw, trainable=True,
                                       embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=KERAS_INIT_SEED))
    hw_sequence_input = Input(shape=(max_hw,), name="headerword_input")
    hw_embedded_sequences = hw_embedding_layer(hw_sequence_input)

    hw_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(hw_embedded_sequences)
    hw_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(hw_embedded_sequences)
    # hw_z_concat = merge([hw_z_pos, hw_embedded_sequences, hw_z_neg], mode='concat', concat_axis=-1)
    hw_z_concat = Concatenate(axis=-1)([hw_z_pos, hw_embedded_sequences, hw_z_neg])

    hw_z = Dense(512, activation='tanh')(hw_z_concat)
    hw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(hw_z)
    # -----bodychar------
    bc_embedding_layer = Embedding( vocab_size_bc,
                                       EMBEDDING_DIM,
                                       input_length=max_bc, trainable=True,
                                       embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=KERAS_INIT_SEED))
    bc_sequence_input = Input(shape=(max_bc,), name="bodychar_input")
    bc_embedded_sequences = bc_embedding_layer(bc_sequence_input)

    bc_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(bc_embedded_sequences)
    bc_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(bc_embedded_sequences)
    # bc_z_concat = merge([bc_z_pos, bc_embedded_sequences, bc_z_neg], mode='concat', concat_axis=-1)
    bc_z_concat = Concatenate(axis=-1)([bc_z_pos, bc_embedded_sequences, bc_z_neg])
    bc_z = Dense(512, activation='tanh')(bc_z_concat)
    bc_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(bc_z)
    # -----bodyword------
    bw_embedding_layer = Embedding( vocab_size_bw,
                                       EMBEDDING_DIM,
                                       input_length=max_bw, trainable=True,
                                       embeddings_initializer=initializers.RandomUniform(minval=-0.2, maxval=0.2, seed=KERAS_INIT_SEED))
    bw_sequence_input = Input(shape=(max_bw,), name="bodyword_input")
    bw_embedded_sequences = bw_embedding_layer(bw_sequence_input)

    bw_z_pos = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(bw_embedded_sequences)
    bw_z_neg = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(bw_embedded_sequences)
    # bw_z_concat = merge([bw_z_pos, bw_embedded_sequences, bw_z_neg], mode='concat', concat_axis=-1)
    bw_z_concat = Concatenate(axis=-1)([bw_z_pos, bw_embedded_sequences, bw_z_neg])

    bw_z = Dense(512, activation='tanh')(bw_z_concat)
    bw_pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(bw_z)

    # ------att----------
    # concat_w_c = merge([hc_pool_rnn, hw_pool_rnn, bc_pool_rnn, bw_pool_rnn], mode='concat')
    concat_w_c = Concatenate()([hc_pool_rnn, hw_pool_rnn, bc_pool_rnn, bw_pool_rnn])
    concat_w_c = Reshape((2, 512 * 2))(concat_w_c)

    attention = Dense(1, activation='tanh')(concat_w_c)
    attention = Flatten()(attention)
    attention = Activation('sigmoid')(attention)
    attention = RepeatVector(512 * 2)(attention)
    attention = Permute([2, 1])(attention)
    # keras.layers.Multiply()([tanh_out, sigmoid_out])
    # sent_representation = merge([concat_w_c, attention], mode='mul')

    multiply_layer = keras.layers.Multiply()
    sent_representation = multiply_layer([concat_w_c, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512 * 2,))(
        sent_representation)  # sent_representation
    # --------merge_4models------------------

    model_final_ = Dense(512, activation='relu')(sent_representation)
    model_final_ = Dropout(0.5)(model_final_)
    model_final = Dense(1, activation='sigmoid')(model_final_)

    model = Model(input=[hc_sequence_input, hw_sequence_input, bc_sequence_input, bw_sequence_input],
                       outputs=model_final)
    adam = optimizers.adam(lr=0.0001)
    model.compile(loss='binary_crossentropy',
                       optimizer=adam,
                       metrics=['binary_accuracy'])

    model.summary()
    # plot_model(model, to_file='dnn.png')
    return model

def encode_sequences(tokenizers, train_header, train_body):
    tokenhw, tokenhc, tokenbw, tokenbc = tokenizers

    max_hw = MAX_TOKENS_HEADER_WORD
    max_hc = MAX_TOKENS_HEADER_CHAR
    max_bw = MAX_TOKENS_BODY_WORD
    max_bc = MAX_TOKENS_BODY_CHAR

    vocab_size_hw = len(tokenhw.word_index)+1
    encoded_hw= tokenhw.texts_to_sequences(train_header)
    hw_sequence_input=pad_sequences(encoded_hw, maxlen=max_hw,padding='post')

    vocab_size_hc = len(tokenhc.word_index)+1
    encoded_hc=tokenhc.texts_to_sequences(train_header)
    hc_sequence_input=pad_sequences(encoded_hc, maxlen=max_hc,padding='post')

    vocab_size_bw = len(tokenbw.word_index)+1
    encoded_bw= tokenbw.texts_to_sequences(train_body)
    bw_sequence_input=pad_sequences(encoded_bw, maxlen=max_bw,padding='post')

    vocab_size_bc = len(tokenbc.word_index)+1
    encoded_bc=tokenhc.texts_to_sequences(train_body)
    bc_sequence_input=pad_sequences(encoded_bc, maxlen=max_bc,padding='post')

    sequences = (hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input)

    return sequences



def train_tokenizers(train_header, train_body):
    #========================================
    # Word-level tokenization on the train_header_text
    # ========================================
    tokenhw=Tokenizer()
    tokenhw.fit_on_texts(train_header)

    #========================================
    # Char-level tokenization on the train_header_tex
    # ========================================
    tokenhc = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tokenhc.fit_on_texts(train_header)

    #========================================
    # Word-level tokenization on the train_body_text
    # ========================================
    tokenbw=Tokenizer()
    tokenbw.fit_on_texts(train_body)
    #========================================
    # Char-level tokenization on the train_body_tex
    # ========================================
    tokenbc = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tokenbc.fit_on_texts(train_body)


    tokenizers = (tokenhw, tokenhc, tokenbw, tokenbc)
    return tokenizers

def get_vocab_sizes(tokenizers):
    tokenhw, tokenhc, tokenbw, tokenbc = tokenizers
    vocab_size_hw = len(tokenhw.word_index)+1
    vocab_size_hc = len(tokenhc.word_index)+1
    vocab_size_bw = len(tokenbw.word_index)+1
    vocab_size_bc = len(tokenbc.word_index)+1
    vocab_sizes = (vocab_size_hw, vocab_size_hc, vocab_size_bw, vocab_size_bc)

    return vocab_sizes


def fit_model(model, sequences, labels):
    hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input = sequences
    model.fit([hc_sequence_input, hw_sequence_input, bc_sequence_input, bw_sequence_input],labels,epochs=NUM_EPOCHS,verbose=2)

def save_tokenizers(tokenizers):
    tokenhw, tokenhc, tokenbw, tokenbc = tokenizers

    pickle.dump(tokenhw, open(TOKEN_HW_FILE, 'wb'))
    pickle.dump(tokenhc,open(TOKEN_HC_FILE,'wb'))
    pickle.dump(tokenbw, open(TOKEN_BW_FILE, 'wb'))
    pickle.dump(tokenbc, open(TOKEN_BC_FILE, 'wb'))

def load_tokenizers():
    tokenhw = pickle.load(open(TOKEN_HW_FILE, 'rb'))
    tokenhc = pickle.load(open(TOKEN_HC_FILE, 'rb'))
    tokenbw = pickle.load(open(TOKEN_BW_FILE, 'rb'))
    tokenbc = pickle.load(open(TOKEN_BC_FILE, 'rb'))

    tokenizers = (tokenhw, tokenhc, tokenbw, tokenbc)
    return tokenizers

def load_model_and_tokenizer():
    print('Using loaded model and tokenizers')
    model = load_model(MODEL_FILE)
    tokenizers = load_tokenizers()
    return (model, tokenizers)

def df_to_header_body_label(train_df):
    train_header = train_df['header'].tolist()
    train_body = train_df['body'].tolist()
    train_labels = train_df['phishy'].tolist()

    return (train_header, train_body, train_labels)

def train_and_save_tokenizers(train_df):
    train_header, train_body, train_labels = df_to_header_body_label(train_df)
    tokenizers = train_tokenizers(train_header, train_body)
    vocab_sizes = get_vocab_sizes(tokenizers)
    print(vocab_sizes)
    
    save_tokenizers(tokenizers)

    return tokenizers

def train_and_save_model(train_df, tokenizers):
    train_header, train_body, train_labels = df_to_header_body_label(train_df)
    sequences = encode_sequences(tokenizers, train_header, train_body)
    vocab_sizes = get_vocab_sizes(tokenizers)
    model = build_model(sequences, vocab_sizes)
    fit_model(model, sequences, train_labels)
    model.save(MODEL_FILE, overwrite=True)
    return model

def predict_classes(model, test_df, tokenizers):
    test_header, test_body, test_labels = df_to_header_body_label(test_df)
    sequences = encode_sequences(tokenizers, test_header, test_body)
    hw_sequence_input, hc_sequence_input, bw_sequence_input, bc_sequence_input = sequences
    predicted_prob = model.predict([hc_sequence_input, hw_sequence_input, bc_sequence_input, bw_sequence_input])

    predicted_labels = np.round(predicted_prob)

    return (test_labels, predicted_labels)

def prediction_metrics(y, y_hat):
    cm = confusion_matrix(y, y_hat, labels=[0,1])

    print("Legit emails correctly predicted as legit:", cm[0][0])
    print("Legit emails incorrectly predicted as phish:", cm[0][1])
    print("Total number of legit emails:",  (cm[0][0]+cm[0][1]))

    print()

    print("Phish emails correctly predicted as phish:", cm[1][1])
    print("Phish emails incorrectly predicted as legit:", cm[1][0])
    print("Total number of phish emails:",  (cm[1][0]+cm[1][1]))


def run():
    legit_emails = read_input_emails(INPUT_DIRECTORY + LEGIT_INPUT_FILE)
    phish_emails = read_input_emails(INPUT_DIRECTORY + PHISH_INPUT_FILE)

    # Filter out emails that passed all quality checks
    legit_emails = [email for email in legit_emails if email['qualify'] == True]
    phish_emails = [email for email in phish_emails if email['qualify'] == True]

    print("Num Legit Emails:", len(legit_emails))
    print("Num Phish Emails:", len(phish_emails))

    legit_headers = [email['header']['Subject'] for email in legit_emails]
    legit_bodies = [email['body'] for email in legit_emails]
    legit_flags = [NOT_PHISHY] * len(legit_emails)

    phish_headers = [email['header']['Subject'] for email in phish_emails]
    phish_bodies = [email['body'] for email in phish_emails]
    phish_flags = [PHISHY] * len(phish_emails)

    legit_df = pd.DataFrame(list(zip(legit_headers, legit_bodies, legit_flags)), columns = ['header', 'body', 'phishy'])
    phish_df = pd.DataFrame(list(zip(phish_headers, phish_bodies, phish_flags)), columns = ['header', 'body', 'phishy'])

    all_df = pd.concat([legit_df, phish_df])
    train_test_df = all_df.sample(SMALL_DATASET_SIZE, replace=False, random_state=1)
    validation_df = all_df.drop(train_test_df.index)
    validation_df = shuffle(validation_df, random_state=1)
    validation_df.reset_index(inplace=True, drop=True)

    # train_legit_df = legit_df.sample(SMALL_DATASET_SIZE, replace=False, random_state=1)
    # train_phish_df = phish_df.sample(SMALL_DATASET_SIZE, replace=False, random_state=1)
    #
    # validation_legit_df = legit_df.drop(train_legit_df.index)
    # validation_phish_df = phish_df.drop(train_phish_df.index)
    #
    # train_test_df = pd.concat([train_legit_df, train_phish_df])
    # train_test_df = shuffle(train_test_df, random_state=1)
    # train_test_df.reset_index(inplace=True, drop=True)
    #
    # validation_df = pd.concat([validation_legit_df, validation_phish_df])
    # validation_df = shuffle(validation_df, random_state=1)
    # validation_df.reset_index(inplace=True, drop=True)

    train_df, test_df = train_test_split(train_test_df, test_size=TRAIN_TEST_SPLIT_PERC)

    print('Shapes of Train and Test Data Sets')
    print(train_df.shape) # 160
    print(test_df.shape) # 40 = 21 phishy + 19 legit
    print(validation_df.shape)

    train_legit = train_df[train_df.phishy == 0]
    train_phish = train_df[train_df.phishy == 1]

    print('Composition of train data set')
    print(train_legit.shape)
    print(train_phish.shape)

    test_legit = test_df[test_df.phishy == 0]
    test_phish = test_df[test_df.phishy == 1]

    print('Composition of test data set')
    print(test_legit.shape)
    print(test_phish.shape)


    tokenizers = train_and_save_tokenizers(train_df)
#    model = train_and_save_model(train_df, tokenizers)

    # model, tokenizers = load_model_and_tokenizer()

    print("Test Run")
#    y_test, y_hat_test = predict_classes(model, test_df, tokenizers)
#    prediction_metrics(y_test, y_hat_test)

    print("Validation Run")
#    y_val, y_hat_val = predict_classes(model, validation_df, tokenizers)
#    prediction_metrics(y_val, y_hat_val)








run()
