import os
import sys
import string
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

EMBEDDING_DIM = 50
MAX_VOCAB_SIZE = 30000
MAX_SEQUENCES_LENGTH = 100
file = pd.read_csv('D:/datasets/quotes.csv')
file = file.quote

input_setences, target_sentences = [], []

for line in file[:2000]:
    line = line.rstrip()
    if line is None:
        continue
    input_sentence = '<sos> ' + line
    target_sentence = line + ' <eos>'

    input_setences.append(input_sentence)
    target_sentences.append(target_sentence)

all_lines = input_setences + target_sentences

tokenizer = Tokenizer(MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_setences)
target_sequences = tokenizer.texts_to_sequences(target_sentences)

max_sequence_length_from_data = max(len(s) for s in input_sequences)
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCES_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
word2idx = tokenizer.word_index


word2vec = {}
with open(os.path.join('D:/datasets/Toxic comments/WORD_VECTORS/glove.6B.%sd.txt') % EMBEDDING_DIM, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

num_words = min(MAX_VOCAB_SIZE, len(word2vec) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx:
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
for i, target_sequence in enumerate(target_sequences):
    for t, word in enumerate(target_sequence):
        if word > 0:
            one_hot_targets[i, t, word] = 1

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix]
)

print('Building a Model...')