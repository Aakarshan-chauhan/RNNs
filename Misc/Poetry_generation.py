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

MAX_SEQUENCE_LENGTH = 15
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 300
EPOCHS = 100
LATENT_DIM = 25

input_texts = []
target_texts = []
file = pd.read_csv('D:/datasets/quotes.csv')
file1 = file['quote']
# for line in open('D:/datasets/poems.txt'):
for line in file1[:2000]:
    line = line.rstrip()
    if not line:
        continue
    input_line = '<sos> ' + line
    target_line = line + ' <eos>'
    input_texts.append(input_line)
    target_texts.append(target_line)

all_lines = input_texts + target_texts

# Tokenizing
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

max_sequence_length_from_data = max(len(s) for s in input_sequences)
print('Max sequence length:', max_sequence_length_from_data)

word2idx = tokenizer.word_index

max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')
print('Shape of data tensor:', input_sequences.shape)

# Load pre-trained word vectors
print('Loading word vectors')
word2vec = {}

assert ('<sos>' in word2idx)
assert ('<eos>' in word2idx)
with open(os.path.join('D:/datasets/Toxic comments/WORD_VECTORS/glove.6B.%sd.txt' % EMBEDDING_DIM),
          encoding='utf8') as f:
    for line in f:
        value = line.split()
        word = value[0]
        vec = np.asarray(value[1:], dtype='float32')
        word2vec[word] = vec
print('Found', len(word2vec), 'Word vectors')

print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2vec) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
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

print('Building Model...')

input_ = Input(shape=(max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
initial_c = Input(shape=(LATENT_DIM,))

x = embedding_layer(input_)
lstm = (LSTM(LATENT_DIM, return_sequences=True, return_state=True))
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(num_words, activation='softmax')
output = dense(x)

model = Model([input_, initial_h, initial_c], output)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy']
)

print('Training model...')
z = np.zeros((len(input_sequences), LATENT_DIM))
r = model.fit(
    [input_sequences, z, z],
    one_hot_targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history['acc'], label="acc")
plt.plot(r.history['val_acc'], label="val_acc")
plt.legend()
plt.show()

input2 = Input(shape=(1,))
x = embedding_layer(input2)
x, h, c = lstm(x, initial_state=[initial_h, initial_c])
output2 = dense(x)
sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])

idx2word = {v: k for k, v in word2idx.items()}


def sample_line():
    np_input = np.array([[word2idx['<sos>']]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))

    eos = word2idx['<eos>']
    output_sentence = []

    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input, h, c])

        probs = o[0, 0]
        if np.argmax(probs) == 0:
            print('wtf')
        probs[0] = 0
        probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        if idx == eos:
            break

        output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

        np_input[0, 0] = idx
    return ' '.join(output_sentence)

model.save('Quote_Generator.h5')
while True:
    for _ in range(1):
        print(sample_line())

    ans = input("===Generate another? [y/n]")
    if ans and ans[0].lower().startswith('n'):
        break

