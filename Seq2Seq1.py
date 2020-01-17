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

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000

input_texts = []
target_texts = []
target_texts_input = []  # sentence in target language offset by 1

t = 0
# for line in open('D:/datasets/poems.txt'):
for line in open('D:/datasets/fra.txt'):

    t += 1
    if t > NUM_SAMPLES:
        break
    if '\t' not in line:
        continue

    input_text, translation = line.split('\t')[0:2]

    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_input.append(target_text_input)
print("num samples:", len(input_texts))

# Tokenizing
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

word2idx_inputs = tokenizer_inputs.word_index
max_len_input = max(len(s) for s in input_sequences)
print("Found", len(word2idx_inputs), "unique input tokens")
########################################################################################

tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_input)
target_sequences = tokenizer_inputs.texts_to_sequences(target_texts)
target_sequences_input = tokenizer_outputs.texts_to_sequences(target_texts_input)

word2idx_outputs = tokenizer_outputs.word_index
num_words_output = len(word2idx_outputs) + 1
max_len_target = max(len(s) for s in target_sequences)
print("FOUND", len(word2idx_outputs), "unique output tokens")
########################################################################################

encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)

decoder_inputs = pad_sequences(target_sequences_input, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# Load pre-trained word vectors
print('Loading word vectors')
word2vec = {}

with open(os.path.join('D:/datasets/Toxic comments/WORD_VECTORS/glove.6B.%sd.txt' % EMBEDDING_DIM),
          encoding='utf8') as f:
    for line in f:
        value = line.split()
        word = value[0]
        vec = np.asarray(value[1:], dtype='float32')
        word2vec[word] = vec
print('Found', len(word2vec), 'Word vectors')

print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2vec) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input
)

decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target, num_words_output))
for i, d in enumerate(target_sequences):
    for t, word in enumerate(d):
        if word > 0:
            decoder_targets_one_hot[i, t, word] = 1

encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LATENT_DIM, return_state=True, dropout=0.5)
encoder_outputs, h, c = encoder(x)

encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(max_len_target,))

decoder_embedding = Embedding(num_words_output, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy']
)
r = model.fit(
    [encoder_inputs, decoder_inputs],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history['acc'], label="acc")
plt.plot(r.history['val_acc'], label="val_acc")
plt.legend()
plt.show()

encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(
    decoder_inputs_single_x,
    initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

idx2word_eng = {v: k for k, v in word2idx_inputs}
idx2word_hin = {v: k for k, v in word2idx_outputs}


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = word2idx_outputs['<sos>']

    eos = word2idx_outputs['<eos>']

    output_sentence = []
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
        )

        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_hin[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx

        states_value = [h, c]
    return ' '.join(output_sentence)

while True:

    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i+1]
    translation = decode_sequence(input_seq)
    print('~')
    print('Input:', input_texts[i])
    print('Translation:', translation)

    ans = input("====Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break