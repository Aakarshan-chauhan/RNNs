import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EPOCHS = 5
word2vec = {}
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
print("Loading word vectors...")
with open(os.path.join('D:/datasets/Toxic comments/WORD_VECTORS/glove.6B.%sd.txt' % EMBEDDING_DIM),
          encoding="utf8") as f:
    # Space separated Text file
    # Word vec[0] vec[1] ....
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print("found ", len(word2vec), " word vectors")

# Prepare text samples and their labels
print("Loading in comments")

train = pd.read_csv("D:/datasets/Toxic comments/train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[possible_labels].values

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index
print("Found", len(word2idx), " unique tokens")

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shaper of data tensor:", data.shape)

print("Filling pre-trained embeddings...")
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)

# END OF PREPROCESSING

print("Building model")

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Bidirectional(LSTM(15, return_sequences=True)(x))
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy']
)

print("training model")
r = model.fit(
    data,
    targets,
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

p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:, j], p[:, j])
    aucs.append(auc)
    print(np.mean(aucs))
