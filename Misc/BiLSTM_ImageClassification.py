import os
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Dense, Lambda, Concatenate
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mnist(limit=None):
    df = pd.read_csv('D:/datasets/MNIST/train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1, 28, 28) / 255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

X, Y = get_mnist()

D = 28
M = 15

input_ = Input(shape=(D, D))

rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_)
x1 = GlobalMaxPooling1D()(x1)

rnn2 = Bidirectional(LSTM(M, return_sequences=True))

permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))
x2 = permutor(input_)
x2 = rnn2(x2)
x2 = GlobalMaxPooling1D()(x2)

concatenator = Concatenate(axis=1)
x = concatenator([x1, x2])

output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_, outputs=output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training model...")
r = model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.3)


plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")
plt.legend()
plt.show()


plt.plot(r.history['acc'], label="acc")
plt.plot(r.history['val_acc'], label="val_acc")
plt.legend()
plt.show()
