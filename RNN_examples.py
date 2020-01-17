from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8
# Sequence Length
D = 2
# Input Dimensionality
M = 3
# Hidden Layer Size

X = np.random.randn(1, T, D)


def lstm1():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print("o:", o)
    print("h:", h)
    print("c:", c)


def lstm2():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print("o:", o)
    print("h:", h)
    print("c:", c)


def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M , return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print("o:", o)
    print("h:", h)


def gru2():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print("o:", o)
    print("h:", h)

print("Lstm1:")
lstm1()

print("Lstm2:")
lstm2()

print("gru1:")
gru1()

print("gru2:")
gru2()
