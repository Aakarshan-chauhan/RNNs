from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os,sys

input_texts = []
target_texts = []
target_texts_inputs = []

t = 0
for line in open(''):
    t += 1
    if t>NUM_SAMPLES:
        break

    if '\t' not in line:
        continue

    input_text , translation = line.split('\t')
    target_text = translation + ' <eos>'
    target_texts_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_texts_input)
print("num samples:", len(input_texts))
