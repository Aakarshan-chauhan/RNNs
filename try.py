print('Loading the libraries...')
import pandas as pd 	

print('Libraries loaded...')
# Load all the libraries required

# Basic Configs (can be changed to fine tune a model or the dataset)
MAX_VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH  = 15
file = pd.read_csv('../abc')
file = file['col1']

input_sentences =[]
target_sentences =[]
for line in file:
	line = line.rstrip()
	if not line:
		continue
	input_sentence = '<sos> ' + line # <sos> = embedding for start of sentences
	target_sentence = line + ' <eos>' # <eos> = embedding for end of sentences
	input_sentences.append(input_sentence)
	target_sentences.append(target_sentence)
all_lines = input_sentences + target_sentences

print('Tokenizing the words...')
# tokenizing the words present in the sentences to create a sequence 
tokenizer = Tokenizer(max_vocab_size=MAX_VOCAB_SIZE, filter='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_sentences)
target_sequences = tokenizer.texts_to_sequences(target_sentences)

# Getting the longest length of a sentence/sequence for the data
max_length_from_data = max(len(s) for s in input_sequences)
print('Sequence length of the longest sentence in data:', max_length_from_data)
# Using the tokenizer to map each word in the dataset to an index
word2idx = tokenizer.word_index

# Finding the least of the specified max length and the data length for memory optimization
max_sequence_length = min(max_length_from_data, MAX_SEQUENCE_LENGTH)
print('Selected maximum sentence length:', max_sequence_length)
# Padding the sequences so that each sentence is of the max_sequence_length
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

print('Loading the word vectors...')
word2vec = {}

# Using the premade glove vectors to map a relevant dictionary
with open(os.path.join('D:/datasets/Toxic comments/WORD_VECTORS/glove.6B.%sd.txt' % EMBEDDING_DIM),
          encoding='utf8') as f:
	for line in f:
		value = line.split()
		word = value[0] # The first element of the line is always the word followed by the vectors
		vec = np.asarray(value[1:], dtype='float32')
		word2vec[word] = vec

print("Length of Unique word vectors:", len(word2vec))

# Finding the total number of words
num_words = min(MAX_VOCAB_SIZE, len(word2vec) + 1)

# Forming an empty embedding matrix 
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

# Filling the embedding matrix with the vectors from word2vec and index i from word2idx.
# Hence embedding matrix is the map of a vectors index and its corresponding word vector.
print('Filling the pre-trained embeddings...')
for word, i in word2idx.items():
	if i < MAX_VOCAB_SIZE:
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
		# For words not having a vector, their respective cell will be having value 0.

# Since one input in this model would give a target sequence(of length t) instead of one target,
# a sparse cross-entropy loss function can not be used here.
# Hence, by one-hotting the targets, we can use another pre defined loss function.
# Essentially, a new loss function can be created for better results.
one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
for i, target_sequences in enumerate(target_sequences):
	for t, word, in enumerate(target_sequences):
		if word> 0:
			one_hot_targets[i, t, word] = 1

# Making an embedding layer to load the embedding targets
embedding_layer = Embedding(
	num_words,
	EMBEDDING_DIM,
	weights=[embedding_matrix]
	)


print("Building model...")