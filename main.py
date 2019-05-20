from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# Load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# Fit a tokenizer, mapping words to integers
def create_tokenizer(lines):
    print(len(lines))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Length of the longest sequence in a list of phrases.
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

# <=======MAIN LOGIC=======>

# load datasets
dataset = load_clean_sentences('english-korean-both.pkl')
train = load_clean_sentences('english-korean-train.pkl')
test = load_clean_sentences('english-korean-test.pkl')
	
# prepare english tokenizer
eng_tokenizer = create_tokenizer(array(dataset)[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(array(dataset)[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare korean tokenizer
ko_tokenizer = create_tokenizer(array(dataset)[:, 1])
ko_vocab_size = len(ko_tokenizer.word_index) + 1
ko_length = max_length(array(dataset)[:, 1])
print('Korean Vocabulary Size: %d' % ko_vocab_size)
print('Korean Max Length: %d' % (ko_length))
	
# Prepare training data
trainX = encode_sequences(ko_tokenizer, ko_length, array(train)[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, array(train)[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# Prepare testing data
testX = encode_sequences(ko_tokenizer, ko_length, array(test)[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, array(test)[:, 0])
testY = encode_output(testY, eng_vocab_size)

print(testX[5])
print(testY[5])