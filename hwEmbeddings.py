
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import KeyedVectors
#%%
model = api.load('glove-wiki-gigaword-300')
model.save("E:\\py599\\wikiEmbedder.npz")
#%%
print("woman-->man == king-->? <take1>")
result = model.most_similar(positive=['woman','king'],negative=['man',])
print(result[0])
print("\n")
print("woman-->man == king-->? <take2>")
result = model.most_similar_cosmul(positive=['woman','king'],negative=['man',])
print(result[0])
print("\n")
print("paris-->france == london-->? <take3>")
result = model.most_similar_cosmul(positive=['paris','france'],negative=['london',])
print(result[0])
print("\n")
print("man-->men == woman-->? <take4>")
result = model.most_similar_cosmul(positive=['man','woman'],negative=['men',])
print(result[0])
print("\n")
print("similar to carolina? <take5>")
result = model.similar_by_word('carolina')
print(result[0])
print("\n")
print("similar to carolina? <take6>")
result = model.most_similar_cosmul(positive=['carolina'],negative=[])
print(result[0])
print("\n")
print("similar to assuage? <take7>")
result = model.most_similar_cosmul(positive=['assuage'],negative=[])
print(result[0])
print("THIS IS A SYNONYM!")
print("\n")
print("similar to lexicon? <take7>")
result = model.most_similar_cosmul(positive=['lexicon'],negative=[])
print(result[0])
print("SO IS THIS!")
print("\n")
print("similarity of ice cream and frozen custard <take8>")
similarity = model.n_similarity(['ice','cream'],['frozen','custard'])
print(str(similarity))
print("\n")
#%%
inWord = 'carolina'
inComp = 'virginia'

inVect = model[inWord]
inSimW = model.most_similar(inWord)[0][0]
inSimV = model[inSimW]

outVnear = model[inComp]-inVect+inSimV
outV = model.most_similar(positive=[outVnear], topn=1)[0][0]
outW = model[outV]


#%%

# loads a dataset of 
dataset = api.load('text8')
A = []
for token in dataset:
    A.append(token)
from keras import layers
from keras import models
from keras import optimizers

corpus_raw = 'He is the king . The king is royal . She is the royal queen '
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)
words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word])
                
# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp
x_train = [] # input word
y_train = [] # output word
for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

#%%
inDimension = vocab_size
embedDim = 2
model = models.Sequential()
model.add(layers.Embedding(10,2,input_length=2))
model.add(layers.Dense(inDimension, activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

model.fit(x_train,y_train,epochs = 12)

weights = model.get_weights()

#%%
from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words 
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
