

# Newest HW 10/18/2019

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import TSNE


seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#%%
num_of_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_of_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_of_pixels).astype('float32')
xCombo = np.concatenate([X_test,X_train])
X_train = X_train / 255
X_test = X_test / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

my_model = Sequential()
my_model.add(Dense(10, input_dim=num_of_pixels, activation='relu'))
my_model.add(Dense(num_classes, activation='softmax'))
 
my_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

my_model.fit(X_train, Y_train, epochs=50, batch_size=100, verbose=1)

scores = my_model.evaluate(X_test, Y_test, verbose=1)
print("Error: %.2f%%" % (100-scores[1]*100))

#%% PCA
# xCombo is made earlier

xCombo = preprocessing.scale(xCombo)

pca = decomposition.PCA(.80)
pca.fit(xCombo)
Xpca = pca.transform(xCombo)

Xpca_test = Xpca[0:10000]
Xpca_train = Xpca[10000:]

newNum_of_pixels = Xpca_train.shape[1]
pca_model = Sequential()
pca_model.add(Dense(10, input_dim=newNum_of_pixels, activation='relu'))
pca_model.add(Dense(num_classes, activation='softmax'))

pca_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

pca_model.fit(Xpca_train, Y_train, epochs=50, batch_size=100, verbose=1)
pcaScores = pca_model.evaluate(Xpca_test, Y_test, verbose=1)
print("Error: %.2f%%" % (100-pcaScores[1]*100))


#%% TSNE attempt

tsneData = pd.DataFrame(Xpca)

dimensions = 2

tsne = TSNE(n_components = dimensions ,verbose=1,perplexity=30,n_iter=2000)
xCombo_tsne = tsne.fit_transform(tsneData)

Xtsne_test = xCombo_tsne[0:10000]
Xtsne_train = xCombo_tsne[10000:]

tsne_model = Sequential()
tsne_model.add(Dense(10, input_dim=2, activation='relu'))
tsne_model.add(Dense(num_classes, activation='softmax'))

tsne_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

tsne_model.fit(Xtsne_train, Y_train, epochs=50, batch_size=100, verbose=1)
tsneScores = tsne_model.evaluate(Xtsne_test, Y_test, verbose=1)
print("Error: %.2f%%" % (100-tsneScores[1]*100))

yFrame = pd.DataFrame(Y_train)
plotY = yFrame.idxmax(axis=1)

if dimensions == 2:
    fig, ax = plt.subplots(figsize = (10,10))
    colors = ['b','g','r','c','m','y','k','0.25','0.50','0.75']
    for i in range(10):
        points = np.array([Xtsne_train[j] for j in range(len(Xtsne_train)) if plotY[j] == i])
        ax.scatter(points[:,0], points[:,1],s = 3 , c= colors[i])
if dimensions == 3:
    fig, ax = plt.subplots(figsize = (10,10))
    colors = ['b','g','r','c','m','y','k','0.25','0.50','0.75']
    for i in range(10):
        points = np.array([Xtsne_train[j] for j in range(len(Xtsne_train)) if plotY[j] == i])
        ax.scatter(points[:,0], points[:,1],s = 3 , c= colors[i])

