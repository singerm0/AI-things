# -*- coding: utf-8 -*-
#day 7 document classification - Naive Bayes

from keras.datasets import imdb
import numpy as np
import pandas as pd


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
#%%

percentGood = sum(y_train)/y_train.shape[0]
goodScores = []
badScores = []
for i in range(y_train.shape[0]):
    if y_train[i] == 1:
        goodScores.append(x_train[i])
    else:
        badScores.append(x_train[i])
        
# figure out max
trueMax = max(max(goodScores))
badMax = max(max(badScores))
if badMax >= trueMax:
    trueMax = badMax

goodScores.append(list(range(trueMax)))
badScores.append(list(range(trueMax)))

goodTotal = [x for y in goodScores for x in y]
badTotal = [x for y in badScores for x in y]

goodDF = pd.DataFrame({'Word':pd.to_numeric(goodTotal)})
badDF = pd.DataFrame({'Word':pd.to_numeric(badTotal)})

goodDf = goodDF.groupby('Word').size().reset_index(name='counts')
badDf = badDF.groupby('Word').size().reset_index(name='counts')

goodDf['percent'] = goodDf['counts'].div(goodDF.shape[0])
badDf['percent'] = badDf['counts'].div(badDF.shape[0])

goodDf['logPercent'] = np.log(goodDf['percent'])
badDf['logPercent'] = np.log(badDf['percent'])

badDict = dict(zip(badDf.Word,badDf.logPercent))
goodDict = dict(zip(goodDf.Word,goodDf.logPercent))
#%%
def betterClassifier(X,badDict,goodDict):
    goodEval = list(map(goodDict.get,X))
    goodEval = list(filter(None,goodEval))
    goodEval = np.sum(np.array(goodEval))
    
    badEval = list(map(badDict.get,X))
    badEval = list(filter(None,badEval))
    badEval = np.sum(np.array(badEval))
    
    if goodEval >= badEval:
        returnVar = 1
    else:
        returnVar = 0
    return returnVar
    
y_classification = [betterClassifier(x,badDict,goodDict) for x in x_test]

score = (x_test.shape[0]-sum((np.array(y_classification-y_test)**2)))/x_test.shape[0]
print(score)

        

