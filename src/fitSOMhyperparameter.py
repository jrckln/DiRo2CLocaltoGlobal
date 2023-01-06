import sys
import pandas as pd
import numpy as np
from diro2c.data_generation.helper import *
import minisom
import pickle
from diro2c.data_generation.helper import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict
import logging

from data.getdata import loaddata
from data.split3fold import split3fold

from sklearn.model_selection import KFold
import itertools
from datetime import date

if __name__ == '__main__':

    data = sys.argv[1]

    logging.basicConfig(filename='SOMhyperparameterfit'+str(date.today())+'.log', level=logging.DEBUG,
                        format = '%(asctime)s %(message)s')
    performancefile = 'SOMhyperparameterperformance'+str(date.today())+'.txt'

    with open(performancefile, 'a') as myfile:
        myfile.write('data sigma learningrate quantization topographic\n')

    dataA, dataB, cols, discrete, continuous, le = loaddata(data)
    blackboxtrainA, trainA, testA = split3fold(dataA, 0.4, 0.2, random_state=1)
    blackboxtrainB, trainB, testB = split3fold(dataB, 0.4, 0.2, random_state=1)

    modelA = pickle.load(open('blackboxes/'+data+'A.sav', 'rb'))
    modelB = pickle.load(open('blackboxes/'+data+'B.sav', 'rb'))

    train = pd.concat([trainA, trainB])
    train['predA'] = modelA.predict(train[cols].values)
    train['predB'] = modelB.predict(train[cols].values)
    train['difference'] = train.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
    train.drop(columns=['predA', 'predB', 'y'], inplace=True, errors='ignore')
    train = train.reset_index(drop=True)
    test = pd.concat([testA, testB])
    test['predA'] = modelA.predict(test[cols].values)
    test['predB'] = modelB.predict(test[cols].values)
    test['difference'] = test.apply(lambda row: str(int(row['predA'])) + '|' + str(int(row['predB'])), axis = 1)
    test.drop(columns=['predA', 'predB', 'y'], inplace=True)
    test = test.reset_index(drop=True)

    discrete_woclassname = discrete.copy()
    discrete.append('difference')

    d = defaultdict(lambda: OneHotEncoder(drop = 'first'))
    trainbinary = train.copy()
    testbinary = test.copy()
    colsbinary = cols.copy()
    for feature in discrete_woclassname:
        uniquevals = np.concatenate((trainbinary[feature].values.reshape(-1,1), testbinary[feature].values.reshape(-1,1)))
        d[feature].fit(uniquevals)
        tmp = d[feature].transform(trainbinary[feature].values.reshape(-1,1)).toarray()
        colnames = [feature + str(i) for i in range(tmp.shape[1])]
        trainbinary[colnames] = tmp
        testbinary[colnames] = d[feature].transform(testbinary[feature].values.reshape(-1,1)).toarray()
        colsbinary = colsbinary + colnames
        colsbinary.remove(feature)
        trainbinary.drop(columns = feature, inplace = True)
        testbinary.drop(columns = feature, inplace = True)

    trainsom = trainbinary[colsbinary].values
    testsom = testbinary[colsbinary].values

    trainsomnorm = trainbinary[colsbinary].copy()
    d = StandardScaler()
    scaler = d.fit(trainsomnorm[continuous].values)
    trainsomnorm[continuous] = scaler.transform(trainsomnorm[continuous].values)
    trainsomnorm = trainsomnorm.values
    trainsomnorm = trainsomnorm[~train.difference.isin(['0|0', '1|1', '2|2'])]
    trainsom = trainsom[~train.difference.isin(['0|0', '1|1', '2|2'])]

    #GridSearchCV for SOM fitting
    sigmas = [0.5, 0.6, 0.7, 0.8, 0.9]
    learning_rates = [0.2, 0.4,0.6, 0.8,1]
    grid = list(itertools.product(*[sigmas, learning_rates]))

    n_nodes = int(np.floor(5*np.sqrt(len(trainsom))))

    for sigma, learning_rate in grid:
        kf = KFold(n_splits=3)
        for train_index, test_index in kf.split(trainsomnorm):
            X_train, X_test = trainsomnorm[train_index], trainsomnorm[test_index]

            som = minisom.MiniSom(1, n_nodes, X_train.shape[1], sigma=sigma, learning_rate=learning_rate,
                                  random_seed = 0)
            som.train(X_train, 100000, verbose = False)

            logging.info('processed: ' + str(sigma) + '/' + str(learning_rate))

            with open(performancefile, 'a') as myfile:
                line = ' '.join([data,
                                 str(sigma),
                                 str(learning_rate),
                                 str(som.quantization_error(X_test)),
                                 str(som.topographic_error(X_test))
                                 ])
                myfile.write(line + '\n')