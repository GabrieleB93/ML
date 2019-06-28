import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.metrics import make_scorer
from matplotlib import gridspec
import os
from time import strftime
import matplotlib.pyplot as plt
import math
import seaborn as sns


def mean_euclidean_error(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.linalg.norm(x - y)
    return sum / X.shape[0]


scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}


def getTrainData():
    data_path = "../DATA/training_set.csv"
    data = pd.read_csv(data_path, header=None, sep=',')
    print(data.shape)
    X = np.array(data.iloc[:, 1:11])
    Y = np.array(data.iloc[:, 11:])

    return X, Y


# Model for NN
def create_model(learn_rate=0.01, units=100, level=1):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(10,)))
    model.add(Dense(units=units, input_dim=10, activation='relu'))

    for l in range(level - 1):
        model.add(Dropout(0.2))
        model.add(Dense(units=units, input_dim=10, activation='relu'))

    model.add(Dense(2, activation='linear'))

    optimizer = SGD(lr=learn_rate, momentum=0.9, nesterov=False, decay=0)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


def print_and_saveGrid(grid_result, save=False, plot=False, nameResult=None, Type=None, ):
    # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    firstScore = 'loss'
    secondScore = 'mee'

    splitPlot = ''
    pivot1 = ''
    pivot2 = ''
    pivot3 = 'mee'
    excluded = 'validation_loss'

    meanTrainLoss = grid_result.cv_results_['mean_train_' + firstScore]
    meanTestLoss = grid_result.cv_results_['mean_test_' + firstScore]
    meanTrainMee = grid_result.cv_results_['mean_train_' + secondScore]
    meanTestMee = grid_result.cv_results_['mean_test_' + secondScore]
    split0_test_Loss = grid_result.cv_results_['split0_test_' + firstScore]
    split1_test_Loss = grid_result.cv_results_['split1_test_' + firstScore]
    split2_test_Loss = grid_result.cv_results_['split2_test_' + firstScore]
    split0_test_Mee = grid_result.cv_results_['split0_test_' + secondScore]
    split1_test_Mee = grid_result.cv_results_['split1_test_' + secondScore]
    split2_test_Mee = grid_result.cv_results_['split2_test_' + secondScore]
    params = grid_result.cv_results_['params']

    # Print on file
    print(Type)
    if save:
        if Type == 'NN':
            results_records = {'n_layers': [], 'hidden_layers_size': [], 'batch_size': [], 'learning_rate': [],
                               'validation_loss': [], 'mee': []}
        elif Type == 'SVR_RBF':
            results_records = {'C': [], 'gamma': [], 'epsilon': [], 'validation_loss': [], 'mee': []}
            splitPlot = 'epsilon'
            pivot2 = 'gamma'
            pivot1 = 'C'
        elif Type == 'SVR_POLY':
            splitPlot = 'epsilon'
            pivot2 = 'degree'
            pivot1 = 'C'
            results_records = {'C': [], 'degree': [],
                               'epsilon': [],
                               'validation_loss': [], 'mee': []}

    #
    for meanTRL, meanTL, meanTRM, meanTM, S0TL, S1TL, S2TL, S0TM, S1TM, S2TM, param in zip(meanTrainLoss, meanTestLoss,
                                                                                           meanTrainMee, meanTestMee,
                                                                                           split0_test_Loss,
                                                                                           split1_test_Loss,
                                                                                           split2_test_Loss,
                                                                                           split0_test_Mee,
                                                                                           split1_test_Mee,
                                                                                           split2_test_Mee, params):
        print("%f %f %f %f %f %f %f %f %f %f with: %r" % (meanTRL, meanTL, meanTRM, meanTM, S0TL, S1TL, S2TL, S0TM,
                                                          S1TM, S2TM, param))
        if save:
            if Type == 'NN':
                results_records['n_layers'].append(param['level'])
                results_records['hidden_layers_size'].append(param['units'])
                results_records['batch_size'].append(param['batch_size'])
                results_records['learning_rate'].append(param['learn_rate'])
            elif Type == 'SVR_RBF':
                results_records['C'].append(param['reg__estimator__C'])
                results_records['gamma'].append(param['reg__estimator__gamma'])
                results_records['epsilon'].append(param['reg__estimator__epsilon'])
            elif Type == 'SVR_POLY':
                results_records['C'].append(param['reg__estimator__C'])
                results_records['epsilon'].append(param['reg__estimator__epsilon'])
                results_records['degree'].append(param['reg__estimator__degree'])

            results_records['validation_loss'].append(-meanTL)
            results_records['mee'].append(meanTM)

    # To generalize
    if plot and save and Type != 'NN':
        plotGrid(pd.DataFrame(data=results_records), splitPlot, pivot1, pivot2, pivot3, excluded, Type)
    if save:
        saveOnCSV(results_records, nameResult)


def saveOnCSV(results_records, nameResult):
    results = pd.DataFrame(data=results_records)
    filepath = "../DATA/" + nameResult
    file = open(filepath, mode='w')
    results.to_csv(file, sep=',', header=True, index=False)
    file.close()


# To generalize
def plotGrid(dataframe, splitData, pivot1, pivot2, pivot3, excluded, Type):
    for d in ([x for _, x in dataframe.groupby(dataframe[splitData])]):
        splitValue = d[splitData].max()
        hm = d.drop([excluded, splitData], 1).sort_values([pivot2, pivot1])
        plot = sns.heatmap(hm.pivot(pivot1, pivot2, pivot3), cmap='binary', vmin=0.90)
        title = splitData + " " + str(splitValue) + " with " + pivot3
        plt.title(title)
        plt.show()

        directory = "../Image/"
        t = strftime("%H_%M")
        file = title.replace(" ", "_") + Type + t + ".png"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot.savefig(directory + file)


def getIntervalHyperP(dataFrame, hyperp):
    sorted = dataFrame.sort_values('mee')

    best_row = dataFrame[dataFrame.mee == dataFrame.mee.min()]

    start = best_row.iloc[0][hyperp]
    end = sorted[sorted[hyperp] != float(best_row[hyperp])].iloc[0][hyperp]

    tmp = []
    for x in np.arange(start, (end + abs(end - start) / 20), abs(end - start) / 20):
        tmp.append(float('%.3f' % x))
    return tmp
