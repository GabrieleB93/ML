import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import os
from time import strftime
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import makedirs


def mean_euclidean_error(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.linalg.norm(x - y)
    return sum / X.shape[0]


def getTrainData(data_path, DimX, DimY, sep):
    startX, endX = DimX.split(':')
    startY, endY = DimY.split(':')
    data = pd.read_csv(data_path, header=None, sep=sep)
    print(data.shape)
    X = np.array(data.iloc[:, int(startX):int(endX)])
    Y = np.array(data.iloc[:, int(startY):int(endY)])

    return X, Y



# Model for NN
def create_model(input_dim=10, output_dim=2, learn_rate=0.01, units=100, level=5, momentum=0.9):
    model = Sequential()
    # model.add(Dropout(0.2, input_shape=(10,)))
    model.add(Dense(units=units, input_dim=input_dim, activation='relu'))

    for l in range(level - 1):
        # model.add(Dropout(0.2))
        # model.add(Dense(units=units, input_dim=10, activation='relu'))
        model.add(Dense(units=units, activation='relu'))

    model.add(Dense(output_dim, activation='relu'))

    optimizer = SGD(lr=learn_rate, momentum=momentum, nesterov=False, decay=0.005)
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
    excluded = ['validation_loss']

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
            splitPlot = ['epsilon']
            pivot2 = 'gamma'
            pivot1 = 'C'
        elif Type == 'SVR_POLY':
            splitPlot = ['epsilon']
            pivot2 = 'degree'
            pivot1 = 'C'
            results_records = {'C': [], 'degree': [],
                               'epsilon': [],
                               # 'gamma': [],
                               'validation_loss': [], 'mee': []}

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
                # results_records['gamma'].append(param['reg__estimator__gamma'])

            results_records['validation_loss'].append(-meanTL)
            results_records['mee'].append(meanTM)

    # To generalize
    if plot and save and Type != 'NN':
        plotGrid(pd.DataFrame(data=results_records), splitPlot, pivot1, pivot2, pivot3, excluded, Type)
    if save:
        saveOnCSV(results_records, nameResult)


def saveOnCSV(results_records, nameResult):
    results = pd.DataFrame(data=results_records)
    filepath = "../DATA/" + nameResult + ".csv"
    file = open(filepath, mode='w')
    results.to_csv(file, sep=',', header=True, index=False)
    file.close()


# To generalize
def plotGrid(dataframe, splitData, pivot1, pivot2, pivot3, excluded, Type):
    for splt in splitData:
        for d in ([x for _, x in dataframe.groupby(dataframe[splt])]):
            splitValue = d[splt].max()
            if splt not in excluded:
                excluded.append(splt)
            hm = d.drop(excluded, 1).sort_values([pivot2, pivot1])
            fig = plt.figure()
            sns.heatmap(hm.pivot(pivot1, pivot2, pivot3), cmap='binary')
            title = splt + " " + str(splitValue) + " with " + pivot3
            plt.title(title)
            plt.show()

            directory = "../Image/"
            t = strftime("%H_%M")
            file = title.replace(" ", "_") + Type + t + ".png"
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(directory + file)


def getIntervalHyperP(dataFrame, hyperp):
    sorted = dataFrame.sort_values('mee')

    best_row = dataFrame[dataFrame.mee == dataFrame.mee.min()]

    start = best_row.iloc[0][hyperp]
    end = sorted[sorted[hyperp] != float(best_row[hyperp])].iloc[0][hyperp]

    End = np.maximum(start, end)
    Start = np.minimum(start, end)
    print(Start)
    print(End)
    tmp = []
    for x in np.arange(Start, (End + abs(End - Start) / 20), abs(End - Start) / 20):
        tmp.append(float('%.3f' % x))

    print(tmp)
    return tmp


def fromCSVToLatexTable(nome1, nome2):
    df = pd.read_csv("../DATA/" + nome1)
    a = df.values
    a = a[:, 1:]
    np.savetxt("../DATA/Latextable" + nome2 + ".csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')



def split_development_set(validation_perc):
    data_path = "../DATA/training_set.csv"

    data = pd.read_csv(data_path, header=None, comment='#')

    remove_n = int(len(data.index) / validation_perc)

    drop_ind = np.random.choice(data.index, remove_n, replace=False)

    val_set = data.iloc[drop_ind, :]

    filepath_val = "../DATA/val_set.csv"
    file = open(filepath_val, mode='w')
    val_set.to_csv(file, sep=',', header=False, index=False)

    tr_set = data.drop(drop_ind)

    filepath_tr = "../DATA/tr_set.csv"
    file = open(filepath_tr, mode='w')
    tr_set.to_csv(file, sep=',', header=False, index=False)

    X_train, Y_train = getTrainData(filepath_tr)
    X_val, Y_val = getTrainData(filepath_val)

    return X_train, Y_train, X_val, Y_val


def train_and_plot_MLP(X_tr, Y_tr, X_val, Y_val, n_layers, n_units, learning_rate, momentum, batch_size, epochs):
    print("Training with: lr=%f, batch size=%f, n layers=%f, units=%f, momentum=%f" % (
    learning_rate, batch_size, n_layers, n_units, momentum))

    model = create_model(learn_rate=learning_rate, units=n_units, level=n_layers, momentum=momentum)
    history = model.fit(X_tr, Y_tr, shuffle=True, epochs=epochs, verbose=2, batch_size=batch_size,
                        validation_data=(X_val, Y_val))

    print("Min loss:", min(history.history['val_loss']))

    fig = plt.figure()

    plt.plot(history.history['loss'], label="TR Loss")
    plt.plot(history.history['val_loss'], label="VL Loss")
    plt.ylim((0, 2))
    plt.legend(loc='upper right')
    # lt.show()

    directory = "../Image/MLP/"
    file = "Eta" + str(learning_rate) + "batch" + str(batch_size) + "m" + str(momentum) + "epochs" + str(epochs) + ".png"
    if not exists(directory):
        makedirs(directory)
    fig.savefig(directory + file)