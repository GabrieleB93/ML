import sys
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import os
from time import strftime
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import makedirs
from config import *
from keras import regularizers


# ---------------------- FROM AND TO CSV -------------------------------------------------------------------------------

# Split the columns of the file to returns features and targets
def getTrainData(data_path, DimX, DimY, sep):
    startX, endX = DimX.split(':')
    startY, endY = DimY.split(':')
    data = pd.read_csv(data_path, header=None, sep=sep, comment='#')
    print(data.shape)
    X = np.array(data.iloc[:, int(startX):int(endX)])
    Y = np.array(data.iloc[:, int(startY):int(endY)])

    return X, Y


# Split the columns of the file to returns features of Blind set
def getBlindData():
    data = pd.read_csv(BLIND_DATA, header=None, sep=',', comment='#')
    print(data.shape)
    X = np.array(data.iloc[:, 1:11])

    return X


def saveOnCSV(results_records, nameResult, Type):
    results = pd.DataFrame(data=results_records)
    filepath = "../../DATA/" + Type + "/" + nameResult + ".csv"
    file = open(filepath, mode='w+')
    if Type.lower() != 'monk':
        results = results.sort_values('mee', ascending=True)
    results.to_csv(file, sep=',', header=True, index=False)
    file.close()


def saveResultsOnCSV(results_records, nameResult):
    results = pd.DataFrame(data=results_records)
    filepath = "../DATA/RESULTS/" + nameResult + ".csv"
    file = open(filepath, mode='w+')
    file.write('# Barreca Gabriele, Bertoncini Gioele\n')
    file.write('# BarBer\n')
    file.write('# ML-CUP18\n')
    file.write('# 15/07/2019\n')
    results.index += 1
    results.to_csv(file, sep=',', header=False, index=True)
    file.close()


# Split the an existing date set in 2 new ones, like split_new.py
def split_development_set(validation_perc):
    data = pd.read_csv(CUP, header=None, comment='#')

    remove_n = int(len(data.index) / validation_perc)

    drop_ind = np.random.choice(data.index, remove_n, replace=False)

    val_set = data.iloc[drop_ind, :]

    file = open(VAL_SET, mode='w')
    val_set.to_csv(file, sep=',', header=False, index=False)

    tr_set = data.drop(drop_ind)

    file = open(TRAIN_SET, mode='w')
    tr_set.to_csv(file, sep=',', header=False, index=False)

    X_train, Y_train = getTrainData(TRAIN_SET, '1:11', '11:13', ',')
    X_val, Y_val = getTrainData(VAL_SET, '1:11', '11:13', ',')

    return X_train, Y_train, X_val, Y_val


# ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------- UTILS FOR NN -------------------------------------------------------------------

# Model for NN
def create_model(input_dim=10, output_dim=2, learn_rate=0.01, units=100, level=5, momentum=0.9, decay=0,
                 activation='relu', lamda=0):
    print(units)
    print(level)

    model = Sequential()
    model.add(Dense(units=units, input_dim=input_dim, activation=activation, kernel_regularizer=regularizers.l2(lamda),
                    # bias_initializer='zeros',
                    use_bias=True))

    for l in range(level - 1):
        model.add(Dense(units=units, activation=activation, kernel_regularizer=regularizers.l2(lamda),
                        # bias_initializer='zeros',
                        use_bias=True))

    if output_dim == 2:
        actv = 'linear'
    else:
        actv = 'sigmoid'

    model.add(
        Dense(output_dim, activation=actv, kernel_regularizer=regularizers.l2(lamda), bias_initializer='zeros',
              use_bias=True))

    optimizer = SGD(lr=learn_rate, momentum=momentum, nesterov=False, decay=decay)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


# one - of - k encoding. From github
def one_of_k(data):
    dist_values = np.array([np.unique(data[:, i]) for i in range(data.shape[1])])
    new_data = []
    First_rec = True
    for record in data:
        new_record = []
        First = True
        indice = 0
        for attribute in record:
            new_attribute = np.zeros(len(dist_values[indice]), dtype=int)
            for j in range(len(dist_values[indice])):
                if dist_values[indice][j] == attribute:
                    new_attribute[j] += 1
            if First:
                new_record = new_attribute
                First = False
            else:
                new_record = np.concatenate((new_record, new_attribute), axis=0)
            indice += 1
        if First_rec:
            new_data = np.array([new_record])
            First_rec = False
        else:
            new_data = np.concatenate((new_data, np.array([new_record])), axis=0)
    return new_data


# ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------ PLOTS ---------------------------------------------------------------------------

def print_and_saveGrid(grid_result, save=False, plot=False, nameResult=None, Type=None, ):
    # summarize results
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
            excluded = ['validation_loss', 'coef0', 'gamma']
            results_records = {'C': [], 'degree': [],
                               'epsilon': [],
                               'gamma': [],
                               'coef0': [],
                               'validation_loss': [], 'mee': []}
        elif Type == 'RFR':
            results_records = {'n_estimators': [], 'max_depth': [],
                               'min_samples_split': [],
                               'max_features': [],
                               'bootstrap': [],
                               'validation_loss': [],
                               'mee': []}
        elif Type == 'ETR':
            results_records = {'n_estimators': [], 'max_depth': [],
                               'min_samples_split': [],
                               'max_features': [],
                               'bootstrap': [],
                               'validation_loss': [],
                               'mee': []}
            splitPlot = ['random_state']
            pivot2 = 'max_depth'
            pivot1 = 'n_estimators'
        elif Type == 'MONK':
            results_records = {'n_layers': [], 'hidden_layers_size': [], 'batch_size': [], 'learning_rate': [],
                               'decay': [],
                               'momentum': [], 'lamda': [], 'activation': [],
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
                print(param['level'])
                print(param['batch_size'])
            elif Type == 'SVR_RBF':
                results_records['C'].append(param['reg__estimator__C'])
                results_records['gamma'].append(param['reg__estimator__gamma'])
                results_records['epsilon'].append(param['reg__estimator__epsilon'])
            elif Type == 'SVR_POLY':
                results_records['C'].append(param['reg__estimator__C'])
                results_records['epsilon'].append(param['reg__estimator__epsilon'])
                results_records['degree'].append(param['reg__estimator__degree'])
                results_records['gamma'].append(param['reg__estimator__gamma'])
                results_records['coef0'].append(param['reg__estimator__coef0'])
            elif Type == "MONK":
                results_records['n_layers'].append(param['level'])
                results_records['hidden_layers_size'].append(param['units'])
                results_records['batch_size'].append(param['batch_size'])
                results_records['learning_rate'].append(param['learn_rate'])
                results_records['momentum'].append(param['momentum'])
                results_records['activation'].append(param['activation'])
                results_records['decay'].append(param['decay'])
                results_records['lamda'].append(param['lamda'])
            elif Type == 'RFR':
                results_records['n_estimators'].append(param['n_estimators'])
                results_records['max_depth'].append(param['max_depth'])
                results_records['min_samples_split'].append(param['min_samples_split'])
                results_records['bootstrap'].append(param['bootstrap'])
                results_records['max_features'].append(param['max_features'])
            elif Type == 'ETR':
                results_records['n_estimators'].append(param['n_estimators'])
                results_records['max_depth'].append(param['max_depth'])
                results_records['min_samples_split'].append(param['min_samples_split'])
                results_records['bootstrap'].append(param['bootstrap'])
                results_records['max_features'].append(param['max_features'])

            results_records['validation_loss'].append(-meanTL)
            results_records['mee'].append(meanTM)

    if plot and save and Type != 'NN':
        plotGrid(pd.DataFrame(data=results_records), splitPlot, pivot1, pivot2, pivot3, excluded, Type)
    if save:
        print(results_records)
        saveOnCSV(results_records, nameResult, Type)


# Plot grid
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

            directory = "../../Image/" + Type + "/"
            t = strftime("%H_%M")
            file = title.replace(" ", "_") + Type + t + ".png"
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(directory + file)


def train_and_plot_MLP(X_tr, Y_tr, X_val, Y_val, n_layers, n_units, learning_rate, momentum, batch_size, epochs, lamda):
    print("Training with: lr=%f, batch size=%f, n layers=%f, units=%f, momentum=%f" % (
        learning_rate, batch_size, n_layers, n_units, momentum))

    model = create_model(learn_rate=learning_rate, units=n_units, level=n_layers, momentum=momentum, lamda=lamda)
    history = model.fit(X_tr, Y_tr, shuffle=True, epochs=epochs, verbose=2, batch_size=batch_size,
                        validation_data=(X_val, Y_val))

    print("Min loss:", min(history.history['val_loss']))

    fig = plt.figure()

    plt.plot(history.history['loss'], label="TR Loss")
    plt.plot(history.history['val_loss'], label="VL Loss", linestyle='dashed')
    plt.ylim((0, 2))
    plt.legend(loc='upper right')
    plt.show()

    directory = "../../Image/MLP/"
    file = "Eta" + str(learning_rate) + "batch" + str(batch_size) + "m" + str(momentum) + "epochs" + str(
        epochs) + "lamda" + str(lamda) + ".png"
    if not exists(directory):
        makedirs(directory)
    fig.savefig(directory + file)


# ----------------------------------------------------------------------------------------------------------------------


# ------------------------------------ UTILS FOR SVM -------------------------------------------------------------------

# The function sorts the dataframe, takes the first two different values and create a list of elements in its range
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


# ----------------------------------------------------------------------------------------------------------------------


# --------------------------------------- UTILS FOR ALL ----------------------------------------------------------------

def parseArg(arg):
    if arg.lower() == 'grid':
        global GRID
        GRID = True
    elif arg.lower() == 'cv':
        global CV
        CV = True
    # elif arg.lower() == 'predict':
    #     global Predict
    #     Predict = True
    else:
        sys.exit("Argument not recognized")

# ----------------------------------------------------------------------------------------------------------------------
