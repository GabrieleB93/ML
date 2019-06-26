import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


def getTrainData():
    data_path = "../DATA/training_set.csv"
    data = pd.read_csv(data_path, header=None, sep=',')
    print(data.shape)
    X = np.array(data.iloc[:, 1:11])
    Y = np.array(data.iloc[:, 11:])

    return X, Y


def create_model(learn_rate=0.01, units=100, level=1):
    # create model
    model = Sequential()
    model.add(Dense(units=units, input_dim=10, activation='relu'))

    for l in range(level - 1):
        model.add(Dense(units=units, input_dim=10, activation='relu'))

    model.add(Dense(2, activation='linear'))

    optimizer = SGD(lr=learn_rate, momentum=0.9, nesterov=False, decay=0)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


def print_and_saveGrid(grid_result):
    # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    meanTrainLoss = grid_result.cv_results_['mean_train_loss']
    meanTestLoss = grid_result.cv_results_['mean_test_loss']
    meanTrainMee = grid_result.cv_results_['mean_train_mee']
    meanTestMee = grid_result.cv_results_['mean_test_mee']
    split0_test_Loss = grid_result.cv_results_['split0_test_loss']
    split1_test_Loss = grid_result.cv_results_['split1_test_loss']
    split2_test_Loss = grid_result.cv_results_['split2_test_loss']
    split0_test_Mee = grid_result.cv_results_['split0_test_mee']
    split1_test_Mee = grid_result.cv_results_['split1_test_mee']
    split2_test_Mee = grid_result.cv_results_['split2_test_mee']
    params = grid_result.cv_results_['params']

    # Stampa su file e print
    results_records = {'n_layers': [], 'hidden_layers_size': [], 'batch_size': [], 'learning_rate': [],
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
        results_records['n_layers'].append(param['level'])
        results_records['hidden_layers_size'].append(param['units'])
        results_records['batch_size'].append(param['batch_size'])
        results_records['learning_rate'].append(param['learn_rate'])
        results_records['validation_loss'].append(-meanTL)
        results_records['mee'].append(meanTM)

    results = pd.DataFrame(data=results_records)
    filepath = "../DATA/grid_search_result_MPL"
    file = open(filepath, mode='w')
    results.to_csv(file, sep=',', header=True, index=False)


def mean_euclidean_error(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.linalg.norm(x - y)
    return sum / X.shape[0]
