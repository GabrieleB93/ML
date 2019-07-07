import multiprocessing
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import product
from utils import *
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from utils import *
from config import *

# from NN2 import train_and_validation

batches = [914]
# learning_rates = [0.001, 0.0002, 0.0001]
# layers = [1, 3, 5]
# layer_size = [100, 500]
# n_epochs = [5000]

batch_size = [64]

learn_rate = [0.001]
# units = [10, 50, 100]
units = [500]
level = [1]
epochs = 2000
grid_result_path= "../DATA/grid_search_MLP_ordered.csv"
def main():
    X, Y = getTrainData(CUP, '1:11', '11:13', ',')
    # print(X.shape[0])
    # print(X.shape[1])
    # print(Y.shape[0])
    # print(Y.shape[1])

    grid_result = pd.read_csv(grid_result_path, sep=',', index_col=False)

    for i,row in grid_result.head(20).iterrows():

        # n_layers = int(i.iloc[0]['n_layers'])
        n_layers = [int(row[1])]
        hidden_layers_units = int(row[2])
        # hidden_layers_units = int(i.iloc[0]['hidden_layers_size'])
        # batch = int(i.iloc[0]['batch_size'])
        batch = [int(row[3])]
        learning_rate = [row[4]]
        # learning_rate = i.iloc[0]['learning_rate']
        epochs = 1000
        momentum = 0.9
        lamda = [0.001, 0.01]

        print("VOLA")
        print(n_layers)
        print(hidden_layers_units)
        print(batch)
        print(learning_rate)
        print(epochs)
        print(lamda)

        model = KerasRegressor(build_fn=create_model, epochs=epochs, verbose=0)

        # define the grid search parameters
        param_grid = dict(learn_rate=learn_rate, units=units, level=level, batch_size=batch_size, lamda=lamda)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=False, return_train_score=True, cv=3,
                        scoring=scoring)

        print_and_saveGrid(grid.fit(X, Y), save=True, plot=False, nameResult='PROVA'+str(i), Type='NN')
    # df2 = df2.drop(['validation_loss', 'epsilon'], 1).sort_values(['C', 'gamma'])
    #
    # df1 = df1.pivot("C", "gamma", "mee")
    # print(df1)
    # df2 = df2.pivot("C", "gamma", "mee")
    # print(df2)
    # sns.heatmap(df1, cmap='binary')
    # sns.heatmap(df2, cmap='binary')
    # # sns.heatmap(df2, annot=True, fmt="g", cmap='viridis')
    # plt.show()

    # print(data.drop('validation_loss',1))
    # for d in data.sort_values('mee').groupby('epsilon', as_index=False):
    #     print(d)
    # ax = sns.heatmap(data.drop('validation_loss'), cmap="binary", vmax=1.4)


def mp_handler():
    data_path = "../DATA/training_set.csv"
    data = pd.read_csv(data_path, header=None)
    print(data.shape)
    x = np.array(data.iloc[:, 1:11])
    y = np.array(data.iloc[:, 11:])
    # scaler = StandardScaler()
    # scaler.fit_transform(x)

    p = multiprocessing.Pool(10)
    numbers = list(range(10))
    # with open('results.txt', 'w') as f:
    # for result, o in p.imap(train_and_validation,
    #                         product(x, y, batch_size, lr, n_layers, hidden_size, n_epochs, activation)):
    #     f.write('%d\n' % result)


if __name__ == '__main__':
    main()
