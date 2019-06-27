import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import product
from NN2 import train_and_validation

batches = [64, 914]
learning_rates = [0.001, 0.0002, 0.0001]
layers = [1, 3, 5]
layer_size = [100, 500]
n_epochs = [5000]

batch_size = [64]
lr = [000.1]
n_layers = [2]
hidden_size = [5000]
activation = "relu"


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
    with open('results.txt', 'w') as f:
        for result, o in p.imap(train_and_validation,
                                product(x,y, batch_size, lr, n_layers, hidden_size, n_epochs, activation)):
            f.write('%d\n' % result)


if __name__ == '__main__':
    mp_handler()
