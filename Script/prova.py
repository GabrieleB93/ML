import multiprocessing
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import product

# from NN2 import train_and_validation

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


def main():
    import seaborn as sns

    data_path = "../DATA/grid_search_result_SVR_RBF"
    data = pd.read_csv(data_path)

    datas = []

    for d in ([x for _, x in data.groupby(data['epsilon'])]):
        split = d['epsilon'].max()
        print(split.max())
        hm = d.drop(['validation_loss', 'epsilon'], 1).sort_values(['gamma', 'C'])
        sns.heatmap(hm.pivot("C", "gamma", "mee"), cmap='binary')
        title = 'epsilon' + " " + str(split) + " with " + 'mee'
        plt.title(title)
        plt.show()

      # print(d.drop(['validation_loss', 'epsilon'], 1).sort_values(['C', 'gamma']))

    # df1 = df1.drop(['validation_loss', 'epsilon'], 1).sort_values(['C', 'gamma'])
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
