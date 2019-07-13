import pandas as pd
from config import *


def main():
    data_path = ALL_DATA

    data = pd.read_csv(data_path, header=None, comment='#')

    remove_n = int(len(data.index) / 10)
    print(remove_n)

    drop_ind = np.random.choice(data.index, remove_n, replace=False)

    print(type(drop_ind))

    test_set = data.iloc[drop_ind, :]
    print(test_set.shape)

    # filepath = "../DATA/test_set.csv"
    # file = open(filepath, mode='w')

    file = open(TEST_SET_NEW2, mode='w')
    test_set.to_csv(file, sep=',', header=False, index=False)

    training_set = data.drop(drop_ind)
    print(training_set.shape)

    # filepath = "../DATA/training_set.csv"
    # file = open(filepath, mode='w')

    file = open(CUP_NEW2, mode='w')
    training_set.to_csv(file, sep=',', header=False, index=False)


if __name__ == '__main__':
    main()
