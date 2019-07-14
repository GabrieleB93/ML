import pandas as pd
from config import *
import sys
from utils import split_development_set


def main():
    if len(sys.argv) == 1:

        data_path = ALL_DATA
        data = pd.read_csv(data_path, header=None, comment='#')

        remove_n = int(len(data.index) / 10)
        print(remove_n)

        drop_ind = np.random.choice(data.index, remove_n, replace=False)

        print(type(drop_ind))

        test_set = data.iloc[drop_ind, :]
        print(test_set.shape)

        file = open(CUP, mode='w')
        test_set.to_csv(file, sep=',', header=False, index=False)

        training_set = data.drop(drop_ind)
        print(training_set.shape)

        file = open(TEST_SET, mode='w')
        training_set.to_csv(file, sep=',', header=False, index=False)

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'new':
        X_train, Y_train, X_val, Y_val = split_development_set(20)

if __name__ == '__main__':
    main()
