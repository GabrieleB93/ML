import pandas as pd
from config import *
import sys
from utils import split_development_set


# According to the arguments, it will have different effects. See more on REadMe.dm
def main():

    # If there is no argument, the script splits the ML2018_TR.csv in training set and test set.
    if len(sys.argv) == 1:
        data_path = ALL_DATA
        data = pd.read_csv(data_path, header=None, comment='#')

        # Choosing and removing the 10% of row from the original file
        remove_n = int(len(data.index) / 10)
        print(remove_n)

        drop_ind = np.random.choice(data.index, remove_n, replace=False)

        print(type(drop_ind))

        test_set = data.iloc[drop_ind, :]
        print(test_set.shape)

        # Save the new files
        file = open(CUP, mode='w')
        test_set.to_csv(file, sep=',', header=False, index=False)

        training_set = data.drop(drop_ind)
        print(training_set.shape)

        file = open(TEST_SET, mode='w')
        training_set.to_csv(file, sep=',', header=False, index=False)

    # If the argument is 'new', then the script will take the Training set (already existing) and draws 2 new sets
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'new':
        X_train, Y_train, X_val, Y_val = split_development_set(20)


if __name__ == '__main__':
    main()
