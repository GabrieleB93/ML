

import pandas as pd
import multiprocessing
from multiprocessing import Pool
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def train(i,u):
    import tensorflow as tf
    import keras
    from keras.models import Sequential

    # ------------------------------
    # # this block enables GPU enabled multiprocessing
    # core_config = tf.ConfigProto()
    # core_config.gpu_options.allow_growth = True
    # session = tf.Session(config=core_config)
    # keras.backend.set_session(session)
    # # ------------------------------
    # prepare input and output values
    batches = 64

    # read file
    data_path = "../DATA/training_set.csv"
    data = pd.read_csv(data_path, header=None)
    print(data.shape)
    x = np.array(data.iloc[:, 1:11])
    y = np.array(data.iloc[:, 11:])
    print(x.shape)
    print(y.shape)

    scaler = StandardScaler()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, shuffle=True)

    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    # define and fit the final model
    model = Sequential()
    model.add(Dense(100, input_dim=10, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='linear'))
    mySGD = SGD(lr=0.00001, momentum=0.9, nesterov=False, decay=0)
    model.compile(loss='mean_squared_error', optimizer=mySGD, metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, shuffle=True, epochs=3000
                     , verbose=2, batch_size=batches, validation_data=(x_test, y_test))

    print("Min loss:", min(hist.history['val_loss']))

    pyplot.plot(hist.history['loss'], label="TR Loss")
    pyplot.plot(hist.history['val_loss'], label="VL Loss")
    pyplot.ylim((0, 2))
    pyplot.legend(loc='upper right')
    pyplot.show()

    # ------------------------------
    # finally, close sessions
    # session.close()
    # keras.backend.clear_session()
    # return 0


# -----------------------------
# main program
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    df = pd.read_csv("../DATA/training_set.csv")

    my_tuple = 1

    with Pool(10) as pool:
        pool.starmap(train, [(1,2), (3,4)])
