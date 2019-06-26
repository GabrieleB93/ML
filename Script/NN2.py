from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import RepeatedKFold, cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
import utils

val_losses = []
mees = []


def crossValidation(X, Y, model, epochs, batch_size, rkfold):
    for tr_index, ts_index in rkfold.split(X, Y):
        x_train, x_test = X[tr_index], X[ts_index]
        y_train, y_test = Y[tr_index], Y[ts_index]
        hist = model.fit(x_train, y_train, shuffle=True, epochs=epochs, verbose=0, batch_size=batch_size,
                         validation_data=(x_test, y_test))

        min_val_loss = min(hist.history['val_loss'])
        # print("Min loss:", min_val_loss)
        val_losses.append(min_val_loss)
        y_guess = model.predict(x_test)
        mee = utils.mean_euclidean_error(y_guess, y_test)
        mees.append(mee)
        # print("MEE: ", mee)

    # all_accur = cross_val_score(model, X, Y, cv=3)
    # all_accur = cross_val_score(KerasClassifier(build_fn=create_model,lr,n_layers_hidden_size,activation), X, Y, cv=3)

    print("COSEEEE")
    # print(all_accur)

    mean_val_loss = np.mean(val_losses)
    mean_mee = np.mean(mees)

    return mean_val_loss, mean_mee


def create_model(lr, n_layers, hidden_size, activation):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=10, activation=activation))
    for i in range(n_layers - 1):
        model.add(Dense(hidden_size, activation=activation))
    model.add(Dense(2, activation='linear'))
    optimizer = SGD(lr=lr, momentum=0.9, nesterov=False, decay=0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_and_validation(X, Y, batch_size=914, lr=0.001, n_layers=5, hidden_size=100, epochs=5000, activation='relu'):
    model = create_model(lr, n_layers, hidden_size, activation)
    rkfold = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1343)

    return crossValidation(X, Y,
                           KerasRegressor(build_fn=create_model, lr=lr, n_layers=n_layers, hidden_size=hidden_size,
                                           activation=activation), epochs, batch_size, rkfold)
