from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
import sys

from sklearn.preprocessing import StandardScaler

from utils import *
from sklearn.svm import SVR


def MLP(xtr, ytr, xts, yts):
    model = create_model(learn_rate=0.0001, units=500, level=5, momentum=0.9, lamda=0, activation='relu')
    model.fit(xtr, ytr, shuffle=True, epochs=5000, verbose=2, batch_size=915
                        # ,validation_data=(xts, yts)
                                                    )
    prd = model.predict(xts)
    # return history.history['val_loss'][len(history.history['val_loss']) - 1]
    return mean_euclidean_error(prd, yts)


def SVM(xtr, ytr, xts, yts):
    SVR_RBF = MultiOutputRegressor(
        SVR(verbose=0, kernel='rbf', C=14.5, epsilon=0.01, gamma=0.1))
    SVR_RBF.fit(xtr, ytr)
    prd = SVR_RBF.predict(xts)
    return mean_euclidean_error(prd, yts)


def ETR(xtr, ytr, xts, yts):
    etr = ExtraTreesRegressor(max_depth=10, n_estimators=100,
                              max_features=10,
                              min_samples_split=3,
                              bootstrap=False,
                              max_leaf_nodes=None,
                              random_state=False, verbose=False)
    etr.fit(xtr, ytr)
    prd = etr.predict(xts)
    return mean_euclidean_error(prd, yts), etr


def main():
    X_All, Y_All = getTrainData(ALL_DATA, '1:11', '11:13', ',')
    X_train, Y_train = getTrainData(CUP1_NEW, '1:11', '11:13', ',')
    X_test, Y_test = getTrainData(TEST_SET_NEW, '1:11', '11:13', ',')
    X_blind = getBlindData()

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # X_All = scaler.fit_transform(X_All)
    # X_blind = scaler.fit_transform(X_blind)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    if len(sys.argv) == 1:
        # mlp = MLP(X_train, Y_train, X_test, Y_test)
        # train_and_plot_MLP(X_train, Y_train, X_test, Y_test, 5, 500, 0.0001, 0.9,
        #                    915, 5000, 0)
        # svr = SVM(X_train, Y_train, X_test, Y_test)
        etr_err, etr_model = ETR(X_train, Y_train, X_test, Y_test)

        # print(mlp)
        # print(svr)
        print(etr_err)

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'predict':
        etr_err, etr_model = ETR(X_All, Y_All, X_blind, Y_test)
        predicted = etr_model.predict(X_blind)
        # predicted = X_blind
        saveResultsOnCSV(predicted, 'BarBer_ML-CUP18-TS')


if __name__ == '__main__':
    main()
