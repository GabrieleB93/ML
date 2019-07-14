from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from time import time
from sklearn.preprocessing import StandardScaler
from utils import *
from sklearn.svm import SVR


def MLP(xtr, ytr, xts, yts):
    start = time()
    model = create_model(learn_rate=0.0001, units=500, level=5, momentum=0.9, lamda=0, activation='relu')
    model.fit(xtr, ytr, shuffle=False, epochs=3000, verbose=2, batch_size=915)
    tmp = time() - start
    prd = model.predict(xts)

    return mean_euclidean_error(prd, yts), tmp


def SVM(xtr, ytr, xts, yts):
    start = time()
    SVR_RBF = MultiOutputRegressor(SVR(verbose=0, kernel='rbf', C=23.5, epsilon=0.01, gamma=0.1))
    SVR_RBF.fit(xtr, ytr)

    tmp = time() - start
    prd = SVR_RBF.predict(xts)
    return mean_euclidean_error(prd, yts), tmp


def ETR(xtr, ytr, xts, yts=None):
    start = time()
    etr = ExtraTreesRegressor(max_depth=10, n_estimators=500,
                              max_features=10,
                              min_samples_split=2,
                              bootstrap=False,
                              max_leaf_nodes=None,
                              random_state=False, verbose=False)
    etr.fit(xtr, ytr)
    tmp = time() - start
    prd = etr.predict(xts)
    if yts is None:
        return etr
    else:
        return mean_euclidean_error(prd, yts), etr, tmp


def main():
    X_train, Y_train = getTrainData(CUP1, '1:11', '11:13', ',')
    X_test, Y_test = getTrainData(CUP1, '1:11', '11:13', ',')

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if len(sys.argv) == 1:
        mlp, mlp_time = MLP(X_train, Y_train, X_test, Y_test)
        svr, svr_time = SVM(X_train, Y_train, X_test, Y_test)
        etr_err, etr_model, etr_time = ETR(X_train, Y_train, X_test, Y_test)

        print("MEE Error MLP: %.3f" % mlp)
        print("Time MLP: %.3f" % mlp_time)
        print("MEE Error SVM: %.3f" % svr)
        print("Time SVM: %.3f" % svr_time)
        print("MEE Error ETR: %.3f" % etr_err)
        print("Time ETR: %.3f" % etr_time)

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'predict':
        X_All, Y_All = getTrainData(ALL_DATA, '1:11', '11:13', ',')
        X_blind = getBlindData()

        scaler1 = StandardScaler()
        scaler1.fit(X_All)
        X_All = scaler1.transform(X_All)
        X_blind = scaler1.transform(X_blind)

        etr_model = ETR(X_All, Y_All, X_blind)
        predicted = etr_model.predict(X_blind)
        saveResultsOnCSV(predicted, 'BarBer_ML-CUP18-TS')

        # fig = plt.figure()
        # plt.plot(predicted[:, 0], predicted[:, 1], 'ro')
        # plt.show()


if __name__ == '__main__':
    main()
